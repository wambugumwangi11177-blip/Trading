"""Make sure TradingView is actually reachable before a run trusts it.

The failure this exists to prevent is silent: TradingView Desktop is closed, the
CDP socket refuses, and every order in the cycle raises. Interactively that is
obvious. On a scheduled run at the opening bell it is a day of no trades that
looks exactly like a day of no signals.

So the sequence is: check CDP, cold-launch the desktop app if it is down, wait
for the page to come up, then wait for the Trading panel to re-attach its broker
-- a fresh launch reports CDP ready well before the broker connection is usable.
Only then does the run proceed, and the paper/live identity is verified one last
time here rather than trusting configuration.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .config import BrokerConfig

logger = logging.getLogger("fable_bot.preflight")

CDP_DOWN_EXIT_CODE = 2  # the CLI reserves this for "TradingView isn't reachable"

# The exact label of TradingView's own simulator in the broker chooser. Matched
# exactly, never fuzzily: the chooser sits real brokerages next to it.
PAPER_BROKER_LABEL = "Paper Trading"


@dataclass
class PreflightResult:
    ok: bool
    detail: str
    cdp_connected: bool = False
    broker_connected: bool = False
    is_paper: bool | None = None
    equity: float | None = None
    broker_name: str | None = None
    launched: bool = False

    def as_event(self) -> dict:
        return {
            "ok": self.ok, "detail": self.detail, "cdp_connected": self.cdp_connected,
            "broker_connected": self.broker_connected, "is_paper": self.is_paper,
            "equity": self.equity, "broker": self.broker_name, "launched": self.launched,
        }


class Preflight:
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.cli_path = Path(config.tv_mcp_path) / "src" / "cli" / "index.js"

    def _run(self, *args: str, timeout: float = 60.0) -> tuple[int, dict | None]:
        """Run a tradingview-mcp CLI command, returning (exit_code, parsed_json)."""
        cmd = [self.config.tv_node_bin, str(self.cli_path), *args]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except FileNotFoundError:
            return 127, {"error": f"'{self.config.tv_node_bin}' not on PATH -- install Node.js or set NODE_BIN"}
        except subprocess.TimeoutExpired:
            return 124, {"error": f"`{' '.join(args)}` timed out after {timeout}s"}

        raw = (proc.stdout or "").strip() or (proc.stderr or "").strip()
        try:
            return proc.returncode, json.loads(raw)
        except json.JSONDecodeError:
            return proc.returncode, {"error": raw[:400]} if raw else None

    # ── individual checks ────────────────────────────────────────────────

    def status(self) -> dict:
        code, payload = self._run("status", timeout=30)
        if code == 0 and (payload or {}).get("success"):
            return payload or {}
        return {}

    def cdp_up(self) -> bool:
        return bool(self.status())

    def chart_ready(self) -> bool:
        """CDP answering *and* the charting API actually loaded.

        These are not the same thing, which is the trap. A cold launch that
        fails to resolve tradingview.com still serves its offline error page
        over CDP, so `cdp_connected` comes back true while nothing works.
        """
        return bool(self.status().get("api_available"))

    def on_error_page(self) -> bool:
        state = self.status()
        blob = f"{state.get('target_url', '')} {state.get('target_title', '')}".lower()
        return "error-view" in blob or "something went wrong" in blob or "load-failed" in blob

    def recover_error_page(self) -> bool:
        """Click the offline page's "Try again". Observed to be enough whenever
        the network was simply not up yet when the app started."""
        hits = self._find_text("Try again", buttons_only=True)
        if not hits:
            return False
        logger.info("TradingView is on its offline error page -- clicking 'Try again'.")
        return self._click_xy(hits[0]["x"], hits[0]["y"])

    def launch(self) -> tuple[bool, str]:
        """Cold-start TradingView Desktop with CDP enabled."""
        code, payload = self._run("launch", timeout=120)
        if code == 0 and (payload or {}).get("success"):
            return True, "launched"
        return False, str((payload or {}).get("error") or f"launch exited {code}")

    def account(self) -> tuple[bool, dict]:
        code, payload = self._run("trade", "account", timeout=self.config.tv_timeout_seconds)
        if code == 0 and (payload or {}).get("success"):
            return True, payload or {}
        return False, payload or {"error": f"`trade account` exited {code}"}

    # ── UI automation, used only to re-attach the paper broker ───────────

    def _ui_eval(self, code: str):
        exit_code, payload = self._run("ui", "eval", "--code", code, timeout=45)
        if exit_code == 0 and (payload or {}).get("success"):
            return (payload or {}).get("result")
        return None

    def _click_xy(self, x: int, y: int) -> bool:
        exit_code, payload = self._run("ui", "mouse", str(x), str(y), timeout=30)
        return exit_code == 0 and bool((payload or {}).get("success"))

    def _press_escape(self) -> bool:
        exit_code, payload = self._run("ui", "keyboard", "Escape", timeout=30)
        return exit_code == 0 and bool((payload or {}).get("success"))

    def _panel_state(self) -> dict:
        """One look at the Trading panel: what is blocking, what is clickable.

        Returns three facts, each scoped by the exact label "Paper Trading":

          blocking  -- a sizeable dialog is open that does NOT contain that
                       label: some other broker's connect dialog (Capital.com
                       and easyMarkets have both been observed popping here,
                       uninvited). Everything under it must wait until it is
                       gone; clicking *through* it is how a real-broker
                       Connect gets pressed by accident.
          connect   -- centre of a Connect button that lives in the SAME
                       dialog or bottom panel as the "Paper Trading" label --
                       the collapsed panel's reconnect button, or the confirm
                       dialog the chooser opens. A Connect button in any other
                       container is deliberately not returned.
          paper     -- centres of every visible element labelled exactly
                       "Paper Trading" (the chooser tile among them).
        """
        return self._ui_eval("""
        (function(){
          var LABEL = 'Paper Trading';
          function visible(el){ var r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0 && !!el.offsetParent; }
          function containerOf(el){
            while (el && el !== document.body){
              if (el.matches('[class*=modal],[class*=dialog],[role=dialog],[class*=layout__area--bottom]')) return el;
              el = el.parentElement;
            }
            return null;
          }
          var dialogs = [];
          var dlgEls = document.querySelectorAll('[class*=modal],[class*=dialog],[role=dialog]');
          for (var i = 0; i < dlgEls.length; i++) {
            var d = dlgEls[i], r = d.getBoundingClientRect();
            if (r.width >= 200 && r.height >= 100 && d.offsetParent) dialogs.push(d);
          }
          var papers = [], connects = [];
          var nodes = document.querySelectorAll('*');
          for (var j = 0; j < nodes.length; j++) {
            var el = nodes[j];
            if (!visible(el)) continue;
            var text = (el.textContent || '').trim();
            var isBtn = el.tagName === 'BUTTON' || el.getAttribute('role') === 'button';
            if (el.children.length === 0 && text === LABEL) papers.push(el);
            if (isBtn && text === 'Connect') connects.push(el);
          }
          var blocking = false;
          for (var k = 0; k < dialogs.length; k++) {
            var hasPaper = papers.some(function(p){ return dialogs[k].contains(p); });
            if (!hasPaper) { blocking = true; break; }
          }
          var connect = null;
          for (var c = 0; c < connects.length && !connect; c++) {
            var cont = containerOf(connects[c]);
            if (!cont) continue;
            if (papers.some(function(p){ return cont.contains(p); })) {
              var r2 = connects[c].getBoundingClientRect();
              connect = { x: Math.round(r2.x + r2.width / 2), y: Math.round(r2.y + r2.height / 2) };
            }
          }
          var paper = papers.map(function(p){
            var r3 = p.getBoundingClientRect();
            return { x: Math.round(r3.x + r3.width / 2), y: Math.round(r3.y + r3.height / 2) };
          });
          return { blocking: blocking, connect: connect, paper: paper };
        })()
        """) or {"blocking": False, "connect": None, "paper": []}

    def _find_text(self, text: str, *, buttons_only: bool = False) -> list[dict]:
        """Centre coordinates of every visible element whose exact text matches.

        The CLI's own text selector misses these -- TradingView nests the label
        in a bare <span> inside the clickable tile -- so this walks the DOM and
        returns viewport coordinates for a real input event instead.
        """
        selector = "button, [role=button]" if buttons_only else "*"
        js = """
        (function(){
          var want = %s;
          var out = [];
          var nodes = document.querySelectorAll(%s);
          for (var i = 0; i < nodes.length; i++) {
            var el = nodes[i];
            if (!%s && el.children.length) continue;
            if ((el.textContent || '').trim() !== want) continue;
            var r = el.getBoundingClientRect();
            if (r.width <= 0 || r.height <= 0) continue;
            out.push({ x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2) });
          }
          return out;
        })()
        """ % (json.dumps(text), json.dumps(selector), "true" if buttons_only else "false")
        return self._ui_eval(js) or []

    def _connect_programmatic(self) -> bool:
        """Ask the chart's own broker registry for the simulator, no clicks.

        `connectBrokerById('Paper')` is what the broker picker itself calls
        when its Paper Trading tile is chosen; using it directly skips every
        dialog, coordinate and timing hazard the UI walk has to survive. The
        id is exact, and the is-paper verification after connecting still
        applies -- as does the rule that this is only ever called while NO
        broker is connected, so it can never switch a real one away.
        """
        result = self._ui_eval("""
        (function(){
          var t = window.TradingViewApi && window.TradingViewApi.trading && window.TradingViewApi.trading();
          if (!t || !t.brokerSelectManager || typeof t.brokerSelectManager.connectBrokerById !== 'function') {
            return 'unavailable';
          }
          try { t.brokerSelectManager.connectBrokerById('Paper'); return 'requested'; }
          catch (e) { return 'error: ' + e.message; }
        })()
        """)
        ok = result == "requested"
        logger.info("programmatic Paper Trading connect: %s", result)
        return ok

    def connect_paper_broker(self, *, timeout: float = 180.0, poll_seconds: float = 3.0) -> tuple[bool, str]:
        """Get a broker answering, starting from the least invasive path.

        Three panel states have been observed behind a failing `trade account`,
        and each needs the opposite of the others:

          * hydrating -- a cold launch restores the layout's Paper Trading
            connection on its own within a couple of minutes, if nobody
            interferes;
          * blocked -- the Trade toolbar button pops a connect dialog for some
            other broker (Capital.com and easyMarkets have both been seen,
            uninvited), which BLOCKS hydration for as long as it sits there,
            and whose Connect button must never, ever be pressed;
          * degraded -- hours or days later the connection drops to a collapsed
            panel whose lone Connect button reconnects the still-selected
            Paper Trading broker.

        So: first ask the chart's own broker registry to connect the simulator
        and give that a quiet window to land alongside natural hydration.
        Only if neither comes through does the UI get walked, and everything
        in that walk is scoped by the exact label "Paper Trading": a dialog
        without it gets dismissed (Escape is safe here -- preflight runs
        before any orders exist), and a Connect button is pressed only when it
        shares its dialog or panel with that label. The label is matched
        exactly, because the chooser sits real brokerages next to the
        simulator and a fuzzy match that wandered onto one of those is the
        single worst thing this file could do.
        """
        deadline = time.monotonic() + timeout

        # The panel has to be up for the UI walk to see anything.
        code, payload = self._run("ui", "panel", "trading", "open", timeout=30)
        if code != 0 or not (payload or {}).get("success"):
            return False, f"could not open the Trading panel: {(payload or {}).get('error')}"

        # 1. Programmatic connect, then a quiet window for it (and natural
        #    hydration) to land -- no clicking while a connection may be
        #    forming under our hands.
        self._connect_programmatic()
        quiet_until = min(deadline, time.monotonic() + 30.0)
        while time.monotonic() < quiet_until:
            ok, _ = self.account()
            if ok:
                logger.info("broker connected after the programmatic request.")
                return True, "broker connected"
            time.sleep(poll_seconds)

        # 2. UI walk.
        connected_clicks = 0
        last_escape = time.monotonic()
        while time.monotonic() < deadline:
            ok, _ = self.account()
            if ok:
                how = "connected" if connected_clicks else "restored on its own"
                logger.info("broker connection %s.", how)
                return True, "broker connected"

            state = self._panel_state()

            if state.get("blocking"):
                logger.info("a dialog that is not Paper Trading is blocking the panel -- dismissing it.")
                self._press_escape()
                last_escape = time.monotonic()
                time.sleep(poll_seconds)
                continue

            connect = state.get("connect")
            if connect:
                logger.info("clicking the Paper Trading Connect button.")
                self._click_xy(connect["x"], connect["y"])
                connected_clicks += 1
                time.sleep(poll_seconds * 2)
                continue

            paper = state.get("paper") or []
            if paper:
                logger.info("selecting the Paper Trading broker.")
                self._click_xy(paper[0]["x"], paper[0]["y"])
                time.sleep(poll_seconds)
                continue

            # Nothing actionable on screen. A periodic Escape keeps stray
            # dialogs from settling in while the stored connection hydrates,
            # and the panel stays up as its surface.
            if time.monotonic() - last_escape >= 15.0:
                self._press_escape()
                last_escape = time.monotonic()
                self._run("ui", "panel", "trading", "open", timeout=30)
            time.sleep(poll_seconds)

        return False, f"broker did not connect within {timeout:g}s"

    # ── the whole sequence ───────────────────────────────────────────────

    def ensure_ready(
        self,
        *,
        cdp_timeout: float = 120.0,
        broker_timeout: float = 180.0,
        poll_seconds: float = 5.0,
        allow_launch: bool = True,
    ) -> PreflightResult:
        if not self.cli_path.exists():
            return PreflightResult(
                ok=False,
                detail=f"tradingview-mcp CLI not found at {self.cli_path}. Set TRADINGVIEW_MCP_PATH in .env.",
            )

        launched = False
        if not self.cdp_up():
            if not allow_launch:
                return PreflightResult(ok=False, detail="TradingView unreachable and auto-launch disabled.")
            logger.info("CDP down -- launching TradingView Desktop.")
            ok, detail = self.launch()
            if not ok:
                return PreflightResult(ok=False, detail=f"Could not launch TradingView: {detail}")
            launched = True
        else:
            logger.info("CDP already connected.")

        # Wait for the charting API, not merely for CDP. A launch that raced the
        # network serves an offline error page that answers CDP quite happily;
        # "Try again" recovers it once DNS is actually up.
        deadline = time.monotonic() + cdp_timeout
        while time.monotonic() < deadline:
            if self.chart_ready():
                break
            if self.on_error_page():
                self.recover_error_page()
            time.sleep(poll_seconds)
        else:
            state = self.status()
            hint = " (stuck on TradingView's offline error page -- check the network)" if self.on_error_page() else ""
            return PreflightResult(
                ok=False, launched=launched, cdp_connected=bool(state),
                detail=f"TradingView did not finish loading within {cdp_timeout:g}s{hint}.",
            )
        logger.info("chart API ready.")

        # The broker does not re-attach on its own after a cold launch, so try
        # the account first and only drive the UI if it is genuinely absent.
        deadline = time.monotonic() + broker_timeout
        last_error = "no response"
        connect_attempted = False
        while True:
            ok, payload = self.account()
            if ok:
                break
            last_error = str(payload.get("error") or payload)

            if not connect_attempted:
                connect_attempted = True
                connected, detail = self.connect_paper_broker(
                    timeout=max(30.0, min(broker_timeout, deadline - time.monotonic())),
                )
                if connected:
                    continue
                last_error = detail

            if time.monotonic() >= deadline:
                return PreflightResult(
                    ok=False, cdp_connected=True, launched=launched,
                    detail=(f"TradingView is up but no broker answered within {broker_timeout:g}s. "
                            f"Connect one in the Trading panel (Trade -> Paper Trading -> Connect). "
                            f"Last error: {last_error}"),
                )
            time.sleep(poll_seconds)

        is_paper = bool(payload.get("is_paper", False))
        broker_name = payload.get("broker")
        equity = float(payload["equity"]) if payload.get("equity") is not None else None

        # Same rule the adapter enforces, checked before any order is built: a
        # paper-configured bot must never find itself pointed at a real account.
        if self.config.is_paper and not is_paper:
            return PreflightResult(
                ok=False, cdp_connected=True, broker_connected=True, is_paper=False,
                equity=equity, broker_name=broker_name, launched=launched,
                detail=(f'TradingView is connected to "{broker_name}", which is not the Paper Trading '
                        "simulator, but the bot is configured for paper. Refusing to trade."),
            )

        return PreflightResult(
            ok=True, cdp_connected=True, broker_connected=True, is_paper=is_paper,
            equity=equity, broker_name=broker_name, launched=launched,
            detail=f"{broker_name or 'broker'} ready, equity {equity:,.2f}" if equity is not None else "ready",
        )
