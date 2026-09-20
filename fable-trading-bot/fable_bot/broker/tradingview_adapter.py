"""TradingView Paper Trading implementation of BrokerAdapter.

Routes orders to TradingView's built-in Paper Trading simulator instead of
Alpaca. The appeal is that it needs no API keys and no signup: if you can see
the Trading panel in TradingView Desktop, the bot can trade through it.

How it talks to TradingView
---------------------------
TradingView has no public trading REST API. What it does have is a Broker API
exposed inside the desktop app's page context, which the sibling
`tradingview-mcp` project drives over the Chrome DevTools Protocol. Rather than
reimplement CDP in Python, this adapter shells out to that project's CLI, which
prints JSON on stdout:

    node <tradingview-mcp>/src/cli/index.js trade account
    node <tradingview-mcp>/src/cli/index.js trade buy AMEX:SPY --qty 12 --stop-loss 602.15

One subprocess per broker call is not free, but this is a daily-bar strategy
that places a handful of orders per day, so the cost is irrelevant next to
keeping a single implementation of the CDP plumbing.

Requirements: TradingView Desktop running with CDP enabled (`tv_launch`, or
`node src/cli/index.js launch`) and a broker connected in the Trading panel
(Trade -> Paper Trading -> Connect). `fable_bot.preflight` does both
automatically for scheduled runs.

Safety
------
Two independent guards have to agree before a real-money order is possible:

  * this adapter only passes `--allow-live` when the bot's own two-flag rule
    says live (TRADING_MODE=live *and* ALLOW_LIVE_TRADING=true), and
  * the CLI refuses any broker whose account is not the Paper Trading simulator
    unless it receives that flag.

So the default configuration cannot place a real order even if a live brokerage
happens to be the one connected in TradingView -- it errors out instead.
"""
from __future__ import annotations

import json
import logging
import math
import subprocess
import time
from pathlib import Path

from ..config import BrokerConfig
from .base import AccountSnapshot, BrokerAdapter, HistoryOrder, OrderResult

logger = logging.getLogger("fable_bot.broker.tradingview")


class TradingViewError(RuntimeError):
    """A trade command failed, or TradingView could not be reached.

    venue_rejection marks the one flavour worth learning from: the CLI reached
    TradingView and got a definite "no" for this order (invalid symbol, market
    closed, qty rules). Connectivity failures, timeouts, and unparseable output
    leave it False -- a 30-day symbol halt learned from "the app was down"
    would be the wrong lesson.
    """

    def __init__(self, message: str, *, venue_rejection: bool = False):
        super().__init__(message)
        self.venue_rejection = venue_rejection


class TradingViewBrokerAdapter(BrokerAdapter):
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.cli_path = Path(config.tv_mcp_path) / "src" / "cli" / "index.js"
        if not self.cli_path.exists():
            raise RuntimeError(
                f"tradingview-mcp CLI not found at {self.cli_path}. Set TRADINGVIEW_MCP_PATH "
                "in .env to the tradingview-mcp checkout (see config/.env.example)."
            )

    # ── plumbing ─────────────────────────────────────────────────────────

    def _run(self, *args: str) -> dict:
        """Run one `tv trade ...` command and return its parsed JSON."""
        cmd = [self.config.tv_node_bin, str(self.cli_path), "trade", *args]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.config.tv_timeout_seconds, check=False,
            )
        except FileNotFoundError as exc:
            raise TradingViewError(
                f"Could not run '{self.config.tv_node_bin}'. Node.js must be installed "
                "and on PATH (or set NODE_BIN in .env)."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TradingViewError(
                f"TradingView command timed out after {self.config.tv_timeout_seconds}s: "
                f"{' '.join(args)}"
            ) from exc

        # The CLI reserves exit code 2 for "TradingView isn't reachable", which
        # is worth distinguishing from a rejected order.
        if proc.returncode == 2:
            raise TradingViewError(
                "Cannot reach TradingView Desktop. Start it with CDP enabled "
                "(node <tradingview-mcp>/src/cli/index.js launch) and connect a broker "
                f"in the Trading panel. Details: {(proc.stderr or '').strip()}"
            )

        raw = proc.stdout.strip() or proc.stderr.strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TradingViewError(
                f"Unparseable response from `trade {' '.join(args)}`: {raw[:400]!r}"
            ) from exc

        if not payload.get("success", False):
            detail = payload.get("error") or payload.get("warning") or raw[:400]
            raise TradingViewError(f"trade {' '.join(args)} failed: {detail}",
                                   venue_rejection=True)
        return payload

    def _live_flag(self) -> list[str]:
        """`--allow-live` only when the bot is explicitly, doubly configured for it."""
        return [] if self.config.is_paper else ["--allow-live"]

    # ── symbol / quantity mapping ────────────────────────────────────────

    def venue_symbol(self, instrument) -> str:
        if not instrument.tv_symbol:
            raise TradingViewError(
                f"{instrument.symbol} has no tv_symbol set in config.INSTRUMENTS, so it "
                "cannot be traded on TradingView."
            )
        return instrument.tv_symbol

    def quantize(self, instrument, qty: float) -> float:
        """Round down to the venue's quantity step.

        Rounding down matters: the ETFs trade in whole shares, so a position
        sized at 12.8 shares becomes 12. Rounding up to 13 would risk more than
        the 1% of equity the sizing was built to respect. A result of 0 means
        the risk budget doesn't cover even one unit -- the caller skips it.
        """
        step = instrument.qty_step or 1.0
        rounded = math.floor(abs(qty) / step) * step
        # Re-round to the step's own precision so 3 * 0.0001 doesn't come back
        # as 0.00030000000000000003 and get rejected by the venue.
        decimals = max(0, -math.floor(math.log10(step))) if step < 1 else 0
        return round(rounded, decimals)

    # ── BrokerAdapter ────────────────────────────────────────────────────

    def get_account(self) -> AccountSnapshot:
        data = self._run("account")
        if not data.get("is_paper") and self.config.is_paper:
            raise TradingViewError(
                f"TradingView is connected to \"{data.get('broker')}\", which is not the "
                "Paper Trading simulator, but the bot is configured for paper. Refusing to "
                "trade. Connect Paper Trading in the Trading panel, or set TRADING_MODE=live "
                "and ALLOW_LIVE_TRADING=true if you truly intend real orders."
            )
        return AccountSnapshot(
            equity=float(data["equity"]),
            cash=float(data.get("available_funds", data["equity"])),
            is_paper=bool(data.get("is_paper", True)),
        )

    def get_open_positions(self) -> dict[str, dict]:
        data = self._run("positions")
        return {
            p["symbol"]: {
                "qty": float(p["qty"]),
                "side": p["side"],  # already "long" / "short"
                "avg_entry_price": float(p.get("avg_price") or 0.0),
            }
            for p in data.get("positions", [])
        }

    def submit_order(self, symbol: str, qty: float, side: str, stop_price: float | None = None) -> OrderResult:
        args = [side, symbol, "--qty", str(abs(qty))]
        if stop_price is not None:
            # Attaches a broker-side protective stop, so the 1%-risk stop is
            # enforced between daily signal checks rather than only inside the
            # backtest loop. TradingView creates a real stop order parented to
            # the position for this.
            args += ["--stop-loss", str(round(stop_price, 4))]
        data = self._run(*args, *self._live_flag())
        order = data.get("order") or {}
        return OrderResult(
            symbol=symbol, side=side, qty=qty,
            status=order.get("status", "submitted"),
            broker_order_id=order.get("id"),
        )

    def close_position(self, symbol: str) -> OrderResult:
        data = self._run("close", symbol, *self._live_flag())
        closed = data.get("closed") or {}
        return OrderResult(
            symbol=symbol, side="close",
            qty=float(closed.get("qty") or 0.0), status="closed",
        )

    def close_all(self) -> list[OrderResult]:
        """Flatten everything — used by the drawdown kill-switch.

        Each position is closed individually and failures are collected rather
        than raised, because a kill-switch that aborts halfway through leaves
        exactly the exposure it was meant to remove.
        """
        results: list[OrderResult] = []
        for symbol in self.get_open_positions():
            try:
                results.append(self.close_position(symbol))
            except TradingViewError as exc:
                logger.error("kill-switch could not close %s: %s", symbol, exc)
                results.append(OrderResult(symbol=symbol, side="close", qty=0, status=f"FAILED: {exc}"))
        return results

    # ── order history / verification ─────────────────────────────────────

    def get_order_history(self, limit: int = 100) -> list[HistoryOrder]:
        data = self._run("history", "--limit", str(int(limit)))
        rows: list[HistoryOrder] = []
        for o in data.get("orders", []):
            fill_price = o.get("avg_fill_price")
            rows.append(HistoryOrder(
                id=str(o["id"]) if o.get("id") is not None else None,
                symbol=str(o.get("symbol", "")),
                side=str(o.get("side", "")),
                type=str(o.get("type", "")),
                status=str(o.get("status", "")),
                qty=float(o.get("qty") or 0.0),
                avg_fill_price=float(fill_price) if fill_price is not None else None,
                placed_at=o.get("placed_at") or None,
            ))
        return rows

    def verify_order(self, symbol: str, side: str, qty: float, *,
                     attempts: int = 3, delay_seconds: float = 2.0) -> HistoryOrder | None:
        """Bounded poll for evidence that a submitted order exists at the venue.

        A match is a non-stop row (bracket protective stops are children of our
        own entries and share the symbol) with the same symbol/side/qty whose
        status says it is real: filled, working, or placing. Rejected/cancelled
        rows do not verify -- reconciliation will surface them separately.
        """
        want_side = str(side).strip().lower()
        want_qty = abs(float(qty))
        seen: list[str] = []

        for attempt in range(max(1, int(attempts))):
            if attempt and delay_seconds > 0:
                time.sleep(delay_seconds)
            try:
                rows = self.get_order_history(limit=50)
            except TradingViewError as exc:
                logger.warning("verify_order poll failed for %s: %s", symbol, exc)
                continue
            seen = [f"{r.symbol}/{r.side}/{r.qty:g}/{r.status}/{r.type}" for r in rows[:10]]
            for row in rows:
                if str(row.type).strip().lower() == "stop":
                    continue
                if (_symbols_match(row.symbol, symbol)
                        and str(row.side).strip().lower() == want_side
                        and _qty_matches(row.qty, want_qty)
                        and str(row.status).strip().lower() in ("filled", "working", "placing")):
                    return row

        # Verification exists to turn a lost submission response into evidence.
        # Returning a bare None made a FAILED match indistinguishable from a
        # successful one that simply was not journalled: every order placed on
        # 2026-09-08 and 2026-09-11 recorded verified=false with no explanation
        # anywhere, so the require_verification lesson was silently inert. Log
        # what the venue actually showed, so the next mismatch is diagnosable.
        logger.warning(
            "verify_order found no match for %s %s qty=%g after %d attempt(s); "
            "most recent history rows: %s",
            symbol, want_side, want_qty, max(1, int(attempts)), seen or "<none>",
        )
        return None


def _symbols_match(row_symbol: str, wanted: str) -> bool:
    """Compare venue symbols tolerantly.

    The bot asks for an exchange-qualified ticker ("AMEX:GLD"); a venue may
    echo the bare ticker ("GLD"), a different exchange prefix, or a different
    case. An exact == made verification depend on cosmetic agreement that
    nothing guarantees, so compare the part after the colon, case-insensitively.
    """
    a = str(row_symbol or "").strip().upper()
    b = str(wanted or "").strip().upper()
    if not a or not b:
        return False
    return a == b or a.rsplit(":", 1)[-1] == b.rsplit(":", 1)[-1]


def _qty_matches(row_qty: float, wanted: float, *, rel_tol: float = 1e-6) -> bool:
    """Quantity equality that survives float and venue rounding.

    The old check was an absolute 1e-9, which is tighter than the precision a
    venue reports for a 300,000-unit FX order -- a rounded echo of the same
    order failed to verify.
    """
    try:
        a, b = abs(float(row_qty)), abs(float(wanted))
    except (TypeError, ValueError):
        return False
    return abs(a - b) <= max(rel_tol * max(a, b), 1e-9)
