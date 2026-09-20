"""Position sizing, correlation exposure filtering, and the drawdown kill-switch.

These are the "non-negotiable" rules from the source strategy design:
  - max 1% of equity risked per trade, sized off ATR-based stop distance
  - block a new position if a highly correlated instrument already has an open
    position in the same direction (avoid doubling up on the same bet)
  - if total equity drops 10% from its peak, flatten everything and halt until
    a human reviews it
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import RiskConfig
from ..timeutil import utc_now_iso

logger = logging.getLogger("fable_bot.risk.manager")


def correlation_matrix(price_history: dict[str, pd.DataFrame], window: int = 60) -> pd.DataFrame:
    """Pairwise correlation of daily returns across the instrument universe."""
    returns = pd.DataFrame({
        symbol: df["close"].pct_change() for symbol, df in price_history.items()
    })
    return returns.tail(window).corr()


@dataclass
class DrawdownMonitor:
    """Peak-to-trough equity kill-switch, durable across processes.

    The `path` is what makes this real. The unattended runner is a ONE-SHOT
    process: the OS scheduler starts a fresh interpreter every session and it
    exits when the cycle ends. An in-memory peak therefore resets to *today's*
    equity every morning, and a drawdown that builds over days -- the only kind
    a daily-bar strategy can actually have -- becomes undetectable. That is
    exactly what happened between 2026-08-25 (peak 117,148) and 2026-09-11
    (113,662): a 3% slide that the switch never saw, and could not have seen,
    because each run's peak was that run's opening equity.

    Persisting peak_equity and halted to memory/ fixes it. `halted` is sticky
    on purpose: once tripped it survives every restart until a human calls
    manual_reset(), because an auto-clearing kill-switch is not a kill-switch.
    """

    max_drawdown_pct: float
    peak_equity: float
    halted: bool = False
    path: Path | None = None

    @classmethod
    def load(cls, path: Path, *, max_drawdown_pct: float, current_equity: float) -> "DrawdownMonitor":
        """Restore the durable peak, or start a new record at current equity.

        Fail-soft: unreadable or malformed state falls back to current equity
        rather than refusing to run. The consequence of a lost peak is one
        session of under-protection, which is strictly better than a bot that
        cannot start because a JSON file got truncated by a hard kill.
        """
        peak, halted = current_equity, False
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            peak = max(float(stored.get("peak_equity", current_equity)), current_equity)
            halted = bool(stored.get("halted", False))
        except (OSError, ValueError, TypeError):
            pass
        monitor = cls(max_drawdown_pct=max_drawdown_pct, peak_equity=peak,
                      halted=halted, path=path)
        monitor._save()
        return monitor

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-replace: a hard kill mid-write must not leave a torn
            # file where the peak used to be.
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps({
                "peak_equity": self.peak_equity,
                "halted": self.halted,
                "updated": utc_now_iso(),
            }, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:  # noqa: BLE001 - persistence is protection, never a gate
            logger.warning("could not persist drawdown state to %s: %s", self.path, exc)

    def update(self, equity: float) -> bool:
        """Update peak equity and return True if the kill-switch should trigger now."""
        self.peak_equity = max(self.peak_equity, equity)
        drawdown_pct = (self.peak_equity - equity) / self.peak_equity * 100
        if drawdown_pct >= self.max_drawdown_pct:
            self.halted = True
        self._save()
        return self.halted

    def drawdown_pct(self, equity: float) -> float:
        """Current peak-to-trough drawdown, for reporting."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - equity) / self.peak_equity * 100

    def manual_reset(self, *, new_peak: float | None = None) -> None:
        """Explicit human action required to resume trading after a halt."""
        self.halted = False
        if new_peak is not None:
            self.peak_equity = new_peak
        self._save()


@dataclass
class RiskManager:
    config: RiskConfig
    correlations: pd.DataFrame = field(default_factory=pd.DataFrame)

    def position_size(self, equity: float, entry_price: float, stop_price: float,
                       point_value: float = 1.0) -> float:
        """Units sized so a stop-out loses exactly max_risk_per_trade_pct of equity.

        point_value is the cash change per 1.00 of price movement, per unit. It
        is 1.0 for shares and ETFs -- one SPY share moves $1 when SPY moves
        $1 -- which is why it defaults to 1.0 and the equity universe is
        unaffected.

        Futures are the reason it exists. One E-mini S&P contract moves $50 per
        index point, so sizing off price distance alone would return 50x too
        many contracts: a 146-point stop on a $111k account would come out as
        7 contracts risking ~$51k (46% of equity) instead of the 1% intended.
        """
        risk_amount = equity * (self.config.max_risk_per_trade_pct / 100)
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0 or point_value <= 0:
            return 0.0
        return risk_amount / (stop_distance * point_value)

    def is_blocked_by_correlation(self, candidate_symbol: str, candidate_side: str,
                                   open_positions: dict[str, str]) -> tuple[bool, str]:
        """Block a new same-direction position if it's highly correlated with one already open."""
        if self.correlations.empty or candidate_symbol not in self.correlations:
            return False, ""
        for symbol, side in open_positions.items():
            if symbol == candidate_symbol or side != candidate_side:
                continue
            if symbol not in self.correlations:
                continue
            corr = self.correlations.loc[candidate_symbol, symbol]
            if pd.notna(corr) and abs(corr) >= self.config.correlation_block_threshold:
                return True, f"blocked: {candidate_symbol}/{symbol} correlation {corr:.2f} >= threshold"
        return False, ""


def sizing_sanity(qty: float, price: float, stop_price: float, equity: float,
                  point_value: float, bounds: dict) -> tuple[bool, str, dict]:
    """Magnitude guardrail for a sized order, learned from the 2026-08 incident.

    The 1%-sizing math can produce a formally-correct but wildly wrong order
    when an input is in the wrong units (EURJPY sized to 628 units next to
    EURUSD 85000). This check can't see currency units, but it can refuse
    orders whose risk or notional lands outside sane bounds, and it records
    the computed magnitudes so the reviewer has evidence either way.

    Returns (ok, reason, metrics). Bounds come from the active
    sizing_notional_bounds lessons (see lessons.py) with defaults applied there.
    """
    stop_distance = abs(price - stop_price)
    risk_amount = qty * stop_distance * point_value
    notional = qty * price * point_value
    risk_pct = (risk_amount / equity * 100) if equity > 0 else float("inf")
    notional_pct = (notional / equity * 100) if equity > 0 else float("inf")

    metrics = {"risk_amount": round(risk_amount, 2), "risk_pct": round(risk_pct, 4),
               "notional": round(notional, 2), "notional_pct_equity": round(notional_pct, 4)}

    risk_min = float(bounds.get("risk_pct_min", 0.25))
    risk_max = float(bounds.get("risk_pct_max", 1.5))
    max_notional_pct = float(bounds.get("max_notional_pct_equity", 2.0)) * 100
    max_notional = equity * float(bounds.get("max_notional_pct_equity", 2.0))

    # The $100 dust floor is an ABSOLUTE number written for a six-figure
    # account, and on a small account it silently becomes a total ban: at $25
    # equity the floor ($100) sits above the notional ceiling (200% = $50), so
    # the admissible window is empty and every order -- any size, any symbol --
    # is refused. Verified against the live bounds on 2026-09-20.
    #
    # The floor's actual job is "don't send dust", which is a statement about
    # proportion, not dollars. Taking the lesser of the absolute floor and a
    # fraction of equity keeps the six-figure behaviour identical (min(100,
    # 10% of 113k) = 100) while letting a small account trade at all.
    floor_pct = float(bounds.get("min_notional_pct_equity", 0.10))
    min_notional = min(float(bounds.get("min_notional_usd", 100.0)), equity * floor_pct)
    metrics["min_notional_applied"] = round(min_notional, 2)

    if min_notional > max_notional:
        return False, (f"bounds are unsatisfiable at equity {equity:,.2f}: floor "
                       f"{min_notional:,.2f} exceeds ceiling {max_notional:,.2f}"), metrics
    if notional < min_notional:
        return False, f"notional {notional:,.2f} below minimum {min_notional:,.2f}", metrics
    if notional_pct > max_notional_pct:
        return False, (f"notional {notional:,.0f} is {notional_pct:.0f}% of equity "
                       f"(limit {max_notional_pct:.0f}%)"), metrics
    if risk_pct > risk_max:
        return False, f"risk {risk_pct:.2f}% of equity exceeds lesson bound {risk_max:.2f}%", metrics
    if risk_pct < risk_min:
        return False, f"risk {risk_pct:.2f}% of equity below lesson bound {risk_min:.2f}%", metrics
    return True, "", metrics


def portfolio_gate(*, open_positions: int, risk_committed: float, equity: float,
                   config: RiskConfig) -> tuple[bool, str]:
    """Refuse a new entry that would breach a book-wide limit.

    Per-trade sizing alone does not bound the book. Each entry is sized to risk
    1% and nothing counted how many fired together, so 2026-09-08 opened eight
    positions in two minutes (~8% of equity at risk in one burst) and
    2026-09-11 carried seven at once. Correlation blocking did not help: it
    only fires when a correlation matrix has both symbols, and across a
    30-instrument mixed universe most pairs never qualified.

    Two independent ceilings, because they fail differently:
      * open_positions  -- how much of the book is exposed at all
      * risk_committed  -- how much NEW risk this run has already sent, which
        is what turns a quiet morning into an eight-order burst

    Returns (allowed, reason).
    """
    if config.max_concurrent_positions > 0 and open_positions >= config.max_concurrent_positions:
        return False, (f"portfolio cap: {open_positions} position(s) already open "
                       f"(limit {config.max_concurrent_positions})")
    if equity > 0 and config.max_new_risk_per_run_pct > 0:
        used_pct = risk_committed / equity * 100
        if used_pct >= config.max_new_risk_per_run_pct:
            return False, (f"portfolio cap: {used_pct:.2f}% of equity already risked this run "
                           f"(limit {config.max_new_risk_per_run_pct:.2f}%)")
    return True, ""
