"""Can this account actually trade this instrument? The arithmetic, stated.

Built for the $25 question. The 1%-per-trade rule is a statement about
proportion; a minimum lot size is a statement about dollars. On a small
account they collide, and the collision needs to be seen in numbers before
any money goes in -- not discovered as a margin call.

For each instrument this reports three quantities and the gap between them:

  ideal_qty   -- what 1% risk sizes to, given the stop distance
  min_qty     -- the smallest order the venue accepts
  risk at min -- what that minimum order actually risks, as % of equity

When ideal_qty < min_qty the rule cannot be honoured; the only choices are
to accept "risk at min" or to not trade the instrument. This module does not
choose. It also reports margin at the minimum lot, because an order the
account cannot MARGIN is rejected before risk is even a question.

Venue parameters default to what TradingView Paper Trading (OANDA feed)
reported on 2026-09-20: 50x leverage, 1,000-unit FX increments, 1-ounce
metals. A live broker will differ -- pass its numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

from .intel.fmt import px


@dataclass(frozen=True)
class VenueSpec:
    symbol: str
    price: float
    min_qty: float
    leverage: float
    # Cash change per 1.00 of price movement per unit. 1.0 for FX quoted in
    # USD (EURUSD, AUDUSD, XAUUSD): one unit moving 1.00 moves $1.00.
    point_value: float = 1.0


@dataclass(frozen=True)
class SizingVerdict:
    spec: VenueSpec
    equity: float
    stop_distance: float
    risk_pct_target: float
    ideal_qty: float
    min_qty: float
    risk_at_min: float
    risk_pct_at_min: float
    margin_at_min: float
    margin_pct_at_min: float
    notional_at_min: float

    @property
    def can_margin(self) -> bool:
        return self.margin_at_min <= self.equity

    @property
    def meets_rule(self) -> bool:
        return self.ideal_qty >= self.min_qty

    @property
    def verdict(self) -> str:
        if not self.can_margin:
            return "CANNOT OPEN -- minimum lot needs more margin than the account holds"
        if self.meets_rule:
            return "OK -- the 1% rule can be honoured"
        return (f"RULE BREAK -- smallest order risks {self.risk_pct_at_min:.1f}% of equity "
                f"({self.risk_pct_at_min / self.risk_pct_target:.0f}x the {self.risk_pct_target:g}% target)")

    def lines(self) -> list[str]:
        s = self.spec
        return [
            f"{s.symbol} @ {px(s.price)}  (min lot {s.min_qty:g}, {s.leverage:g}x)",
            f"  stop distance {px(self.stop_distance, s.price)}  ->  1% sizes to {self.ideal_qty:.4f} units",
            f"  at the minimum lot: risk ${self.risk_at_min:,.2f} ({self.risk_pct_at_min:.1f}%), "
            f"notional ${self.notional_at_min:,.2f}, margin ${self.margin_at_min:,.2f} "
            f"({self.margin_pct_at_min:.0f}% of equity)",
            f"  {self.verdict}",
        ]


def size_check(spec: VenueSpec, *, equity: float, stop_distance: float,
               risk_pct: float = 1.0) -> SizingVerdict:
    risk_amount = equity * risk_pct / 100
    per_unit_risk = stop_distance * spec.point_value
    ideal = risk_amount / per_unit_risk if per_unit_risk > 0 else 0.0
    risk_min = spec.min_qty * per_unit_risk
    notional_min = spec.min_qty * spec.price * spec.point_value
    margin_min = notional_min / spec.leverage if spec.leverage > 0 else notional_min
    return SizingVerdict(
        spec=spec, equity=equity, stop_distance=stop_distance, risk_pct_target=risk_pct,
        ideal_qty=ideal, min_qty=spec.min_qty,
        risk_at_min=risk_min, risk_pct_at_min=(risk_min / equity * 100) if equity else float("inf"),
        margin_at_min=margin_min, margin_pct_at_min=(margin_min / equity * 100) if equity else float("inf"),
        notional_at_min=notional_min,
    )


# What TradingView Paper (OANDA) reported on 2026-09-20. Override per broker.
PAPER_MIN_QTY = {"XAUUSD": 1.0, "EURUSD": 1000.0, "AUDUSD": 1000.0}
PAPER_LEVERAGE = 50.0
