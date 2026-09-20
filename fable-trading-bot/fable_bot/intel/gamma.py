"""Dealer gamma exposure (GEX): the red/green profile, and what it constrains.

WHY THIS EXISTS
---------------
Every strategy in this repo reads price. Price is what the machinery emits.
Gamma exposure reads a constraint on the machinery itself.

An option market maker does not choose to hedge. Having sold or bought options,
they carry delta they are mandated to neutralise, and gamma is the rate at which
that delta changes as spot moves. The sign of their aggregate gamma decides the
direction of their forced flow:

  NET GAMMA POSITIVE (dealers long gamma)
      Spot rises -> their delta rises -> they SELL to re-neutralise.
      Spot falls -> their delta falls -> they BUY.
      Their hedging leans against the move. Realised volatility is suppressed,
      ranges compress, and price gravitates to the strikes holding the most
      open interest. This is the classic "pin".

  NET GAMMA NEGATIVE (dealers short gamma)
      Spot rises -> they BUY. Spot falls -> they SELL.
      Their hedging leans WITH the move. Volatility is amplified, gaps extend,
      and stops cascade. Nearly every violent session happens here.

The level where net gamma crosses zero -- the GAMMA FLIP -- is therefore a
regime boundary, not a support line. Above it, fade extremes. Below it, do not.

This is the closest thing in public data to reading the rulebook the largest
continuous hedging flow is forced to obey. It is not a forecast, and this module
never emits a trade signal. It emits the constraint, and the caller decides.

WHAT THE NUMBERS MEAN
---------------------
Per-strike gamma exposure is reported in dollars of dealer delta bought or sold
per 1% move in spot:

    GEX_strike = gamma * open_interest * contract_size * spot^2 * 0.01

The spot^2 term is not decoration. Gamma is d(delta)/d(spot); converting it to
cash flow needs one factor of spot to turn delta into dollars and another to
turn a percentage move into a price move.

SIGN CONVENTION, STATED PLAINLY
-------------------------------
This module uses the standard naive convention: dealers are assumed LONG calls
and SHORT puts, so call gamma counts positive and put gamma negative. That
assumption is an approximation -- real dealer inventory is not observable from
open interest alone, and on single names it is frequently wrong. It holds up
best on broad, heavily-hedged index and large-ETF underlyings (SPY, QQQ, GLD,
IWM), which is exactly where this repo uses it.

Anyone reading a GEX chart anywhere is looking at this same assumption, usually
without being told. `dealer_convention` records it on every profile so the
assumption travels with the number.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

logger = logging.getLogger("fable_bot.intel.gamma")

# Equity/ETF listed options deliver 100 shares. Index options differ; pass
# contract_size explicitly for those rather than relying on this.
DEFAULT_CONTRACT_SIZE = 100.0

# A risk-free rate this strategy is not sensitive to: gamma's dependence on r
# is third-order next to its dependence on time and moneyness, and using a
# fixed value keeps the profile reproducible rather than silently re-rating
# when a rate feed moves. Override per call if that ever stops being true.
DEFAULT_RISK_FREE_RATE = 0.04

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    """Standard normal density. Hand-rolled: scipy is not a dependency here."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def black_scholes_gamma(spot: float, strike: float, time_to_expiry_years: float,
                        volatility: float, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> float:
    """Black-Scholes gamma. Identical for a call and a put at the same strike.

    Returns 0.0 for degenerate inputs (expired, zero vol, non-positive prices)
    rather than raising: an option chain routinely contains rows with a missing
    or zero implied volatility, and one bad row must not destroy a profile
    built from a thousand good ones.
    """
    # NaN must be rejected explicitly. Every ordering comparison against NaN is
    # False, so a `volatility <= 0` guard alone lets a NaN implied vol straight
    # through, and a single NaN strike turns the whole summed profile into NaN
    # -- which then reads as "short gamma", because `nan >= 0` is also False.
    # A live GLD chain contained exactly such a row.
    if not all(math.isfinite(v) for v in (spot, strike, time_to_expiry_years, volatility)):
        return 0.0
    if spot <= 0 or strike <= 0 or time_to_expiry_years <= 0 or volatility <= 0:
        return 0.0
    sqrt_t = math.sqrt(time_to_expiry_years)
    try:
        d1 = ((math.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2)
               * time_to_expiry_years) / (volatility * sqrt_t))
    except (ValueError, ZeroDivisionError):
        return 0.0
    gamma = _norm_pdf(d1) / (spot * volatility * sqrt_t)
    return gamma if math.isfinite(gamma) else 0.0


@dataclass(frozen=True)
class StrikeGamma:
    """One strike's contribution to dealer gamma, in dollars per 1% spot move."""

    strike: float
    call_oi: float
    put_oi: float
    call_gex: float
    put_gex: float

    @property
    def net_gex(self) -> float:
        return self.call_gex + self.put_gex

    @property
    def total_oi(self) -> float:
        return self.call_oi + self.put_oi


@dataclass(frozen=True)
class GammaProfile:
    """The full picture for one underlying at one moment.

    `by_strike` is the red/green profile: the bars of the mountain. Everything
    else is a summary statistic computed from it.
    """

    symbol: str
    spot: float
    as_of: datetime
    by_strike: list[StrikeGamma]
    net_gex: float
    gamma_flip: float | None
    call_wall: float | None
    put_wall: float | None
    expiries_used: list[str] = field(default_factory=list)
    contracts_used: int = 0
    dealer_convention: str = "long_calls_short_puts"

    @property
    def regime(self) -> str:
        """'long_gamma' (vol suppressed, pinning) or 'short_gamma' (vol amplified)."""
        return "long_gamma" if self.net_gex >= 0 else "short_gamma"

    @property
    def distance_to_flip_pct(self) -> float | None:
        if self.gamma_flip is None or self.spot <= 0:
            return None
        return (self.spot - self.gamma_flip) / self.spot * 100

    def summary(self) -> str:
        """One-paragraph plain reading of the regime, for the daily report."""
        lines = [
            f"{self.symbol} spot {self.spot:,.2f} -- dealer gamma {self.regime.replace('_', ' ').upper()}",
            f"  net GEX {self.net_gex / 1e6:+,.1f}M per 1% move "
            f"({self.contracts_used} contracts across {len(self.expiries_used)} expiries)",
        ]
        if self.gamma_flip is not None:
            side = "above" if self.spot >= self.gamma_flip else "below"
            lines.append(f"  gamma flip {self.gamma_flip:,.2f} -- spot is {side} it "
                         f"({self.distance_to_flip_pct:+.2f}%)")
        else:
            lines.append("  gamma flip: not found inside the scanned price range")
        if self.call_wall is not None:
            lines.append(f"  call wall (largest positive gamma) {self.call_wall:,.2f}")
        if self.put_wall is not None:
            lines.append(f"  put wall (largest negative gamma) {self.put_wall:,.2f}")
        if self.regime == "long_gamma":
            lines.append("  reading: dealer hedging LEANS AGAINST moves. Expect range "
                         "compression and pinning toward high-OI strikes.")
        else:
            lines.append("  reading: dealer hedging LEANS WITH moves. Expect extension, "
                         "gaps and stop cascades; fading extremes is expensive here.")
        return "\n".join(lines)


def _years_to_expiry(expiry: str, now: datetime) -> float:
    """Calendar years until an ISO expiry date, floored at zero.

    Options expire at the close, not at midnight, so a same-day expiry keeps a
    small positive stub rather than collapsing to zero and zeroing its gamma --
    0DTE gamma is the largest and most consequential of the set.
    """
    try:
        expiry_date = date.fromisoformat(str(expiry)[:10])
    except ValueError:
        return 0.0
    expiry_dt = datetime.combine(expiry_date, datetime.min.time(), tzinfo=timezone.utc)
    # 21:00 UTC ~ 16:00 ET close.
    expiry_dt = expiry_dt.replace(hour=21)
    seconds = (expiry_dt - now).total_seconds()
    if seconds <= 0:
        return 0.0
    return seconds / (365.25 * 24 * 3600)


def compute_gamma_profile(
    symbol: str,
    spot: float,
    chains: list[dict],
    *,
    now: datetime | None = None,
    contract_size: float = DEFAULT_CONTRACT_SIZE,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    strike_range_pct: float = 20.0,
    flip_scan_pct: float = 15.0,
    flip_scan_steps: int = 241,
) -> GammaProfile:
    """Build a gamma profile from already-fetched option chains.

    Pure computation, deliberately separated from the network call in
    `fetch_gamma_profile`, so the maths is testable without a live feed and the
    same profile can be rebuilt from a stored chain.

    `chains` is a list of dicts, one per expiry:
        {"expiry": "2026-10-16",
         "calls": [{"strike": 390.0, "open_interest": 1200, "implied_volatility": 0.21}, ...],
         "puts":  [...]}

    `strike_range_pct` discards strikes far from spot. Deep wings carry huge
    open interest but negligible gamma, and including them adds noise to the
    walls without moving net GEX.
    """
    now = now or datetime.now(timezone.utc)
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")

    lo = spot * (1 - strike_range_pct / 100)
    hi = spot * (1 + strike_range_pct / 100)

    call_gex: dict[float, float] = {}
    put_gex: dict[float, float] = {}
    call_oi: dict[float, float] = {}
    put_oi: dict[float, float] = {}
    expiries_used: list[str] = []
    contracts_used = 0

    for chain in chains:
        expiry = str(chain.get("expiry", ""))
        tte = _years_to_expiry(expiry, now)
        if tte <= 0:
            continue
        used_this_expiry = False
        for kind, gex_acc, oi_acc, sign in (
            ("calls", call_gex, call_oi, +1.0),
            ("puts", put_gex, put_oi, -1.0),
        ):
            for row in chain.get(kind) or []:
                try:
                    strike = float(row.get("strike"))
                    oi = float(row.get("open_interest") or 0.0)
                    iv = float(row.get("implied_volatility") or 0.0)
                except (TypeError, ValueError):
                    continue
                if not all(math.isfinite(v) for v in (strike, oi, iv)):
                    continue
                if strike < lo or strike > hi or oi <= 0 or iv <= 0:
                    continue
                gamma = black_scholes_gamma(spot, strike, tte, iv, risk_free_rate)
                if gamma <= 0:
                    continue
                # Dollars of dealer delta per 1% spot move.
                gex = sign * gamma * oi * contract_size * spot * spot * 0.01
                gex_acc[strike] = gex_acc.get(strike, 0.0) + gex
                oi_acc[strike] = oi_acc.get(strike, 0.0) + oi
                contracts_used += 1
                used_this_expiry = True
        if used_this_expiry:
            expiries_used.append(expiry)

    strikes = sorted(set(call_gex) | set(put_gex))
    by_strike = [
        StrikeGamma(
            strike=k,
            call_oi=call_oi.get(k, 0.0),
            put_oi=put_oi.get(k, 0.0),
            call_gex=call_gex.get(k, 0.0),
            put_gex=put_gex.get(k, 0.0),
        )
        for k in strikes
    ]
    net_gex = sum(s.net_gex for s in by_strike)

    call_wall = max(by_strike, key=lambda s: s.net_gex).strike if by_strike else None
    put_wall = min(by_strike, key=lambda s: s.net_gex).strike if by_strike else None
    # A "wall" only means something if it is actually signed that way.
    if call_wall is not None and max(s.net_gex for s in by_strike) <= 0:
        call_wall = None
    if put_wall is not None and min(s.net_gex for s in by_strike) >= 0:
        put_wall = None

    flip = _find_gamma_flip(chains, spot, now, contract_size, risk_free_rate,
                            flip_scan_pct, flip_scan_steps, lo, hi)

    return GammaProfile(
        symbol=symbol, spot=spot, as_of=now, by_strike=by_strike, net_gex=net_gex,
        gamma_flip=flip, call_wall=call_wall, put_wall=put_wall,
        expiries_used=expiries_used, contracts_used=contracts_used,
    )


def _net_gex_at(chains: list[dict], hypothetical_spot: float, now: datetime,
                contract_size: float, risk_free_rate: float,
                lo: float, hi: float) -> float:
    """Net dealer GEX if spot were `hypothetical_spot`, open interest unchanged.

    Holding open interest fixed while moving spot is the standard construction
    and its limitation: real positioning changes as price moves. It answers
    "given today's book, where does the sign flip", not "where will the sign
    flip tomorrow".
    """
    total = 0.0
    for chain in chains:
        tte = _years_to_expiry(str(chain.get("expiry", "")), now)
        if tte <= 0:
            continue
        for kind, sign in (("calls", +1.0), ("puts", -1.0)):
            for row in chain.get(kind) or []:
                try:
                    strike = float(row.get("strike"))
                    oi = float(row.get("open_interest") or 0.0)
                    iv = float(row.get("implied_volatility") or 0.0)
                except (TypeError, ValueError):
                    continue
                if not all(math.isfinite(v) for v in (strike, oi, iv)):
                    continue
                if strike < lo or strike > hi or oi <= 0 or iv <= 0:
                    continue
                gamma = black_scholes_gamma(hypothetical_spot, strike, tte, iv, risk_free_rate)
                if gamma <= 0:
                    continue
                total += (sign * gamma * oi * contract_size
                          * hypothetical_spot * hypothetical_spot * 0.01)
    return total


def _find_gamma_flip(chains: list[dict], spot: float, now: datetime,
                     contract_size: float, risk_free_rate: float,
                     scan_pct: float, steps: int,
                     lo: float, hi: float) -> float | None:
    """Price at which net dealer gamma crosses zero, by scan then bisection.

    Returns None when the sign never changes across the scanned band, which is
    a real and common answer: a book can be long gamma everywhere nearby. A
    fabricated level would be worse than no level.
    """
    if steps < 3:
        return None
    low = spot * (1 - scan_pct / 100)
    high = spot * (1 + scan_pct / 100)
    step = (high - low) / (steps - 1)

    prev_price = low
    prev_val = _net_gex_at(chains, low, now, contract_size, risk_free_rate, lo, hi)
    for i in range(1, steps):
        price = low + i * step
        val = _net_gex_at(chains, price, now, contract_size, risk_free_rate, lo, hi)
        if prev_val == 0.0:
            return prev_price
        if (prev_val < 0) != (val < 0):
            # Bracketed: bisect for a level worth quoting to two decimals.
            a, fa, b = prev_price, prev_val, price
            for _ in range(40):
                mid = (a + b) / 2
                fm = _net_gex_at(chains, mid, now, contract_size, risk_free_rate, lo, hi)
                if fm == 0.0:
                    return mid
                if (fa < 0) != (fm < 0):
                    b = mid
                else:
                    a, fa = mid, fm
            return (a + b) / 2
        prev_price, prev_val = price, val
    return None
