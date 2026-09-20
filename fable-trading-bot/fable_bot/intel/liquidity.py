"""Liquidity map: where resting stops are pooled, and whether they've been taken.

THE PREMISE, STATED THE WAY THE USER STATES IT
----------------------------------------------
The chart is produced by code, the code is written by people, and the people
who write it are the ones with the size to move price. Price does not wander
to a level; it is DRIVEN to where the resting orders are, because filling a
large order needs the other side, and the other side is densest where retail
stops cluster.

Stops cluster in predictable places:

  * just above a swing high / just below a swing low  (breakout entries + the
    stops of anyone who shorted the high / bought the low)
  * above EQUAL highs / below EQUAL lows  (two or three touches of the same
    level read as "support" or "resistance" to everyone, so everyone puts the
    stop on the far side of it -- the most obvious pool of all)
  * prior day high/low, prior week high/low, session extremes
  * round numbers

A level is a magnet until it is SWEPT: price trades through it, fills the
pooled orders, and (often) reverses because the order flow that drove it
there is spent. A swept level is no longer a target; the untouched ones are.

This module lists the untouched pools nearest to spot on each side, marks the
ones taken recently, and says nothing about direction. Nearest untouched pool
on each side is where the machinery has a reason to go next. Which one it
goes to first is the trade, and that is not decided here.

Everything is on daily bars and lags by the swing-confirmation window, for
the same look-ahead reason documented in strategies/smc.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..strategies.smc import swing_points
from .fmt import px

# How close two extremes must be, as a fraction of price, to count as EQUAL.
# 0.1% on gold at 4,400 is ~4.4 dollars; on EURUSD at 1.17 it is ~12 pips.
EQUAL_TOLERANCE_PCT = 0.10


@dataclass(frozen=True)
class LiquidityPool:
    price: float
    kind: str            # swing_high | swing_low | equal_highs | equal_lows | pdh | pdl | pwh | pwl
    side: str            # "above" spot (buy-side liquidity) or "below" (sell-side)
    touches: int         # how many bars set this extreme (2+ = equal highs/lows)
    bar: str             # date of the (most recent) bar that set it
    swept: bool          # has price traded through it since?
    distance_pct: float  # signed % from spot (+ above, - below)

    def label(self) -> str:
        tag = {"swing_high": "swing high", "swing_low": "swing low",
               "equal_highs": f"EQUAL HIGHS x{self.touches}",
               "equal_lows": f"EQUAL LOWS x{self.touches}",
               "pdh": "prior day high", "pdl": "prior day low",
               "pwh": "prior week high", "pwl": "prior week low"}[self.kind]
        state = "SWEPT" if self.swept else "untouched"
        return f"{px(self.price):>12}  {tag:<18} {self.distance_pct:+.2f}%  {state}  ({self.bar})"


@dataclass(frozen=True)
class LiquidityMap:
    symbol: str
    spot: float
    as_of_bar: str
    above: list[LiquidityPool] = field(default_factory=list)   # nearest first
    below: list[LiquidityPool] = field(default_factory=list)   # nearest first
    recently_swept: list[LiquidityPool] = field(default_factory=list)

    @property
    def nearest_above(self) -> LiquidityPool | None:
        return next((p for p in self.above if not p.swept), None)

    @property
    def nearest_below(self) -> LiquidityPool | None:
        return next((p for p in self.below if not p.swept), None)

    # Prior-day extremes are always the closest pools on both sides -- they are
    # yesterday's range -- so any "which way is the nearer target" question
    # asked of them comes back "equidistant" every day. Structural pools
    # (swing points, equal highs/lows, prior WEEK) are the ones price gets
    # driven to; these accessors answer the directional question with those.
    STRUCTURAL = ("swing_high", "swing_low", "equal_highs", "equal_lows", "pwh", "pwl")

    @property
    def nearest_structural_above(self) -> LiquidityPool | None:
        return next((p for p in self.above if not p.swept and p.kind in self.STRUCTURAL), None)

    @property
    def nearest_structural_below(self) -> LiquidityPool | None:
        return next((p for p in self.below if not p.swept and p.kind in self.STRUCTURAL), None)

    def summary(self, limit: int = 4) -> str:
        lines = [f"{self.symbol} liquidity map (bar {self.as_of_bar}) -- spot {px(self.spot)}"]
        lines.append("  buy-side pools ABOVE (stops of shorts, breakout buys):")
        for pool in [p for p in self.above if not p.swept][:limit] or []:
            lines.append(f"    {pool.label()}")
        if not any(not p.swept for p in self.above):
            lines.append("    (none untouched inside the scanned range)")
        lines.append("  sell-side pools BELOW (stops of longs, breakdown sells):")
        for pool in [p for p in self.below if not p.swept][:limit] or []:
            lines.append(f"    {pool.label()}")
        if not any(not p.swept for p in self.below):
            lines.append("    (none untouched inside the scanned range)")
        if self.recently_swept:
            lines.append("  taken in the last 5 bars:")
            for pool in self.recently_swept[:limit]:
                lines.append(f"    {pool.label()}")
        na, nb = self.nearest_above, self.nearest_below
        if na and nb:
            lines.append(f"  nearest untouched: {px(na.price)} above ({na.distance_pct:+.2f}%) vs "
                         f"{px(nb.price)} below ({nb.distance_pct:+.2f}%)")
        return "\n".join(lines)


def _cluster_equal(levels: list[tuple[float, str]], tol_pct: float) -> list[tuple[float, int, str]]:
    """Merge extremes within tolerance into (price, touches, latest_bar)."""
    if not levels:
        return []
    ordered = sorted(levels, key=lambda x: x[0])
    clusters: list[list[tuple[float, str]]] = [[ordered[0]]]
    for price, bar in ordered[1:]:
        anchor = clusters[-1][0][0]
        if abs(price - anchor) / anchor * 100 <= tol_pct:
            clusters[-1].append((price, bar))
        else:
            clusters.append([(price, bar)])
    out = []
    for cluster in clusters:
        prices = [p for p, _ in cluster]
        bars = sorted(b for _, b in cluster)
        out.append((sum(prices) / len(prices), len(cluster), bars[-1]))
    return out


def build_liquidity_map(symbol: str, df: pd.DataFrame, *, window: int = 5,
                        lookback: int = 120, scan_pct: float = 8.0,
                        equal_tol_pct: float = EQUAL_TOLERANCE_PCT) -> LiquidityMap:
    """Locate untouched and swept liquidity pools around spot from daily bars."""
    if df is None or len(df) < window * 2 + 10:
        raise ValueError(f"{symbol}: not enough bars for a liquidity map")
    frame = df.tail(lookback).copy()
    spot = float(frame["close"].iloc[-1])
    as_of = str(pd.Timestamp(frame.index[-1]).date())
    lo_bound, hi_bound = spot * (1 - scan_pct / 100), spot * (1 + scan_pct / 100)

    is_sh, is_sl = swing_points(frame, window)
    # Confirmation flags are delayed by `window`, so the extreme lives at i - window.
    highs: list[tuple[float, str, int]] = []
    lows: list[tuple[float, str, int]] = []
    for i, (sh, sl) in enumerate(zip(is_sh.values, is_sl.values)):
        j = i - window
        if j < 0:
            continue
        bar = str(pd.Timestamp(frame.index[j]).date())
        if sh:
            highs.append((float(frame["high"].iloc[j]), bar, j))
        if sl:
            lows.append((float(frame["low"].iloc[j]), bar, j))

    def swept_after(level: float, idx: int, above: bool) -> bool:
        later = frame.iloc[idx + 1:]
        if later.empty:
            return False
        return bool((later["high"] > level).any()) if above else bool((later["low"] < level).any())

    def recent_sweep(level: float, above: bool, bars: int = 5, set_idx: int | None = None) -> bool:
        """True only if the level was first taken inside the last `bars` bars.

        Testing the tail alone marks a swing high from June as "just swept"
        every day price sits above it. A sweep is an EVENT: the level must
        have been untouched from the bar that set it up to the start of the
        window, and broken inside the window.
        """
        tail = frame.tail(bars)
        hit_now = bool((tail["high"] > level).any()) if above else bool((tail["low"] < level).any())
        if not hit_now:
            return False
        start = (set_idx + 1) if set_idx is not None else 0
        before = frame.iloc[start:len(frame) - bars]
        if before.empty:
            return True
        hit_before = bool((before["high"] > level).any()) if above else bool((before["low"] < level).any())
        return not hit_before

    pools: list[LiquidityPool] = []
    set_index: dict[float, int] = {}   # pool price -> bar index that set it

    # Swing pools, then merge equal ones.
    for price, touches, bar in _cluster_equal([(p, b) for p, b, _ in highs], equal_tol_pct):
        if not (lo_bound <= price <= hi_bound):
            continue
        idxs = [k for p, _, k in highs if abs(p - price) / price * 100 <= equal_tol_pct]
        swept = swept_after(price, max(idxs), above=True)
        set_index[price] = max(idxs)
        pools.append(LiquidityPool(
            price=price, kind="equal_highs" if touches >= 2 else "swing_high",
            side="above" if price >= spot else "below", touches=touches, bar=bar,
            swept=swept, distance_pct=(price - spot) / spot * 100))
    for price, touches, bar in _cluster_equal([(p, b) for p, b, _ in lows], equal_tol_pct):
        if not (lo_bound <= price <= hi_bound):
            continue
        idxs = [k for p, _, k in lows if abs(p - price) / price * 100 <= equal_tol_pct]
        swept = swept_after(price, max(idxs), above=False)
        set_index[price] = max(idxs)
        pools.append(LiquidityPool(
            price=price, kind="equal_lows" if touches >= 2 else "swing_low",
            side="above" if price >= spot else "below", touches=touches, bar=bar,
            swept=swept, distance_pct=(price - spot) / spot * 100))

    # Prior day / prior week extremes: always relevant, always fresh.
    if len(frame) >= 2:
        prev = frame.iloc[-2]
        pbar = str(pd.Timestamp(frame.index[-2]).date())
        for price, kind, above in ((float(prev["high"]), "pdh", True), (float(prev["low"]), "pdl", False)):
            set_index[price] = len(frame) - 2
            pools.append(LiquidityPool(
                price=price, kind=kind, side="above" if price >= spot else "below",
                touches=1, bar=pbar, swept=recent_sweep(price, above, bars=1, set_idx=len(frame) - 2),
                distance_pct=(price - spot) / spot * 100))
    if len(frame) >= 10:
        week = frame.iloc[-10:-5]
        wbar = str(pd.Timestamp(week.index[-1]).date())
        # The level is "set" on the LAST bar of the prior week, so anything
        # earlier in that week touching it is formation, not a sweep.
        w_idx = len(frame) - 6
        for price, kind, above in ((float(week["high"].max()), "pwh", True),
                                   (float(week["low"].min()), "pwl", False)):
            set_index[price] = w_idx
            pools.append(LiquidityPool(
                price=price, kind=kind, side="above" if price >= spot else "below",
                touches=1, bar=wbar, swept=recent_sweep(price, above, bars=5, set_idx=w_idx),
                distance_pct=(price - spot) / spot * 100))

    above = sorted([p for p in pools if p.side == "above"], key=lambda p: p.distance_pct)
    below = sorted([p for p in pools if p.side == "below"], key=lambda p: -p.distance_pct)
    # Sweeps are judged against the side the level was on when it was SET:
    # a swing high is broken UPWARD regardless of where spot sits now.
    recently = [p for p in pools if p.swept and recent_sweep(
        p.price, p.kind in ("swing_high", "equal_highs", "pdh", "pwh"),
        set_idx=set_index.get(p.price))]
    return LiquidityMap(symbol=symbol, spot=spot, as_of_bar=as_of,
                        above=above, below=below, recently_swept=recently)
