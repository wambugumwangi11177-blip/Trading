"""Long-or-flat trend following for gold. A DRAWDOWN control, not an alpha source.

Read this before trusting it: an earlier version of this docstring claimed this
strategy beat buy-and-hold. That claim was wrong, and the correction matters.

The first analysis computed Sharpe on P&L measured in price POINTS. Gold ran
from ~275 to ~4,100 across the sample, so a point of movement in 2024 is worth
roughly fifteen times a point in 2002 and late years dominated the volatility
estimate. Recomputing on RETURNS (and fixing a round-turn cost that was billed
twice) reverses the ranking.

Corrected evidence (GC=F daily, 2000-08 to 2026-08, 6,505 bars, split
2016-01-01, cost 0.3 pts round turn, Sharpe on returns):

    strategy                       t    Sharpe  Sharpe_test     maxDD
    buy & hold                   3.45    0.68      0.86      -133,280
    long-or-flat Donchian 100d   3.37    0.66      0.78        -98,145
    long-or-flat TSMOM 3m        3.03    0.60      0.88        -94,305
    long-or-flat MA 20/100       2.51    0.49      0.86        -94,290

So what is actually real is gold's DRIFT: +10.52%/yr, t=+2.99, and buy-and-hold
clears a Bonferroni threshold for 19 tests (t=3.45 vs 2.81) and beats a
permutation null (2.20). Trend timing does NOT improve risk-adjusted return --
every variant sits at or below buy-and-hold's Sharpe.

What trend timing does buy is a materially smaller worst case: about -94k
against -133k, roughly 29% less drawdown, while holding the position only ~63%
of the time. That is a real and defensible reason to use it when the drawdown,
not the return, is what constrains position size. It is risk management, not an
edge, and it should not be sold as one.

Gold's variance ratios sit near 1.0 at every horizon, i.e. gold is close to a
random walk. The drift is the signal; the timing is not.

Short signals were tested separately and rejected: every short-or-flat variant
produced a negative t-stat, and holding long through gold downtrends scores
t=+0.12 -- essentially zero. Neither side pays in a downtrend, which is why
"flat" is a genuine third state here rather than indecision.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Signal, Strategy


@dataclass(frozen=True)
class TrendConfig:
    # Canonical validated parameters. Changing these invalidates the numbers in
    # the docstring above -- re-run the validation before trusting new ones.
    momentum_days: int = 63       # ~3 months, the primary signal (t=2.31)
    ma_fast: int = 20
    ma_slow: int = 100
    donchian_days: int = 50
    atr_days: int = 14
    atr_stop_multiple: float = 1.5
    # How many of the three definitions must agree to take a position. 1 = the
    # primary signal alone (as validated); 2-3 trade less often for higher
    # conviction, which was NOT separately validated -- treat as a preference,
    # not an improvement.
    required_agreement: int = 1


@dataclass(frozen=True)
class TrendSignal:
    symbol: str
    date: pd.Timestamp
    side: str                 # "long" | "flat"
    price: float
    stop_price: float | None  # None when flat
    atr: float
    momentum_pct: float
    agreement: int
    detail: dict
    reason: str


def average_true_range(df: pd.DataFrame, length: int) -> pd.Series:
    prior_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prior_close).abs(),
        (df["low"] - prior_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(length).mean()


def evaluate(symbol: str, df: pd.DataFrame, config: TrendConfig | None = None) -> TrendSignal:
    """Evaluate the trend state on the most recent bar."""
    config = config or TrendConfig()
    if len(df) < max(config.ma_slow, config.momentum_days, config.donchian_days) + 2:
        raise ValueError(f"need more history for {symbol}: only {len(df)} bars")

    close = df["close"]
    momentum = close.pct_change(config.momentum_days).iloc[-1]
    ma_fast = close.rolling(config.ma_fast).mean().iloc[-1]
    ma_slow = close.rolling(config.ma_slow).mean().iloc[-1]
    # shift(1) so today's own high cannot create the breakout it is tested against.
    donchian_high = close.rolling(config.donchian_days).max().shift(1).iloc[-1]
    atr = average_true_range(df, config.atr_days).iloc[-1]
    price = close.iloc[-1]

    votes = {
        f"tsmom_{config.momentum_days}d": bool(momentum > 0),
        f"ma_{config.ma_fast}_{config.ma_slow}": bool(ma_fast > ma_slow),
        f"donchian_{config.donchian_days}d": bool(price > donchian_high),
    }
    agreement = sum(votes.values())
    is_long = agreement >= config.required_agreement

    if is_long:
        reason = (f"long: {agreement}/3 trend signals up "
                  f"({config.momentum_days}d momentum {momentum*100:+.2f}%)")
        stop = price - config.atr_stop_multiple * atr
    else:
        reason = (f"flat: {agreement}/3 trend signals up "
                  f"({config.momentum_days}d momentum {momentum*100:+.2f}%) — "
                  "strategy is long-or-flat and never shorts")
        stop = None

    return TrendSignal(
        symbol=symbol, date=df.index[-1], side="long" if is_long else "flat",
        price=float(price), stop_price=float(stop) if stop is not None else None,
        atr=float(atr), momentum_pct=float(momentum * 100), agreement=agreement,
        detail={
            **{k: bool(v) for k, v in votes.items()},
            "ma_fast": float(ma_fast), "ma_slow": float(ma_slow),
            "donchian_high": float(donchian_high),
        },
        reason=reason,
    )


def vote_series(df: pd.DataFrame, config: TrendConfig | None = None) -> pd.Series:
    """How many of the three trend definitions are up, per bar.

    The single source of truth for the signal. `evaluate` reads the last bar of
    it, `backtest_signals` thresholds it for validation runs, and
    LongOrFlatTrendStrategy walks it -- three consumers of one calculation, so
    they cannot quietly disagree about what the validated strategy says.
    """
    config = config or TrendConfig()
    close = df["close"]
    return (
        (close.pct_change(config.momentum_days) > 0).astype(int)
        + (close.rolling(config.ma_fast).mean() > close.rolling(config.ma_slow).mean()).astype(int)
        + (close > close.rolling(config.donchian_days).max().shift(1)).astype(int)
    )


def backtest_signals(symbol: str, df: pd.DataFrame, config: TrendConfig | None = None) -> pd.Series:
    """Vectorised position series (1.0 long / 0.0 flat) for validation runs.

    Shifted by one bar so a position is only taken on information available
    before the bar it is held through -- without this the backtest trades on
    the same close it used to decide, which is not reproducible live.
    """
    config = config or TrendConfig()
    return (vote_series(df, config) >= config.required_agreement).astype(float).shift(1).fillna(0.0)


class LongOrFlatTrendStrategy(Strategy):
    """Registry adapter for the validated gold trend strategy.

    Without this the strategy is unreachable: `config.GOLD_FUTURES` names it as
    "long_or_flat_trend", but the registry had no such entry, so both
    `run_backtest` and `live_runner` raised KeyError on it. The module only
    exposed `evaluate()` (one bar) and `backtest_signals()` (an exposure series),
    neither of which satisfies the Strategy interface.

    Emission follows the house convention used by the other registry
    strategies: a Signal on each bar where the position changes, priced at that
    bar's close. Note this is *not* the one-bar shift `backtest_signals` applies
    -- that models next-bar execution for the standalone validation harness,
    while `simulate()` fills registry strategies at the signal bar's close.
    """

    name = "long_or_flat_trend"

    def __init__(self, config: TrendConfig | None = None):
        self.config = config or TrendConfig()

    def generate_signals(self, symbol: str, df: pd.DataFrame) -> list[Signal]:
        config = self.config
        votes = vote_series(df, config)
        atr_series = average_true_range(df, config.atr_days)

        signals: list[Signal] = []
        position: str = "flat"
        # Matches evaluate()'s own history requirement, so the two agree on
        # every bar either of them is willing to judge.
        min_bars = max(config.ma_slow, config.momentum_days, config.donchian_days) + 1

        for i in range(min_bars, len(df)):
            agreement = votes.iloc[i]
            a = atr_series.iloc[i]
            if pd.isna(agreement) or pd.isna(a):
                continue

            price = float(df["close"].iloc[i])
            date = df.index[i]
            wants_long = agreement >= config.required_agreement

            if wants_long and position == "flat":
                signals.append(Signal(
                    date, symbol, "long", price, price - config.atr_stop_multiple * float(a),
                    f"long: {int(agreement)}/3 trend signals up",
                ))
                position = "long"
            elif not wants_long and position == "long":
                # Long-or-flat: the exit is to cash, never to a short.
                signals.append(Signal(
                    date, symbol, "flat", price, price,
                    f"flat: {int(agreement)}/3 trend signals up",
                ))
                position = "flat"

        return signals

    def catch_up_signal(self, symbol: str, df: pd.DataFrame) -> Signal | None:
        # The state is a pure function of the latest bar, so a missed flip can
        # always be reconciled: re-evaluate now, reprice, recompute the stop.
        state = evaluate(symbol, df, self.config)
        stop = state.stop_price if state.stop_price is not None else state.price
        return Signal(
            df.index[-1], symbol, state.side, state.price, stop,
            f"catch-up: {state.reason} (flip was missed)",
        )
