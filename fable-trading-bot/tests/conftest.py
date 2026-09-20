import numpy as np
import pandas as pd
import pytest


def _make_ohlc(close: np.ndarray, start: str = "2026-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(close), freq="D")
    close = pd.Series(close, index=dates)
    high = close * 1.005
    low = close * 0.995
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(1_000_000, index=dates)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
    df.index.name = "date"
    return df


@pytest.fixture
def random_walk_df():
    rng = np.random.default_rng(42)
    steps = rng.normal(0, 1, 200)
    close = 100 + np.cumsum(steps)
    return _make_ohlc(close)


@pytest.fixture
def mean_reverting_df():
    rng = np.random.default_rng(7)
    t = np.arange(200)
    close = 100 + 8 * np.sin(t / 8) + rng.normal(0, 0.3, 200)
    return _make_ohlc(close)


@pytest.fixture
def uptrend_df():
    rng = np.random.default_rng(3)
    t = np.arange(200)
    close = 100 + 0.3 * t + 2 * np.sin(t / 5) + rng.normal(0, 0.2, 200)
    return _make_ohlc(close)


@pytest.fixture
def breakout_df():
    rng = np.random.default_rng(11)
    flat = 100 + rng.normal(0, 0.2, 100)
    breakout = 100 + np.linspace(0, 40, 100) + rng.normal(0, 0.3, 100)
    close = np.concatenate([flat, breakout])
    return _make_ohlc(close)
