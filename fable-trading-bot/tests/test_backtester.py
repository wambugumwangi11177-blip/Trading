import pandas as pd

from fable_bot.backtester import enforce_hard_stops, simulate
from fable_bot.config import Instrument
from fable_bot.portfolio import PortfolioTracker


def test_enforce_hard_stops_closes_long_on_intrabar_breach():
    p = PortfolioTracker(cash=100_000)
    p.open_position("SPY", "long", qty=10, price=100.0, stop_price=98.0, date=pd.Timestamp("2026-01-01"))
    df = pd.DataFrame(
        {"open": [99.0], "high": [99.5], "low": [97.0], "close": [97.5], "volume": [1_000_000]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-02")], name="date"),
    )
    enforce_hard_stops(p, {"SPY": df}, pd.Timestamp("2026-01-02"))
    assert "SPY" not in p.positions
    assert p.trades[-1].exit_price == 98.0  # exits at the stop, not the bar's close
    assert p.trades[-1].pnl == -20.0  # (98-100)*10, exactly the sized risk, not the closing price's loss


def test_enforce_hard_stops_closes_short_on_intrabar_breach():
    p = PortfolioTracker(cash=100_000)
    p.open_position("BTC-USD", "short", qty=1, price=100.0, stop_price=105.0, date=pd.Timestamp("2026-01-01"))
    df = pd.DataFrame(
        {"open": [101.0], "high": [110.0], "low": [100.5], "close": [108.0], "volume": [1_000_000]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-02")], name="date"),
    )
    enforce_hard_stops(p, {"BTC-USD": df}, pd.Timestamp("2026-01-02"))
    assert "BTC-USD" not in p.positions
    assert p.trades[-1].exit_price == 105.0
    assert p.trades[-1].pnl == -5.0  # capped at the stop, not the much worse close


def test_enforce_hard_stops_leaves_position_open_when_not_breached():
    p = PortfolioTracker(cash=100_000)
    p.open_position("SPY", "long", qty=10, price=100.0, stop_price=98.0, date=pd.Timestamp("2026-01-01"))
    df = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1_000_000]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-02")], name="date"),
    )
    enforce_hard_stops(p, {"SPY": df}, pd.Timestamp("2026-01-02"))
    assert "SPY" in p.positions


def test_simulate_runs_end_to_end_on_synthetic_universe():
    import numpy as np

    rng = np.random.default_rng(1)
    dates = pd.date_range("2026-01-01", periods=150, freq="D")

    def make_df(drift):
        close = pd.Series(100 + drift * np.arange(150) + rng.normal(0, 1.5, 150), index=dates)
        return pd.DataFrame({
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        }, index=dates)

    instruments = [
        Instrument("SPY", "SPY", "mean_reversion", "equity", "S&P 500 ETF"),
        Instrument("QQQ", "QQQ", "mean_reversion", "equity", "Nasdaq 100 ETF"),
        Instrument("BTC-USD", "BTC/USD", "momentum", "crypto", "Bitcoin"),
        Instrument("GLD", "GLD", "trend_following", "equity", "Gold ETF"),
        Instrument("USO", "USO", "trend_following", "equity", "Oil ETF"),
    ]
    price_history = {
        "SPY": make_df(0.0), "QQQ": make_df(0.0), "BTC-USD": make_df(0.3),
        "GLD": make_df(0.1), "USO": make_df(-0.1),
    }

    result = simulate(price_history, instruments)
    assert len(result.equity_curve) == 150
    assert isinstance(result.sharpe_ratio, float)
    assert isinstance(result.max_drawdown_pct, float)
    assert result.max_drawdown_pct <= 0
