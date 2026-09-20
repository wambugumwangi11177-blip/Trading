import pandas as pd

from fable_bot.portfolio import PortfolioTracker


def test_open_and_close_long_position_pnl():
    p = PortfolioTracker(cash=100_000)
    p.open_position("SPY", "long", qty=10, price=100.0, stop_price=98.0, date=pd.Timestamp("2026-01-01"))
    pnl = p.close_position("SPY", price=105.0, date=pd.Timestamp("2026-01-05"), reason="test exit")
    assert pnl == 50.0
    assert p.cash == 100_050.0
    assert "SPY" not in p.positions
    assert len(p.trades) == 1


def test_open_and_close_short_position_pnl():
    p = PortfolioTracker(cash=100_000)
    p.open_position("BTC-USD", "short", qty=1, price=100.0, stop_price=105.0, date=pd.Timestamp("2026-01-01"))
    pnl = p.close_position("BTC-USD", price=90.0, date=pd.Timestamp("2026-01-05"), reason="test exit")
    assert pnl == 10.0
    assert p.cash == 100_010.0


def test_unrealized_pnl_and_equity():
    p = PortfolioTracker(cash=100_000)
    p.open_position("GLD", "long", qty=5, price=200.0, stop_price=190.0, date=pd.Timestamp("2026-01-01"))
    assert p.unrealized_pnl("GLD", 210.0) == 50.0
    assert p.equity({"GLD": 210.0}) == 100_050.0


def test_futures_pnl_uses_the_contract_multiplier():
    """2 MES contracts, 10 points, at $5/point = $100 -- not $20."""
    p = PortfolioTracker(cash=100_000)
    p.open_position("MES=F", "long", qty=2, price=7600.0, stop_price=7550.0,
                    date=pd.Timestamp("2026-08-04"), point_value=5.0)
    assert p.unrealized_pnl("MES=F", 7610.0) == 100.0
    pnl = p.close_position("MES=F", price=7610.0, date=pd.Timestamp("2026-08-04"), reason="test exit")
    assert pnl == 100.0
    assert p.cash == 100_100.0


def test_short_futures_pnl_uses_the_multiplier_too():
    p = PortfolioTracker(cash=100_000)
    p.open_position("MNQ=F", "short", qty=1, price=29_100.0, stop_price=29_400.0,
                    date=pd.Timestamp("2026-08-04"), point_value=2.0)
    # Down 50 points on a short at $2/point = +$100.
    assert p.close_position("MNQ=F", price=29_050.0, date=pd.Timestamp("2026-08-04"), reason="x") == 100.0


def test_point_value_defaults_to_one_for_shares():
    p = PortfolioTracker(cash=100_000)
    p.open_position("SPY", "long", qty=10, price=100.0, stop_price=98.0, date=pd.Timestamp("2026-01-01"))
    assert p.positions["SPY"].point_value == 1.0
    assert p.unrealized_pnl("SPY", 105.0) == 50.0


def test_record_equity_appends_curve():
    p = PortfolioTracker(cash=100_000)
    p.record_equity(pd.Timestamp("2026-01-01"), {})
    p.record_equity(pd.Timestamp("2026-01-02"), {})
    assert len(p.equity_curve) == 2


def test_open_sides_reflects_positions():
    p = PortfolioTracker(cash=100_000)
    p.open_position("SPY", "long", qty=1, price=100, stop_price=98, date=pd.Timestamp("2026-01-01"))
    p.open_position("QQQ", "short", qty=1, price=100, stop_price=102, date=pd.Timestamp("2026-01-01"))
    assert p.open_sides() == {"SPY": "long", "QQQ": "short"}
