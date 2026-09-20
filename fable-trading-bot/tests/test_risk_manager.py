import pandas as pd

from fable_bot.config import RiskConfig
from fable_bot.lessons import DEFAULT_SIZING_BOUNDS
from fable_bot.risk.manager import DrawdownMonitor, RiskManager, correlation_matrix, sizing_sanity


def _risk_manager(**overrides):
    config = RiskConfig(**{
        "max_risk_per_trade_pct": 1.0,
        "max_drawdown_pct": 10.0,
        "correlation_block_threshold": 0.7,
        "atr_stop_multiple": 1.5,
        "starting_equity": 100_000,
        **overrides,
    })
    return RiskManager(config=config)


def test_position_size_risks_exactly_target_pct():
    rm = _risk_manager()
    equity = 100_000
    entry, stop = 100.0, 98.0  # $2 stop distance
    qty = rm.position_size(equity, entry, stop)
    risked = qty * abs(entry - stop)
    assert abs(risked - equity * 0.01) < 1e-6


def test_position_size_zero_when_no_stop_distance():
    rm = _risk_manager()
    assert rm.position_size(100_000, 100.0, 100.0) == 0.0


def test_position_size_accounts_for_futures_contract_multiplier():
    """One E-mini S&P contract moves $50 per index point, not $1."""
    rm = _risk_manager()
    equity = 111_822.0
    entry, stop = 7642.0, 7496.0  # 146-point stop, ~1.5x daily ATR
    qty = rm.position_size(equity, entry, stop, point_value=50.0)
    risked = qty * abs(entry - stop) * 50.0
    assert abs(risked - equity * 0.01) < 1e-6
    # Ignoring the multiplier would have returned ~7.7 contracts risking ~46%
    # of the account. Sizing correctly, the budget doesn't even cover one.
    assert qty < 1


def test_micro_contract_fits_where_the_full_size_does_not():
    rm = _risk_manager()
    equity = 111_822.0
    entry, stop = 7642.0, 7496.0
    assert rm.position_size(equity, entry, stop, point_value=50.0) < 1   # MES's big brother
    assert rm.position_size(equity, entry, stop, point_value=5.0) >= 1   # MES


def test_point_value_defaults_to_one_so_equities_are_unchanged():
    rm = _risk_manager()
    assert rm.position_size(100_000, 100.0, 98.0) == rm.position_size(100_000, 100.0, 98.0, point_value=1.0)


def test_position_size_zero_for_nonsense_point_value():
    rm = _risk_manager()
    assert rm.position_size(100_000, 100.0, 98.0, point_value=0.0) == 0.0


def test_correlation_blocks_same_direction_high_correlation():
    rm = _risk_manager()
    rm.correlations = pd.DataFrame({"SPY": [1.0, 0.95], "QQQ": [0.95, 1.0]}, index=["SPY", "QQQ"])
    blocked, reason = rm.is_blocked_by_correlation("QQQ", "long", {"SPY": "long"})
    assert blocked
    assert "correlation" in reason


def test_correlation_allows_opposite_direction():
    rm = _risk_manager()
    rm.correlations = pd.DataFrame({"SPY": [1.0, 0.95], "QQQ": [0.95, 1.0]}, index=["SPY", "QQQ"])
    blocked, _ = rm.is_blocked_by_correlation("QQQ", "short", {"SPY": "long"})
    assert not blocked


def test_correlation_matrix_shape():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    a = pd.DataFrame({"close": range(100, 130)}, index=dates)
    b = pd.DataFrame({"close": range(200, 230)}, index=dates)
    corr = correlation_matrix({"A": a, "B": b})
    assert corr.loc["A", "B"] > 0.9  # both are linear trends, perfectly correlated


def test_drawdown_monitor_triggers_kill_switch():
    monitor = DrawdownMonitor(max_drawdown_pct=10.0, peak_equity=100_000)
    assert not monitor.update(98_000)
    assert monitor.update(89_000)  # 11% drawdown from peak
    assert monitor.halted


def test_drawdown_monitor_manual_reset():
    monitor = DrawdownMonitor(max_drawdown_pct=10.0, peak_equity=100_000)
    monitor.update(85_000)
    assert monitor.halted
    monitor.manual_reset()
    assert not monitor.halted


# ── sizing_sanity (2026-08 incident lesson) ──────────────────────────────

def test_sizing_sanity_passes_a_normal_1pct_order():
    ok, reason, metrics = sizing_sanity(
        qty=500, price=100.0, stop_price=98.0, equity=100_000,
        point_value=1.0, bounds=DEFAULT_SIZING_BOUNDS)
    assert ok and reason == ""
    assert metrics["risk_pct"] == 1.0
    assert metrics["notional"] == 50_000


def test_sizing_sanity_blocks_ten_times_equity_notional():
    ok, reason, _ = sizing_sanity(
        qty=20_000, price=100.0, stop_price=99.9, equity=100_000,
        point_value=1.0, bounds=DEFAULT_SIZING_BOUNDS)
    assert not ok
    assert "equity" in reason


def test_sizing_sanity_blocks_oversized_risk():
    # Notional is fine (20% of equity) but the stop makes it a 2% risk.
    ok, reason, _ = sizing_sanity(
        qty=200, price=100.0, stop_price=90.0, equity=100_000,
        point_value=1.0, bounds=DEFAULT_SIZING_BOUNDS)
    assert not ok
    assert "risk" in reason


def test_sizing_sanity_blocks_dust_notional():
    ok, reason, _ = sizing_sanity(
        qty=0.5, price=100.0, stop_price=90.0, equity=100_000,
        point_value=1.0, bounds=DEFAULT_SIZING_BOUNDS)
    assert not ok
    assert "below minimum" in reason


def test_sizing_sanity_blocks_near_zero_risk():
    # Stop so wide the "risk" rounds to dust -- sizing input is wrong somewhere.
    ok, reason, _ = sizing_sanity(
        qty=1, price=100.0, stop_price=99.99, equity=100_000,
        point_value=1.0, bounds=DEFAULT_SIZING_BOUNDS)
    assert not ok


def test_sizing_sanity_respects_futures_point_value():
    # One MES micro at a 146-point stop: $730 = 0.65% of equity, 34% notional.
    ok, reason, metrics = sizing_sanity(
        qty=1, price=7642.0, stop_price=7496.0, equity=111_822,
        point_value=5.0, bounds=DEFAULT_SIZING_BOUNDS)
    assert ok, reason
    assert metrics["risk_amount"] == 730.0

