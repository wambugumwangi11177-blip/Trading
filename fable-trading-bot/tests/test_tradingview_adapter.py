"""Tests for the TradingView Paper Trading adapter.

The subprocess boundary is faked, so what gets exercised is the part that can
actually lose money: quantity rounding, the paper/live guard, and how failures
from the CLI are surfaced.
"""
from __future__ import annotations

import json
import subprocess

import pytest

from fable_bot.broker import build_broker
from fable_bot.broker.tradingview_adapter import TradingViewBrokerAdapter, TradingViewError
from fable_bot.config import BrokerConfig, Instrument

SPY = Instrument("SPY", "SPY", "mean_reversion", "equity", "S&P 500 ETF",
                 tv_symbol="AMEX:SPY", qty_step=1.0)
BTC = Instrument("BTC-USD", "BTC/USD", "momentum", "crypto", "Bitcoin",
                 tv_symbol="BITSTAMP:BTCUSD", qty_step=0.0001)
NO_TV = Instrument("XYZ", "XYZ", "momentum", "equity", "Unmapped")

ACCOUNT_OK = {
    "success": True, "broker": "Paper Trading", "account_id": "00000000", "is_paper": True,
    "balance": 111822.25, "equity": 111822.25, "realized_pnl": 11822.25,
    "unrealized_pnl": 0, "account_margin": 0, "available_funds": 111700.0,
    "margin_buffer_pct": 100,
}


def _config(tmp_path, **overrides) -> BrokerConfig:
    # The adapter checks the CLI exists before doing anything else.
    cli = tmp_path / "src" / "cli"
    cli.mkdir(parents=True, exist_ok=True)
    (cli / "index.js").write_text("// stub", encoding="utf-8")
    return BrokerConfig(**{
        "provider": "tradingview", "trading_mode": "paper", "allow_live_trading": False,
        "tv_mcp_path": str(tmp_path), "tv_node_bin": "node", "tv_timeout_seconds": 5,
        **overrides,
    })


def _adapter(tmp_path, monkeypatch, responses, *, returncode=0):
    """Patch subprocess.run to replay canned CLI responses in order."""
    calls = []
    queue = list(responses)

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        payload = queue.pop(0) if queue else {"success": True}
        if isinstance(payload, Exception):
            raise payload
        code = payload.pop("__returncode", returncode) if isinstance(payload, dict) else returncode
        return subprocess.CompletedProcess(cmd, code, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return TradingViewBrokerAdapter(_config(tmp_path)), calls


# ── Quantity rounding ────────────────────────────────────────────────────

def test_equity_quantity_rounds_down_to_whole_shares(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path, monkeypatch, [])
    # Rounding 12.8 up to 13 would risk more than the position was sized for.
    assert adapter.quantize(SPY, 12.8) == 12
    assert adapter.quantize(SPY, 12.0) == 12


def test_quantity_below_one_share_becomes_zero(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path, monkeypatch, [])
    assert adapter.quantize(SPY, 0.6) == 0


def test_crypto_quantity_rounds_to_its_finer_step(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path, monkeypatch, [])
    assert adapter.quantize(BTC, 0.0123456) == 0.0123


def test_rounding_does_not_leak_float_noise(tmp_path, monkeypatch):
    """0.0003 must not come back as 0.00030000000000000003."""
    adapter, _ = _adapter(tmp_path, monkeypatch, [])
    assert str(adapter.quantize(BTC, 0.00031)) == "0.0003"


# ── Symbol mapping ───────────────────────────────────────────────────────

def test_maps_instrument_to_exchange_qualified_symbol(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path, monkeypatch, [])
    assert adapter.venue_symbol(SPY) == "AMEX:SPY"
    assert adapter.venue_symbol(BTC) == "BITSTAMP:BTCUSD"


def test_unmapped_instrument_is_rejected_not_guessed(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path, monkeypatch, [])
    with pytest.raises(TradingViewError, match="no tv_symbol"):
        adapter.venue_symbol(NO_TV)


# ── Account / positions ──────────────────────────────────────────────────

def test_reads_account_snapshot(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path, monkeypatch, [dict(ACCOUNT_OK)])
    snap = adapter.get_account()
    assert snap.equity == 111822.25
    assert snap.cash == 111700.0
    assert snap.is_paper is True


def test_refuses_a_live_broker_while_configured_for_paper(tmp_path, monkeypatch):
    live = dict(ACCOUNT_OK, broker="Interactive Brokers", is_paper=False)
    adapter, _ = _adapter(tmp_path, monkeypatch, [live])
    with pytest.raises(TradingViewError, match="not the Paper Trading simulator"):
        adapter.get_account()


def test_positions_are_keyed_by_venue_symbol(tmp_path, monkeypatch):
    resp = {"success": True, "count": 1, "positions": [
        {"id": "AMEX:SPY", "symbol": "AMEX:SPY", "side": "long", "qty": 12, "avg_price": 604.2},
    ]}
    adapter, _ = _adapter(tmp_path, monkeypatch, [resp])
    positions = adapter.get_open_positions()
    assert positions == {"AMEX:SPY": {"qty": 12.0, "side": "long", "avg_entry_price": 604.2}}


# ── Orders ───────────────────────────────────────────────────────────────

def test_submit_order_attaches_a_broker_side_stop(tmp_path, monkeypatch):
    resp = {"success": True, "order": {"id": "999", "status": "filled"}}
    adapter, calls = _adapter(tmp_path, monkeypatch, [resp])
    result = adapter.submit_order("AMEX:SPY", 12, "buy", stop_price=598.1234)
    cmd = calls[0]
    assert cmd[2:] == ["trade", "buy", "AMEX:SPY", "--qty", "12", "--stop-loss", "598.1234"]
    assert result.broker_order_id == "999"
    assert result.status == "filled"


def test_submit_order_without_a_stop_omits_the_flag(tmp_path, monkeypatch):
    adapter, calls = _adapter(tmp_path, monkeypatch, [{"success": True, "order": {}}])
    adapter.submit_order("AMEX:SPY", 5, "sell")
    assert "--stop-loss" not in calls[0]


def test_paper_mode_never_passes_allow_live(tmp_path, monkeypatch):
    adapter, calls = _adapter(tmp_path, monkeypatch, [{"success": True, "order": {}}])
    adapter.submit_order("AMEX:SPY", 1, "buy")
    assert "--allow-live" not in calls[0]


def test_allow_live_requires_both_bot_flags(tmp_path, monkeypatch):
    """TRADING_MODE=live alone is not enough — ALLOW_LIVE_TRADING must agree."""
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout=json.dumps({"success": True, "order": {}}), stderr=""))

    half = TradingViewBrokerAdapter(_config(tmp_path, trading_mode="live", allow_live_trading=False))
    assert half._live_flag() == []

    both = TradingViewBrokerAdapter(_config(tmp_path, trading_mode="live", allow_live_trading=True))
    assert both._live_flag() == ["--allow-live"]


# ── Failure handling ─────────────────────────────────────────────────────

def test_unreachable_tradingview_gives_an_actionable_error(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path, monkeypatch,
                          [{"success": False, "error": "CDP connection failed", "__returncode": 2}])
    with pytest.raises(TradingViewError, match="Cannot reach TradingView Desktop"):
        adapter.get_account()


def test_rejected_order_raises_with_the_reason(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path, monkeypatch,
                          [{"success": False, "error": "qty must be positive"}])
    with pytest.raises(TradingViewError, match="qty must be positive"):
        adapter.submit_order("AMEX:SPY", 1, "buy")


def test_order_refusal_is_marked_as_a_venue_rejection(tmp_path, monkeypatch):
    """This flag is what turns a refusal into a halt lesson; it must be set
    exactly when the venue said no to this order."""
    adapter, _ = _adapter(tmp_path, monkeypatch,
                          [{"success": False, "error": "Invalid symbol"}])
    with pytest.raises(TradingViewError) as excinfo:
        adapter.submit_order("NASDAQ:NDX", 1, "buy")
    assert excinfo.value.venue_rejection is True


def test_connectivity_failure_is_not_a_venue_rejection(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path, monkeypatch,
                          [{"success": False, "error": "CDP connection failed", "__returncode": 2}])
    with pytest.raises(TradingViewError) as excinfo:
        adapter.submit_order("AMEX:SPY", 1, "buy")
    assert excinfo.value.venue_rejection is False


def test_missing_node_is_reported_clearly(tmp_path, monkeypatch):
    adapter, _ = _adapter(tmp_path, monkeypatch, [FileNotFoundError("node")])
    with pytest.raises(TradingViewError, match="Node.js must be installed"):
        adapter.get_account()


def test_unparseable_output_is_not_silently_swallowed(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout="<html>login</html>", stderr=""))
    adapter = TradingViewBrokerAdapter(_config(tmp_path))
    with pytest.raises(TradingViewError, match="Unparseable response"):
        adapter.get_account()


def test_missing_cli_path_is_caught_at_construction(tmp_path):
    config = BrokerConfig(provider="tradingview", tv_mcp_path=str(tmp_path / "nope"))
    with pytest.raises(RuntimeError, match="tradingview-mcp CLI not found"):
        TradingViewBrokerAdapter(config)


def test_kill_switch_keeps_closing_after_one_position_fails(tmp_path, monkeypatch):
    """A kill-switch that aborts halfway leaves the exposure it exists to remove."""
    responses = [
        {"success": True, "positions": [
            {"symbol": "AMEX:SPY", "side": "long", "qty": 12, "avg_price": 604.2},
            {"symbol": "AMEX:GLD", "side": "long", "qty": 30, "avg_price": 250.0},
        ]},
        {"success": False, "error": "market closed"},          # SPY close fails
        {"success": True, "closed": {"symbol": "AMEX:GLD", "qty": 30}},  # GLD still closes
    ]
    adapter, _ = _adapter(tmp_path, monkeypatch, responses)
    results = adapter.close_all()
    assert len(results) == 2
    assert results[0].status.startswith("FAILED")
    assert results[1].status == "closed"


# ── Order history / verification ─────────────────────────────────────────

def _history_resp(orders):
    return {"success": True, "count": len(orders), "orders": orders}


def test_get_order_history_normalizes_rows(tmp_path, monkeypatch):
    resp = _history_resp([
        {"id": "1", "symbol": "OANDA:EURUSD", "side": "buy", "type": "market",
         "status": "filled", "qty": 85000, "avg_fill_price": 1.16449,
         "placed_at": "2026-08-27T12:11:13.000Z"},
        {"id": "2", "symbol": "AMEX:GLD", "side": "sell", "type": "stop",
         "status": "cancelled", "qty": 69},
    ])
    adapter, calls = _adapter(tmp_path, monkeypatch, [resp])
    rows = adapter.get_order_history(limit=100)
    assert calls[0][2:] == ["trade", "history", "--limit", "100"]
    assert len(rows) == 2
    assert rows[0].symbol == "OANDA:EURUSD" and rows[0].avg_fill_price == 1.16449
    assert rows[1].avg_fill_price is None and rows[1].placed_at is None


def test_verify_order_finds_the_fill(tmp_path, monkeypatch):
    resp = _history_resp([
        {"id": "7", "symbol": "AMEX:SPY", "side": "buy", "type": "market",
         "status": "filled", "qty": 12, "avg_fill_price": 604.2,
         "placed_at": "2026-08-31T13:31:57.000Z"},
    ])
    adapter, _ = _adapter(tmp_path, monkeypatch, [resp])
    row = adapter.verify_order("AMEX:SPY", "buy", 12, attempts=1, delay_seconds=0)
    assert row is not None and row.id == "7"


def test_verify_order_matches_on_a_later_poll(tmp_path, monkeypatch):
    empty = _history_resp([])
    found = _history_resp([
        {"id": "8", "symbol": "AMEX:SPY", "side": "buy", "type": "market",
         "status": "working", "qty": 12},
    ])
    adapter, calls = _adapter(tmp_path, monkeypatch, [empty, found])
    row = adapter.verify_order("AMEX:SPY", "buy", 12, attempts=3, delay_seconds=0)
    assert row is not None and len(calls) == 2


def test_verify_order_gives_up_after_bounded_polls(tmp_path, monkeypatch):
    adapter, calls = _adapter(tmp_path, monkeypatch, [_history_resp([])] * 3)
    assert adapter.verify_order("AMEX:SPY", "buy", 12, attempts=3, delay_seconds=0) is None
    assert len(calls) == 3


def test_verify_order_ignores_bracket_stops(tmp_path, monkeypatch):
    # The protective stop shares our symbol; it must not count as the entry.
    resp = _history_resp([
        {"id": "9", "symbol": "AMEX:SPY", "side": "sell", "type": "stop",
         "status": "working", "qty": 12},
    ])
    adapter, _ = _adapter(tmp_path, monkeypatch, [resp])
    assert adapter.verify_order("AMEX:SPY", "sell", 12, attempts=1, delay_seconds=0) is None


def test_verify_order_ignores_rejected_rows(tmp_path, monkeypatch):
    resp = _history_resp([
        {"id": "10", "symbol": "NASDAQ:NDX", "side": "buy", "type": "market",
         "status": "rejected", "qty": 1.44},
    ])
    adapter, _ = _adapter(tmp_path, monkeypatch, [resp])
    assert adapter.verify_order("NASDAQ:NDX", "buy", 1.44, attempts=1, delay_seconds=0) is None


# ── Factory ──────────────────────────────────────────────────────────────

def test_factory_defaults_to_tradingview(tmp_path):
    broker = build_broker(_config(tmp_path))
    assert isinstance(broker, TradingViewBrokerAdapter)


def test_factory_rejects_an_unknown_provider(tmp_path):
    with pytest.raises(ValueError, match="Unknown BROKER_PROVIDER"):
        build_broker(_config(tmp_path, provider="etrade"))
