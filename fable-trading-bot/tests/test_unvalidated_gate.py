"""The unvalidated-universe quarantine must block ENTRIES ONLY.

A gate placed above the flat/close branch strands every open position in a
quarantined instrument: the bot can never exit what it can no longer evaluate.
When the quarantine was introduced, USDCHF, UKX and EURGBP were all open in the
paper account, so an exit-blocking gate would have left three live positions
with no way out except by hand.

This is the same invariant the lesson gate already documents, and it is easy to
break by adding a new gate in the obvious-looking place at the top of the loop.
"""
from __future__ import annotations

import dataclasses

import pytest

from fable_bot.config import INSTRUMENTS


def _by_symbol(symbol):
    for inst in INSTRUMENTS:
        if inst.symbol == symbol:
            return inst
    raise AssertionError(f"{symbol} not in INSTRUMENTS")


def test_the_quarantine_actually_covers_the_2026_08_additions():
    """The instruments config.py marks NOT backtest-validated must be excluded."""
    for symbol in ("EURGBP=X", "USDCHF=X", "^FTSE", "GC=F", "^SOX"):
        assert _by_symbol(symbol).validated is False, symbol


def test_the_documented_core_stays_tradable():
    for symbol in ("SPY", "QQQ", "BTC-USD", "GLD", "USO", "MGC=F"):
        inst = _by_symbol(symbol)
        assert inst.validated is True, symbol
        assert inst.tradable is True, symbol


def test_spot_gold_lot_size_is_no_longer_a_silent_zero():
    """qty_step=1000 made every spot-gold order quantize down to nothing."""
    gold = _by_symbol("GC=F")
    assert gold.qty_step == 1.0
    # ~12 ounces at this account size must survive quantization.
    assert (12.6 // gold.qty_step) * gold.qty_step > 0


def test_entry_is_blocked_for_an_unvalidated_instrument(cycle):
    cycle.signals["EURGBP=X"] = ("short", 0.8580, 0.8620)
    cycle.lr.run_signal_check(dry_run=True, journal=cycle.journal)
    events = cycle.events()
    gates = [e for e in events if e.get("gate") == "unvalidated"]
    assert gates, "expected an unvalidated skip"
    assert not [e for e in events if e["kind"] == "order"], "no order may be placed"


def test_exit_is_allowed_for_an_unvalidated_instrument(cycle):
    """The regression this file exists for: a flat signal must still close."""
    cycle.open_position("OANDA:EURGBP", "short")
    cycle.signals["EURGBP=X"] = ("flat", 0.8580, 0.8620)

    cycle.lr.run_signal_check(dry_run=False, journal=cycle.journal)

    closes = [e for e in cycle.events() if e["kind"] == "close"]
    assert closes, "an open unvalidated position must still be closeable"
    assert closes[0]["symbol"] == "OANDA:EURGBP"
    assert "OANDA:EURGBP" in cycle.closed


def test_exit_precedes_the_gate_for_every_open_unvalidated_position(cycle):
    """All three that were live when the quarantine landed."""
    for venue, sym, side in (("OANDA:USDCHF", "USDCHF=X", "long"),
                             ("TVC:UKX", "^FTSE", "long"),
                             ("OANDA:EURGBP", "EURGBP=X", "short")):
        cycle.open_position(venue, side)
        cycle.signals[sym] = ("flat", 100.0, 99.0)

    cycle.lr.run_signal_check(dry_run=False, journal=cycle.journal)

    assert set(cycle.closed) == {"OANDA:USDCHF", "TVC:UKX", "OANDA:EURGBP"}


def test_trade_unvalidated_flag_restores_entries(cycle, monkeypatch):
    import dataclasses as dc

    from fable_bot import live_runner as lr
    monkeypatch.setattr(lr, "RISK", dc.replace(lr.RISK, trade_unvalidated=True,
                                               max_concurrent_positions=99))
    # Stop distance chosen so the order clears sizing_sanity: a tighter stop
    # sizes to 250k units (214% of equity notional) and is refused there
    # instead, which would pass this test for the wrong reason.
    cycle.signals["EURGBP=X"] = ("short", 0.8580, 0.8780)
    cycle.lr.run_signal_check(dry_run=True, journal=cycle.journal)

    events = cycle.events()
    assert not [e for e in events if e.get("gate") == "unvalidated"],         "the quarantine must be off when TRADE_UNVALIDATED is set"
    orders = [e for e in events if e["kind"] == "order"]
    assert any(e["symbol"] == "OANDA:EURGBP" for e in orders),         f"no EURGBP order; skips were {[(e.get('symbol'), e.get('gate')) for e in events if e['kind']=='skip']}"
