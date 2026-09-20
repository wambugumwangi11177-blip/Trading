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


# ── shared live_runner cycle harness ──────────────────────────────────────
# Lives here rather than in one test module because more than one suite needs
# it: the lesson-gate tests and the unvalidated-quarantine tests both drive a
# full run_signal_check with the broker, feed and strategies stubbed.

import json as _json
import math as _math
from types import SimpleNamespace as _NS


@pytest.fixture
def cycle(monkeypatch, tmp_path):
    from fable_bot.broker.base import AccountSnapshot, OrderResult
    from fable_bot.broker.tradingview_adapter import TradingViewError
    from fable_bot.journal import Journal
    from fable_bot.strategies.base import Signal
    import fable_bot.lessons as lessons_mod
    import fable_bot.live_runner as lr

    memory_dir = tmp_path / "memory"
    monkeypatch.setattr(lessons_mod, "MEMORY_DIR", memory_dir)
    # live_runner keeps the durable drawdown peak under MEMORY_DIR too; without
    # redirecting it the suite reads the real peak and halts on stub equity.
    monkeypatch.setattr(lr, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(lr, "_drawdown_monitor", None)

    requested: list[list[str]] = []

    def fake_fetch(symbols, period="1y"):
        requested.append(list(symbols))
        frames = {}
        for symbol in symbols:
            index = pd.date_range(end="2026-08-28", periods=60, freq="D")
            close = pd.Series(np.linspace(100, 110, 60), index=index)
            frames[symbol] = pd.DataFrame(
                {"open": close, "high": close * 1.01, "low": close * 0.99,
                 "close": close, "volume": 1_000}, index=index)
        return frames

    monkeypatch.setattr(lr, "fetch_universe_history", fake_fetch)

    signals: dict[str, tuple[str, float, float]] = {}

    class StubStrategy:
        name = "stub"

        def generate_signals(self, symbol, df):
            spec = signals.get(symbol)
            if not spec:
                return []
            side, price, stop = spec
            return [Signal(df.index[-1], symbol, side, price, stop, "stub signal")]

        def catch_up_signal(self, symbol, df):
            return None

    monkeypatch.setattr(lr, "build_strategy", lambda name: StubStrategy())
    monkeypatch.setattr(lr, "build_event_filter", lambda cfg: _NS(
        available=True, is_blocked=lambda currencies: (False, "")))

    positions: dict[str, dict] = {}
    closed: list[str] = []

    class FakeBroker:
        def __init__(self):
            self.submitted = []
            self.submit_exc = None
            self.order_id = "111"
            self.verify_row = None

        def get_account(self):
            return AccountSnapshot(equity=100_000.0, cash=100_000.0, is_paper=True)

        def get_open_positions(self):
            return dict(positions)

        def venue_symbol(self, inst):
            if not inst.tv_symbol:
                raise TradingViewError(f"{inst.symbol} has no tv_symbol")
            return inst.tv_symbol

        def quantize(self, inst, qty):
            step = inst.qty_step or 1.0
            decimals = max(0, -_math.floor(_math.log10(step))) if step < 1 else 0
            return round(_math.floor(abs(qty) / step) * step, decimals)

        def submit_order(self, symbol, qty, side, stop_price=None):
            self.submitted.append((symbol, side, qty))
            if self.submit_exc is not None:
                raise self.submit_exc
            return OrderResult(symbol=symbol, side=side, qty=qty,
                               status="filled", broker_order_id=self.order_id)

        def close_position(self, symbol):
            closed.append(symbol)
            positions.pop(symbol, None)
            return OrderResult(symbol=symbol, side="flat", qty=0.0,
                               status="filled", broker_order_id="close-1")

        def verify_order(self, symbol, side, qty, *, attempts=3, delay_seconds=2.0):
            return self.verify_row

    broker = FakeBroker()
    monkeypatch.setattr(lr, "build_broker", lambda config: broker)

    journal = Journal(run_id="lesson-test", log_dir=tmp_path / "logs")

    def events():
        text = journal.path.read_text(encoding="utf-8")
        return [_json.loads(line) for line in text.strip().splitlines() if line.strip()]

    def open_position(venue_symbol, side, qty=1.0, avg_entry_price=100.0):
        positions[venue_symbol] = {"side": side, "qty": qty,
                                   "avg_entry_price": avg_entry_price}

    return _NS(lr=lr, broker=broker, journal=journal, signals=signals,
               memory_dir=memory_dir, requested=requested, events=events,
               open_position=open_position, closed=closed, positions=positions)
