"""Run notifications.

The property that matters is not that messages get delivered -- it is that
nothing about messaging can affect trading. An unconfigured Telegram must be a
silent no-op, and a Telegram that is configured but broken (network down, token
revoked, chat deleted) must not take a trading cycle down with it. The trade is
the job; the message is commentary.

The second concern is coverage: the whole reason for alerting is that a run
which *could not trade* is otherwise indistinguishable from a run with nothing
to do. Both are silence.
"""
from __future__ import annotations

from datetime import date

import pytest

from fable_bot.auto_run import ANNOUNCED, _announce
from fable_bot.config import TelegramConfig
from fable_bot.telegram_bot import notify


# ── nothing about messaging may break trading ────────────────────────────


def test_unconfigured_telegram_is_a_silent_no_op():
    assert notify("hello", TelegramConfig(bot_token=None, chat_id=None)) is False


def test_half_configured_telegram_is_also_a_no_op():
    assert notify("hello", TelegramConfig(bot_token="123:abc", chat_id=None)) is False
    assert notify("hello", TelegramConfig(bot_token=None, chat_id="42")) is False


def test_a_failing_send_is_swallowed(monkeypatch):
    """A revoked token or a dead network must not raise into the caller."""
    import fable_bot.telegram_bot as tg

    class Exploding:
        def __init__(self, config):
            raise RuntimeError("telegram is down")

    monkeypatch.setattr(tg, "TelegramBriefing", Exploding)
    assert notify("hello", TelegramConfig(bot_token="123:abc", chat_id="42")) is False


def test_announce_never_raises_when_unconfigured():
    """_announce runs inside finish(), which every exit path goes through."""
    for status in ANNOUNCED:
        _announce(status, "some detail", ["ORDER AMEX:SPY sell qty=78"], date(2026, 8, 6))


# ── the failures are the point ───────────────────────────────────────────


@pytest.mark.parametrize("status", ["preflight_failed", "error", "too_early", "locked", "market_closed"])
def test_failure_outcomes_are_announced(status):
    """A run that could not trade must not be silent -- silence reads as
    'no signals today', which is the opposite conclusion."""
    assert status in ANNOUNCED
    assert ANNOUNCED[status], f"{status} needs a headline explaining it did not trade"


def test_successful_and_dry_runs_are_announced():
    assert "ok" in ANNOUNCED
    assert "dry_run" in ANNOUNCED


@pytest.mark.parametrize("status", ["no_session", "already_ran"])
def test_benign_no_ops_stay_quiet(status):
    """Weekends and repeat invocations must not spam, or the alerts stop
    carrying information."""
    assert status not in ANNOUNCED


def test_the_message_carries_the_actions(monkeypatch):
    sent: list[str] = []
    import fable_bot.auto_run as ar

    monkeypatch.setattr("fable_bot.telegram_bot.notify", lambda text, config=None: sent.append(text))
    ar._announce("ok", "1 action(s), Equity 113,637.55",
                 ["ORDER AMEX:SPY sell qty=78 status=filled"], date(2026, 8, 6))

    assert len(sent) == 1
    body = sent[0]
    assert "2026-08-06" in body
    assert "ORDER AMEX:SPY sell qty=78 status=filled" in body
    assert "Equity 113,637.55" in body


def test_a_quiet_day_says_so_rather_than_sending_an_empty_message(monkeypatch):
    sent: list[str] = []
    import fable_bot.auto_run as ar

    monkeypatch.setattr("fable_bot.telegram_bot.notify", lambda text, config=None: sent.append(text))
    ar._announce("ok", "no signals on today's bar", [], date(2026, 8, 6))

    assert "No signals on today's bar" in sent[0]


def test_a_missed_session_is_announced(monkeypatch):
    """The machine being asleep at the open leaves no error anywhere -- it is
    the failure most likely to pass unnoticed, so it must speak up."""
    sent: list[str] = []
    import fable_bot.auto_run as ar

    monkeypatch.setattr("fable_bot.telegram_bot.notify", lambda text, config=None: sent.append(text))
    ar._announce("market_closed", "Session for 2026-08-06 already closed at 16:00 ET", [], date(2026, 8, 6))

    assert "MISSED THE SESSION" in sent[0]


def test_a_preflight_failure_explains_itself(monkeypatch):
    sent: list[str] = []
    import fable_bot.auto_run as ar

    monkeypatch.setattr("fable_bot.telegram_bot.notify", lambda text, config=None: sent.append(text))
    ar._announce("preflight_failed", "TradingView is up but no broker answered", [], date(2026, 8, 6))

    body = sent[0]
    assert "COULD NOT TRADE" in body
    assert "no broker answered" in body
