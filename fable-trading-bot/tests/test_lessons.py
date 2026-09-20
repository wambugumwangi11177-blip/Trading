import json
from datetime import timedelta

import pytest

from fable_bot.lessons import (
    ACTIVE,
    DISABLED,
    InvalidLesson,
    LessonBook,
    LessonGate,
    halt_until_param,
    mistake_signature,
    validate_lesson,
)
from fable_bot.timeutil import utc_now


@pytest.fixture
def book(tmp_path):
    return LessonBook(memory_dir=tmp_path)


class TestRuleRegistry:
    def test_unknown_rule_type_rejected(self):
        with pytest.raises(InvalidLesson):
            validate_lesson("make_me_a_sandwich", {})

    def test_block_symbol_requires_symbols(self):
        with pytest.raises(InvalidLesson):
            validate_lesson("block_symbol", {"symbols": []})

    def test_block_symbol_canonicalizes(self):
        params = validate_lesson("block_symbol", {"symbols": ["B:X", "A:Y", "A:Y", " "]})
        assert params == {"symbols": ["A:Y", "B:X"]}

    def test_sizing_params_clamped(self):
        params = validate_lesson("sizing_notional_bounds", {
            "risk_pct_min": 5.0, "risk_pct_max": 99.0,
            "max_notional_pct_equity": 500.0, "min_notional_usd": -1.0,
        })
        assert params["risk_pct_min"] <= params["risk_pct_max"] <= 1.5
        assert params["max_notional_pct_equity"] == 10.0
        assert params["min_notional_usd"] == 0.0

    def test_max_adds_clamped_to_range(self):
        assert validate_lesson("max_adds_per_position", {"max_adds": 99}) == {"max_adds": 3}
        assert validate_lesson("max_adds_per_position", {"max_adds": -2}) == {"max_adds": 0}

    def test_verification_params_clamped(self):
        params = validate_lesson("require_verification",
                                 {"verify_attempts": 500, "verify_delay_seconds": 900})
        assert params == {"verify_attempts": 10, "verify_delay_seconds": 30.0}


class TestLessonBook:
    def test_add_and_roundtrip(self, book):
        lesson = book.add("block_symbol", {"symbols": ["NASDAQ:NDX"]},
                          source="reviewer", incident="test", note="first")
        assert lesson is not None and lesson.status == ACTIVE

        reloaded = LessonBook(memory_dir=book.dir)
        assert len(reloaded.all_lessons()) == 1
        got = reloaded.get(lesson.id)
        assert got is not None and got.params == {"symbols": ["NASDAQ:NDX"]}

    def test_signature_dedupe(self, book):
        first = book.add("alert_on_gap", {"max_missing_sessions": 1},
                         source="reviewer", incident="test")
        second = book.add("alert_on_gap", {"max_missing_sessions": 1},
                          source="reviewer", incident="test again")
        assert first is not None
        assert second is None

    def test_dedupe_respects_disabled_veto(self, book):
        lesson = book.add("alert_on_gap", {"max_missing_sessions": 1},
                          source="reviewer", incident="test")
        assert book.set_status(lesson.id, DISABLED)
        # Same mistake resurfaces: the vetoed signature must not come back to life.
        assert book.add("alert_on_gap", {"max_missing_sessions": 1},
                        source="reviewer", incident="test again") is None

    def test_status_change_appends_and_folds(self, book):
        lesson = book.add("max_adds_per_position", {"max_adds": 0},
                          source="seeded", incident="test")
        assert book.set_status(lesson.id, DISABLED)
        reloaded = LessonBook(memory_dir=book.dir)
        assert reloaded.get(lesson.id).status == DISABLED
        assert reloaded.active_lessons() == []

        lines = book.path.read_text(encoding="utf-8").strip().splitlines()
        assert json.loads(lines[-1])["kind"] == "status_change"

    def test_torn_final_line_tolerated(self, book):
        lesson = book.add("note", {"topic": "x"}, source="seeded", incident="test")
        assert lesson is not None
        with book.path.open("a", encoding="utf-8") as fh:
            fh.write('{"kind": "lesson", "id": "L-broken", "rule_type"')  # torn line
        reloaded = LessonBook(memory_dir=book.dir)
        assert len(reloaded.all_lessons()) == 1
        assert reloaded.load_errors

    def test_invalid_rule_in_file_is_skipped_not_fatal(self, book):
        good = book.add("note", {"topic": "ok"}, source="seeded", incident="test")
        with book.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "lesson", "id": "L-bad", "ts": "t",
                                 "rule_type": "unknown_type", "params": {},
                                 "mistake_signature": "sha256:zz"}) + "\n")
        reloaded = LessonBook(memory_dir=book.dir)
        assert len(reloaded.all_lessons()) == 1
        assert reloaded.get(good.id) is not None

    def test_seed_is_idempotent(self, book):
        first = book.seed()
        second = book.seed()
        assert len(first) == 6
        assert second == []
        assert len(book.all_lessons()) == 6


class TestLessonGate:
    def test_block_symbol_blocks_entry(self, book):
        book.add("block_symbol", {"symbols": ["NASDAQ:NDX"]},
                 source="seeded", incident="test")
        gate = LessonGate(book)
        blocked, reason, lesson_id = gate.blocked_entry("NASDAQ:NDX")
        assert blocked and lesson_id
        assert gate.blocked_entry("AMEX:SPY")[0] is False

    def test_expired_halt_does_not_block(self, book):
        past = (utc_now() - timedelta(days=1)).isoformat(timespec="seconds")
        book.add("halt_entries_for_symbol", {"symbol": "AMEX:GLD", "until": past},
                 source="reviewer", incident="test")
        gate = LessonGate(book)
        assert gate.blocked_entry("AMEX:GLD")[0] is False

    def test_unexpired_halt_blocks(self, book):
        book.add("halt_entries_for_symbol",
                 {"symbol": "AMEX:GLD", "until": halt_until_param(30)},
                 source="reviewer", incident="test")
        gate = LessonGate(book)
        blocked, reason, lesson_id = gate.blocked_entry("AMEX:GLD")
        assert blocked and lesson_id

    def test_most_restrictive_combination(self, book):
        book.add("sizing_notional_bounds",
                 {"risk_pct_min": 0.1, "risk_pct_max": 1.4,
                  "max_notional_pct_equity": 5.0, "min_notional_usd": 50.0},
                 source="seeded", incident="a")
        book.add("sizing_notional_bounds",
                 {"risk_pct_min": 0.4, "risk_pct_max": 1.2,
                  "max_notional_pct_equity": 1.5, "min_notional_usd": 200.0},
                 source="reviewer", incident="b")
        gate = LessonGate(book)
        bounds = gate.sizing_bounds()
        assert bounds == {"risk_pct_min": 0.4, "risk_pct_max": 1.2,
                          "max_notional_pct_equity": 1.5, "min_notional_usd": 200.0,
                          "min_notional_pct_equity": 0.10}

    def test_max_adds_authority_is_most_restrictive(self, book):
        book.add("max_adds_per_position", {"max_adds": 2}, source="seeded", incident="a")
        book.add("max_adds_per_position", {"max_adds": 0}, source="reviewer", incident="b")
        gate = LessonGate(book)
        assert gate.max_adds_authority() is not None

    def test_empty_book_defaults(self, book):
        gate = LessonGate(book)
        assert gate.active_count == 0
        assert gate.sizing_bounds()["risk_pct_max"] == 1.5
        assert gate.verification_config()["verify_attempts"] == 3
        assert gate.gap_config()["max_missing_sessions"] == 1

    def test_signature_stable_across_param_order(self):
        # Canonicalization happens in validation, so signatures are order-independent.
        sig1 = mistake_signature("block_symbol",
                                 validate_lesson("block_symbol", {"symbols": ["A:X", "B:Y"]}))
        sig2 = mistake_signature("block_symbol",
                                 validate_lesson("block_symbol", {"symbols": ["B:Y", "A:X"]}))
        assert sig1 == sig2
