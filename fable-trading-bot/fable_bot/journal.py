"""Append-only trade journal.

An unattended run leaves nobody watching stdout, so every decision it makes has
to be recoverable afterwards. This writes one JSON object per line to
`logs/journal-YYYY-MM-DD.jsonl` -- append-only, never rewritten, flushed per
event so a crash mid-run still leaves the events that preceded it on disk.

Why JSONL rather than the log file: the log is for reading, this is for
answering questions ("did it trade on the 12th, and why not?") without parsing
prose. Both get written; they serve different purposes.

The journal is also what makes the runner idempotent. `completed_session_days`
reads back which sessions already finished a run, so a Task Scheduler retry or a
manual re-run on the same day exits instead of doubling a position.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import date, datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT

LOG_DIR = Path(os.getenv("FABLE_LOG_DIR", str(PROJECT_ROOT / "logs")))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Journal:
    """One run's worth of events, appended to today's journal file."""

    def __init__(self, run_id: str | None = None, log_dir: Path | None = None):
        self.run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.dir = log_dir or LOG_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"journal-{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"

    def event(self, kind: str, **fields: Any) -> dict:
        """Append one event. Never raises -- journalling must not break a run."""
        record = {"ts": _utc_now_iso(), "run_id": self.run_id, "kind": kind, **fields}
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
                fh.flush()
        except OSError as exc:  # noqa: BLE001 - a failed write must not abort trading
            logging.getLogger("fable_bot.journal").error("journal write failed: %s", exc)
        return record


def setup_file_logging(level: int = logging.INFO, log_dir: Path | None = None) -> Path:
    """Send the logging tree to a rotating file as well as the console.

    5 MB x 5 files is months of daily-bar runs; unattended processes are exactly
    the ones nobody prunes by hand, so the cap is not optional.
    """
    directory = log_dir or LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "fable-bot.log"

    handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.setLevel(level)

    # Under pythonw.exe -- which is how the scheduled task runs, so no console
    # window appears -- there is no stdout or stderr. A StreamHandler built
    # around None fails on every single record. logging happens to swallow that,
    # but silently broken handlers are not worth keeping.
    for existing in list(root.handlers):
        if isinstance(existing, logging.StreamHandler) and not isinstance(existing, RotatingFileHandler):
            if getattr(existing, "stream", None) is None:
                root.removeHandler(existing)

    # Re-running in the same process (tests, `serve`) must not stack handlers.
    if not any(isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == path for h in root.handlers):
        root.addHandler(handler)
    return path


def completed_session_days(log_dir: Path | None = None, lookback_files: int = 5) -> set[str]:
    """Session days (ISO dates) that already have a finished, order-placing run.

    Only `run_end` events with status "ok" count. A run that died in preflight
    never reached the order stage, so re-running it later the same day is safe
    and should be allowed.
    """
    directory = log_dir or LOG_DIR
    if not directory.exists():
        return set()

    days: set[str] = set()
    for path in sorted(directory.glob("journal-*.jsonl"))[-lookback_files:]:
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # a torn final line from a hard kill; skip it
                    if record.get("kind") == "run_end" and record.get("status") == "ok":
                        session_day = record.get("session_day")
                        if session_day:
                            days.add(str(session_day))
        except OSError:
            continue
    return days


def already_ran(session_day: date, log_dir: Path | None = None) -> bool:
    return session_day.isoformat() in completed_session_days(log_dir)
