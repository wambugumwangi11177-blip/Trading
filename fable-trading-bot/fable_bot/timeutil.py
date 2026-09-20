"""Shared time helpers.

The bot lives across two clock conventions: broker order history arrives as
UTC ISO strings with a trailing ``Z`` (JavaScript ``toISOString``), while the
journal writes UTC ISO with a ``+00:00`` offset. Python 3.10's
``fromisoformat`` rejects the ``Z`` form, so everything parses through one
helper that normalizes both.
"""
from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat(timespec="seconds")


def parse_utc_iso(value: str) -> datetime:
    """Parse either ``...Z`` or ``...+00:00`` UTC ISO strings to an aware datetime."""
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
