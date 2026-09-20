from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import pandas as pd

Side = Literal["long", "short", "flat"]


@dataclass
class Signal:
    date: pd.Timestamp
    symbol: str
    side: Side
    price: float
    stop_price: float
    reason: str


class Strategy(ABC):
    name: str

    @abstractmethod
    def generate_signals(self, symbol: str, df: pd.DataFrame) -> list[Signal]:
        """Return one Signal per bar where the strategy wants to enter or flip.

        `df` must have columns open/high/low/close/volume and a date index,
        already sorted ascending. Implementations should only look at data up
        to and including the current row (no lookahead).
        """
        raise NotImplementedError

    def catch_up_signal(self, symbol: str, df: pd.DataFrame) -> Signal | None:
        """Current desired state, repriced on the last bar.

        State-based strategies whose position is fully determined by the latest
        bar override this so the live runner can reconcile the book after a
        missed flip (watcher down, instrument added mid-trend). Returning None
        keeps the historical behavior: stale signals are simply skipped.
        """
        return None
