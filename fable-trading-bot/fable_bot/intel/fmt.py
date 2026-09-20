"""Price formatting that respects the instrument.

Two decimals is right for gold at 4,400 and destroys FX at 1.15: a 12-pip
stop prints as 0.00 and every pool in a liquidity map collapses to "1.15".
Decimals follow price magnitude -- there is no per-instrument table to keep
in sync, and the rule matches how each market quotes itself.
"""
from __future__ import annotations


def decimals_for(price: float) -> int:
    p = abs(price)
    if p < 10:
        return 5      # FX majors, most crosses
    if p < 1000:
        return 3      # JPY crosses, silver, small ETFs
    return 2          # gold, indices


def px(price: float, ref: float | None = None) -> str:
    """Format `price` with decimals chosen from `ref` (default: the price itself)."""
    d = decimals_for(ref if ref is not None else price)
    return f"{price:,.{d}f}"
