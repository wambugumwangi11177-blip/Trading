"""Dealer gamma exposure: the maths, the sign convention, and the NaN trap.

compute_gamma_profile is pure, so all of this runs without a feed.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from fable_bot.intel.gamma import black_scholes_gamma, compute_gamma_profile

NOW = datetime(2026, 9, 21, 13, 30, tzinfo=timezone.utc)


def expiry(days: int) -> str:
    return (NOW + timedelta(days=days)).date().isoformat()


def chain(calls, puts, days=30):
    return [{"expiry": expiry(days), "calls": calls, "puts": puts}]


def opt(strike, oi, iv=0.20):
    return {"strike": strike, "open_interest": oi, "implied_volatility": iv}


# ── Black-Scholes gamma ───────────────────────────────────────────────────

def test_gamma_peaks_at_the_money():
    atm = black_scholes_gamma(100.0, 100.0, 0.25, 0.2)
    otm = black_scholes_gamma(100.0, 130.0, 0.25, 0.2)
    itm = black_scholes_gamma(100.0, 70.0, 0.25, 0.2)
    assert atm > otm and atm > itm


def test_gamma_rises_as_expiry_approaches_for_atm():
    near = black_scholes_gamma(100.0, 100.0, 1 / 365, 0.2)
    far = black_scholes_gamma(100.0, 100.0, 1.0, 0.2)
    assert near > far


def test_degenerate_inputs_return_zero_not_an_exception():
    assert black_scholes_gamma(0.0, 100.0, 0.25, 0.2) == 0.0
    assert black_scholes_gamma(100.0, 100.0, 0.0, 0.2) == 0.0
    assert black_scholes_gamma(100.0, 100.0, 0.25, 0.0) == 0.0
    assert black_scholes_gamma(-1.0, 100.0, 0.25, 0.2) == 0.0


def test_nan_inputs_return_zero():
    """A NaN implied vol slips past `iv <= 0` -- every comparison to NaN is False.

    One such row turned a whole live GLD profile into NaN, which then read as
    "short gamma" because `nan >= 0` is False too.
    """
    nan = float("nan")
    assert black_scholes_gamma(100.0, 100.0, 0.25, nan) == 0.0
    assert black_scholes_gamma(nan, 100.0, 0.25, 0.2) == 0.0
    assert black_scholes_gamma(100.0, nan, 0.25, 0.2) == 0.0


# ── profile construction ──────────────────────────────────────────────────

def test_calls_are_positive_gamma_puts_negative():
    """The stated convention: dealers long calls, short puts."""
    calls_only = compute_gamma_profile("X", 100.0, chain([opt(100, 1000)], []), now=NOW)
    puts_only = compute_gamma_profile("X", 100.0, chain([], [opt(100, 1000)]), now=NOW)
    assert calls_only.net_gex > 0
    assert puts_only.net_gex < 0
    assert calls_only.regime == "long_gamma"
    assert puts_only.regime == "short_gamma"


def test_balanced_book_nets_to_zero():
    profile = compute_gamma_profile(
        "X", 100.0, chain([opt(100, 1000)], [opt(100, 1000)]), now=NOW)
    assert profile.net_gex == pytest.approx(0.0, abs=1e-6)


def test_nan_row_does_not_poison_the_profile():
    rows = [opt(100, 1000), {"strike": 105.0, "open_interest": 500,
                             "implied_volatility": float("nan")}]
    profile = compute_gamma_profile("X", 100.0, chain(rows, []), now=NOW)
    assert math.isfinite(profile.net_gex)
    assert profile.net_gex > 0
    assert profile.regime == "long_gamma"


def test_walls_are_the_extreme_signed_strikes():
    profile = compute_gamma_profile(
        "X", 100.0,
        chain([opt(105, 5000), opt(100, 100)], [opt(95, 5000)]),
        now=NOW)
    assert profile.call_wall == pytest.approx(105.0)
    assert profile.put_wall == pytest.approx(95.0)


def test_expired_chains_are_ignored():
    past = [{"expiry": (NOW - timedelta(days=2)).date().isoformat(),
             "calls": [opt(100, 10_000)], "puts": []}]
    profile = compute_gamma_profile("X", 100.0, past, now=NOW)
    assert profile.net_gex == 0.0
    assert profile.contracts_used == 0


def test_strikes_far_from_spot_are_excluded():
    profile = compute_gamma_profile(
        "X", 100.0, chain([opt(100, 100), opt(400, 999_999)], []),
        now=NOW, strike_range_pct=20.0)
    assert [s.strike for s in profile.by_strike] == [100.0]


def test_zero_open_interest_contributes_nothing():
    profile = compute_gamma_profile("X", 100.0, chain([opt(100, 0)], []), now=NOW)
    assert profile.net_gex == 0.0


def test_gamma_flip_found_between_put_and_call_heavy_zones():
    """Puts below, calls above: net gamma must cross zero in between."""
    profile = compute_gamma_profile(
        "X", 100.0,
        chain([opt(104, 4000), opt(106, 4000)], [opt(96, 4000), opt(94, 4000)]),
        now=NOW)
    assert profile.gamma_flip is not None
    assert 94.0 < profile.gamma_flip < 106.0


def test_gamma_flip_is_none_when_sign_never_changes():
    """Calls only: long gamma everywhere nearby. None is the honest answer."""
    profile = compute_gamma_profile("X", 100.0, chain([opt(100, 5000)], []), now=NOW)
    assert profile.gamma_flip is None


def test_non_positive_spot_is_rejected():
    with pytest.raises(ValueError):
        compute_gamma_profile("X", 0.0, chain([opt(100, 1)], []), now=NOW)


def test_convention_is_recorded_on_every_profile():
    profile = compute_gamma_profile("X", 100.0, chain([opt(100, 10)], []), now=NOW)
    assert profile.dealer_convention == "long_calls_short_puts"


def test_summary_states_the_regime_and_never_raises():
    profile = compute_gamma_profile(
        "X", 100.0, chain([opt(104, 3000)], [opt(96, 3000)]), now=NOW)
    text = profile.summary()
    assert "dealer gamma" in text.lower()
    assert "net GEX" in text
