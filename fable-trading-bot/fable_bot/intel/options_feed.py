"""Option chain retrieval, kept separate from the gamma maths.

compute_gamma_profile() is pure and testable; this is the part that touches the
network and therefore the part that fails. Keeping them apart means a feed
outage degrades the report instead of breaking the calculation, and a stored
chain can be replayed through the same maths later.

Source is yfinance, which exposes the OCC-reported chain (strike, open interest,
implied volatility) for listed US options. Open interest is the number that
matters here and it is end-of-day: the chain you read intraday reflects
YESTERDAY's settled positioning. That lag is inherent to free OI data and is
recorded on every profile rather than papered over -- a gamma profile is a map
of positioning as of the last settlement, not a live tape.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from .gamma import DEFAULT_CONTRACT_SIZE, DEFAULT_RISK_FREE_RATE, GammaProfile, compute_gamma_profile

logger = logging.getLogger("fable_bot.intel.options_feed")


class OptionsFeedError(RuntimeError):
    """The chain could not be retrieved. Callers degrade; they do not crash."""


def _rows(frame, kind: str) -> list[dict]:
    """Normalise a yfinance options DataFrame into plain dicts.

    Column names are yfinance's, not ours, so they are mapped here and nowhere
    else. Rows missing open interest or implied volatility are kept with zeros
    and dropped downstream by compute_gamma_profile, which already treats a
    zero-IV row as unusable.
    """
    if frame is None or getattr(frame, "empty", True):
        return []
    out: list[dict] = []
    for record in frame.to_dict("records"):
        try:
            out.append({
                "strike": float(record.get("strike")),
                "open_interest": float(record.get("openInterest") or 0.0),
                "implied_volatility": float(record.get("impliedVolatility") or 0.0),
                "volume": float(record.get("volume") or 0.0),
                "kind": kind,
            })
        except (TypeError, ValueError):
            continue
    return out


def fetch_option_chains(symbol: str, *, max_expiries: int = 6) -> list[dict]:
    """Fetch up to `max_expiries` nearest expiries for `symbol`.

    Near-dated expiries carry almost all the gamma -- gamma decays with time to
    expiry, so a 2-year LEAP contributes essentially nothing next to a weekly.
    Six covers the front weeklies plus the front monthlies, which is where
    dealer hedging actually bites, and keeps the request count small.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise OptionsFeedError("yfinance is not installed") from exc

    try:
        ticker = yf.Ticker(symbol)
        expiries = list(ticker.options or [])
    except Exception as exc:  # noqa: BLE001 - any feed failure is the same to us
        raise OptionsFeedError(f"could not list expiries for {symbol}: {exc}") from exc

    if not expiries:
        raise OptionsFeedError(f"{symbol} has no listed option expiries")

    chains: list[dict] = []
    for expiry in expiries[:max_expiries]:
        try:
            chain = ticker.option_chain(expiry)
        except Exception as exc:  # noqa: BLE001
            # One bad expiry must not lose the other five.
            logger.warning("chain fetch failed for %s %s: %s", symbol, expiry, exc)
            continue
        chains.append({
            "expiry": str(expiry),
            "calls": _rows(getattr(chain, "calls", None), "call"),
            "puts": _rows(getattr(chain, "puts", None), "put"),
        })

    if not chains:
        raise OptionsFeedError(f"no usable option chains retrieved for {symbol}")
    return chains


def fetch_spot(symbol: str) -> float:
    """Last traded price for the underlying."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        history = ticker.history(period="5d", interval="1d")
        if history is not None and not history.empty:
            return float(history["close" if "close" in history.columns else "Close"].iloc[-1])
        info = getattr(ticker, "fast_info", None)
        if info is not None:
            price = info.get("last_price") if hasattr(info, "get") else None
            if price:
                return float(price)
    except Exception as exc:  # noqa: BLE001
        raise OptionsFeedError(f"could not read spot for {symbol}: {exc}") from exc
    raise OptionsFeedError(f"no spot price available for {symbol}")


def fetch_gamma_profile(
    symbol: str,
    *,
    spot: float | None = None,
    max_expiries: int = 6,
    contract_size: float = DEFAULT_CONTRACT_SIZE,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    now: datetime | None = None,
) -> GammaProfile:
    """Fetch chains and build the profile. Raises OptionsFeedError on feed failure."""
    resolved_spot = spot if spot is not None else fetch_spot(symbol)
    chains = fetch_option_chains(symbol, max_expiries=max_expiries)
    return compute_gamma_profile(
        symbol, resolved_spot, chains,
        now=now or datetime.now(timezone.utc),
        contract_size=contract_size,
        risk_free_rate=risk_free_rate,
    )
