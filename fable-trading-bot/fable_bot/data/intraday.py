"""Session-aware intraday bars for index futures.

Everything intraday hinges on knowing where the trading session starts, so this
module does the unglamorous work the daily feed never needed: localise to
exchange time, cut the overnight Globex tape down to the regular session, label
each bar with the session it belongs to, and derive the per-session anchors
(open, VWAP, minutes elapsed) that the strategy references.

Regular Trading Hours are used deliberately. Index futures trade nearly 24h,
but the intraday momentum literature (Gao et al. 2018; Zarattini, Aziz &
Barbon 2024) is entirely about the RTH session -- the open is what resets
positioning, and overnight Globex volume is thin enough that its "breakouts"
are mostly noise.

Two data facts drive how this gets used, and both are traps:

  * yfinance serves ~60 days of sub-hourly history but ~730 days of hourly.
  * Hourly bars are aligned to the hour, so there is NO 09:30 bar -- the RTH
    open sits inside the excluded 09:00 bar and the first available bar is
    10:00. 30m and 15m bars do align to 09:30.

So an open-anchored strategy is only honestly testable on 30m/15m (correct
anchor, short sample), while hourly offers a long sample at a 10:00 anchor.
`fetch_intraday` records which anchor it actually got in `df.attrs["anchor"]`
rather than quietly pretending it starts at the open.
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

EXCHANGE_TZ = "America/New_York"
RTH_OPEN = "09:30"
RTH_CLOSE = "16:00"

# yfinance caps intraday history by interval; requesting more silently returns
# less, so the ceilings are explicit here.
MAX_PERIOD = {"1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d", "1h": "730d"}


def fetch_intraday(symbol: str, interval: str = "1h", period: str | None = None,
                    rth_only: bool = True) -> pd.DataFrame:
    """Fetch intraday bars indexed in exchange time, optionally RTH-only.

    Returns columns: open, high, low, close, volume, plus the session helpers
    `session` (date), `bar_of_session` (0-based), and `minutes_from_open`.
    """
    period = period or MAX_PERIOD.get(interval, "60d")
    raw = yf.Ticker(symbol).history(period=period, interval=interval)
    if raw.empty:
        raise ValueError(f"No intraday data for {symbol!r} at {interval} over {period}")

    df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
    df.index = (df.index.tz_convert(EXCHANGE_TZ) if df.index.tz is not None
                else df.index.tz_localize("UTC").tz_convert(EXCHANGE_TZ))
    df.index.name = "timestamp"

    if rth_only:
        df = df.between_time(RTH_OPEN, RTH_CLOSE, inclusive="left")

    df = add_session_columns(df)
    # Record the anchor actually obtained. Hourly data has no 09:30 bar, so a
    # caller asking for an open-anchored strategy needs to know it is really
    # getting a 10:00 anchor rather than discovering it in the results.
    anchors = pd.Series([ts.strftime("%H:%M") for ts in df.index]).groupby(
        pd.Series(df["session"].values)).first()
    anchor = anchors.mode().iloc[0] if len(anchors) else None
    df.attrs["anchor"] = anchor
    df.attrs["anchored_at_open"] = anchor == RTH_OPEN
    df.attrs["interval"] = interval
    # Sessions that do not start at the common anchor are truncated days
    # (holidays, half sessions); their "open" is not comparable.
    keep = anchors[anchors == anchor].index
    return df[df["session"].isin(set(keep))].copy()


def add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Label each bar with its session and position within that session."""
    out = df.copy()
    out["session"] = out.index.date
    open_ts = out.groupby("session").apply(lambda g: g.index[0], include_groups=False)
    out["minutes_from_open"] = [
        (ts - open_ts[sess]).total_seconds() / 60.0
        for ts, sess in zip(out.index, out["session"])
    ]
    out["bar_of_session"] = out.groupby("session").cumcount()
    return out


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Running VWAP, reset each session.

    Used as the strategy's trailing stop: the paper closes a long once price
    loses VWAP, on the logic that intraday momentum has failed when the average
    participant is offside.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (typical * df["volume"]).groupby(df["session"]).cumsum()
    vol = df["volume"].groupby(df["session"]).cumsum()
    # Some futures bars report zero volume; fall back to typical price rather
    # than producing an inf/NaN VWAP that would silently disable the stop.
    return (pv / vol.replace(0, pd.NA)).fillna(typical)


def session_opens(df: pd.DataFrame) -> pd.Series:
    """Opening price of each session, broadcast back onto every bar."""
    opens = df.groupby("session")["open"].first()
    return pd.Series([opens[s] for s in df["session"]], index=df.index)
