"""Central configuration: instrument universe, risk parameters, and credentials.

Credentials are read from environment variables (populated via a .env file in the
project root, see config/.env.example). Nothing in this module fabricates or
guesses secrets -- missing values simply mean the paper/live broker and Telegram
briefing features stay disabled until the user supplies them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Persistent bot memory: lessons learned from mistakes (lessons.py) and
# reconciliation state (reconcile.py). Separate from logs/ on purpose -- logs
# are forensic output, memory is living state the bot consults every run.
MEMORY_DIR = Path(os.getenv("FABLE_MEMORY_DIR", str(PROJECT_ROOT / "memory")))


@dataclass(frozen=True)
class Instrument:
    symbol: str  # yfinance / backtest ticker
    broker_symbol: str  # ticker as understood by Alpaca
    strategy: str  # "mean_reversion" | "momentum" | "trend_following"
    asset_class: str  # "equity" | "crypto"
    display_name: str
    # Exchange-qualified ticker for TradingView. Adapters map an Instrument to
    # their own venue's symbol via BrokerAdapter.venue_symbol, so nothing
    # outside the adapters needs to know which naming scheme is in play.
    tv_symbol: str = ""
    # Smallest tradable increment at the venue. TradingView paper reports
    # step 1 (whole shares / whole contracts) for the ETFs and futures, and
    # 0.0001 for BTC; ATR sizing produces fractions, so orders are rounded DOWN
    # to this -- rounding up would risk more than the 1% the position was sized
    # for.
    qty_step: float = 1.0
    # Cash change per 1.00 of price movement, per unit. 1.0 for shares/ETFs;
    # for futures it is the contract multiplier (E-mini S&P = $50/point).
    # Position sizing multiplies stop distance by this -- get it wrong and the
    # 1%-per-trade rule silently becomes 50%. See risk/manager.position_size.
    point_value: float = 1.0
    # Currencies whose macro releases move this instrument, for the economic
    # calendar blackout (see risk/event_filter.py). Everything here is
    # USD-denominated and trades on US macro -- including BTC, which reacts to
    # FOMC and CPI like any other risk asset.
    event_currencies: tuple[str, ...] = ("USD",)
    # Whether the venue can actually fill orders on this instrument. Some
    # series exist for intelligence only (indicator series like VIX/DXY, bare
    # index values like ^NDX): their signals inform the others, but an order
    # sent for them is rejected by the broker. False keeps them out of the
    # order path entirely (see live_runner); history is still fetched for them
    # nowhere -- they are skipped end to end. Learned 2026-08: rejected
    # NASDAQ:NDX/IXIC orders came from this exact leak.
    tradable: bool = True
    # Whether this instrument's strategy has documented backtest support in
    # this repo. False means "paper-only experiment" -- the comments below say
    # NOT backtest-validated for every 2026-08 addition, and those additions
    # are what the account traded through its round trip from 113,637 (Aug 5)
    # up to 117,148 (Aug 25) and back to 113,662 (Sep 11).
    #
    # Sizing does not care whether an edge exists: an unvalidated signal is
    # risked at the same 1% as a validated one, so a universe of 30 mostly
    # unvalidated instruments is 30 ways to pay spread and stop-loss for no
    # expected return. TRADE_UNVALIDATED=true re-enables them deliberately.
    validated: bool = False


# The 5-instrument universe from the source strategy design:
# mean reversion on SPY/QQQ, momentum breakout on BTC, trend following on gold/oil.
# Gold and oil are represented via GLD/USO ETF proxies rather than spot XAUUSD/WTI
# CFDs so the whole universe can run through a single equities+crypto paper broker
# (Alpaca) without needing a separate forex/CFD account. Swap broker_symbol if a
# CFD/forex broker is wired in later.
INSTRUMENTS: list[Instrument] = [
    Instrument("SPY", "SPY", "mean_reversion", "equity", "S&P 500 ETF",
               tv_symbol="AMEX:SPY", qty_step=1.0, validated=True),
    Instrument("QQQ", "QQQ", "mean_reversion", "equity", "Nasdaq 100 ETF",
               tv_symbol="NASDAQ:QQQ", qty_step=1.0, validated=True),
    Instrument("BTC-USD", "BTC/USD", "momentum", "crypto", "Bitcoin",
               tv_symbol="BITSTAMP:BTCUSD", qty_step=0.0001, validated=True),
    Instrument("GLD", "GLD", "trend_following", "equity", "Gold ETF (proxy for XAUUSD)",
               tv_symbol="AMEX:GLD", qty_step=1.0, validated=True),
    Instrument("USO", "USO", "trend_following", "equity", "Oil ETF (proxy for WTI)",
               tv_symbol="AMEX:USO", qty_step=1.0, validated=True),
    # Forex majors, added 2026-08-25 at the user's request. History comes from
    # yfinance ("GBPUSD=X" style), orders route to TradingView Paper Trading's
    # OANDA paper feed. Trend following, same as the gold/oil proxies. NOTE:
    # unlike the ETF universe these are NOT backtest-validated -- paper only.
    Instrument("GBPUSD=X", "GBPUSD", "trend_following", "forex", "British Pound / USD",
               tv_symbol="OANDA:GBPUSD", qty_step=1000, point_value=1.0,
               event_currencies=("USD", "GBP")),
    Instrument("EURUSD=X", "EURUSD", "trend_following", "forex", "Euro / USD",
               tv_symbol="OANDA:EURUSD", qty_step=1000, point_value=1.0,
               event_currencies=("USD", "EUR")),
    Instrument("USDJPY=X", "USDJPY", "trend_following", "forex", "USD / Japanese Yen",
               tv_symbol="OANDA:USDJPY", qty_step=1000, point_value=1.0,
               event_currencies=("USD", "JPY")),
    Instrument("AUDUSD=X", "AUDUSD", "trend_following", "forex", "Australian Dollar / USD",
               tv_symbol="OANDA:AUDUSD", qty_step=1000, point_value=1.0,
               event_currencies=("USD", "AUD")),
    # Remaining majors + liquid crosses, added 2026-08-26 at the user's
    # request. Same trend-following setup and OANDA paper routing as above.
    # NOT backtest-validated -- paper only.
    Instrument("USDCAD=X", "USDCAD", "trend_following", "forex", "USD / Canadian Dollar",
               tv_symbol="OANDA:USDCAD", qty_step=1000, point_value=1.0,
               event_currencies=("USD", "CAD")),
    Instrument("USDCHF=X", "USDCHF", "trend_following", "forex", "USD / Swiss Franc",
               tv_symbol="OANDA:USDCHF", qty_step=1000, point_value=1.0,
               event_currencies=("USD", "CHF")),
    Instrument("NZDUSD=X", "NZDUSD", "trend_following", "forex", "New Zealand Dollar / USD",
               tv_symbol="OANDA:NZDUSD", qty_step=1000, point_value=1.0,
               event_currencies=("USD", "NZD")),
    Instrument("EURGBP=X", "EURGBP", "trend_following", "forex", "Euro / British Pound",
               tv_symbol="OANDA:EURGBP", qty_step=1000, point_value=1.0,
               event_currencies=("EUR", "GBP")),
    Instrument("EURJPY=X", "EURJPY", "trend_following", "forex", "Euro / Japanese Yen",
               tv_symbol="OANDA:EURJPY", qty_step=1000, point_value=1.0,
               event_currencies=("EUR", "JPY")),
    Instrument("GBPJPY=X", "GBPJPY", "trend_following", "forex", "British Pound / Japanese Yen",
               tv_symbol="OANDA:GBPJPY", qty_step=1000, point_value=1.0,
               event_currencies=("GBP", "JPY")),
    Instrument("AUDJPY=X", "AUDJPY", "trend_following", "forex", "Australian Dollar / Japanese Yen",
               tv_symbol="OANDA:AUDJPY", qty_step=1000, point_value=1.0,
               event_currencies=("AUD", "JPY")),
    # Spot metals, added 2026-08-26 at the user's request.
    # Index & gold micro futures, promoted into the active universe 2026-08-26
    # at the user's request (were defined in FUTURES/GOLD_FUTURES below but
    # never traded). Sizing notes above still apply: MES fits the 1% rule;
    # MNQ correctly sizes to ~zero. Paper routing via CME_MINI/COMEX_MINI.
    Instrument("MES=F", "MES", "mean_reversion", "futures", "Micro E-mini S&P 500",
               tv_symbol="CME_MINI:MES1!", qty_step=1.0, point_value=5.0,
               event_currencies=("USD",)),
    Instrument("MNQ=F", "MNQ", "mean_reversion", "futures", "Micro E-mini Nasdaq-100",
               tv_symbol="CME_MINI:MNQ1!", qty_step=1.0, point_value=2.0,
               event_currencies=("USD",)),
    Instrument("MGC=F", "MGC", "long_or_flat_trend", "futures", "Micro Gold Futures",
               tv_symbol="COMEX_MINI:MGC1!", qty_step=1.0, point_value=10.0,
               event_currencies=("USD",), validated=True),
    # Yahoo has no spot
    # XAUUSD=X feed, so history comes from the futures continuous contracts
    # (GC=F / SI=F), which track spot ~1:1; orders route to TradingView Paper
    # Trading's OANDA spot feed. Trend following, same as the FX pairs. NOT
    # backtest-validated -- paper only, like the forex majors.
    # qty_step was 1000 for both, copied from the FX block where a 1,000-unit
    # micro lot is correct. Metals are not quoted that way: spot gold sizes to
    # ~12 ounces on this account, and quantize() rounds DOWN, so 12 -> 0 and
    # the order was silently dropped. Spot gold has NEVER been orderable by
    # this bot -- not once, in any session -- and nothing surfaced it because
    # "sized below the venue's minimum" reads like an ordinary skip. One ounce
    # is the correct increment for OANDA spot metals.
    Instrument("GC=F", "XAUUSD", "trend_following", "forex", "Spot Gold / USD",
               tv_symbol="OANDA:XAUUSD", qty_step=1.0, point_value=1.0),
    Instrument("SI=F", "XAGUSD", "trend_following", "forex", "Spot Silver / USD",
               tv_symbol="OANDA:XAGUSD", qty_step=1.0, point_value=1.0),
    # World market indices, added 2026-08-26 at the user's request. History
    # from Yahoo (^-prefixed indices); orders route to TradingView. Trend
    # following, qty_step sized so index units trade fractionally. NOT
    # backtest-validated -- paper only. DXY/VIX/VXN/TNX are indicator series
    # with no direct cash instrument and ^NDX/^IXIC are bare index values the
    # paper broker cannot fill (rejected orders, 2026-08-27) -- all six carry
    # tradable=False so they never reach the order path.
    Instrument("^NSEI", "NIFTY50", "trend_following", "index", "India Nifty 50",
               tv_symbol="NSE:NIFTY", qty_step=0.01),
    Instrument("DX-Y.NYB", "DXY", "trend_following", "index", "US Dollar Index",
               tv_symbol="TVC:DXY", qty_step=0.01, tradable=False),
    Instrument("^VIX", "VIX", "mean_reversion", "index", "CBOE Volatility Index",
               tv_symbol="CBOE:VIX", qty_step=0.01, tradable=False),
    Instrument("^VXN", "VXN", "mean_reversion", "index", "CBOE Nasdaq Volatility Index",
               tv_symbol="CBOE:VXN", qty_step=0.01, tradable=False),
    Instrument("^TNX", "TNX", "trend_following", "index", "US 10-Year Treasury Yield x10",
               tv_symbol="TVC:TNX", qty_step=0.01, tradable=False),
    Instrument("^NDX", "NAS100", "trend_following", "index", "Nasdaq 100 Index",
               tv_symbol="NASDAQ:NDX", qty_step=0.01, tradable=False),
    Instrument("^HSI", "HSI", "trend_following", "index", "Hang Seng Index",
               tv_symbol="HKEX:HSI", qty_step=0.01),
    Instrument("^RUT", "RUT", "trend_following", "index", "Russell 2000",
               tv_symbol="CBOE:RUT", qty_step=0.01),
    Instrument("^IXIC", "COMPOSITE", "trend_following", "index", "Nasdaq Composite",
               tv_symbol="NASDAQ:IXIC", qty_step=0.01, tradable=False),
    Instrument("^N225", "JPN225", "trend_following", "index", "Nikkei 225",
               tv_symbol="TVC:NI225", qty_step=0.01),
    Instrument("^SOX", "SOX", "trend_following", "index", "PHLX Semiconductor Index",
               tv_symbol="NASDAQ:SOX", qty_step=0.01),
    Instrument("^FTSE", "UK100", "trend_following", "index", "FTSE 100",
               tv_symbol="TVC:UKX", qty_step=0.01),
]

# Index futures, traded via the MICRO contracts.
#
# The full-size E-minis do not fit this account under the 1% rule. Measured
# 2026-08-04 with equity ~$111.8k (so a $1,118 risk budget) and the strategy's
# 1.5x-ATR stop, using daily ATR(14) from TradingView:
#
#   ES1!  ATR 97.5 pts -> 146 pt stop x $50/pt = $7,310  =  6.5% per contract
#   NQ1!  ATR 695 pts  -> 1043 pt stop x $20/pt = $20,856 = 18.7% per contract
#   MES1! same stop    x $5/pt  = $731   = 0.65%  <- fits
#   MNQ1! same stop    x $2/pt  = $2,086 = 1.87%  <- still over 1%
#
# So MES is tradable as-is; MNQ needs either a tighter stop or a smaller risk
# budget, and correct sizing will simply return 0 contracts rather than break
# the rule. Point values are the CME contract multipliers, cross-checked
# against TradingView's own symbolInfo (pipValue per 0.25 tick x 4).
FUTURES: list[Instrument] = [
    Instrument("MES=F", "MES", "mean_reversion", "futures", "Micro E-mini S&P 500",
               tv_symbol="CME_MINI:MES1!", qty_step=1.0, point_value=5.0),
    Instrument("MNQ=F", "MNQ", "mean_reversion", "futures", "Micro E-mini Nasdaq-100",
               tv_symbol="CME_MINI:MNQ1!", qty_step=1.0, point_value=2.0),
]

# Full-size E-minis. Defined so they can be traded deliberately (they are what
# "trade ES/NQ" usually means), but deliberately NOT in INSTRUMENTS: at this
# account size a single contract breaks the 1%-per-trade rule, so the runner
# would size them to zero and skip every signal anyway.
# Gold futures, traded by the long-or-flat trend strategy (the one strategy that
# cleared validation: t=2.31 over 26y, out-of-sample Sharpe 0.65, ~29% less
# drawdown than buy-and-hold). Point values are the COMEX contract multipliers,
# cross-checked against TradingView symbolInfo (pipValue per 0.1 tick x 10).
#
# At ~$112k equity a 1.5x-ATR gold stop is ~93 points, which costs $9,289 on a
# full GC contract (8.3% of equity) versus $929 on a micro. Only the micro is
# tradable under the 1% rule, exactly as with the equity indices.
GOLD_FUTURES: list[Instrument] = [
    Instrument("MGC=F", "MGC", "long_or_flat_trend", "futures", "Micro Gold Futures",
               tv_symbol="COMEX_MINI:MGC1!", qty_step=1.0, point_value=10.0),
]

GOLD_FULL: list[Instrument] = [
    Instrument("GC=F", "GC", "long_or_flat_trend", "futures", "Gold Futures",
               tv_symbol="COMEX:GC1!", qty_step=1.0, point_value=100.0),
]

EMINI_FUTURES: list[Instrument] = [
    Instrument("ES=F", "ES", "mean_reversion", "futures", "E-mini S&P 500",
               tv_symbol="CME_MINI:ES1!", qty_step=1.0, point_value=50.0),
    Instrument("NQ=F", "NQ", "mean_reversion", "futures", "E-mini Nasdaq-100",
               tv_symbol="CME_MINI:NQ1!", qty_step=1.0, point_value=20.0),
]


@dataclass(frozen=True)
class RiskConfig:
    max_risk_per_trade_pct: float = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "1.0"))
    max_drawdown_pct: float = float(os.getenv("MAX_DRAWDOWN_PCT", "10.0"))
    correlation_block_threshold: float = float(os.getenv("CORRELATION_BLOCK_THRESHOLD", "0.7"))
    atr_stop_multiple: float = 1.5
    starting_equity: float = float(os.getenv("STARTING_EQUITY", "100000"))
    # Book-wide ceilings. Per-trade sizing bounds ONE trade; nothing bounded
    # the book, so 2026-09-08 fired eight entries in two minutes (~8% of
    # equity at risk) and 2026-09-11 carried seven positions at once.
    # See risk.manager.portfolio_gate.
    max_concurrent_positions: int = int(os.getenv("MAX_CONCURRENT_POSITIONS", "3"))
    max_new_risk_per_run_pct: float = float(os.getenv("MAX_NEW_RISK_PER_RUN_PCT", "2.0"))
    # Trade instruments whose strategy has no documented backtest support.
    # Default off: see Instrument.validated.
    trade_unvalidated: bool = os.getenv("TRADE_UNVALIDATED", "false").lower() == "true"
    # Refuse entries that fight the positioning briefing (risk/intel_gate.py).
    # Fail-open on missing coverage; never consulted on exits.
    intel_gate_enabled: bool = os.getenv("INTEL_GATE_ENABLED", "true").lower() == "true"
    # Below this equity the 1% rule is unreachable on standard minimum lots
    # (see sizing_check.py) and every run says so in the journal and the
    # briefing, so a small live account is never traded in silence.
    small_account_equity: float = float(os.getenv("SMALL_ACCOUNT_EQUITY", "1000"))


@dataclass(frozen=True)
class BrokerConfig:
    # "tradingview" routes orders to TradingView's Paper Trading simulator via
    # the tradingview-mcp CLI; "alpaca" uses the Alpaca REST API. TradingView is
    # the default because it needs no API keys -- just the desktop app running
    # with a broker connected.
    provider: str = os.getenv("BROKER_PROVIDER", "tradingview").lower()
    api_key: str | None = os.getenv("ALPACA_API_KEY") or None
    secret_key: str | None = os.getenv("ALPACA_SECRET_KEY") or None
    trading_mode: str = os.getenv("TRADING_MODE", "paper")
    allow_live_trading: bool = os.getenv("ALLOW_LIVE_TRADING", "false").lower() == "true"
    # Path to the tradingview-mcp checkout (provider="tradingview" only).
    # Defaults to a sibling directory of this repo.
    tv_mcp_path: str = os.getenv(
        "TRADINGVIEW_MCP_PATH", str(PROJECT_ROOT.parent / "tradingview-mcp"),
    )
    tv_node_bin: str = os.getenv("NODE_BIN", "node")
    tv_timeout_seconds: float = float(os.getenv("TRADINGVIEW_TIMEOUT_SECONDS", "60"))

    @property
    def is_configured(self) -> bool:
        if self.provider == "tradingview":
            return bool(self.tv_mcp_path)
        return bool(self.api_key and self.secret_key)

    @property
    def is_paper(self) -> bool:
        # Fail safe: anything other than an explicit, doubly-confirmed live setup
        # is treated as paper.
        return not (self.trading_mode == "live" and self.allow_live_trading)


@dataclass(frozen=True)
class EventFilterConfig:
    """Blackout windows around scheduled macro releases (risk/event_filter.py)."""

    enabled: bool = os.getenv("EVENT_FILTER_ENABLED", "true").lower() == "true"
    # 1 low / 2 medium / 3 high. High-only by default: medium-impact releases
    # rarely move a daily-bar position enough to justify skipping the entry.
    min_importance: int = int(os.getenv("EVENT_MIN_IMPORTANCE", "3"))
    blackout_minutes_before: int = int(os.getenv("EVENT_BLACKOUT_MINUTES_BEFORE", "60"))
    blackout_minutes_after: int = int(os.getenv("EVENT_BLACKOUT_MINUTES_AFTER", "30"))
    # What to do when the calendar can't be reached. Defaults to trading
    # through it (with a warning): this is a daily-bar strategy whose entries
    # are already infrequent, and halting the whole book because a scraped
    # endpoint had a bad afternoon trades one risk for a worse one. Set true to
    # refuse new entries whenever the calendar is unknown.
    fail_closed: bool = os.getenv("EVENT_FILTER_FAIL_CLOSED", "false").lower() == "true"


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str | None = os.getenv("TELEGRAM_BOT_TOKEN") or None
    chat_id: str | None = os.getenv("TELEGRAM_CHAT_ID") or None

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)


RISK = RiskConfig()
BROKER = BrokerConfig()
TELEGRAM = TelegramConfig()
EVENT_FILTER = EventFilterConfig()
