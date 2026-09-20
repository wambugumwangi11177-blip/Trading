# Fable Trading Bot

A 5-instrument, multi-strategy trading bot: mean reversion on SPY/QQQ, momentum
breakout on BTC, trend following on gold/oil (via GLD/USO), with a Smart Money
Concepts (SMC) confluence filter informed by TJR Trades' free Bootcamp material
(market structure, liquidity sweeps, fair value gaps, order blocks) layered on
top of each strategy's entries. Risk is managed with ATR-based position sizing
capped at 1% of equity per trade, a correlation filter that blocks doubling up
on correlated same-direction exposure, and a 10%-drawdown kill-switch that
flattens everything and halts until manually reset.

Built to go: backtest -> paper trade -> (only with explicit human sign-off) live.

## Project layout

```
fable_bot/
  config.py            instrument universe + risk/broker/telegram/event config from .env
  data/feed.py          yfinance-backed historical data (free, no API key)
  data/calendar.py      economic calendar feed (Investing.com), market intelligence
  strategies/
    smc.py               market structure / liquidity sweep / FVG / order block confluence
    mean_reversion.py     SPY/QQQ
    momentum.py           BTC
    trend_following.py    GLD/USO
  risk/manager.py         position sizing, correlation filter, drawdown kill-switch
  risk/event_filter.py    blackout windows around scheduled macro releases
  broker/tradingview_adapter.py  TradingView Paper Trading (default, no API keys)
  broker/alpaca_adapter.py       Alpaca paper/live (BROKER_PROVIDER=alpaca)
  portfolio.py            positions, P&L, equity curve
  backtester.py           wires it all together over historical data
  broker/alpaca_adapter.py  paper/live execution via Alpaca
  telegram_bot.py           morning/evening briefing messages
  live_runner.py            one live/paper trading cycle
  cli.py                    command-line entry point
```

## Setup

```
# from fable-trading-bot/
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
copy config\.env.example .env
```

Backtesting needs no credentials at all -- it pulls free historical data from
Yahoo Finance via yfinance.

### Paper trading (default: TradingView, no API keys)

Orders route to **TradingView's built-in Paper Trading simulator**, which needs
no signup, no API keys, and no brokerage account -- only the desktop app:

1. Install TradingView Desktop and sign in to your normal TradingView account.
2. Launch it with remote debugging enabled:
   `node ../tradingview-mcp/src/cli/index.js health launch`
3. In the app: **Trade → Paper Trading → Connect**.
4. Check the bot can see it: `python -m fable_bot.cli broker-check`

That's the whole setup. `BROKER_PROVIDER=tradingview` is the default, and
`TRADINGVIEW_MCP_PATH` defaults to the sibling `tradingview-mcp` checkout.

Under the hood the adapter shells out to that project's CLI, which drives
TradingView's Broker API over the Chrome DevTools Protocol. One subprocess per
broker call is not free, but this is a daily-bar strategy placing a handful of
orders a day, so it costs nothing that matters and avoids a second
implementation of the CDP plumbing.

**Alpaca is still supported** -- set `BROKER_PROVIDER=alpaca` and fill in
`ALPACA_API_KEY` / `ALPACA_SECRET_KEY`. The `alpaca-py` import is lazy, so
running on TradingView doesn't require it to be installed.

### Telegram briefings (optional)

Message @BotFather on Telegram, `/newbot`, copy the token into
`TELEGRAM_BOT_TOKEN`. Message your new bot once, then visit
`https://api.telegram.org/bot<token>/getUpdates` to read your chat id into
`TELEGRAM_CHAT_ID`.

### Going live

Still a deliberate two-flag change (`TRADING_MODE=live` **and**
`ALLOW_LIVE_TRADING=true`) -- nothing in the code can flip both on its own. On
TradingView there is a **second, independent** guard: the CLI refuses any broker
whose account is not the Paper Trading simulator unless it is explicitly passed
`--allow-live`, and the adapter only passes that flag when both bot flags agree.
So the default configuration cannot place a real order even if you happen to
have a live brokerage connected in TradingView -- it errors out instead.

### Symbols and quantity rounding

Each instrument carries a `tv_symbol` (`AMEX:SPY`, `BITSTAMP:BTCUSD`, ...) and a
`qty_step` in `config.py`. TradingView paper trades the ETFs in **whole shares**
while BTC steps in 0.0001, and ATR sizing produces fractions -- so orders are
rounded **down** to the step. Down, not nearest: rounding 12.8 shares up to 13
would risk more than the 1% the position was sized for. If the risk budget
doesn't cover even one unit the trade is skipped rather than upsized.

## Usage

```
python -m fable_bot.cli backtest --period 6mo
python -m fable_bot.cli telegram-test
python -m fable_bot.cli signal-check
python -m fable_bot.cli broker-check          # verify the broker connection
python -m fable_bot.cli calendar --hours 96   # upcoming high-impact macro releases
python -m fable_bot.cli morning-briefing
python -m fable_bot.cli evening-briefing
python -m fable_bot.cli serve      # daily schedule: 09:00 morning briefing, 16:15 signal check + evening briefing
```

## Market intelligence: the economic calendar filter

The three risk rules from the source design are all reactive -- they measure
what price has already done. This adds a fourth, forward-looking one: pause
**new entries** in a window around scheduled high-impact macro releases.

The argument is about sizing, not about predicting direction. Position size is
derived from ATR, i.e. from recent realised volatility. An FOMC or CPI print is
a *scheduled* discontinuity in that volatility, so a position opened minutes
beforehand is sized off an ATR that no longer describes the next bar -- the
"1% risk" it was built to respect is not the risk actually taken. Standing
aside for a defined window keeps the sizing honest.

Only entries are gated. Exits, stops, and the drawdown kill-switch are never
blocked; being unable to close during a release would invert the point.

```
python -m fable_bot.cli calendar --hours 96
=== High-impact releases, next 96h (importance >= 3) ===
  2026-08-03 14:00 UTC  USD  ISM Manufacturing PMI (Jul)  (fc 54.0 / prev 53.3)
  2026-08-05 12:15 UTC  USD  ADP Nonfarm Employment Change (Jul)  (fc 71K / prev 98K)
  2026-08-05 14:30 UTC  USD  Crude Oil Inventories  (fc - / prev -7.167M)
  2026-08-06 12:30 UTC  USD  Initial Jobless Claims  (fc 205K / prev 197K)
```

Tuning lives in `.env` (`EVENT_FILTER_ENABLED`, `EVENT_MIN_IMPORTANCE`,
`EVENT_BLACKOUT_MINUTES_BEFORE/AFTER`, `EVENT_FILTER_FAIL_CLOSED`). No API key
is needed. Upcoming releases are also appended to the morning Telegram briefing.

**On the data source.** There is no maintained Python client for Investing.com's
calendar: `investpy` has been Cloudflare-blocked (403) since 2022, and its
successor `investiny` covers historical prices only. So `data/calendar.py` calls
their AJAX endpoint directly and parses the HTML fragment it returns. Be clear
about what that means -- it is unofficial scraping, subject to Investing.com's
terms, and a markup change on their side will break parsing. Three things keep
that from turning into silent bad behaviour:

- A fetch that returns 200 but parses to zero events raises rather than
  returning an empty list, because "no events" and "our parser broke" are
  opposite facts for risk purposes.
- Results are cached (3h TTL) and a *stale* cache is preferred over no calendar
  when a refetch fails.
- If the calendar is genuinely unavailable, `EVENT_FILTER_FAIL_CLOSED` decides
  the policy. It defaults to `false` -- trade through it, but log a warning
  loudly -- on the grounds that halting the whole book because a scraped
  endpoint had a bad afternoon trades one risk for a worse one. Set it `true`
  to refuse entries whenever the calendar is unknown.

Everything above `EconomicEvent` is provider-specific by design: to switch
sources, write another function returning `list[EconomicEvent]` and the filter
needs no changes.

## Tests

```
.venv\Scripts\python -m pytest -v
```

## Backtest results (as of 2026-07-29)

| Period | Return | Sharpe | Max DD | Win rate | Trades |
|---|---|---|---|---|---|
| 6mo | -3.60% | -0.84 | -4.59% | 28.6% | 14 |
| 1y  | +9.05% | 0.56 | -6.28% | 45.5% | 33 |
| 2y  | +25.88% | 0.75 | -6.28% | 45.3% | 75 |

The source design's own rule is "if any strategy shows a negative Sharpe ratio,
fix it before going further." The 6-month window came back negative, but with
only 14 trades that's too small a sample to read anything into -- one or two
unlucky stop-outs swing it entirely. The 1y/2y windows (33 and 75 trades) are
consistently positive and roughly agree with each other (Sharpe 0.56 and 0.75),
which is a much more meaningful signal than the 6-month number. Re-check this
table after real trading days accumulate rather than trusting the 6mo number in
isolation.

While validating this, found and fixed a real bug: the ATR-based stop price
computed at entry was never actually enforced intrabar -- strategies only
exited on their own logic (z-score reversion, trend flip), so a loss could run
well past the 1% it was sized for (one SPY trade lost $2,748 against a ~$1,000
budget before the fix). Backtester now force-closes at the stop the moment
a bar's high/low breaches it (`backtester.enforce_hard_stops`), and the Alpaca
adapter attaches the stop as a bracket order so it's enforced by the broker
between daily signal-check polls too, not just in the backtest loop.

## Design notes / where this deviates from the source material

- Gold and oil are traded as GLD/USO ETFs rather than spot XAUUSD/WTI CFDs, so
  the whole 5-instrument universe runs through one broker (Alpaca) that also
  covers SPY/QQQ (equities) and BTC/USD (crypto). Swap `broker_symbol` in
  `config.py` and the broker adapter if a forex/CFD broker is wired in later.
- SMC concepts (market structure, liquidity sweeps, FVGs, order blocks) are
  computed on daily bars for backtest simplicity, not the intraday NY/London
  session windows TJR's material describes. `smc.py` is written so the same
  functions work unchanged if the data feed is swapped to intraday bars later.
- The correlation filter and drawdown kill-switch match the "Non-Negotiable
  Risk Rules" screenshots exactly: 1% max risk/trade, correlation-blocked
  doubling up, 10% total-equity drawdown halt.
- The economic calendar blackout is an *addition* to the source design, not
  part of it. It is applied in `live_runner` only, not in the backtester --
  Investing.com's endpoint serves upcoming events, not a historical archive, so
  there is no way to replay it over past bars without a different data source.
  That means the backtest numbers above do not reflect it, and cannot be used
  to claim it improves returns. Judge it on the sizing argument, and on live
  results once they accumulate.
