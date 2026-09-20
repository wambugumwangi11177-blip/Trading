"""Morning / evening Telegram briefings, matching the source design's two-message
daily cadence: open positions + risk flags in the morning, trades + P&L in the
evening.
"""
from __future__ import annotations

import asyncio
import logging

from telegram import Bot

from .config import TELEGRAM, TelegramConfig
from .portfolio import PortfolioTracker

logger = logging.getLogger("fable_bot.telegram")


def notify(text: str, config: TelegramConfig | None = None) -> bool:
    """Best-effort alert. Never raises.

    Called from the unattended run, where two rules apply. A failed
    notification must never abort a trading cycle -- the trade is the job, the
    message is commentary. And an unconfigured Telegram must be a silent no-op
    rather than an error, so the bot runs identically for someone who never
    sets it up.
    """
    config = config or TELEGRAM
    if not config.is_configured:
        return False
    try:
        TelegramBriefing(config).send(text)
        return True
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning("Telegram notification failed: %s", exc)
        return False


class TelegramBriefing:
    def __init__(self, config: TelegramConfig):
        if not config.is_configured:
            raise RuntimeError(
                "Telegram not configured. Create a bot via @BotFather, set "
                "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env (see config/.env.example), "
                "or run: python -m fable_bot.cli telegram-setup"
            )
        self.config = config
        self.bot = Bot(token=config.bot_token)

    def send(self, text: str) -> None:
        asyncio.run(self.bot.send_message(chat_id=self.config.chat_id, text=text))

    # Retained: live_runner's briefings call this name.
    _send = send

    def send_test_message(self) -> None:
        self._send("Fable trading bot: Telegram briefing wired up correctly.")

    def send_morning_briefing(self, portfolio: PortfolioTracker, prices: dict[str, float],
                               halted: bool) -> None:
        lines = ["Good morning. Fable trading bot -- morning briefing.", ""]
        if halted:
            lines.append("RISK HALT ACTIVE: drawdown kill-switch triggered. No new trades until manually reset.")
        if not portfolio.positions:
            lines.append("Open positions: none.")
        else:
            lines.append("Open positions:")
            for symbol, pos in portfolio.positions.items():
                price = prices.get(symbol, pos.entry_price)
                pnl = portfolio.unrealized_pnl(symbol, price)
                lines.append(f"  {symbol} {pos.side} qty={pos.qty:.4f} entry={pos.entry_price:.2f} "
                             f"pnl={pnl:+.2f}")
        equity = portfolio.equity(prices)
        lines.append("")
        lines.append(f"Equity: {equity:,.2f}")
        self._send("\n".join(lines))

    def send_evening_briefing(self, portfolio: PortfolioTracker, todays_trades: list) -> None:
        lines = ["Evening briefing -- trades executed today.", ""]
        if not todays_trades:
            lines.append("No trades closed today.")
        else:
            for t in todays_trades:
                lines.append(f"  {t.symbol} {t.side} pnl={t.pnl:+.2f} ({t.reason})")
            best = max(todays_trades, key=lambda t: t.pnl)
            worst = min(todays_trades, key=lambda t: t.pnl)
            lines.append("")
            lines.append(f"Best: {best.symbol} {best.pnl:+.2f}")
            lines.append(f"Worst: {worst.symbol} {worst.pnl:+.2f}")
        self._send("\n".join(lines))
