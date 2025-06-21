import asyncio
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import Dict, List, Optional
from api_endpoints import router as api_router
from exchange_interface import BloFinExchange
from notifications import Notifications
from performance_analytics import PerformanceAnalytics
from portfolio_management import PortfolioManagement
from api_utils import BloFinAPIUtils
from config import Config
from database import Database
from event_bus import EventBus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(api_router)

class Signal(BaseModel):
    symbol: str
    signal: str
    confidence: float
    patterns: List[str]
    timeframes: List[str]
    bot_name: str

class MainBot:
    def __init__(self):
        self.config = Config()
        self.api_utils = BloFinAPIUtils(
            api_key=os.getenv("API_KEY"),
            api_secret=os.getenv("API_SECRET"),
            api_passphrase=os.getenv("API_PASSPHRASE")
        )
        self.exchange = BloFinExchange(
            api_key=os.getenv("API_KEY"),
            api_secret=os.getenv("API_SECRET"),
            api_passphrase=os.getenv("API_PASSPHRASE")
        )
        self.notifications = Notifications(webhook_url=os.getenv("WEBHOOK_URL"))
        self.performance = PerformanceAnalytics()
        self.portfolio = PortfolioManagement()
        self.database = Database(db_path=self.config.DB_PATH)
        self.event_bus = EventBus(redis_url=self.config.REDIS_URL)
        self.account_balance = self.config.DEFAULT_BALANCE
        self.signals: Dict[str, List[Signal]] = {s: [] for s in self.config.SYMBOLS}
        self.candle_history: Dict[str, List[Dict]] = {s: [] for s in self.config.SYMBOLS}

    async def initialize(self):
        await self.database.initialize()
        await self.event_bus.connect()
        for symbol in self.config.SYMBOLS:
            candles = await self.api_utils.get_candles(symbol, "1h", limit=100)
            self.candle_history[symbol] = candles
            await self.database.store_candles(symbol, candles)

    async def process_signal(self, signal: Signal):
        self.signals[signal.symbol].append(signal)
        if len(self.signals[signal.symbol]) >= 3:
            await self.evaluate_signals(signal.symbol)

    async def evaluate_signals(self, symbol: str):
        signals = self.signals[symbol][-3:]
        score = sum(1 if s.signal == "buy" else -1 for s in signals)
        confidence = sum(s.confidence for s in signals) / len(signals)
        if abs(score) >= 3 and confidence > 0.5:
            action = "buy" if score > 0 else "sell"
            price = self.candle_history[symbol][-1]["close"]
            risk_amount = self.account_balance * self.config.RISK_PER_TRADE
            margin, leverage = self.portfolio.calculate_margin(
                symbol, risk_amount, confidence, abs(price - self.candle_history[symbol][-2]["close"]) / price,
                [s.patterns for s in signals], self
            )
            size = margin / price
            order = await self.exchange.place_order(symbol, action, size, leverage, price)
            if order:
                self.performance.log_trade(symbol, action, price, size, leverage)
                await self.notifications.send_alert(f"Trade executed: {action} {symbol} at ${price:.2f}")
                self.account_balance += order.get("profit", 0)
            self.signals[symbol].clear()

    async def run(self):
        await self.initialize()
        while True:
            for symbol in self.config.SYMBOLS:
                candles = await self.api_utils.get_candles(symbol, "1h", limit=1)
                if candles:
                    self.candle_history[symbol].append(candles[0])
                    await self.database.store_candles(symbol, candles)
            await asyncio.sleep(60)

if __name__ == "__main__":
    bot = MainBot()
    asyncio.run(bot.run())