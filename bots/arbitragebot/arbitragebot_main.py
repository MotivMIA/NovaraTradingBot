import asyncio
import os
import logging
import requests
from exchange_interface import MultiExchange
from portfolio_management import PortfolioManagement
from config import Config
from database import Database
from event_bus import EventBus
from api_utils import BloFinAPIUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArbitrageBot:
    def __init__(self):
        self.config = Config()
        self.api_utils = BloFinAPIUtils(
            api_key=os.getenv("API_KEY"),
            api_secret=os.getenv("API_SECRET"),
            api_passphrase=os.getenv("API_PASSPHRASE")
        )
        self.exchange = MultiExchange(
            api_key=os.getenv("API_KEY"),
            api_secret=os.getenv("API_SECRET"),
            api_passphrase=os.getenv("API_PASSPHRASE")
        )
        self.portfolio = PortfolioManagement()
        self.database = Database(db_path=self.config.DB_PATH)
        self.event_bus = EventBus(redis_url=self.config.REDIS_URL)
        self.bot_name = "ArbitrageBot"

    async def initialize(self):
        await self.database.initialize()
        await self.event_bus.connect()

    async def find_arbitrage(self, symbol):
        prices = await self.exchange.get_multi_prices(symbol)
        if len(prices) < 2:
            return None
        max_price, min_price = max(prices.values()), min(prices.values())
        max_ex, min_ex = max(prices, key=prices.get), min(prices, key=prices.get)
        if (max_price - min_price) / min_price > 0.005:
            logger.info(f"Arbitrage: Buy {symbol} on {min_ex} at ${min_price}, sell on {max_ex} at ${max_price}")
            return "buy", min_price, min_ex, "sell", max_price, max_ex

    async def send_signal(self, symbol, signal, confidence, patterns, timeframes):
        payload = {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "patterns": patterns,
            "timeframes": timeframes,
            "bot_name": self.bot_name
        }
        try:
            response = requests.post(f"{self.config.MAIN_BOT_URL}/receive", json=payload)
            response.raise_for_status()
            logger.info(f"Signal sent for {symbol}: {signal} with confidence {confidence:.2f}")
        except Exception as e:
            logger.error(f"Failed to send signal for {symbol}: {e}")

    async def run(self):
        await self.initialize()
        while True:
            for symbol in self.config.SYMBOLS:
                arbitrage_opp = await self.find_arbitrage(symbol)
                if arbitrage_opp:
                    signal, min_price, min_ex, _, max_price, max_ex = arbitrage_opp
                    confidence = (max_price - min_price) / min_price * 10
                    patterns = [f"arbitrage_{min_ex}_to_{max_ex}"]
                    timeframes = ["1h"]
                    await self.send_signal(symbol, signal, min(confidence, 0.9), patterns, timeframes)
            await asyncio.sleep(60)

if __name__ == "__main__":
    bot = ArbitrageBot()
    asyncio.run(bot.run())