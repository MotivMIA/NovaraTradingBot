import asyncio
import os
import logging
import requests
from indicators import Indicators
from machine_learning import MachineLearning
from sentiment_analysis import SentimentAnalysis
from trading_logic import TradingLogic
from config import Config
from database import Database
from event_bus import EventBus
from api_utils import BloFinAPIUtils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictBot:
    def __init__(self):
        self.config = Config()
        self.api_utils = BloFinAPIUtils(
            api_key=os.getenv("API_KEY"),
            api_secret=os.getenv("API_SECRET"),
            api_passphrase=os.getenv("API_PASSPHRASE")
        )
        self.indicators = Indicators()
        self.ml = MachineLearning()
        self.sentiment = SentimentAnalysis()
        self.trading_logic = TradingLogic()
        self.database = Database(db_path=self.config.DB_PATH)
        self.event_bus = EventBus(redis_url=self.config.REDIS_URL)
        self.candle_history = {s: [] for s in self.config.SYMBOLS}
        self.bot_name = "PredictBot"

    async def initialize(self):
        await self.database.initialize()
        await self.event_bus.connect()
        for symbol in self.config.SYMBOLS:
            candles = await self.api_utils.get_candles(symbol, "1h", limit=100)
            self.candle_history[symbol] = candles
            await self.ml.train_ml_model(symbol, self)

    async def send_signal(self, signal_info, symbol, price):
        signal, confidence, patterns, timeframes = signal_info
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
                candles = await self.api_utils.get_candles(symbol, "1h", limit=1)
                if candles:
                    self.candle_history[symbol].append(candles[0])
                    price = candles[0]["close"]
                    signal_info = self.trading_logic.generate_signal(symbol, price, self)
                    if signal_info:
                        await self.send_signal(signal_info, symbol, price)
            await asyncio.sleep(60)

if __name__ == "__main__":
    bot = PredictBot()
    asyncio.run(bot.run())