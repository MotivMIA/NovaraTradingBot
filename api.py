import asyncio
import logging
import os
import socket
import sys
import time
import argparse
from datetime import datetime
import glob
from dotenv import load_dotenv
from logtail import LogtailHandler
from features.config import *
from features.indicators import Indicators
from features.trading_logic import TradingLogic
from features.machine_learning import MachineLearning
from features.sentiment_analysis import SentimentAnalysis
from features.portfolio_management import PortfolioManagement
from features.performance_analytics import PerformanceAnalytics
from features.backtesting import Backtesting
from features.database import Database
from features.api_utils import APIUtils
from features.notifications import Notifications
import websockets
import json

# Custom formatter for microseconds
class MicrosecondFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = f"{t},{int(record.msecs * 1000):06d}"
        return s

# Configure logging
def setup_logging():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"trading_bot_{timestamp}.log")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(MicrosecondFormatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S,%f"))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    
    # Logtail handler
    logtail_token = os.getenv('LOGTAIL_TOKEN')
    if logtail_token:
        logtail_handler = LogtailHandler(source_token=logtail_token)
        logtail_handler.setFormatter(MicrosecondFormatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S,%f"))
        logger.addHandler(logtail_handler)
        logger.info("Logtail integration enabled")
    else:
        logger.warning("LOGTAIL_TOKEN not set, skipping Logtail integration")
    
    # Clean old logs
    log_files = glob.glob(os.path.join(log_dir, "trading_bot_*.log"))
    log_files.sort(key=os.path.getctime, reverse=True)
    for old_log in log_files[5:]:
        try:
            os.remove(old_log)
            logger.debug(f"Deleted old log file: {old_log}")
        except Exception as e:
            logger.error(f"Failed to delete old log file {old_log}: {e}")
    return logger

logger = setup_logging()
logger.debug(f"Local hostname: {socket.gethostname()}")

# Load environment variables
load_dotenv()

# Configuration
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# Credentials
API_KEY = os.getenv("DEMO_API_KEY" if DEMO_MODE else "API_KEY", "1068db9f2fd8486dad50c5e304b0a150")
API_SECRET = os.getenv("DEMO_API_SECRET" if DEMO_MODE else "API_SECRET", "")
API_PASSPHRASE = os.getenv("DEMO_API_PASSPHRASE" if DEMO_MODE else "API_PASSPHRASE", "0NS8e3oKfiW2G4O9x")

if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
    logger.error("Missing API credentials in .env")
    exit(1)

logger.debug(f"api-key: {API_KEY[:4]}...{API_KEY[-4:]}")
logger.debug(f"access-passphrase: {API_PASSPHRASE[:4]}...{API_PASSPHRASE[-4:]}")

class TradingBot:
    def __init__(self):
        self.symbols = ["BTC-USDT", "ETH-USDT", "XRP-USDT"]
        self.candle_history = {symbol: [] for symbol in self.symbols}
        self.price_history = {symbol: [] for symbol in self.symbols}
        self.volume_history = {symbol: [] for symbol in self.symbols}
        self.last_candle_fetch = {symbol: {tf: 0 for tf in TIMEFRAMES} for symbol in self.symbols}
        self.account_balance = None
        self.initial_balance = None
        self.is_model_trained = {symbol: False for symbol in self.symbols}
        self.last_model_train = {symbol: 0 for symbol in self.symbols}
        self.leverage_info = {symbol: None for symbol in self.symbols}
        self.open_orders = {symbol: {} for symbol in self.symbols}
        self.sentiment_cache = {symbol: {"score": 0.0, "timestamp": 0} for symbol in self.symbols}
        
        self.indicators = Indicators()
        self.trading_logic = TradingLogic()
        self.ml = MachineLearning()
        self.sentiment = SentimentAnalysis()
        self.portfolio = PortfolioManagement()
        self.analytics = PerformanceAnalytics()
        self.backtesting = Backtesting()
        self.database = Database()
        self.api_utils = APIUtils(api_key=API_KEY, api_secret=API_SECRET, api_passphrase=API_PASSPHRASE)
        self.notifications = Notifications(webhook_url=WEBHOOK_URL)

    async def ws_connect(self, max_retries: int = 10):
        retry_count = 0
        while retry_count < max_retries:
            logger.debug(f"WebSocket connection attempt to {WS_URL} at {datetime.now(LOCAL_TZ)}")
            try:
                async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                    logger.info("WebSocket connected")
                    for symbol in self.symbols:
                        await ws.send(json.dumps({
                            "op": "subscribe",
                            "args": [{"channel": "tickers", "instId": symbol}]
                        }))
                        logger.info(f"Subscribed to {symbol} ticker")
                        self.database.load_candles(symbol, self.candle_history)
                        for tf in TIMEFRAMES:
                            candles = self.api_utils.get_candles(symbol, timeframe=tf)
                            if candles:
                                self.candle_history[symbol] = candles
                                self.database.save_candles(symbol, self.candle_history[symbol])
                                logger.info(f"Fetched {len(candles)} initial {tf} candles for {symbol}")
                    
                    last_order_time = {symbol: 0 for symbol in self.symbols}
                    last_symbol_update = 0
                    order_interval = 60
                    last_price = {symbol: None for symbol in self.symbols}
                    
                    while True:
                        try:
                            if time.time() - last_symbol_update > 3600:
                                new_symbols = self.portfolio.select_top_symbols(self)
                                for symbol in new_symbols:
                                    if symbol not in self.symbols:
                                        self.candle_history[symbol] = []
                                        self.price_history[symbol] = []
                                        self.volume_history[symbol] = []
                                        self.last_candle_fetch[symbol] = {tf: 0 for tf in TIMEFRAMES}
                                        self.is_model_trained[symbol] = False
                                        self.last_model_train[symbol] = 0
                                        self.leverage_info[symbol] = None
                                        self.open_orders[symbol] = {}
                                        self.sentiment_cache[symbol] = {"score": 0.0, "timestamp": 0}
                                        await ws.send(json.dumps({
                                            "op": "subscribe",
                                            "args": [{"channel": "tickers", "instId": symbol}]
                                        }))
                                self.symbols = new_symbols
                                last_symbol_update = time.time()
                            
                            data = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                            logger.debug(f"WebSocket data: {json.dumps(data, indent=2)}")
                            if "data" in data and data["data"]:
                                symbol = data["arg"]["instId"]
                                price = float(data["data"][0]["last"])
                                volume = float(data["data"][0]["lastSize"])
                                logger.info(f"WebSocket price for {symbol}: ${price}")
                                self.price_history[symbol].append(price)
                                self.volume_history[symbol].append(volume)
                                if len(self.price_history[symbol]) > 100:
                                    self.price_history[symbol] = self.price_history[symbol][-100:]
                                    self.volume_history[symbol] = self.volume_history[symbol][-100:]
                                
                                candles = self.api_utils.get_candles(symbol, limit=1, timeframe="1m")
                                if candles:
                                    self.candle_history[symbol].append(candles[0])
                                    if len(self.candle_history[symbol]) > 100:
                                        self.candle_history[symbol] = self.candle_history[symbol][-100:]
                                    self.database.save_candles(symbol, self.candle_history[symbol])
                                
                                self.trading_logic.manage_cost_averaging(symbol, price, self)
                                self.trading_logic.manage_trailing_stop(symbol, price, self)
                                
                                current_time = time.time()
                                if last_price[symbol] is None:
                                    last_price[symbol] = price
                                    continue
                                
                                price_change = abs(price - last_price[symbol]) / last_price[symbol]
                                if current_time - last_order_time[symbol] >= order_interval and price_change >= VOLATILITY_THRESHOLD:
                                    signal_info = self.trading_logic.generate_signal(symbol, price, self)
                                    if signal_info:
                                        signal, confidence, patterns, timeframes = signal_info
                                        logger.info(f"Signal for {symbol}: {signal} with confidence {confidence:.2f}")
                                        await self.trading_logic.process_trade(symbol, price, signal, price_change, confidence, patterns, timeframes, self)
                                        last_order_time[symbol] = current_time
                                        last_price[symbol] = price
                            await asyncio.sleep(0.2)
                        except (websockets.exceptions.ConnectionClosed, json.JSONDecodeError, asyncio.TimeoutError) as e:
                            logger.error(f"WebSocket error: {e}")
                            break
            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying WebSocket (attempt {retry_count + 1}/{max_retries})")
                    await asyncio.sleep(2 ** retry_count)
                else:
                    logger.error(f"Failed after {max_retries} attempts")
                    break

    async def health_check(self):
        logger.info("Running health check")
        try:
            logger.debug("Attempting to validate credentials")
            if not self.api_utils.validate_credentials():
                logger.error("Health check failed: Credential validation")
                return False
            logger.debug("Credentials validated, attempting to retrieve balance")
            balance = self.api_utils.get_account_balance(self)
            if balance is None:
                logger.error("Health check failed: Could not retrieve balance")
                return False
            self.account_balance = balance
            logger.info(f"Health check passed: Balance ${balance:.2f} USDT")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--health-check", action="store_true", help="Run health check and exit")
        args = parser.parse_args()

        if args.health_check:
            if await self.health_check():
                logger.info("Health check completed successfully")
                sys.exit(0)
            else:
                logger.error("Health check failed")
                sys.exit(1)

        self.database.initialize_db()
        self.symbols = self.portfolio.select_top_symbols(self)
        for symbol in self.symbols:
            self.candle_history[symbol] = []
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
            self.last_candle_fetch[symbol] = {tf: 0 for tf in TIMEFRAMES}
            self.is_model_trained[symbol] = False
            self.last_model_train[symbol] = 0
            self.leverage_info[symbol] = None
            self.open_orders[symbol] = {}
            self.sentiment_cache[symbol] = {"score": 0.0, "timestamp": 0}
        
        if not self.api_utils.validate_credentials():
            logger.error("Credential validation failed, exiting")
            return
        
        self.account_balance = self.api_utils.get_account_balance(self)
        if not self.account_balance:
            logger.error("Failed to get initial account balance, using default")
            self.account_balance = DEFAULT_BALANCE
        self.initial_balance = self.account_balance
        
        for symbol in self.symbols:
            price_volume = self.api_utils.get_price(symbol)
            if not price_volume:
                logger.error(f"Failed to get initial price for {symbol}")
                continue
            price, volume = price_volume
            logger.info(f"Current price for {symbol}: ${price}")
            self.price_history[symbol].append(price)
            self.volume_history[symbol].append(volume)
            self.database.load_candles(symbol, self.candle_history)
            for tf in TIMEFRAMES:
                candles = self.api_utils.get_candles(symbol, timeframe=tf)
                if candles:
                    self.candle_history[symbol] = candles
                    self.database.save_candles(symbol, self.candle_history[symbol])
                    logger.info(f"Fetched {len(candles)} initial {tf} candles for {symbol}")
            await self.backtesting.backtest_strategy(symbol, self)
            signal_info = self.trading_logic.generate_signal(symbol, price, self)
            if signal_info:
                signal, confidence, patterns, timeframes = signal_info
                price_change = self.trading_logic.check_volatility(symbol, price, self.price_history)[1]
                logger.info(f"Initial signal for {symbol}: {signal} with confidence {confidence:.2f}")
                await self.trading_logic.process_trade(symbol, price, signal, price_change, confidence, patterns, timeframes, self)
        
        await self.ws_connect()

if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.main())