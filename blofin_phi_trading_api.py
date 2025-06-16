import asyncio
import base64
import hmac
import hashlib
import json
import logging
import os
import time
import requests
import websockets
import pandas as pd
import numpy as np
import socket
import sqlite3
import glob
import sys
import argparse
import nltk
from datetime import datetime
import pytz
from dotenv import load_dotenv
from uuid import uuid4
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Configure logging with a new file per run
def setup_logging():
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"trading_bot_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S,%f",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    for handler in logger.handlers:
        handler.setFormatter(MicrosecondFormatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S,%f"))
    
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
DEMO_MODE = os.getenv("DEMO_MODE", "True").lower() == "true"
BASE_URL = "https://demo-trading-openapi.blofin.com" if DEMO_MODE else "https://openapi.blofin.com"
WS_URL = "wss://demo-trading-openapi.blofin.com/ws/public" if DEMO_MODE else "wss://openapi.blofin.com/ws/public"
SYMBOLS = []  # Dynamically populated
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))
SIZE_PRECISION = 8
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
EMA_FAST = 12
EMA_SLOW = 26
BB_PERIOD = 20
BB_STD = 2
ML_LOOKBACK = 50
MAX_DRAWDOWN = 0.10
DEFAULT_BALANCE = 10000.0
MIN_PRICE_POINTS = 1
VWAP_PERIOD = 20
VOLATILITY_THRESHOLD = 0.01
CANDLE_TIMEFRAME = "1m"
CANDLE_FETCH_INTERVAL = 60
CANDLE_LIMIT = 1000
DB_PATH = os.path.join("/opt/render/project/src/db", "market_data.db") if os.getenv("RENDER") else os.path.join(os.path.dirname(__file__), "market_data.db")
MAX_LEVERAGE = 10.0
TIMEFRAMES = ["1m", "5m", "15m", "1h", "1d"]
SENTIMENT_WEIGHT = 0.1
MAX_SYMBOLS = 10
CORRELATION_THRESHOLD = 0.7
COST_AVERAGE_DIP = 0.02
COST_AVERAGE_LIMIT = 2
TRAILING_STOP_MULTIPLIER = 1.5
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")  # Add to .env for Discord/Slack

# Credentials
API_KEY = os.getenv("DEMO_API_KEY" if DEMO_MODE else "API_KEY")
API_SECRET = os.getenv("DEMO_API_SECRET" if DEMO_MODE else "API_SECRET")
API_PASSPHRASE = os.getenv("DEMO_API_PASSPHRASE" if DEMO_MODE else "API_PASSPHRASE")

if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
    logger.error("Missing API credentials in .env")
    exit(1)

logger.debug(f"api-key: {API_KEY[:4]}...{API_KEY[-4:]}")
logger.debug(f"access-passphrase: {API_PASSPHRASE[:4]}...{API_PASSPHRASE[-4:]}")

# Timezone
LOCAL_TZ = pytz.timezone("America/Los_Angeles")

class TradingBot:
    def __init__(self):
        self.candle_history = {}
        self.price_history = {}
        self.volume_history = {}
        self.last_candle_fetch = {}
        self.account_balance = None
        self.initial_balance = None
        self.ml_model = LogisticRegression()
        self.rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.is_model_trained = {}
        self.last_model_train = {}
        self.leverage_info = {}
        self.open_orders = {}
        self.sentiment_cache = {}
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sid = SentimentIntensityAnalyzer()
            logger.info("NLTK vader_lexicon initialized")
        except Exception as e:
            logger.error(f"Failed to initialize NLTK vader_lexicon: {e}")
            self.sid = None

    def send_webhook_alert(self, message: str):
        if not WEBHOOK_URL:
            return
        try:
            payload = {"content": message}
            requests.post(WEBHOOK_URL, json=payload, timeout=5)
            logger.debug(f"Sent webhook alert: {message}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    def initialize_db(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    symbol TEXT,
                    entry_time INTEGER,
                    exit_time INTEGER,
                    entry_price REAL,
                    exit_price REAL,
                    size REAL,
                    leverage REAL,
                    profit_loss REAL,
                    features_used TEXT,
                    timeframes TEXT
                )
            """)
            conn.commit()
            conn.close()
            logger.info("Initialized trades table in database")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")

    def save_candles(self, symbol: str):
        try:
            conn = sqlite3.connect(DB_PATH)
            os.chmod(DB_PATH, 0o666) if os.path.exists(DB_PATH) else None
            df = pd.DataFrame(self.candle_history[symbol])
            if not df.empty:
                table_name = symbol.replace("-", "_") + "_candles"
                df.to_sql(table_name, conn, if_exists='append', index=False)
                logger.debug(f"Saved {len(df)} candles for {symbol} to {DB_PATH} (table: {table_name})")
            else:
                logger.warning(f"No candles to save for {symbol}")
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Failed to save candles for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving candles for {symbol}: {e}")

    def load_candles(self, symbol: str):
        try:
            conn = sqlite3.connect(DB_PATH)
            table_name = symbol.replace("-", "_") + "_candles"
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            self.candle_history[symbol] = df.to_dict('records')
            logger.info(f"Loaded {len(self.candle_history[symbol])} candles for {symbol} from {DB_PATH} (table: {table_name})")
            conn.close()
        except sqlite3.Error as e:
            logger.debug(f"No stored candles found for {symbol} or database error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading candles for {symbol}: {e}")

    def log_trade(self, symbol: str, entry_time: int, exit_time: int, entry_price: float, exit_price: float, size: float, leverage: float, features_used: dict, timeframes: list, side: str):
        try:
            profit_loss = (exit_price - entry_price) * size if side == "buy" else (entry_price - exit_price) * size
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (symbol, entry_time, exit_time, entry_price, exit_price, size, leverage, profit_loss, features_used, timeframes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, entry_time, exit_time, entry_price, exit_price, size, leverage, profit_loss, json.dumps(features_used), json.dumps(timeframes)))
            conn.commit()
            conn.close()
            logger.info(f"Logged trade for {symbol}: P/L ${profit_loss:.2f}")
            self.send_webhook_alert(f"Trade closed for {symbol}: P/L ${profit_loss:.2f}, Leverage: {leverage:.2f}x")
        except sqlite3.Error as e:
            logger.error(f"Failed to log trade for {symbol}: {e}")

    def analyze_performance(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT * FROM trades", conn)
            conn.close()
            if df.empty:
                logger.info("No trades to analyze")
                return
            
            win_rate = len(df[df["profit_loss"] > 0]) / len(df)
            avg_pl = df["profit_loss"].mean()
            sharpe_ratio = df["profit_loss"].mean() / df["profit_loss"].std() * np.sqrt(252) if df["profit_loss"].std() != 0 else 0
            feature_impact = {}
            for feature in ["atr", "sentiment", "multi_timeframe"]:
                feature_impact[feature] = df[df["features_used"].str.contains(feature)]["profit_loss"].mean()
            
            report = {
                "win_rate": win_rate,
                "avg_profit_loss": avg_pl,
                "sharpe_ratio": sharpe_ratio,
                "feature_impact": feature_impact
            }
            logger.info(f"Performance report: {json.dumps(report, indent=2)}")
            self.send_webhook_alert(f"Performance Report: Win Rate {win_rate:.2%}, Sharpe {sharpe_ratio:.2f}")
            return report
        except Exception as e:
            logger.error(f"Failed to analyze performance: {e}")
            return None

    def generate_performance_plots(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT * FROM trades", conn)
            conn.close()
            if df.empty:
                logger.info("No trades to plot")
                return
            
            fig = make_subplots(rows=3, cols=1, subplot_titles=("Profit/Loss Over Time", "Win/Loss Distribution", "Leverage vs. Return"))
            
            fig.add_trace(go.Scatter(x=pd.to_datetime(df["exit_time"], unit="s"), y=df["profit_loss"].cumsum(), mode="lines", name="Cumulative P/L"), row=1, col=1)
            fig.add_trace(go.Histogram(x=df["profit_loss"], name="P/L Distribution"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df["leverage"], y=df["profit_loss"], mode="markers", name="Leverage vs. P/L"), row=3, col=1)
            
            plot_dir = os.path.join(os.path.dirname(__file__), "plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_file = os.path.join(plot_dir, f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            fig.write_html(plot_file)
            logger.info(f"Generated performance plot: {plot_file}")
            self.send_webhook_alert(f"New performance plot generated: {plot_file}")
        except Exception as e:
            logger.error(f"Failed to generate performance plots: {e}")

    def select_top_symbols(self) -> list:
        path = "/api/v1/market/tickers"
        try:
            response = requests.get(f"{BASE_URL}{path}", timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("code") != "0" or not data.get("data"):
                logger.error(f"Failed to fetch tickers: {data}")
                return ["BTC-USDT", "ETH-USDT", "XRP-USDT"]
            
            df = pd.DataFrame(data["data"])
            df["vol24h"] = df["vol24h"].astype(float)
            df["price_change"] = (df["last"].astype(float) - df["open24h"].astype(float)) / df["open24h"].astype(float)
            df["atr"] = 0.0
            for symbol in df["instId"]:
                candles = self.get_candles(symbol, limit=50)
                if candles:
                    df_candles = pd.DataFrame(candles)
                    df_candles["tr"] = df_candles[["high", "low", "close"]].apply(
                        lambda x: max(x["high"] - x["low"], abs(x["high"] - x["close"].shift(1)), abs(x["low"] - x["close"].shift(1))), axis=1
                    )
                    df.loc[df["instId"] == symbol, "atr"] = df_candles["tr"].rolling(window=14).mean().iloc[-1]
            
            df["score"] = 0.5 * df["price_change"] + 0.3 * df["vol24h"] / df["vol24h"].max() - 0.2 * df["atr"] / df["atr"].max()
            top_symbols = df[df["vol24h"] > 1_000_000]["instId"].nlargest(MAX_SYMBOLS).tolist()
            logger.info(f"Selected top symbols: {top_symbols}")
            self.send_webhook_alert(f"Updated top symbols: {top_symbols}")
            return top_symbols
        except Exception as e:
            logger.error(f"Failed to select top symbols: {e}")
            return ["BTC-USDT", "ETH-USDT", "XRP-USDT"]

    def calculate_portfolio_allocation(self) -> dict:
        allocations = {symbol: RISK_PER_TRADE / len(SYMBOLS) for symbol in SYMBOLS}
        try:
            correlations = {}
            for i, symbol1 in enumerate(SYMBOLS):
                for symbol2 in SYMBOLS[i+1:]:
                    df1 = pd.DataFrame(self.candle_history.get(symbol1, []))["close"]
                    df2 = pd.DataFrame(self.candle_history.get(symbol2, []))["close"]
                    if len(df1) > 14 and len(df2) > 14:
                        corr = df1.tail(14).corr(df2.tail(14))
                        if corr > CORRELATION_THRESHOLD:
                            correlations[(symbol1, symbol2)] = corr
            
            atrs = {symbol: self.calculate_atr(symbol) for symbol in SYMBOLS}
            total_inverse_atr = sum(1 / atr if atr > 0 else 1 for atr in atrs.values())
            for symbol in SYMBOLS:
                atr = atrs[symbol] if atrs[symbol] > 0 else 1
                allocations[symbol] = (1 / atr / total_inverse_atr) * RISK_PER_TRADE
            
            for (s1, s2), corr in correlations.items():
                total_alloc = allocations[s1] + allocations[s2]
                allocations[s1] = total_alloc * 0.6
                allocations[s2] = total_alloc * 0.4
            
            logger.debug(f"Portfolio allocations: {allocations}")
            return allocations
        except Exception as e:
            logger.error(f"Failed to calculate portfolio allocations: {e}")
            return allocations

    def get_candles(self, symbol: str, limit: int = CANDLE_LIMIT, timeframe: str = "1m") -> list | None:
        current_time = time.time()
        if current_time - self.last_candle_fetch.get(symbol, {}).get(timeframe, 0) < CANDLE_FETCH_INTERVAL:
            logger.debug(f"Skipping candle fetch for {symbol} ({timeframe}): within {CANDLE_FETCH_INTERVAL}s interval")
            return None
        path = "/api/v1/market/candles"
        params = {"instId": symbol, "bar": timeframe, "limit": str(limit)}
        for attempt in range(3):
            try:
                response = requests.get(f"{BASE_URL}{path}", params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Candle response for {symbol} ({timeframe}): {json.dumps(data, indent=2)}")
                if data.get("code") == "0" and data.get("data"):
                    self.last_candle_fetch.setdefault(symbol, {})[timeframe] = current_time
                    candles = [{
                        "timestamp": int(candle[0]),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5])
                    } for candle in data["data"]]
                    return candles
                logger.error(f"No candle data for {symbol} ({timeframe})")
                return None
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed after {attempt + 1} attempts: {e}")
                    return None
        return None

    def get_instrument_info(self, symbol: str) -> dict | None:
        path = "/api/v1/market/instruments"
        params = {"instType": "SWAP", "instId": symbol}
        for attempt in range(3):
            try:
                response = requests.get(f"{BASE_URL}{path}", params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                logger.info(f"Instrument info for {symbol}: {json.dumps(data, indent=2)}")
                if data.get("code") == "0" and data.get("data"):
                    for inst in data["data"]:
                        if inst["instId"] == symbol:
                            return {
                                "minSize": float(inst.get("minSize", 0.1)),
                                "lotSize": float(inst.get("lotSize", 0.1)),
                                "tickSize": float(inst.get("tickSize", 0.1)),
                                "contractValue": float(inst.get("contractValue", 0.001))
                            }
                logger.error(f"No instrument info for {symbol}")
                return None
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed after {attempt + 1} attempts: {e}")
                    return None
        return None

    def get_price(self, symbol: str) -> tuple[float, float] | None:
        path = "/api/v1/market/tickers"
        params = {"instId": symbol}
        for attempt in range(3):
            try:
                response = requests.get(f"{BASE_URL}{path}", params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                logger.info(f"Ticker response for {symbol}: {json.dumps(data, indent=2)}")
                if data.get("code") == "0" and data.get("data") and "last" in data["data"][0]:
                    return float(data["data"][0]["last"]), float(data["data"][0]["lastSize"])
                logger.error(f"Unexpected response: {data}")
                return None
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed after {attempt + 1} attempts: {e}")
                    return None
        return None

    def get_account_balance(self) -> float | None:
        import urllib.request
        try:
            outbound_ip = urllib.request.urlopen('https://api.ipify.org').read().decode()
            logger.debug(f"Outbound IP: {outbound_ip}")
        except Exception as e:
            logger.error(f"Failed to get outbound IP: {e}")
        path = "/api/v1/account/balance"
        for attempt in range(5):
            headers, _, _ = self.sign_request("GET", path)
            try:
                response = requests.get(f"{BASE_URL}{path}", headers=headers, timeout=5)
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Balance response: {data}")
                if data.get("code") == "0" and data.get("data") and "details" in data["data"]:
                    for asset in data["data"]["details"]:
                        if isinstance(asset, dict) and asset.get("currency") == "USDT":
                            balance = float(asset.get("balance", 0))
                            logger.info(f"Account balance: ${balance:.2f} USDT")
                            return balance
                    logger.error("No USDT balance found in response")
                    return None
                logger.error(f"Unexpected balance response: {data}")
                if data.get("code") == "152409":
                    logger.error("Signature verification failed, possible credential mismatch")
                if data.get("code") == "152401":
                    logger.error("Access key does not exist, verify API key")
                if data.get("code") == "152406":
                    logger.error("IP whitelisting issue, verify IP settings")
                time.sleep(2 ** attempt)
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed after 5 attempts: {e}")
                    return None
        logger.warning(f"Using default balance: ${DEFAULT_BALANCE:.2f} USDT")
        return DEFAULT_BALANCE

    def get_max_leverage(self, symbol: str) -> float | None:
        path = "/api/v1/account/leverage-info"
        params = {"instId": symbol, "marginMode": "cross"}
        headers, _, _ = self.sign_request("GET", path, params=params)
        for attempt in range(3):
            try:
                response = requests.get(f"{BASE_URL}{path}", headers=headers, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Leverage info for {symbol}: {json.dumps(data, indent=2)}")
                if data.get("code") == "0" and data.get("data"):
                    max_leverage = float(data["data"][0].get("maxLeverage", 1.0))
                    self.leverage_info[symbol] = max_leverage
                    logger.info(f"Max leverage for {symbol}: {max_leverage}x")
                    return max_leverage
                logger.error(f"No leverage info for {symbol}")
                return 1.0
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed after {attempt + 1} attempts: {e}")
                    return 1.0
        return 1.0

    def calculate_margin(self, symbol: str, size_usd: float, confidence: float, volatility: float, patterns: dict) -> tuple[float, float]:
        max_leverage = self.leverage_info.get(symbol) or self.get_max_leverage(symbol)
        risk_amount = size_usd
        
        leverage = 1.0
        leverage_percentage = 0.0
        if confidence > 0.7 or patterns.get("bullish_engulfing") or patterns.get("bearish_engulfing") or patterns.get("bullish_pin") or patterns.get("bearish_pin"):
            leverage_percentage = 0.5 + (confidence - 0.7) * 1.5
            leverage = max_leverage * min(leverage_percentage, 0.8)
        elif confidence > 0.5 or patterns.get("doji") or patterns.get("inside_bar"):
            leverage = min(max_leverage, 3.0)
            leverage_percentage = leverage / max_leverage if max_leverage > 0 else 1.0
        if volatility > 2 * VOLATILITY_THRESHOLD:
            leverage = min(leverage, 2.0)
            leverage_percentage = leverage / max_leverage if max_leverage > 0 else 1.0
        
        if confidence <= 0.7:
            leverage = min(leverage, MAX_LEVERAGE)
        
        margin_amount = risk_amount * leverage * 0.8
        logger.debug(f"Calculated margin for {symbol}: ${margin_amount:.2f} at {leverage:.2f}x leverage ({leverage_percentage*100:.1f}% of {max_leverage}x)")
        return margin_amount, leverage

    def calculate_atr(self, symbol: str, period: int = 14) -> float:
        candles = self.candle_history.get(symbol, [])
        if len(candles) < period + 1:
            return 0.0
        df = pd.DataFrame(candles[-period-1:])
        df["high_low"] = df["high"] - df["low"]
        df["high_prev_close"] = abs(df["high"] - df["close"].shift(1))
        df["low_prev_close"] = abs(df["low"] - df["close"].shift(1))
        df["tr"] = df[["high_low", "high_prev_close", "low_prev_close"]].max(axis=1)
        atr = df["tr"].rolling(window=period).mean().iloc[-1]
        logger.debug(f"ATR for {symbol}: ${atr:.2f}")
        return atr if not np.isnan(atr) else 0.0

    def get_x_sentiment(self, symbol: str) -> float:
        if not self.sid:
            return 0.0
        current_time = time.time()
        if current_time - self.sentiment_cache.get(symbol, {}).get("timestamp", 0) < 300:
            return self.sentiment_cache[symbol]["score"]
        
        mock_posts = [
            f"{symbol.replace('-USDT', '')} to the moon! 🚀",
            f"Bearish on {symbol.replace('-USDT', '')} this week.",
            f"Buying {symbol.replace('-USDT', '')} at dip!"
        ]
        scores = [self.sid.polarity_scores(post)["compound"] for post in mock_posts]
        sentiment_score = sum(scores) / len(scores) if scores else 0.0
        self.sentiment_cache[symbol] = {"score": sentiment_score, "timestamp": current_time}
        logger.debug(f"X sentiment for {symbol}: {sentiment_score:.2f}")
        return sentiment_score

    def calculate_indicators(self, symbol: str, timeframe: str = "1m") -> dict | None:
        candles = self.get_candles(symbol, timeframe=timeframe) or self.candle_history.get(symbol, [])
        if len(candles) < MIN_PRICE_POINTS:
            logger.warning(f"Insufficient candle data for {symbol} ({timeframe}): {len(candles)} candles")
            return {"price": candles[-1]["close"] if candles else None}
        
        df = pd.DataFrame(candles)
        indicators = {"price": df["close"].iloc[-1]}
        vwap = self.calculate_vwap(symbol)
        if vwap:
            indicators["vwap"] = vwap
        
        if len(candles) >= RSI_PERIOD:
            indicators["rsi"] = RSIIndicator(df["close"], window=RSI_PERIOD).rsi().iloc[-1]
        if len(candles) >= max(MACD_SLOW + MACD_SIGNAL, EMA_SLOW, BB_PERIOD):
            macd = MACD(df["close"], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
            indicators["macd"] = macd.macd().iloc[-1]
            indicators["macd_signal"] = macd.macd_signal().iloc[-1]
            indicators["ema_fast"] = EMAIndicator(df["close"], window=EMA_FAST).ema_indicator().iloc[-1]
            indicators["ema_slow"] = EMAIndicator(df["close"], window=EMA_SLOW).ema_indicator().iloc[-1]
            bb = BollingerBands(df["close"], window=BB_PERIOD, window_dev=BB_STD)
            indicators["bb_upper"] = bb.bollinger_hband().iloc[-1]
            indicators["bb_lower"] = bb.bollinger_lband().iloc[-1]
        indicators["price_lag1"] = df["close"].iloc[-2] if len(df) > 1 else np.nan
        indicators["price_lag2"] = df["close"].iloc[-3] if len(df) > 2 else np.nan
        indicators["volume_change"] = df["volume"].pct_change().iloc[-1] if len(df) > 1 else 0.0
        
        logger.debug(f"{symbol} ({timeframe}) indicators - RSI: {indicators.get('rsi', 'N/A')}")
        return indicators

    def train_ml_model(self, symbol: str):
        current_time = time.time()
        if current_time - self.last_model_train.get(symbol, 0) < 300:
            return
        
        prices = self.price_history.get(symbol, [])
        volumes = self.volume_history.get(symbol, [])
        if len(prices) < ML_LOOKBACK + RSI_PERIOD + MACD_SLOW + MACD_SIGNAL:
            logger.warning(f"Insufficient data for {symbol}: {len(prices)} points")
            return
        
        df = pd.DataFrame({"price": prices, "volume": volumes})
        for tf in TIMEFRAMES:
            tf_indicators = self.calculate_indicators(symbol, timeframe=tf)
            if tf_indicators:
                df[f"rsi_{tf}"] = tf_indicators.get("rsi", 50.0)
                df[f"macd_{tf}"] = tf_indicators.get("macd", 0.0)
                df[f"macd_signal_{tf}"] = tf_indicators.get("macd_signal", 0.0)
                df[f"bb_upper_{tf}"] = tf_indicators.get("bb_upper", df["price"].iloc[-1] * 1.02)
                df[f"bb_lower_{tf}"] = tf_indicators.get("bb_lower", df["price"].iloc[-1] * 0.98)
        
        df["atr"] = pd.DataFrame(self.candle_history[symbol])[["high", "low", "close"]].apply(
            lambda x: max(x["high"] - x["low"], abs(x["high"] - x["close"].shift(1)), abs(x["low"] - x["close"].shift(1))), axis=1
        ).rolling(window=14).mean()
        df["volume_change"] = df["volume"].pct_change()
        df["sentiment"] = self.get_x_sentiment(symbol)
        df["price_lag1"] = df["price"].shift(1)
        df["price_lag2"] = df["price"].shift(2)
        df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
        
        df = df.dropna()
        if len(df) < 10:
            logger.warning(f"Insufficient valid data for {symbol}: {len(df)} rows")
            return
        
        features = [
            f"rsi_{tf}" for tf in TIMEFRAMES
        ] + [
            f"macd_{tf}" for tf in TIMEFRAMES
        ] + [
            f"macd_signal_{tf}" for tf in TIMEFRAMES
        ] + [
            f"bb_upper_{tf}" for tf in TIMEFRAMES
        ] + [
            f"bb_lower_{tf}" for tf in TIMEFRAMES
        ] + [
            "atr", "volume_change", "sentiment", "price_lag1", "price_lag2"
        ]
        X = df[features].values
        y = df["target"].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.partial_fit(X_scaled, y, classes=[0, 1])
        self.rf_model.partial_fit(X_scaled, y, classes=[0, 1])
        self.ensemble_model = VotingClassifier(
            estimators=[('lr', self.ml_model), ('rf', self.rf_model)], voting='soft'
        )
        self.ensemble_model.fit(X_scaled, y)
        self.is_model_trained[symbol] = True
        self.last_model_train[symbol] = current_time
        logger.info(f"ML ensemble trained for {symbol} with {len(df)} data points")

    def predict_ml_signal(self, symbol: str, indicators: dict) -> tuple[str, float] | None:
        if not self.is_model_trained.get(symbol, False):
            self.train_ml_model(symbol)
            if not self.is_model_trained[symbol]:
                return None
        
        features = []
        for tf in TIMEFRAMES:
            tf_indicators = self.calculate_indicators(symbol, timeframe=tf)
            features.extend([
                tf_indicators.get("rsi", 50.0),
                tf_indicators.get("macd", 0.0),
                tf_indicators.get("macd_signal", 0.0),
                tf_indicators.get("bb_upper", indicators["price"] * 1.02),
                tf_indicators.get("bb_lower", indicators["price"] * 0.98)
            ])
        features.extend([
            self.calculate_atr(symbol),
            indicators.get("volume_change", 0.0),
            self.get_x_sentiment(symbol),
            indicators.get("price_lag1", indicators["price"]),
            indicators.get("price_lag2", indicators["price"])
        ])
        
        if any(np.isnan(f) for f in features):
            logger.warning(f"Invalid features for {symbol}: {features}")
            return None
        
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        prediction = self.ensemble_model.predict_proba(X_scaled)[0]
        confidence = max(prediction)
        
        if confidence < 0.6:
            return None
        
        return ("buy" if prediction[1] > prediction[0] else "sell", confidence)

    def analyze_indicators_and_patterns(self, symbol):
        indicators = self.calculate_indicators(symbol)
        patterns = self.detect_price_action_patterns(symbol)
        if not indicators or not indicators["price"]:
            logger.warning(f"No indicators available for {symbol}")
            return None, None
        return indicators, patterns
    
    def analyze_indicators(self, symbol, indicators):
        vwap = indicators.get("vwap")
        rsi = indicators.get("rsi")
        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        ema_fast = indicators.get("ema_fast")
        ema_slow = indicators.get("ema_slow")
        bb_upper = indicators.get("bb_upper")
        bb_lower = indicators.get("bb_lower")
        price = indicators.get("price")
        
        confidence = 0.0
        signal = None
        return signal, confidence, vwap, rsi, macd, macd_signal, ema_fast, ema_slow, bb_upper, bb_lower, price
    
    def check_volatility(self, symbol: str, price: float) -> tuple[bool, float]:
        volatility = 0.0
        if len(self.price_history[symbol]) >= 2:
            volatility = abs(price - self.price_history[symbol][-2]) / self.price_history[symbol][-2]
            if volatility < VOLATILITY_THRESHOLD:
                logger.debug(f"Low volatility for {symbol}: {volatility:.4f}")
                return False, volatility
        logger.debug(f"Volatility check passed for {symbol}: {volatility:.4f}")
        return True, volatility

    def analyze_patterns_and_indicators(self, patterns, symbol, price, vwap, confidence, signal):
        if patterns:
            previous_price = self.price_history[symbol][-2] if len(self.price_history[symbol]) >= 2 else price
            if patterns.get("bullish_engulfing") or patterns.get("bullish_pin") or (patterns.get("inside_bar") and price > previous_price):
                signal = "buy"
                confidence += 0.4
            elif patterns.get("bearish_engulfing") or patterns.get("bearish_pin") or (patterns.get("inside_bar") and price < previous_price):
                signal = "sell"
                confidence += 0.4
            elif patterns.get("doji") and vwap is not None:
                if price > vwap:
                    signal = "buy"
                    confidence += 0.3
                else:
                    signal = "sell"
                    confidence += 0.3
        return signal, confidence
    
    def analyze_vwap(self, vwap, price, signal, confidence):
        if vwap is not None:
            if price > vwap:
                if signal == "buy":
                    confidence += 0.3
                elif not signal:
                    signal = "buy"
                    confidence += 0.3
            elif price < vwap:
                if signal == "sell":
                    confidence += 0.3
                elif not signal:
                    signal = "sell"
                    confidence += 0.3
        return signal, confidence
    
    def check_indicators(self, symbol: str, macd: float, macd_signal: float, ema_fast: float, ema_slow: float, rsi: float, bb_upper: float, bb_lower: float, price: float, signal: str, confidence: float):
        if len(self.candle_history[symbol]) >= RSI_PERIOD:
            if macd is not None and macd_signal is not None and ema_fast is not None and ema_slow is not None:
                if macd > macd_signal and ema_fast > ema_slow:
                    if signal == "buy":
                        confidence += 0.2
                    elif not signal:
                        signal = "buy"
                        confidence += 0.2
                elif macd < macd_signal and ema_fast < ema_slow:
                    if signal == "sell":
                        confidence += 0.2
                    elif not signal:
                        signal = "sell"
                        confidence += 0.2
            
            if rsi is not None and bb_upper is not None and bb_lower is not None:
                if rsi < RSI_OVERSOLD and price <= bb_lower:
                    if signal == "buy":
                        confidence += 0.2
                    elif not signal:
                        signal = "buy"
                        confidence += 0.2
                elif rsi > RSI_OVERBOUGHT and price >= bb_upper:
                    if signal == "sell":
                        confidence += 0.2
                    elif not signal:
                        signal = "sell"
                        confidence += 0.2
        return signal, confidence

    def evaluate_signal(self, symbol, signal, confidence):
        if not signal or confidence < 0.5:
            logger.debug(f"No valid signal for {symbol}: confidence {confidence:.2f}")
            return None
        logger.info(f"Generated signal for {symbol}: {signal} with confidence {confidence:.2f}")
        return signal, confidence

    def generate_signal(self, symbol: str, current_price: float) -> tuple[str, float, dict, list] | None:
        logger.debug(f"Generating signal for {symbol} at current_price: {current_price}")
        if len(self.price_history[symbol]) < MIN_PRICE_POINTS or len(self.candle_history[symbol]) < MIN_PRICE_POINTS:
            logger.warning(f"Skipping signal for {symbol}: insufficient data (price: {len(self.price_history[symbol])}, candles: {len(self.candle_history[symbol])})")
            return None
        
        indicators, patterns = self.analyze_indicators_and_patterns(symbol)
        if not indicators:
            return None
        
        signal, confidence, vwap, rsi, macd, macd_signal, ema_fast, ema_slow, bb_upper, bb_lower, price = self.analyze_indicators(symbol, indicators)
        
        is_volatile, volatility = self.check_volatility(symbol, current_price)
        if not is_volatile:
            return None
        
        timeframes_used = ["1m"]
        timeframe_weights = {"1m": 0.4, "5m": 0.2, "15m": 0.15, "1h": 0.15, "1d": 0.1}
        for tf in TIMEFRAMES[1:]:
            tf_indicators = self.calculate_indicators(symbol, timeframe=tf)
            if tf_indicators:
                tf_rsi = tf_indicators.get("rsi", 50.0)
                if tf_rsi < RSI_OVERSOLD and signal == "buy":
                    confidence += timeframe_weights[tf]
                    timeframes_used.append(tf)
                elif tf_rsi > RSI_OVERBOUGHT and signal == "sell":
                    confidence += timeframe_weights[tf]
                    timeframes_used.append(tf)
        
        signal, confidence = self.analyze_patterns_and_indicators(patterns, symbol, current_price, vwap, confidence, signal)
        
        signal, confidence = self.analyze_vwap(vwap, current_price, signal, confidence)
        
        signal, confidence = self.check_indicators(symbol, macd, macd_signal, ema_fast, ema_slow, rsi, bb_upper, bb_lower, current_price, signal, confidence)
        
        ml_signal = self.predict_ml_signal(symbol, indicators)
        if ml_signal:
            ml_side, ml_confidence = ml_signal
            if signal == ml_side:
                confidence += ml_confidence * 0.2
            elif not signal:
                signal = ml_side
                confidence += ml_confidence * 0.2
        
        sentiment_score = self.get_x_sentiment(symbol)
        if signal == "buy" and sentiment_score > 0.3:
            confidence += SENTIMENT_WEIGHT
        elif signal == "sell" and sentiment_score < -0.3:
            confidence += SENTIMENT_WEIGHT
        
        signal_info = self.evaluate_signal(symbol, signal, confidence)
        if signal_info:
            return signal_info[0], signal_info[1], patterns or {}, timeframes_used
        return None

    def place_order(self, symbol: str, price: float, size_usd: float, side: str, confidence: float, patterns: dict, volatility: float, max_retries: int = 3) -> str | None:
        logger.info(f"Attempting to place {side} order for {symbol}: ${size_usd:.2f} at ${price}")
        path = "/api/v1/trade/order"
        inst_info = self.get_instrument_info(symbol)
        if not inst_info:
            logger.error(f"Failed to get instrument info for {symbol}")
            return None
        
        min_size = inst_info["minSize"]
        lot_size = inst_info["lotSize"]
        contract_value = inst_info["contractValue"]
        
        atr = self.calculate_atr(symbol)
        risk_multiplier = 1.0 if atr == 0 else max(0.5, min(2.0, 100 / atr))
        adjusted_size_usd = size_usd * risk_multiplier
        logger.debug(f"Adjusted size for {symbol}: ${adjusted_size_usd:.2f} based on ATR ${atr:.2f}")
        
        margin_amount, leverage = self.calculate_margin(symbol, adjusted_size_usd, confidence, volatility, patterns)
        size = (margin_amount / price) / contract_value
        size = max(round(size / lot_size) * lot_size, min_size)
        logger.info(f"Calculated order size for {symbol}: {size} contracts at ${price} with {leverage:.2f}x leverage")
        
        stop_loss = price * (0.99 if side == "buy" else 1.01)
        take_profit = price * (1.02 if side == "buy" else 0.98)
        
        order_request = {
            "instId": symbol,
            "instType": "SWAP",
            "marginMode": "cross",
            "leverage": str(leverage),
            "positionSide": "net",
            "side": side,
            "orderType": "limit",
            "price": str(round(price, SIZE_PRECISION)),
            "size": str(size)
        }
        
        for attempt in range(max_retries):
            try:
                headers, _, _ = self.sign_request("POST", path, body=order_request)
                response = requests.post(f"{BASE_URL}{path}", headers=headers, json=order_request, timeout=5)
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Order response for {symbol}: {data}")
                if data.get("code") == "0" and data.get("data"):
                    order_id = data["data"][0]["orderId"]
                    logger.info(f"Placed {side} order for {symbol}: {size} contracts at ${price}, SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}, Leverage: {leverage:.2f}x")
                    self.send_webhook_alert(f"Placed {side} order for {symbol}: ${size_usd:.2f} at ${price}, Leverage: {leverage:.2f}x")
                    self.open_orders.setdefault(symbol, {})[order_id] = {
                        "entry_price": price,
                        "size": size,
                        "leverage": leverage,
                        "side": side,
                        "confidence": confidence,
                        "patterns": patterns,
                        "volatility": volatility,
                        "cost_average_count": 0,
                        "entry_time": int(time.time())
                    }
                    return order_id
                logger.error(f"Order failed for {symbol}: {data}")
                if data.get("code") == "152406":
                    logger.error("IP whitelisting issue, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                return None
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    return None
        return None

    def manage_cost_averaging(self, symbol: str, current_price: float):
        if symbol not in self.open_orders:
            return
        
        for order_id, order in list(self.open_orders[symbol].items()):
            if order["confidence"] < 0.7:
                continue
            entry_price = order["entry_price"]
            size = order["size"]
            count = order.get("cost_average_count", 0)
            if count >= COST_AVERAGE_LIMIT:
                continue
            dip_threshold = entry_price * (1 - COST_AVERAGE_DIP if order["side"] == "buy" else 1 + COST_AVERAGE_DIP)
            if (order["side"] == "buy" and current_price <= dip_threshold) or (order["side"] == "sell" and current_price >= dip_threshold):
                new_size = size * 0.5
                new_leverage = order["leverage"] * 0.5
                new_order_id = self.place_order(
                    symbol, current_price, new_size * current_price, order["side"],
                    order["confidence"], order["patterns"], order["volatility"]
                )
                if new_order_id:
                    self.open_orders[symbol][new_order_id] = {
                        "entry_price": current_price,
                        "size": new_size,
                        "leverage": new_leverage,
                        "side": order["side"],
                        "confidence": order["confidence"],
                        "patterns": order["patterns"],
                        "volatility": order["volatility"],
                        "cost_average_count": count + 1,
                        "entry_time": int(time.time())
                    }
                    logger.info(f"Cost-averaged {order['side']} order for {symbol} at ${current_price}")

    def manage_trailing_stop(self, symbol: str, current_price: float):
        if symbol not in self.open_orders:
            return
        
        atr = self.calculate_atr(symbol)
        for order_id, order in list(self.open_orders[symbol].items()):
            if order["side"] == "buy":
                new_stop = max(order.get("trailing_stop", order["entry_price"] * 0.99), current_price - atr * TRAILING_STOP_MULTIPLIER)
                if new_stop > order.get("trailing_stop", 0):
                    order["trailing_stop"] = new_stop
                    logger.debug(f"Updated trailing stop for {symbol} buy order {order_id} to ${new_stop:.2f}")
                if current_price <= new_stop:
                    self.log_trade(
                        symbol, order["entry_time"], int(time.time()), order["entry_price"],
                        current_price, order["size"], order["leverage"],
                        {"atr": atr, "sentiment": self.get_x_sentiment(symbol), "multi_timeframe": True},
                        TIMEFRAMES, order["side"]
                    )
                    del self.open_orders[symbol][order_id]
            else:
                new_stop = min(order.get("trailing_stop", order["entry_price"] * 1.01), current_price + atr * TRAILING_STOP_MULTIPLIER)
                if new_stop < order.get("trailing_stop", float("inf")):
                    order["trailing_stop"] = new_stop
                    logger.debug(f"Updated trailing stop for {symbol} sell order {order_id} to ${new_stop:.2f}")
                if current_price >= new_stop:
                    self.log_trade(
                        symbol, order["entry_time"], int(time.time()), order["entry_price"],
                        current_price, order["size"], order["leverage"],
                        {"atr": atr, "sentiment": self.get_x_sentiment(symbol), "multi_timeframe": True},
                        TIMEFRAMES, order["side"]
                    )
                    del self.open_orders[symbol][order_id]

    async def backtest_strategy(self, symbol: str, lookback_days: int = 7):
        try:
            start_time = int(time.time() - lookback_days * 86400)
            conn = sqlite3.connect(DB_PATH)
            table_name = symbol.replace("-", "_") + "_candles"
            df = pd.read_sql(f"SELECT * FROM {table_name} WHERE timestamp >= {start_time}", conn)
            conn.close()
            if df.empty:
                logger.info(f"No historical data for backtesting {symbol}")
                return
            
            simulated_balance = self.account_balance or DEFAULT_BALANCE
            open_orders = {}
            for i in range(len(df) - 1):
                price = df["close"].iloc[i]
                signal_info = self.generate_signal(symbol, price)
                if signal_info:
                    signal, confidence, patterns, timeframes = signal_info
                    price_change = abs(price - df["close"].iloc[i-1]) / df["close"].iloc[i-1] if i > 0 else VOLATILITY_THRESHOLD
                    risk_amount = simulated_balance * RISK_PER_TRADE
                    margin_amount, leverage = self.calculate_margin(symbol, risk_amount, confidence, price_change, patterns)
                    size = margin_amount / price
                    atr = self.calculate_atr(symbol)
                    stop_loss = price * (0.99 if signal == "buy" else 1.01)
                    open_orders[i] = {
                        "entry_price": price,
                        "size": size,
                        "leverage": leverage,
                        "side": signal,
                        "trailing_stop": stop_loss,
                        "entry_time": df["timestamp"].iloc[i]
                    }
                
                for idx, order in list(open_orders.items()):
                    new_stop = (order["entry_price"] - atr * TRAILING_STOP_MULTIPLIER if order["side"] == "buy" else
                                order["entry_price"] + atr * TRAILING_STOP_MULTIPLIER)
                    order["trailing_stop"] = max(new_stop, order["trailing_stop"]) if order["side"] == "buy" else min(new_stop, order["trailing_stop"])
                    next_price = df["close"].iloc[i+1]
                    if (order["side"] == "buy" and next_price <= order["trailing_stop"]) or (order["side"] == "sell" and next_price >= order["trailing_stop"]):
                        pl = (next_price - order["entry_price"]) * order["size"] if order["side"] == "buy" else (order["entry_price"] - next_price) * order["size"]
                        simulated_balance += pl
                        logger.debug(f"Backtest trade for {symbol}: {order['side']} at ${order['entry_price']}, P/L ${pl:.2f}")
                        del open_orders[idx]
                    elif (order["side"] == "buy" and next_price <= order["entry_price"] * (1 - COST_AVERAGE_DIP)) or (
                            order["side"] == "sell" and next_price >= order["entry_price"] * (1 + COST_AVERAGE_DIP)):
                        new_size = order["size"] * 0.5
                        open_orders[i + 0.5] = {
                            "entry_price": next_price,
                            "size": new_size,
                            "leverage": order["leverage"] * 0.5,
                            "side": order["side"],
                            "trailing_stop": next_price * (0.99 if order["side"] == "buy" else 1.01),
                            "entry_time": df["timestamp"].iloc[i+1]
                        }
            
            logger.info(f"Backtest result for {symbol}: Final balance ${simulated_balance:.2f}")
        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}")

    async def process_trade(self, symbol: str, price: float, side: str, price_change: float, confidence: float, patterns: dict, timeframes: list):
        if not self.account_balance:
            self.account_balance = self.get_account_balance()
            if not self.account_balance:
                logger.error("Failed to get account balance")
                return
            if not self.initial_balance:
                self.initial_balance = self.account_balance
        
        if self.account_balance < self.initial_balance * (1 - MAX_DRAWDOWN):
            logger.error(f"Max drawdown reached for {symbol}: {self.account_balance:.2f}/{self.initial_balance:.2f}")
            return
        
        allocations = self.calculate_portfolio_allocation()
        risk_amount = self.account_balance * allocations.get(symbol, RISK_PER_TRADE / len(SYMBOLS))
        logger.debug(f"Dynamic risk for {symbol}: ${risk_amount:.2f}")
        
        order_id = self.place_order(symbol, price, risk_amount, side, confidence, patterns, price_change)
        if order_id:
            logger.info(f"Trade executed for {symbol}: {side} ${risk_amount:.2f} at ${price}")
            self.account_balance -= risk_amount * 0.01

    async def ws_connect(self, max_retries: int = 10):
        retry_count = 0
        while retry_count < max_retries:
            logger.debug(f"WebSocket connection attempt to {WS_URL} at {datetime.now(LOCAL_TZ)}")
            try:
                async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                    logger.info("WebSocket connected")
                    for symbol in SYMBOLS:
                        await ws.send(json.dumps({
                            "op": "subscribe",
                            "args": [{"channel": "tickers", "instId": symbol}]
                        }))
                        logger.info(f"Subscribed to {symbol} ticker")
                        self.load_candles(symbol)
                        for tf in TIMEFRAMES:
                            candles = self.get_candles(symbol, timeframe=tf)
                            if candles:
                                self.candle_history[symbol] = candles
                                self.save_candles(symbol)
                                logger.info(f"Fetched {len(candles)} initial {tf} candles for {symbol}")
                    
                    last_order_time = {symbol: 0 for symbol in SYMBOLS}
                    last_symbol_update = 0
                    order_interval = 60
                    last_price = {symbol: None for symbol in SYMBOLS}
                    price_change_threshold = VOLATILITY_THRESHOLD
                    
                    while True:
                        try:
                            if time.time() - last_symbol_update > 3600:
                                global SYMBOLS
                                new_symbols = self.select_top_symbols()
                                for symbol in new_symbols:
                                    if symbol not in SYMBOLS:
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
                                SYMBOLS = new_symbols
                                last_symbol_update = time.time()
                            
                            data = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                            logger.info(f"WebSocket data: {json.dumps(data, indent=2)}")
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
                                
                                candles = self.get_candles(symbol, limit=1)
                                if candles:
                                    self.candle_history[symbol].append(candles[0])
                                    if len(self.candle_history[symbol]) > 100:
                                        self.candle_history[symbol] = self.candle_history[symbol][-100:]
                                    self.save_candles(symbol)
                                
                                self.manage_cost_averaging(symbol, price)
                                self.manage_trailing_stop(symbol, price)
                                
                                current_time = time.time()
                                if last_price[symbol] is None:
                                    last_price[symbol] = price
                                    continue
                                
                                price_change = abs(price - last_price[symbol]) / last_price[symbol]
                                if current_time - last_order_time[symbol] >= order_interval and price_change >= price_change_threshold:
                                    signal_info = self.generate_signal(symbol, price)
                                    if signal_info:
                                        signal, confidence, patterns, timeframes = signal_info
                                        logger.info(f"Signal for {symbol}: {signal} with confidence {confidence:.2f}")
                                        await self.process_trade(symbol, price, signal, price_change, confidence, patterns, timeframes)
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
            if not self.validate_credentials():
                logger.error("Health check failed: Credential validation")
                return False
            balance = self.get_account_balance()
            if balance is None:
                logger.error("Health check failed: Could not retrieve balance")
                return False
            logger.info("Health check passed: API reachable, balance retrieved")
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

        self.initialize_db()
        global SYMBOLS
        SYMBOLS = self.select_top_symbols()
        for symbol in SYMBOLS:
            self.candle_history[symbol] = []
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
            self.last_candle_fetch[symbol] = {tf: 0 for tf in TIMEFRAMES}
            self.is_model_trained[symbol] = False
            self.last_model_train[symbol] = 0
            self.leverage_info[symbol] = None
            self.open_orders[symbol] = {}
            self.sentiment_cache[symbol] = {"score": 0.0, "timestamp": 0}
        
        if not self.validate_credentials():
            logger.error("Credential validation failed, exiting")
            return
        
        self.account_balance = self.get_account_balance()
        if not self.account_balance:
            logger.error("Failed to get initial account balance, using default")
            self.account_balance = DEFAULT_BALANCE
        self.initial_balance = self.account_balance
        
        for symbol in SYMBOLS:
            price_volume = self.get_price(symbol)
            if not price_volume:
                logger.error(f"Failed to get initial price for {symbol}")
                continue
            price, volume = price_volume
            logger.info(f"Current price for {symbol}: ${price}")
            self.price_history[symbol].append(price)
            self.volume_history[symbol].append(volume)
            self.load_candles(symbol)
            for tf in TIMEFRAMES:
                candles = self.get_candles(symbol, timeframe=tf)
                if candles:
                    self.candle_history[symbol] = candles
                    self.save_candles(symbol)
                    logger.info(f"Fetched {len(candles)} initial {tf} candles for {symbol}")
            await self.backtest_strategy(symbol)
            signal_info = self.generate_signal(symbol, price)
            if signal_info:
                signal, confidence, patterns, timeframes = signal_info
                price_change = abs(price - self.price_history[symbol][-2]) / self.price_history[symbol][-2] if len(self.price_history[symbol]) >= 2 else VOLATILITY_THRESHOLD
                logger.info(f"Initial signal for {symbol}: {signal} with confidence {confidence:.2f}")
                await self.process_trade(symbol, price, signal, price_change, confidence, patterns, timeframes)
        
        await self.ws_connect()

if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.main())