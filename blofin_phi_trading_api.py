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
from datetime import datetime
import pytz
from dotenv import load_dotenv
from uuid import uuid4
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
SYMBOLS = os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,XRP-USDT").split(",")
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
MAX_LEVERAGE = 10.0  # Global cap, overridden for high-confidence signals

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
        self.candle_history = {symbol: [] for symbol in SYMBOLS}
        self.price_history = {symbol: [] for symbol in SYMBOLS}
        self.volume_history = {symbol: [] for symbol in SYMBOLS}
        self.last_candle_fetch = {symbol: 0 for symbol in SYMBOLS}
        self.account_balance = None
        self.initial_balance = None
        self.ml_model = LogisticRegression()
        self.scaler = StandardScaler()
        self.is_model_trained = {symbol: False for symbol in SYMBOLS}
        self.last_model_train = {symbol: 0 for symbol in SYMBOLS}
        self.leverage_info = {symbol: None for symbol in SYMBOLS}

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

    def sign_request(self, method: str, path: str, body: dict | None = None, params: dict | None = None) -> tuple[dict, str, str]:
        local_time = datetime.now(LOCAL_TZ)
        utc_time = local_time.astimezone(pytz.UTC)
        timestamp_ms = int(utc_time.timestamp() * 1000)
        system_time_ms = int(time.time() * 1000)
        if abs(timestamp_ms - system_time_ms) > 30000:
            logger.warning(f"Timestamp offset: {timestamp_ms} ms vs system {system_time_ms} ms")
        timestamp = str(timestamp_ms)
        nonce = str(uuid4())
        
        path_with_params = path
        if params:
            query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            path_with_params += f"?{query}"
        
        body_str = json.dumps(body, separators=(',', ':'), sort_keys=True) if body else ""
        content = path_with_params + method.upper() + timestamp + nonce + body_str
        
        logger.debug(f"Local time: {local_time.strftime('%Y-%m-%d %H:%M:%S,%f %Z')}")
        logger.debug(f"UTC time: {utc_time.strftime('%Y-%m-%d %H:%M:%S,%f %Z')}")
        logger.debug(f"Timestamp (ms): {timestamp_ms}")
        logger.debug(f"Signature message: {content}")
        
        sign_token = hmac.new(
            API_SECRET.encode(),
            content.encode(),
            hashlib.sha256
        ).hexdigest().encode()
        signature = base64.b64encode(sign_token).decode()
        
        logger.debug(f"Generated signature: {signature}")
        
        headers = {
            "access-key": API_KEY.strip(),
            "access-sign": signature,
            "access-timestamp": timestamp,
            "access-nonce": nonce,
            "access-passphrase": API_PASSPHRASE.strip(),
            "content-type": "application/json"
        }
        logger.debug(f"Request headers: {headers}")
        return headers, timestamp, nonce

    def validate_credentials(self) -> bool:
        path = "/api/v1/market/tickers"
        try:
            response = requests.get(f"{BASE_URL}{path}?instId={SYMBOLS[0]}", timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("code") == "0":
                logger.info("Public API request successful, credentials likely valid")
                return True
            logger.error(f"Credential validation failed: {data}")
            return False
        except requests.RequestException as e:
            logger.error(f"Credential validation error: {e}")
            return False

    def get_candles(self, symbol: str, limit: int = CANDLE_LIMIT) -> list | None:
        current_time = time.time()
        if current_time - self.last_candle_fetch[symbol] < CANDLE_FETCH_INTERVAL:
            logger.debug(f"Skipping candle fetch for {symbol}: within {CANDLE_FETCH_INTERVAL}s interval")
            return None
        path = "/api/v1/market/candles"
        params = {"instId": symbol, "bar": CANDLE_TIMEFRAME, "limit": str(limit)}
        for attempt in range(3):
            try:
                response = requests.get(f"{BASE_URL}{path}", params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Candle response for {symbol}: {json.dumps(data, indent=2)}")
                if data.get("code") == "0" and data.get("data"):
                    self.last_candle_fetch[symbol] = current_time
                    candles = [{
                        "timestamp": int(candle[0]),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5])
                    } for candle in data["data"]]
                    return candles
                logger.error(f"No candle data for {symbol}")
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

    def calculate_margin(self, symbol: str, size_usd: float, confidence: float, volatility: float, patterns: dict) -> float:
        max_leverage = self.leverage_info.get(symbol) or self.get_max_leverage(symbol)
        risk_amount = size_usd
        
        leverage = 1.0
        leverage_percentage = 0.0
        if confidence > 0.7 or patterns.get("bullish_engulfing") or patterns.get("bearish_engulfing") or patterns.get("bullish_pin") or patterns.get("bearish_pin"):
            leverage_percentage = 0.5 + (confidence - 0.7) * 1.5  # 50% at 0.7, 80% at 0.9
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

    def detect_price_action_patterns(self, symbol: str) -> dict | None:
        candles = self.candle_history.get(symbol, [])
        if len(candles) < 3:
            logger.warning(f"Insufficient candle data for {symbol}: {len(candles)} candles")
            return None
        
        patterns = {}
        current = candles[-1]
        previous = candles[-2]
        prev_prev = candles[-3] if len(candles) >= 3 else None
        
        current_body = abs(current["close"] - current["open"])
        previous_body = abs(previous["close"] - previous["open"])
        if current_body > previous_body:
            if (current["close"] > current["open"] and previous["close"] < previous["open"] and
                    current["open"] <= previous["close"] and current["close"] >= previous["open"]):
                patterns["bullish_engulfing"] = True
            elif (current["close"] < current["open"] and previous["close"] > previous["open"] and
                    current["open"] >= previous["close"] and current["close"] <= previous["open"]):
                patterns["bearish_engulfing"] = True
        
        total_range = current["high"] - current["low"]
        upper_wick = current["high"] - max(current["open"], current["close"])
        lower_wick = min(current["open"], current["close"]) - current["low"]
        if total_range > 0 and current_body / total_range < 0.3:
            if upper_wick > 2 * current_body:
                patterns["bearish_pin"] = True
            elif lower_wick > 2 * current_body:
                patterns["bullish_pin"] = True
        
        if current_body / total_range < 0.05:
            patterns["doji"] = True
        
        if prev_prev and current["high"] <= previous["high"] and current["low"] >= previous["low"]:
            patterns["inside_bar"] = True
        
        if patterns:
            logger.debug(f"Price action patterns for {symbol}: {patterns}")
            return patterns
        return None

    def calculate_vwap(self, symbol: str) -> float | None:
        candles = self.candle_history.get(symbol, [])
        if len(candles) < VWAP_PERIOD:
            logger.warning(f"Insufficient candle data for VWAP calculation for {symbol}: {len(candles)} candles")
            return None
        
        df = pd.DataFrame(candles[-VWAP_PERIOD:])
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["pv"] = df["typical_price"] * df["volume"]
        vwap = df["pv"].sum() / df["volume"].sum()
        logger.debug(f"VWAP for {symbol}: ${vwap:.2f}")
        return vwap

    def train_ml_model(self, symbol: str):
        current_time = time.time()
        if current_time - self.last_model_train[symbol] < 3600:
            return
        
        prices = self.price_history.get(symbol, [])
        if len(prices) < ML_LOOKBACK + RSI_PERIOD + MACD_SLOW + MACD_SIGNAL:
            logger.warning(f"Insufficient data for {symbol}: {len(prices)} points")
            return
        
        df = pd.DataFrame(prices, columns=["price"])
        df["rsi"] = RSIIndicator(df["price"], window=RSI_PERIOD).rsi()
        macd = MACD(df["price"], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["ema_fast"] = EMAIndicator(df["price"], window=EMA_FAST).ema_indicator()
        df["ema_slow"] = EMAIndicator(df["price"], window=EMA_SLOW).ema_indicator()
        bb = BollingerBands(df["price"], window=BB_PERIOD, window_dev=BB_STD)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        
        df["price_lag1"] = df["price"].shift(1)
        df["price_lag2"] = df["price"].shift(2)
        df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
        
        df = df.dropna()
        if len(df) < 10:
            logger.warning(f"Insufficient valid data for {symbol}: {len(df)} rows")
            return
        
        features = ["rsi", "macd", "macd_signal", "ema_fast", "ema_slow", "bb_upper", "bb_lower", "price_lag1", "price_lag2"]
        X = df[features].values
        y = df["target"].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.fit(X_scaled, y)
        self.is_model_trained[symbol] = True
        self.last_model_train[symbol] = current_time
        logger.info(f"ML model trained for {symbol} with {len(df)} data points")

    def calculate_indicators(self, symbol: str) -> dict | None:
        candles = self.candle_history.get(symbol, [])
        if len(candles) < MIN_PRICE_POINTS:
            logger.warning(f"Insufficient candle data for {symbol}: {len(candles)} candles")
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
        
        logger.debug(f"{symbol} indicators - VWAP: {indicators.get('vwap', 'N/A')}, RSI: {indicators.get('rsi', 'N/A')}")
        return indicators

    def predict_ml_signal(self, symbol: str, indicators: dict) -> tuple[str, float] | None:
        if not self.is_model_trained.get(symbol, False):
            self.train_ml_model(symbol)
            if not self.is_model_trained[symbol]:
                return None
        
        features = [
            indicators.get("rsi", 50.0),
            indicators.get("macd", 0.0),
            indicators.get("macd_signal", 0.0),
            indicators.get("ema_fast", indicators["price"]),
            indicators.get("ema_slow", indicators["price"]),
            indicators.get("bb_upper", indicators["price"] * 1.02),
            indicators.get("bb_lower", indicators["price"] * 0.98),
            indicators.get("price_lag1", indicators["price"]),
            indicators.get("price_lag2", indicators["price"])
        ]
        
        if any(np.isnan(f) for f in features):
            logger.warning(f"Invalid features for {symbol}: {features}")
            return None
        
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        prediction = self.ml_model.predict_proba(X_scaled)[0]
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

    def generate_signal(self, symbol: str, current_price: float) -> tuple[str, float, dict] | None:
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
        
        signal_info = self.evaluate_signal(symbol, signal, confidence)
        if signal_info:
            return signal_info[0], signal_info[1], patterns or {}
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
        
        margin_amount, leverage = self.calculate_margin(symbol, size_usd, confidence, volatility, patterns)
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
                        candles = self.get_candles(symbol)
                        if candles:
                            self.candle_history[symbol] = candles
                            self.save_candles(symbol)
                            logger.info(f"Fetched {len(candles)} initial candles for {symbol}")
                    
                    last_order_time = {symbol: 0 for symbol in SYMBOLS}
                    order_interval = 60
                    last_price = {symbol: None for symbol in SYMBOLS}
                    price_change_threshold = VOLATILITY_THRESHOLD
                    
                    while True:
                        try:
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
                                
                                current_time = time.time()
                                if last_price[symbol] is None:
                                    last_price[symbol] = price
                                    continue
                                
                                price_change = abs(price - last_price[symbol]) / last_price[symbol]
                                if current_time - last_order_time[symbol] >= order_interval and price_change >= price_change_threshold:
                                    signal_info = self.generate_signal(symbol, price)
                                    if signal_info:
                                        signal, confidence, patterns = signal_info
                                        logger.info(f"Signal for {symbol}: {signal} with confidence {confidence:.2f}")
                                        await self.process_trade(symbol, price, signal, price_change, confidence, patterns)
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

    async def process_trade(self, symbol: str, price: float, side: str, price_change: float, confidence: float, patterns: dict):
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
        
        risk_factor = min(2, max(0.5, price_change / VOLATILITY_THRESHOLD))
        risk_amount = self.account_balance * RISK_PER_TRADE * risk_factor
        logger.debug(f"Dynamic risk for {symbol}: {risk_factor:.2f}x, amount: ${risk_amount:.2f}")
        
        order_id = self.place_order(symbol, price, risk_amount, side, confidence, patterns, price_change)
        if order_id:
            logger.info(f"Trade executed for {symbol}: {side} ${risk_amount:.2f} at ${price}")
            self.account_balance -= risk_amount * 0.01

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
            candles = self.get_candles(symbol)
            if candles:
                self.candle_history[symbol] = candles
                self.save_candles(symbol)
                logger.info(f"Fetched {len(candles)} initial candles for {symbol}")
            signal_info = self.generate_signal(symbol, price)
            if signal_info:
                signal, confidence, patterns = signal_info
                price_change = abs(price - self.price_history[symbol][-2]) / self.price_history[symbol][-2] if len(self.price_history[symbol]) >= 2 else VOLATILITY_THRESHOLD
                logger.info(f"Initial signal for {symbol}: {signal} with confidence {confidence:.2f}")
                await self.process_trade(symbol, price, signal, price_change, confidence, patterns)
        
        await self.ws_connect()

if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.main())