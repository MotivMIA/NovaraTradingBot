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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,%f",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
for handler in logger.handlers:
    handler.setFormatter(MicrosecondFormatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S,%f"))

# Load environment variables
load_dotenv()

# Configuration
DEMO_MODE = os.getenv("DEMO_MODE", "True").lower() == "true"
BASE_URL = "https://demo-trading-openapi.blofin.com" if DEMO_MODE else "https://openapi.blofin.com"
WS_URL = "wss://demo-trading-openapi.blofin.com/ws/public" if DEMO_MODE else "wss://openapi.blofin.com/ws/public"
SYMBOLS = os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,XRP-USDT").split(",")
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))  # 1% risk
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
MAX_DRAWDOWN = 0.10  # 10% max drawdown
DEFAULT_BALANCE = 10000.0  # Fallback balance
MIN_PRICE_POINTS = 5  # Minimum data for initial signals
VWAP_PERIOD = 20  # VWAP calculation period
VOLATILITY_THRESHOLD = 0.02  # Min price change for volatile pairs
CANDLE_TIMEFRAME = "1m"  # 1-minute candles for price action

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
        self.candle_history = {symbol: [] for symbol in SYMBOLS}  # OHLC candles
        self.price_history = {symbol: [] for symbol in SYMBOLS}  # Last prices
        self.volume_history = {symbol: [] for symbol in SYMBOLS}
        self.account_balance = None
        self.initial_balance = None
        self.ml_model = LogisticRegression()
        self.scaler = StandardScaler()
        self.is_model_trained = {symbol: False for symbol in SYMBOLS}
        self.last_model_train = {symbol: 0 for symbol in SYMBOLS}

    def sign_request(self, method: str, path: str, body: dict | None = None, params: dict | None = None) -> tuple[dict, str, str]:
        """Generate BloFin API request signature per api.py."""
        local_time = datetime.now(LOCAL_TZ)
        utc_time = local_time.astimezone(pytz.UTC)
        timestamp_ms = int(utc_time.timestamp() * 1000)
        system_time_ms = int(time.time() * 1000)
        if abs(timestamp_ms - system_time_ms) > 30000:
            logger.warning(f"Timestamp offset: {timestamp_ms} ms vs system {system_time_ms} ms")
        timestamp = str(timestamp_ms)
        nonce = str(uuid4())  # Use UUID per api.py
        
        # Build path with query parameters for GET
        path_with_params = path
        if params:
            query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            path_with_params += f"?{query}"
        
        # Build signature content
        body_str = json.dumps(body, separators=(',', ':'), sort_keys=True) if body else ""
        content = path_with_params + method.upper() + timestamp + nonce + body_str
        
        logger.debug(f"Local time: {local_time.strftime('%Y-%m-%d %H:%M:%S,%f %Z')}")
        logger.debug(f"UTC time: {utc_time.strftime('%Y-%m-%d %H:%M:%S,%f %Z')}")
        logger.debug(f"Timestamp (ms): {timestamp_ms}")
        logger.debug(f"Signature message: {content}")
        
        # Generate signature per api.py
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
        """Validate API credentials with a test request."""
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

    def get_candles(self, symbol: str, limit: int = 100) -> list | None:
        """Fetch candlestick data for price action analysis."""
        path = "/api/v1/market/candles"
        params = {"instId": symbol, "bar": CANDLE_TIMEFRAME, "limit": str(limit)}
        for attempt in range(3):
            try:
                response = requests.get(f"{BASE_URL}{path}", params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Candle response for {symbol}: {json.dumps(data, indent=2)}")
                if data.get("code") == "0" and data.get("data"):
                    return [{
                        "timestamp": int(candle[0]),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5])
                    } for candle in data["data"]]
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
        """Get instrument details."""
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
        """Get current market price and volume."""
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
        """Get account balance in USDT with retry."""
        path = "/api/v1/account/balance"
        for attempt in range(5):
            headers, _, _ = self.sign_request("GET", path)
            try:
                response = requests.get(f"{BASE_URL}{path}", headers=headers, timeout=5)
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Balance response: {data}")
                if data.get("code") == "0" and data.get("data"):
                    for asset in data["data"]:
                        if asset["currency"] == "USDT":
                            balance = float(asset["total"])
                            logger.info(f"Account balance: ${balance:.2f} USDT")
                            return balance
                logger.error(f"Unexpected balance response: {data}")
                if data.get("code") == "152409":
                    logger.error("Signature verification failed, possible credential mismatch")
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

    def detect_price_action_patterns(self, symbol: str) -> dict | None:
        """Detect candlestick patterns for price action."""
        candles = self.candle_history.get(symbol, [])
        if len(candles) < 2:
            logger.warning(f"Insufficient candle data for {symbol}: {len(candles)} candles")
            return None
        
        current = candles[-1]
        previous = candles[-2]
        patterns = {}
        
        # Bullish/Bearish Engulfing
        current_body = abs(current["close"] - current["open"])
        previous_body = abs(previous["close"] - previous["open"])
        if current_body > previous_body:
            if current["close"] > current["open"] and previous["close"] < previous["open"]:
                if current["open"] <= previous["close"] and current["close"] >= previous["open"]:
                    patterns["bullish_engulfing"] = True
            elif current["close"] < current["open"] and previous["close"] > previous["open"]:
                if current["open"] >= previous["close"] and current["close"] <= previous["open"]:
                    patterns["bearish_engulfing"] = True
        
        # Pin Bar
        body = abs(current["close"] - current["open"])
        upper_wick = current["high"] - max(current["open"], current["close"])
        lower_wick = min(current["open"], current["close"]) - current["low"]
        total_range = current["high"] - current["low"]
        if body / total_range < 0.3:
            if upper_wick > 2 * body:
                patterns["bearish_pin"] = True
            elif lower_wick > 2 * body:
                patterns["bullish_pin"] = True
        
        # Doji
        if body / total_range < 0.05:
            patterns["doji"] = True
        
        if patterns:
            logger.debug(f"Price action patterns for {symbol}: {patterns}")
            return patterns
        return None

    def calculate_vwap(self, symbol: str) -> float | None:
        """Calculate VWAP for the last VWAP_PERIOD candles."""
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
        """Train logistic regression model for a symbol."""
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
        """Calculate technical indicators including VWAP."""
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

    def predict_ml_signal(self, symbol: str, indicators: dict) -> str | None:
        """Generate ML-based signal."""
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
        
        return "buy" if prediction[1] > prediction[0] else "sell"

    def generate_signal(self, symbol: str, current_price: float) -> tuple[str, float] | None:
        """Generate VWAP and price action-driven trading signal."""
        indicators = self.calculate_indicators(symbol)
        patterns = self.detect_price_action_patterns(symbol)
        if not indicators or not indicators["price"]:
            return None
        
        vwap = indicators.get("vwap")
        rsi = indicators.get("rsi")
        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        ema_fast = indicators.get("ema_fast")
        ema_slow = indicators.get("ema_slow")
        bb_upper = indicators.get("bb_upper")
        bb_lower = indicators.get("bb_lower")
        price = indicators["price"]
        
        ml_signal = self.predict_ml_signal(symbol, indicators)
        confidence = 0.0
        signal = None
        
        # Volatility check
        if len(self.price_history[symbol]) >= 2:
            price_change = abs(price - self.price_history[symbol][-2]) / self.price_history[symbol][-2]
            if price_change < VOLATILITY_THRESHOLD:
                logger.debug(f"Low volatility for {symbol}: {price_change:.4f}")
                return None
        
        # Price action signals
        if patterns:
            if patterns.get("bullish_engulfing") or patterns.get("bullish_pin"):
                signal = "buy"
                confidence += 0.3
            elif patterns.get("bearish_engulfing") or patterns.get("bearish_pin"):
                signal = "sell"
                confidence += 0.3
            elif patterns.get("doji") and vwap is not None:
                if price > vwap:
                    signal = "buy"
                    confidence += 0.2
                else:
                    signal = "sell"
                    confidence += 0.2
        
        # VWAP-based signals
        if vwap is not None:
            if price > vwap and len(self.candle_history[symbol]) >= MIN_PRICE_POINTS:
                if signal == "buy":
                    confidence += 0.4
                elif not signal:
                    signal = "buy"
                    confidence += 0.4
            elif price < vwap and len(self.candle_history[symbol]) >= MIN_PRICE_POINTS:
                if signal == "sell":
                    confidence += 0.4
                elif not signal:
                    signal = "sell"
                    confidence += 0.4
            elif abs(price - vwap) / vwap < 0.005:
                if price > self.price_history[symbol][-2]:
                    if signal == "buy":
                        confidence += 0.3
                    elif not signal:
                        signal = "buy"
                        confidence += 0.3
                else:
                    if signal == "sell":
                        confidence += 0.3
                    elif not signal:
                        signal = "sell"
                        confidence += 0.3
        
        # Additional indicators
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
        
        if ml_signal:
            if signal and signal == ml_signal:
                confidence += 0.2
            elif not signal:
                signal = ml_signal
                confidence += 0.2
        
        if not signal or confidence < 0.5:
            return None
        
        return signal, confidence

    def place_order(self, symbol: str, price: float, size_usd: float, side: str, max_retries: int = 3) -> str | None:
        """Place a limit order with stop-loss and take-profit."""
        path = "/api/v1/trade/order"
        inst_info = self.get_instrument_info(symbol)
        if not inst_info:
            logger.error(f"Failed to get instrument info for {symbol}")
            return None
        
        min_size = inst_info["minSize"]
        lot_size = inst_info["lotSize"]
        contract_value = inst_info["contractValue"]
        
        size = (size_usd / price) / contract_value
        size = max(round(size / lot_size) * lot_size, min_size)
        logger.info(f"Calculated order size for {symbol}: {size} contracts at ${price}")
        
        stop_loss = price * (0.99 if side == "buy" else 1.01)
        take_profit = price * (1.02 if side == "buy" else 0.98)
        
        order_request = {
            "instId": symbol,
            "instType": "SWAP",
            "marginMode": "cross",
            "positionSide": "net",
            "side": side,
            "orderType": "limit",
            "price": str(round(price, 2)),
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
                    logger.info(f"Placed {side} order for {symbol}: {size} contracts at ${price}, SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}")
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

    async def ws_connect(self, max_retries: int = 3):
        """Connect to WebSocket for multiple symbols."""
        retry_count = 0
        while retry_count < max_retries:
            try:
                async with websockets.connect(WS_URL) as ws:
                    for symbol in SYMBOLS:
                        await ws.send(json.dumps({
                            "op": "subscribe",
                            "args": [{"channel": "tickers", "instId": symbol}]
                        }))
                        logger.info(f"Subscribed to {symbol} ticker")
                        candles = self.get_candles(symbol)
                        if candles:
                            self.candle_history[symbol] = candles
                            logger.info(f"Fetched {len(candles)} initial candles for {symbol}")
                    
                    last_order_time = {symbol: 0 for symbol in SYMBOLS}
                    order_interval = 60
                    last_price = {symbol: None for symbol in SYMBOLS}
                    price_change_threshold = VOLATILITY_THRESHOLD
                    
                    while True:
                        try:
                            data = json.loads(await ws.recv())
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
                                
                                current_time = time.time()
                                if last_price[symbol] is None:
                                    last_price[symbol] = price
                                    continue
                                
                                price_change = abs(price - last_price[symbol]) / last_price[symbol]
                                if current_time - last_order_time[symbol] >= order_interval and price_change >= price_change_threshold:
                                    signal_info = self.generate_signal(symbol, price)
                                    if signal_info:
                                        signal, confidence = signal_info
                                        logger.info(f"Signal for {symbol}: {signal} with confidence {confidence:.2f}")
                                        await self.process_trade(symbol, price, signal)
                                        last_order_time[symbol] = current_time
                                        last_price[symbol] = price
                            await asyncio.sleep(0.2)
                        except (websockets.exceptions.ConnectionClosed, json.JSONDecodeError) as e:
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

    async def process_trade(self, symbol: str, price: float, side: str):
        """Process trade with risk management."""
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
        
        risk_amount = self.account_balance * RISK_PER_TRADE
        order_id = self.place_order(symbol, price, risk_amount, side)
        if order_id:
            logger.info(f"Trade executed for {symbol}: {side} ${risk_amount:.2f} at ${price}")
            self.account_balance -= risk_amount * 0.01  # Approximate fee impact

    async def main(self):
        """Main trading loop."""
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
            candles = self.get_candles(symbol)
            if candles:
                self.candle_history[symbol] = candles
                logger.info(f"Fetched {len(candles)} initial candles for {symbol}")
            signal_info = self.generate_signal(symbol, price)
            if signal_info:
                signal, confidence = signal_info
                logger.info(f"Initial signal for {symbol}: {signal} with confidence {confidence:.2f}")
                await self.process_trade(symbol, price, signal)
        
        await self.ws_connect()

if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.main())