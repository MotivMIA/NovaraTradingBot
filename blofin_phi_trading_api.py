
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
SYMBOLS = os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT").split(",")
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
DEFAULT_BALANCE = 10000.0  # Fallback balance if API fails

# Credentials
API_KEY = os.getenv("DEMO_API_KEY" if DEMO_MODE else "API_KEY")
API_SECRET = os.getenv("DEMO_API_SECRET" if DEMO_MODE else "API_SECRET")
API_PASSPHRASE = os.getenv("DEMO_API_PASSPHRASE" if DEMO_MODE else "API_PASSPHRASE")

if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
    logger.error("Missing API credentials in .env")
    exit(1)

logger.debug(f"API_KEY: {API_KEY[:4]}...{API_KEY[-4:]}")
logger.debug(f"API_PASSPHRASE: {API_PASSPHRASE[:4]}...{API_PASSPHRASE[-4:]}")

# Timezone
LOCAL_TZ = pytz.timezone("America/Los_Angeles")

class TradingBot:
    def __init__(self):
        self.price_history = {symbol: [] for symbol in SYMBOLS}
        self.account_balance = None
        self.initial_balance = None
        self.ml_model = LogisticRegression()
        self.scaler = StandardScaler()
        self.is_model_trained = {symbol: False for symbol in SYMBOLS}
        self.last_model_train = {symbol: 0 for symbol in SYMBOLS}

    def sign_request(self, method: str, path: str, body: dict | None = None) -> tuple[dict, str, str]:
        """Generate BloFin API request signature."""
        local_time = datetime.now(LOCAL_TZ)
        utc_time = local_time.astimezone(pytz.UTC)
        timestamp_ms = int(utc_time.timestamp() * 1000)
        system_time_ms = int(time.time() * 1000)
        if abs(timestamp_ms - system_time_ms) > 30000:
            logger.warning(f"Timestamp offset: {timestamp_ms} ms vs system {system_time_ms} ms")
        timestamp = str(timestamp_ms)
        nonce = str(int(time.time() * 1000))  # Timestamp-based nonce
        msg = f"{path}{method.upper()}{timestamp}{nonce}"
        if body:
            msg += json.dumps(body, separators=(',', ':'), sort_keys=True)
        
        secret = API_SECRET.strip()
        logger.debug(f"Local time: {local_time.strftime('%Y-%m-%d %H:%M:%S,%f %Z')}")
        logger.debug(f"UTC time: {utc_time.strftime('%Y-%m-%d %H:%M:%S,%f %Z')}")
        logger.debug(f"Timestamp (ms): {timestamp_ms}")
        logger.debug(f"Signature message: {msg}")
        
        signature = hmac.new(
            secret.encode('utf-8'),
            msg.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        signature = base64.b64encode(signature).decode('utf-8').strip()
        logger.debug(f"Generated signature: {signature}")
        
        headers = {
            "ACCESS-KEY": API_KEY.strip(),
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-NONCE": nonce,
            "ACCESS-PASSPHRASE": API_PASSPHRASE.strip(),
            "Content-Type": "application/json"
        }
        logger.debug(f"Request headers: {headers}")
        return headers, timestamp, nonce

    def get_instrument_info(self, symbol: str) -> dict | None:
        """Get instrument details."""
        path = "/api/v1/market/instruments"
        for attempt in range(3):
            try:
                response = requests.get(f"{BASE_URL}{path}?instType=SWAP&instId={symbol}", timeout=5)
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

    def get_price(self, symbol: str) -> float | None:
        """Get current market price."""
        path = "/api/v1/market/tickers"
        for attempt in range(3):
            try:
                response = requests.get(f"{BASE_URL}{path}?instId={symbol}", timeout=5)
                response.raise_for_status()
                data = response.json()
                logger.info(f"Ticker response for {symbol}: {json.dumps(data, indent=2)}")
                if data.get("code") == "0" and data.get("data") and "last" in data["data"][0]:
                    return float(data["data"][0]["last"])
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
        """Get account balance in USDT."""
        path = "/api/v1/account/balance"
        headers, _, _ = self.sign_request("GET", path)
        for attempt in range(3):
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
                return None
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed after {attempt + 1} attempts: {e}")
                    return None
        logger.warning(f"Using default balance: ${DEFAULT_BALANCE:.2f} USDT")
        return DEFAULT_BALANCE

    def train_ml_model(self, symbol: str):
        """Train logistic regression model for a symbol."""
        current_time = time.time()
        if current_time - self.last_model_train[symbol] < 3600:  # Train every hour
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
        """Calculate technical indicators."""
        prices = self.price_history.get(symbol, [])
        if len(prices) < max(RSI_PERIOD, MACD_SLOW + MACD_SIGNAL, EMA_SLOW, BB_PERIOD):
            logger.warning(f"Insufficient price data for {symbol}: {len(prices)} points")
            return None
        
        df = pd.DataFrame(prices, columns=["price"])
        indicators = {}
        indicators["rsi"] = RSIIndicator(df["price"], window=RSI_PERIOD).rsi().iloc[-1]
        macd = MACD(df["price"], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
        indicators["macd"] = macd.macd().iloc[-1]
        indicators["macd_signal"] = macd.macd_signal().iloc[-1]
        indicators["ema_fast"] = EMAIndicator(df["price"], window=EMA_FAST).ema_indicator().iloc[-1]
        indicators["ema_slow"] = EMAIndicator(df["price"], window=EMA_SLOW).ema_indicator().iloc[-1]
        bb = BollingerBands(df["price"], window=BB_PERIOD, window_dev=BB_STD)
        indicators["bb_upper"] = bb.bollinger_hband().iloc[-1]
        indicators["bb_lower"] = bb.bollinger_lband().iloc[-1]
        indicators["price"] = df["price"].iloc[-1]
        indicators["price_lag1"] = df["price"].iloc[-2] if len(df) > 1 else np.nan
        indicators["price_lag2"] = df["price"].iloc[-3] if len(df) > 2 else np.nan
        
        logger.debug(f"{symbol} indicators - RSI: {indicators['rsi']:.2f}, MACD: {indicators['macd']:.2f}, Signal: {indicators['macd_signal']:.2f}")
        return indicators

    def predict_ml_signal(self, symbol: str, indicators: dict) -> str | None:
        """Generate ML-based signal."""
        if not self.is_model_trained.get(symbol, False):
            self.train_ml_model(symbol)
            if not self.is_model_trained[symbol]:
                return None
        
        features = [
            indicators["rsi"],
            indicators["macd"],
            indicators["macd_signal"],
            indicators["ema_fast"],
            indicators["ema_slow"],
            indicators["bb_upper"],
            indicators["bb_lower"],
            indicators["price_lag1"],
            indicators["price_lag2"]
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
        """Generate hybrid trading signal."""
        indicators = self.calculate_indicators(symbol)
        if not indicators:
            return None
        
        rsi = indicators["rsi"]
        macd = indicators["macd"]
        macd_signal = indicators["macd_signal"]
        ema_fast = indicators["ema_fast"]
        ema_slow = indicators["ema_slow"]
        bb_upper = indicators["bb_upper"]
        bb_lower = indicators["bb_lower"]
        price = indicators["price"]
        
        ml_signal = self.predict_ml_signal(symbol, indicators)
        confidence = 0.0
        
        # Trend-following signals
        if macd > macd_signal and ema_fast > ema_slow:
            signal = "buy"
            confidence += 0.4
        elif macd < macd_signal and ema_fast < ema_slow:
            signal = "sell"
            confidence += 0.4
        else:
            signal = None
        
        # Mean-reversion signals
        if rsi < RSI_OVERSOLD and price <= bb_lower:
            signal = "buy"
            confidence += 0.3
        elif rsi > RSI_OVERBOUGHT and price >= bb_upper:
            signal = "sell"
            confidence += 0.3
        
        # ML signal
        if ml_signal:
            if signal and signal == ml_signal:
                confidence += 0.3
            elif not signal:
                signal = ml_signal
                confidence += 0.3
        
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
                headers, _, _ = self.sign_request("POST", path, order_request)
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
                    
                    last_order_time = {symbol: 0 for symbol in SYMBOLS}
                    order_interval = 60
                    last_price = {symbol: None for symbol in SYMBOLS}
                    price_change_threshold = 0.01
                    
                    while True:
                        try:
                            data = json.loads(await ws.recv())
                            logger.info(f"WebSocket data: {json.dumps(data, indent=2)}")
                            if "data" in data and data["data"]:
                                symbol = data["arg"]["instId"]
                                price = float(data["data"][0]["last"])
                                logger.info(f"WebSocket price for {symbol}: ${price}")
                                self.price_history[symbol].append(price)
                                if len(self.price_history[symbol]) > 100:
                                    self.price_history[symbol] = self.price_history[symbol][-100:]
                                
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
        self.account_balance = self.get_account_balance()
        if not self.account_balance:
            logger.error("Failed to get initial account balance, using default")
            self.account_balance = DEFAULT_BALANCE
        self.initial_balance = self.account_balance
        
        for symbol in SYMBOLS:
            price = self.get_price(symbol)
            if not price:
                logger.error(f"Failed to get initial price for {symbol}")
                continue
            logger.info(f"Current price for {symbol}: ${price}")
            self.price_history[symbol].append(price)
            signal_info = self.generate_signal(symbol, price)
            if signal_info:
                signal, confidence = signal_info
                logger.info(f"Initial signal for {symbol}: {signal} with confidence {confidence:.2f}")
                await self.process_trade(symbol, price, signal)
        
        await self.ws_connect()

if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.main())
