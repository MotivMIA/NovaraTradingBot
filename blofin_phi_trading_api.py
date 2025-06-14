
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
from ta.trend import MACD

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s,%f - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
DEMO_MODE = os.getenv("DEMO_MODE", "True").lower() == "true"
BASE_URL = "https://demo-trading-openapi.blofin.com" if DEMO_MODE else "https://openapi.blofin.com"
WS_URL = "wss://demo-trading-openapi.blofin.com/ws/public" if DEMO_MODE else "wss://openapi.blofin.com/ws/public"
SYMBOLS = os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT").split(",")  # Multiple trading pairs
INITIAL_BID_USD = float(os.getenv("INITIAL_BID_USD", 100.0))
PHI = 1.618  # Golden ratio
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))  # 1% risk per trade
SIZE_PRECISION = 8
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

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
LOCAL_TZ = pytz.timezone("America/Los_Angeles")  # PDT

class TradingBot:
    def __init__(self):
        self.price_history = {symbol: [] for symbol in SYMBOLS}
        self.account_balance = None

    def sign_request(self, method: str, path: str, body: dict | None = None) -> tuple[dict, str, str]:
        """Generate BloFin API request signature."""
        local_time = datetime.now(LOCAL_TZ)
        utc_time = local_time.astimezone(pytz.UTC)
        timestamp_ms = int(utc_time.timestamp() * 1000)
        system_time_ms = int(time.time() * 1000)
        if abs(timestamp_ms - system_time_ms) > 30000:
            logger.warning(f"Timestamp offset: {timestamp_ms} ms vs system {system_time_ms} ms")
        timestamp = str(timestamp_ms)
        nonce = str(uuid4())
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
        """Get instrument details for a symbol."""
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
        """Get current market price for a symbol."""
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
                            return float(asset["total"])
                logger.error(f"Unexpected balance response: {data}")
                return None
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed after {attempt + 1} attempts: {e}")
                    return None
        return None

    def calculate_indicators(self, symbol: str) -> tuple[float, float, float] | None:
        """Calculate RSI and MACD for a symbol."""
        prices = self.price_history.get(symbol, [])
        if len(prices) < max(RSI_PERIOD, MACD_SLOW + MACD_SIGNAL):
            logger.warning(f"Insufficient price data for {symbol}: {len(prices)} points")
            return None
        
        df = pd.DataFrame(prices, columns=["price"])
        rsi = RSIIndicator(df["price"], window=RSI_PERIOD).rsi().iloc[-1]
        macd = MACD(df["price"], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
        macd_value = macd.macd().iloc[-1]
        signal_value = macd.macd_signal().iloc[-1]
        
        logger.debug(f"{symbol} - RSI: {rsi:.2f}, MACD: {macd_value:.2f}, Signal: {signal_value:.2f}")
        return rsi, macd_value, signal_value

    def generate_signal(self, symbol: str, current_price: float) -> str | None:
        """Generate buy/sell signal based on RSI and MACD."""
        indicators = self.calculate_indicators(symbol)
        if not indicators:
            return None
        
        rsi, macd_value, signal_value = indicators
        if rsi < RSI_OVERSOLD and macd_value > signal_value:
            return "buy"
        elif rsi > RSI_OVERBOUGHT and macd_value < signal_value:
            return "sell"
        return None

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
        
        stop_loss = price * (0.99 if side == "buy" else 1.01)  # 1% stop-loss
        take_profit = price * (1.02 if side == "buy" else 0.98)  # 2% take-profit
        
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
        """Connect to WebSocket for multiple symbols with reconnection."""
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
                                    signal = self.generate_signal(symbol, price)
                                    if signal:
                                        await self.process_bid(symbol, price, signal)
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

    async def process_bid(self, symbol: str, price: float, side: str):
        """Process phi-based bidding strategy with risk management."""
        if not self.account_balance:
            self.account_balance = self.get_account_balance()
            if not self.account_balance:
                logger.error("Failed to get account balance")
                return
        
        risk_amount = self.account_balance * RISK_PER_TRADE
        bid_amount_usd = min(INITIAL_BID_USD, risk_amount)
        order_id = self.place_order(symbol, price, bid_amount_usd, side)
        if order_id:
            logger.info(f"Initial {side} bid for {symbol}: ${bid_amount_usd:.2f} at ${price}")
            next_bid_usd = bid_amount_usd * PHI
            logger.info(f"Next phi-based bid for {symbol}: ${next_bid_usd:.2f} at ${price}")

    async def main(self):
        """Main trading loop."""
        self.account_balance = self.get_account_balance()
        if not self.account_balance:
            logger.error("Failed to get initial account balance")
            return
        
        for symbol in SYMBOLS:
            price = self.get_price(symbol)
            if not price:
                logger.error(f"Failed to get initial price for {symbol}")
                continue
            logger.info(f"Current price for {symbol}: ${price}")
            self.price_history[symbol].append(price)
            signal = self.generate_signal(symbol, price)
            if signal:
                await self.process_bid(symbol, price, signal)
        
        await self.ws_connect()

if __name__ == "__main__":
    bot = TradingBot()
    asyncio.run(bot.main())
