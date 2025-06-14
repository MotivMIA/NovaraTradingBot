
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
from datetime import datetime
import pytz
from dotenv import load_dotenv
from uuid import uuid4

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s,%f - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Environment toggle: True for demo, False for live
DEMO_MODE = os.getenv("DEMO_MODE", "True").lower() == "true"

# API configuration
BASE_URL = "https://demo-trading-openapi.blofin.com" if DEMO_MODE else "https://openapi.blofin.com"
WS_URL = "wss://demo-trading-openapi.blofin.com/ws/public" if DEMO_MODE else "wss://openapi.blofin.com/ws/public"
SYMBOL = "BTC-USDT"
INITIAL_BID_USD = 100.0  # Initial bid amount in USD
PHI = 1.618  # Golden ratio for bid progression
SIZE_PRECISION = 8  # Decimal places for size

# Timezone configuration
LOCAL_TZ = pytz.timezone("America/Los_Angeles")  # PDT

# Credentials
API_KEY = os.getenv("DEMO_API_KEY" if DEMO_MODE else "API_KEY")
API_SECRET = os.getenv("DEMO_API_SECRET" if DEMO_MODE else "API_SECRET")
API_PASSPHRASE = os.getenv("DEMO_API_PASSPHRASE" if DEMO_MODE else "API_PASSPHRASE")

if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
    logger.error("Missing API credentials in .env")
    exit(1)

# Log credentials (redacted for security)
logger.debug(f"API_KEY: {API_KEY[:4]}...{API_KEY[-4:]}")
logger.debug(f"API_PASSPHRASE: {API_PASSPHRASE[:4]}...{API_PASSPHRASE[-4:]}")

def sign_request(secret: str, method: str, path: str, body: dict | None = None) -> tuple[dict, str, str]:
    """Generate BloFin API request signature."""
    local_time = datetime.now(LOCAL_TZ)
    utc_time = local_time.astimezone(pytz.UTC)
    timestamp_ms = int(utc_time.timestamp() * 1000)
    system_time_ms = int(time.time() * 1000)
    if abs(timestamp_ms - system_time_ms) > 30000:
        logger.warning(f"Timestamp offset too large: {timestamp_ms} ms vs system {system_time_ms} ms")
    timestamp = str(timestamp_ms)
    nonce = str(uuid4())
    msg = f"{path}{method.upper()}{timestamp}{nonce}"
    if body:
        msg += json.dumps(body, separators=(',', ':'), sort_keys=True)
    
    secret = secret.strip()
    logger.debug(f"Local time (PDT): {local_time.strftime('%Y-%m-%d %H:%M:%S,%f %Z')}")
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

async def sign_websocket_login(secret: str, api_key: str, passphrase: str) -> tuple[str, str, str]:
    """Generate WebSocket login signature."""
    local_time = datetime.now(LOCAL_TZ)
    utc_time = local_time.astimezone(pytz.UTC)
    timestamp = str(int(utc_time.timestamp() * 1000))
    nonce = timestamp
    method = "GET"
    path = "/users/self/verify"
    msg = f"{path}{method}{timestamp}{nonce}"
    
    secret = secret.strip()
    signature = hmac.new(
        secret.encode('utf-8'),
        msg.encode('utf-8'),
        hashlib.sha256
    ).digest()
    signature = base64.b64encode(signature).decode('utf-8').strip()
    logger.debug(f"WebSocket signature: {signature}")
    return signature, timestamp, nonce

def get_instrument_info():
    """Get instrument details for SYMBOL."""
    path = "/api/v1/market/instruments"
    for attempt in range(3):
        try:
            response = requests.get(f"{BASE_URL}{path}?instType=SWAP&instId={SYMBOL}", timeout=5)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Instrument info response: {json.dumps(data, indent=2)}")
            if data.get("code") == "0" and data.get("data"):
                for inst in data["data"]:
                    if inst["instId"] == SYMBOL:
                        return {
                            "minSize": float(inst.get("minSize", 0.1)),
                            "lotSize": float(inst.get("lotSize", 0.1)),
                            "tickSize": float(inst.get("tickSize", 0.1)),
                            "contractValue": float(inst.get("contractValue", 0.001))
                        }
            logger.error(f"No instrument info for {SYMBOL}")
            return None
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Failed to get instrument info after {attempt + 1} attempts: {e}")
                return None
    return None

def get_price():
    """Get current market price via REST with retries."""
    path = "/api/v1/market/tickers"
    for attempt in range(3):
        try:
            response = requests.get(f"{BASE_URL}{path}?instId={SYMBOL}", timeout=5)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Raw ticker response: {json.dumps(data, indent=2)}")
            if data.get("code") == "0" and data.get("data") and "last" in data["data"][0]:
                return float(data["data"][0]["last"])
            else:
                logger.error(f"Unexpected response structure: {data}")
                return None
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Failed to get price after {attempt + 1} attempts: {e}")
                return None
    return None

def place_order(price: float, size: float, side: str = "buy", max_retries: int = 3):
    """Place a limit order for SWAP contract with retries."""
    path = "/api/v1/trade/order"
    inst_info = get_instrument_info()
    if not inst_info:
        logger.error("Failed to get instrument info, cannot place order")
        return None
    
    min_size = inst_info["minSize"]
    lot_size = inst_info["lotSize"]
    contract_value = inst_info["contractValue"]
    
    size = (size / price) / contract_value
    size = max(round(size / lot_size) * lot_size, min_size)
    logger.info(f"Calculated order size: {size} contracts at ${price} (minSize: {min_size}, lotSize: {lot_size}, contractValue: {contract_value})")
    
    order_request = {
        "instId": SYMBOL,
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
            headers, _, _ = sign_request(API_SECRET, "POST", path, order_request)
            response = requests.post(f"{BASE_URL}{path}", headers=headers, json=order_request, timeout=5)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Order response: {data}")
            if data.get("code") == "0" and data.get("data"):
                order_id = data["data"][0]["orderId"]
                logger.info(f"Placed {side} order: {size} contracts at ${price}")
                return order_id
            else:
                logger.error(f"Order placement failed: {data}")
                if data.get("code") == "152406":
                    logger.error("IP whitelisting issue detected, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                return None
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Failed to place order after {max_retries} attempts: {e}")
                return None
    return None

async def ws_connect(max_retries: int = 3):
    """Connect to WebSocket and subscribe to tickers channel with reconnection."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            async with websockets.connect(WS_URL) as ws:
                await ws.send(json.dumps({
                    "op": "subscribe",
                    "args": [{"channel": "tickers", "instId": SYMBOL}]
                }))
                logger.info(f"Subscribed to {SYMBOL} ticker")
                last_order_time = 0
                order_interval = 60
                last_price = None
                price_change_threshold = 0.01
                
                while True:
                    try:
                        data = json.loads(await ws.recv())
                        logger.info(f"WebSocket data: {json.dumps(data, indent=2)}")
                        if "data" in data and data["data"]:
                            price = float(data["data"][0]["last"])
                            logger.info(f"WebSocket price: ${price}")
                            current_time = time.time()
                            if last_price is None:
                                last_price = price
                                continue
                            price_change = abs(price - last_price) / last_price
                            if current_time - last_order_time >= order_interval and price_change >= price_change_threshold:
                                await process_bid(price)
                                last_order_time = current_time
                                last_price = price
                        await asyncio.sleep(0.1)
                    except (websockets.exceptions.ConnectionClosed, json.JSONDecodeError) as e:
                        logger.error(f"WebSocket error: {e}")
                        break
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Retrying WebSocket connection (attempt {retry_count + 1}/{max_retries})")
                await asyncio.sleep(2 ** retry_count)
            else:
                logger.error(f"Failed to connect to WebSocket after {max_retries} attempts")
                break

async def process_bid(price: float):
    """Process phi-based bidding strategy."""
    bid_amount_usd = INITIAL_BID_USD
    order_id = place_order(price, bid_amount_usd)
    if order_id:
        logger.info(f"Initial bid: ${bid_amount_usd:.2f} at ${price}")
        next_bid_usd = bid_amount_usd * PHI
        logger.info(f"Next phi-based bid: ${next_bid_usd:.2f} at ${price}")

async def main():
    """Main trading loop."""
    price = get_price()
    if not price:
        logger.error("Failed to get initial price, exiting.")
        return
    
    logger.info(f"Current price: ${price}")
    await process_bid(price)
    await ws_connect()

if __name__ == "__main__":
    asyncio.run(main())
