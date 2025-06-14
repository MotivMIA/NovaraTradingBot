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
from dotenv import load_dotenv
from uuid import uuid4

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,%f"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Environment toggle: True for demo, False for live
DEMO_MODE = os.getenv("DEMO_MODE", "False").lower() == "true"

# API configuration
BASE_URL = "https://demo-trading-openapi.blofin.com" if DEMO_MODE else "https://openapi.blofin.com"
WS_URL = "wss://demo-trading-openapi.blofin.com/ws/public" if DEMO_MODE else "wss://openapi.blofin.com/ws/public"
SYMBOL = "BTC-USDT"
INITIAL_BID_USD = 100.0  # Initial bid amount in USD
PHI = 1.618  # Golden ratio for bid progression
SIZE_PRECISION = 8  # Decimal places for size

# Credentials
API_KEY = os.getenv("DEMO_API_KEY" if DEMO_MODE else "API_KEY")
API_SECRET = os.getenv("DEMO_API_SECRET" if DEMO_MODE else "API_SECRET")
API_PASSPHRASE = os.getenv("DEMO_API_PASSPHRASE" if DEMO_MODE else "API_PASSPHRASE")

if not all([API_KEY, API_SECRET, API_PASSPHRASE]):
    logger.error("Missing API credentials in .env")
    exit(1)

def sign_request(secret: str, method: str, path: str, body: dict | None = None) -> tuple[dict, str, str]:
    """Generate BloFin API request signature."""
    timestamp = str(int(datetime.now().timestamp() * 1000))
    nonce = str(uuid4())
    msg = f"{path}{method}{timestamp}{nonce}"
    if body:
        msg += json.dumps(body, separators=(',', ':'))
    
    # Ensure secret is stripped of whitespace
    secret = secret.strip()
    logger.debug(f"Signature message: {msg}")
    
    # Generate HMAC-SHA256 signature
    signature = hmac.new(
        secret.encode('utf-8'),
        msg.encode('utf-8'),
        hashlib.sha256
    ).digest()
    
    # Base64 encode the signature
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
    return headers, timestamp, nonce

async def sign_websocket_login(secret: str, api_key: str, passphrase: str) -> tuple[str, str, str]:
    """Generate WebSocket login signature."""
    timestamp = str(int(time.time() * 1000))
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
    return base64.b64encode(signature).decode('utf-8').strip(), timestamp, nonce

def get_instrument_info():
    """Get instrument details for SYMBOL."""
    path = "/api/v1/market/instruments"
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
        logger.error(f"Failed to get instrument info: {e}")
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
            logger.error(f"Response: {response.text if 'response' in locals() else 'No response'}")
            if attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to get price after {attempt + 1} attempts: {e}")
                return None
    return None

def place_order(price: float, size: float, side: str = "buy"):
    """Place a limit order for SWAP contract."""
    path = "/api/v1/trade/order"
    inst_info = get_instrument_info()
    min_size = inst_info["minSize"] if inst_info else 0.1
    lot_size = inst_info["lotSize"] if inst_info else 0.1
    contract_value = inst_info["contractValue"] if inst_info else 0.001
    
    # Convert size from USD to contracts: (USD / price) / contractValue
    size = (size / price) / contract_value
    # Round to lotSize
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
        "size": str(size),
        "leverage": "3"
    }
    
    headers, _, _ = sign_request(API_SECRET, "POST", path, order_request)
    logger.debug(f"Request headers: {headers}")
    try:
        response = requests.post(f"{BASE_URL}{path}", headers=headers, json=order_request, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data.get("code") == "0" and data.get("data"):
            order_id = data["data"][0]["orderId"]
            logger.info(f"Placed {side} order: {size} contracts at ${price}")
            return order_id
        else:
            logger.error(f"Order placement failed: {data}")
            return None
    except requests.RequestException as e:
        logger.error(f"Order placement error: {e}")
        logger.error(f"Response: {response.text if 'response' in locals() else 'No response'}")
        return None

async def ws_connect():
    """Connect to WebSocket and subscribe to tickers channel."""
    try:
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({
                "op": "subscribe",
                "args": [{"channel": "tickers", "instId": SYMBOL}]
            }))
            logger.info(f"Subscribed to {SYMBOL} ticker")
            last_order_time = 0
            order_interval = 60  # Place orders every 60 seconds
            last_price = None  # Initialize as None
            price_change_threshold = 0.01  # 1% price change
            
            while True:
                try:
                    data = json.loads(await ws.recv())
                    logger.info(f"WebSocket data: {json.dumps(data, indent=2)}")
                    if "data" in data and data["data"]:
                        price = float(data["data"][0]["last"])
                        logger.info(f"WebSocket price: ${price}")
                        # Place order only after initial bid, sufficient time, and significant price change
                        current_time = time.time()
                        if last_price is None:
                            last_price = price  # Set initial price
                            continue  # Skip first WebSocket update
                        price_change = abs(price - last_price) / last_price
                        if current_time - last_order_time >= order_interval and price_change >= price_change_threshold:
                            await process_bid(price)
                            last_order_time = current_time
                            last_price = price
                    time.sleep(0.1)  # Rate limit
                except (websockets.exceptions.ConnectionClosed, json.JSONDecodeError) as e:
                    logger.error(f"WebSocket error: {e}")
                    break
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
        logger.error(f"Handshake details: {e.__dict__ if hasattr(e, '__dict__') else str(e)}")

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
    # Place initial bid
    await process_bid(price)
    
    # Start WebSocket for real-time updates
    await ws_connect()

if __name__ == "__main__":
    asyncio.run(main())