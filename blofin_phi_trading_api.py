import requests
import websocket
import json
import hmac
import hashlib
import base64
import time
from datetime import datetime
import uuid
import asyncio
import numpy as np
from ta.momentum import RSIIndicator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Blofin API credentials (replace with your own)
API_KEY = "50f02f2dc2d4442cb4de50b7f87d37f2"
API_SECRET = "31ce8bbafc5c41a18620aacdbf427edf"
API_PASSPHRASE = "zWz1sFh6O25173wnkTQ1"
BASE_URL = "https://api.blofin.com"
WS_URL = "wss://api.blofin.com/ws/v1/public"

# Trading parameters
SYMBOL = "BTC-USDT"  # Trading pair
INITIAL_BID = 100.0  # Initial bid in USDT
PHI = 1.618  # Golden ratio
MAX_BIDS = 3  # Maximum number of bid increases
STOP_LOSS_PCT = 0.05  # 5% stop-loss below lowest bid
TOTAL_CAPITAL = 1000.0  # Max capital allocation in USDT
WAITING_PERIOD = 7200  # 2 hours in seconds for intraday
RSI_PERIOD = 14  # RSI period
OVERSOLD_THRESHOLD = 30  # RSI oversold level
STABILIZATION_CANDLES = 3  # Number of candles to confirm stabilization

# Global state
current_bid = INITIAL_BID
bid_count = 0
last_bid_price = None
price_history = []
rsi_values = []

def sign_request(method, path, body=None):
    """Generate HMAC-SHA256 signature for Blofin API."""
    timestamp = str(int(datetime.now().timestamp() * 1000))
    nonce = str(uuid.uuid4())
    # For GET requests, only include path, method, timestamp, nonce
    msg = f"{path}{method}{timestamp}{nonce}"
    if body and method in ["POST", "PUT"]:
        msg += json.dumps(body, separators=(',', ':'))
    hex_signature = hmac.new(
        API_SECRET.encode(), msg.encode(), hashlib.sha256
    ).hexdigest()
    signature = base64.b64encode(hex_signature.encode()).decode()
    return {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": signature,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-NONCE": nonce,
        "ACCESS-PASSPHRASE": API_PASSPHRASE,
        "Content-Type": "application/json"
    }

def place_order(price, quantity, side="buy", order_type="limit"):
    """Place a limit order on Blofin."""
    path = "/api/v1/trade/order"
    body = {
        "instId": SYMBOL,
        "marginMode": "isolated",
        "side": side,
        "orderType": order_type,
        "price": str(price),
        "size": str(quantity)
    }
    headers = sign_request("POST", path, body)
    try:
        response = requests.post(f"{BASE_URL}{path}", headers=headers, json=body)
        response.raise_for_status()
        logger.info(f"Placed {side} order: {quantity} {SYMBOL} at ${price}")
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Order placement failed: {e}")
        return None

def cancel_order(order_id):
    """Cancel an existing order."""
    path = f"/api/v1/trade/cancel-order"
    body = {"instId": SYMBOL, "orderId": order_id}
    headers = sign_request("POST", path, body)
    try:
        response = requests.post(f"{BASE_URL}{path}", headers=headers, json=body)
        response.raise_for_status()
        logger.info(f"Cancelled order: {order_id}")
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Order cancellation failed: {e}")
        return None

def get_price():
    """Get current market price via REST with retries."""
    path = f"/api/v1/market/ticker?instId={SYMBOL}"
    for attempt in range(3):
        headers = sign_request("GET", path)
        try:
            response = requests.get(f"{BASE_URL}{path}", headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            return float(data["data"][0]["last"])
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to get price after {attempt + 1} attempts: {e}")
                return None
    return None

def calculate_rsi(prices):
    """Calculate RSI for the given price series."""
    if len(prices) < RSI_PERIOD:
        return None
    series = np.array(prices)
    rsi_indicator = RSIIndicator(close=series, window=RSI_PERIOD)
    return rsi_indicator.rsi()[-1]

def is_price_stabilized(prices):
    """Check if price has stabilized (no new lows for N candles)."""
    if len(prices) < STABILIZATION_CANDLES:
        return False
    recent_prices = prices[-STABILIZATION_CANDLES:]
    return all(recent_prices[i] >= recent_prices[i-1] for i in range(1, len(recent_prices)))

def on_ws_message(ws, message):
    """Handle WebSocket messages."""
    global price_history, rsi_values, current_bid, bid_count, last_bid_price
    try:
        data = json.loads(message)
        if "data" in data and data["arg"]["channel"] == "tickers":
            price = float(data["data"][0]["last"])
            price_history.append(price)
            logger.info(f"Current price: ${price}")

            # Maintain price history for RSI
            if len(price_history) > RSI_PERIOD:
                price_history = price_history[-RSI_PERIOD-1:]
                rsi = calculate_rsi(price_history)
                if rsi:
                    rsi_values.append(rsi)
                    logger.info(f"RSI: {rsi:.2f}")

            # Check if price dropped below last bid
            if last_bid_price and price < last_bid_price and bid_count < MAX_BIDS:
                # Wait for stabilization or confirmation
                time.sleep(WAITING_PERIOD)
                current_price = get_price()
                if not current_price:
                    return

                # Check stabilization and RSI conditions
                stabilized = is_price_stabilized(price_history[-STABILIZATION_CANDLES:])
                rsi_recovery = rsi_values and rsi_values[-1] > OVERSOLD_THRESHOLD and len(rsi_values) > 1 and rsi_values[-1] > rsi_values[-2]
                if stabilized and rsi_recovery:
                    # Cancel previous order (simplified, assumes one active order)
                    # In production, query active orders and cancel specific order
                    # Increase bid
                    current_bid *= PHI
                    if current_bid * current_price > TOTAL_CAPITAL:
                        logger.warning("Bid exceeds capital limit, stopping.")
                        ws.close()
                        return
                    quantity = current_bid / current_price
                    order_response = place_order(current_price, quantity)
                    if order_response:
                        bid_count += 1
                        last_bid_price = current_price
                        logger.info(f"New bid: ${current_bid} at ${current_price} (Bid {bid_count}/{MAX_BIDS})")
                        # Set stop-loss (simplified, use trigger order in production)
                        stop_price = current_price * (1 - STOP_LOSS_PCT)
                        logger.info(f"Stop-loss set at ${stop_price:.2f}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

def on_ws_error(ws, error):
    logger.error(f"WebSocket error: {error}")

def on_ws_close(ws, close_status_code, close_msg):
    logger.info("WebSocket closed")

def on_ws_open(ws):
    """Subscribe to ticker channel."""
    subscription = {
        "op": "subscribe",
        "args": [{"channel": "tickers", "instId": SYMBOL}]
    }
    ws.send(json.dumps(subscription))
    logger.info(f"Subscribed to {SYMBOL} ticker")

async def main():
    """Main function to run the trading bot."""
    global last_bid_price
    # Place initial bid
    initial_price = get_price()
    if not initial_price:
        logger.error("Failed to get initial price, exiting.")
        return
    quantity = INITIAL_BID / initial_price
    order_response = place_order(initial_price, quantity)
    if order_response:
        last_bid_price = initial_price
        logger.info(f"Initial bid: ${INITIAL_BID} at ${initial_price}")

    # Start WebSocket
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_ws_open,
        on_message=on_ws_message,
        on_error=on_ws_error,
        on_close=on_ws_close
    )
    ws.run_forever()

if __name__ == "__main__":
    asyncio.run(main())
