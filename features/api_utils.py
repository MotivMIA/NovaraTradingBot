import hmac
import hashlib
import time
import requests
import logging
import os
from typing import List, Dict
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BloFinAPIUtils:
    def __init__(self, api_key: str = None, api_secret: str = None, api_passphrase: str = None):
        self.config = Config()
        self.api_key = api_key or self.config.API_KEY
        self.api_secret = api_secret or self.config.API_SECRET
        self.api_passphrase = api_passphrase or self.config.API_PASSPHRASE
        self.base_url = "https://api.blofin.com" if not self.config.DEMO_MODE else "https://api-demo.blofin.com"
        self.last_candle_fetch: Dict[str, float] = {}

    def sign_request(self, method: str, path: str, params: dict = None) -> dict:
        try:
            timestamp = str(int(time.time() * 1000))
            query = "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
            message = timestamp + method.upper() + path + (f"?{query}" if query else "")
            signature = hmac.new(
                self.api_secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            return {
                "X-BLOFIN-APIKEY": self.api_key,
                "X-BLOFIN-SIGN": signature,
                "X-BLOFIN-TIMESTAMP": timestamp,
                "X-BLOFIN-PASSPHRASE": self.api_passphrase
            }
        except Exception as e:
            logger.error(f"Error signing request: {e}")
            return {}

    async def get_candles(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict]:
        try:
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.last_candle_fetch and time.time() - self.last_candle_fetch[cache_key] < 60:
                return []

            path = "/v1/market/candles"
            params = {"symbol": symbol, "timeframe": timeframe, "limit": limit}
            headers = self.sign_request("GET", path, params)
            response = requests.get(f"{self.base_url}{path}", params=params, headers=headers)
            response.raise_for_status()
            candles = response.json()
            self.last_candle_fetch[cache_key] = time.time()
            return [
                {
                    "timestamp": int(c["timestamp"]),
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    "volume": float(c["volume"])
                }
                for c in candles
            ]
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return []

    def get_max_leverage(self, symbol: str, bot) -> float:
        return 20.0  # Placeholder