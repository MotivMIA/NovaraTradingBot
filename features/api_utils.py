# API authentication, credential validation, balance/leverage queries
import base64
import hmac
import hashlib
import json
import requests
import time
import urllib.request
from datetime import datetime
from uuid import uuid4
from logging import getLogger
from features.config import CANDLE_LIMIT, CANDLE_FETCH_INTERVAL

logger = getLogger(__name__)

class APIUtils:
    def __init__(self, api_key: str, api_secret: str, api_passphrase: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase

    def validate_credentials(self) -> bool:
        path = "/api/v1/market/tickers"
        try:
            response = requests.get(f"{BASE_URL}{path}?instId=BTC-USDT", timeout=5)
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
            self.api_secret.encode(),
            content.encode(),
            hashlib.sha256
        ).hexdigest().encode()
        signature = base64.b64encode(sign_token).decode()
        
        logger.debug(f"Generated signature: {signature}")
        
        headers = {
            "access-key": self.api_key.strip(),
            "access-sign": signature,
            "access-timestamp": timestamp,
            "access-nonce": nonce,
            "access-passphrase": self.api_passphrase.strip(),
            "content-type": "application/json"
        }
        logger.debug(f"Request headers: {headers}")
        return headers, timestamp, nonce

    def get_account_balance(self, bot) -> float | None:
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

    def get_max_leverage(self, symbol: str, bot) -> float | None:
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
                    bot.leverage_info[symbol] = max_leverage
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

    def get_candles(self, symbol: str, limit: int = CANDLE_LIMIT, timeframe: str = "1m") -> list | None:
        current_time = time.time()
        if current_time - bot.last_candle_fetch.get(symbol, {}).get(timeframe, 0) < CANDLE_FETCH_INTERVAL:
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
                    bot.last_candle_fetch.setdefault(symbol, {})[timeframe] = current_time
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