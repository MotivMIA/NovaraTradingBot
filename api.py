# Ensure the file is complete and remove invalid characters
from features.config import BASE_URL, API_KEY, API_SECRET, API_PASSPHRASE, SYMBOLS, CANDLE_LIMIT, CANDLE_TIMEFRAME
from features.database import Database
import requests  # Ensure the 'requests' library is installed
import hmac
import base64
import time

def fetch_historical_candles(symbol):
	timestamp = str(int(time.time()))
	method = "GET"
	endpoint = "/v1/market/candles"
	params = f"symbol={symbol}&interval={CANDLE_TIMEFRAME}&limit={CANDLE_LIMIT}"
	sign_str = timestamp + method + endpoint + "?" + params
	signature = base64.b64encode(hmac.new(API_SECRET.encode(), sign_str.encode(), digestmod="sha256").digest()).decode()
	headers = {
		"X-BLOFIN-API-KEY": API_KEY,
		"X-BLOFIN-PASSPHRASE": API_PASSPHRASE,
		"X-BLOFIN-SIGNATURE": signature,
		"X-BLOFIN-TIMESTAMP": timestamp
	}
	url = f"{BASE_URL}{endpoint}?{params}"
	response = requests.get(url, headers=headers)
	if response.status_code == 200:
		data = response.json().get("data", [])
		candles = [
			{
				"timestamp": int(candle[0]),
				"open": float(candle[1]),
				"high": float(candle[2]),
				"low": float(candle[3]),
				"close": float(candle[4]),
				"volume": float(candle[5])
			}
			for candle in data
		]
		Database().save_candles(symbol, candles)
		return len(candles)
	print(f"Failed to fetch candles for {symbol}: {response.status_code} {response.text}")
	return 0

for symbol in SYMBOLS:
	num_candles = fetch_historical_candles(symbol)
	print(f"Fetched {num_candles} candles for {symbol}")

if __name__ == "__main__":
	bot = TradingBot()
	asyncio.run(bot.main())
