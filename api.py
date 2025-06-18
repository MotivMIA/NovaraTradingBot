import asyncio
import websockets
import json
import time
import hmac
import base64
import requests
import logging
import os
import socket
import sys
import argparse
from datetime import datetime
import glob
from dotenv import load_dotenv
# Commenting out unused imports
# from logtail import LogtailHandler
import uvicorn
from fastapi import FastAPI
from features.config import BASE_URL, API_KEY, API_SECRET, API_PASSPHRASE, SYMBOLS, CANDLE_LIMIT, CANDLE_TIMEFRAME
# Commenting out unused imports
# from features.indicators import Indicators
# from features.trading_logic import TradingLogic
# from features.machine_learning import MachineLearning
# from features.sentiment_analysis import SentimentAnalysis
# from features.portfolio_management import PortfolioManagement
# from features.performance_analytics import PerformanceAnalytics
from features.performance_analytics import PerformanceAnalytics
logger = logging.getLogger(__name__)
load_dotenv(".env.local")
logger = getLogger(name)
app = FastAPI()

logger.debug(f"api-key: {API_KEY[:4]}...{API_KEY[-4:]}")
class TradingBot:
	def __init__(self):
		self.symbols = SYMBOLS
		self.candle_history = {symbol: [] for symbol in self.symbols}
		self.price_history = {symbol: [] for symbol in self.symbols}
		self.volume_history = {symbol: [] for symbol in self.symbols}
self.price_history = {symbol: [] for symbol in self.symbols}
async def ws_connect(self):
	uri = "wss://demo-trading-openapi.blofin.com/ws/public"
	try:
		async with websockets.connect(uri) as websocket:
			for symbol in self.symbols:
				subscribe_msg = {"op": "subscribe", "args": [{"channel": "tickers", "symbol": symbol}]}
				await websocket.send(json.dumps(subscribe_msg))
				logger.info(f"Subscribed to {symbol} ticker")
			while True:
				data = json.loads(await websocket.recv())
				logger.debug(f"Received data: {data}")
	except Exception as e:
		logger.error(f"WebSocket error: {e}")
		sys.exit(1)
logger.error(f"WebSocket error: {e}")
async def health_check(self):
	try:
		return True
	except Exception as e:
		logger.error(f"Health check failed: {e}")
		return False
logger.error(f"Health check failed: {e}")
async def main(self):
	parser = argparse.ArgumentParser(description="NovaraTradingBot")
	parser.add_argument("--health-check", action="store_true", help="Run health check and exit")
	args = parser.parse_args()

	if args.health_check:
		if await self.health_check():
			logger.info("Health check passed")
			sys.exit(0)
		else:
			logger.error("Health check failed")
			sys.exit(1)

	async def start_server(self):
		server = uvicorn.Server(uvicorn.Config("features.api_endpoints:app", host="0.0.0.0", port=8000))
		await asyncio.gather(self.ws_connect(), server.serve())

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
		logger.info(f"Saved {len(candles)} candles for {symbol}")
		return len(candles)
	logger.error(f"Failed to fetch candles for {symbol}: {response.status_code} {response.text}")
	return 0

for symbol in SYMBOLS:
	num_candles = fetch_historical_candles(symbol)
if __name__ == "__main__":
	bot = TradingBot()
	asyncio.run(bot.main())
bot = TradingBot()
asyncio.run(bot.main())
