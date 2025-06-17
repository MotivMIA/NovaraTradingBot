import os
import pytz

# Configuration constants for NovaraTradingBot
DEMO_MODE = os.getenv("DEMO_MODE", "True").lower() == "true"
BASE_URL = "https://demo-trading-openapi.blofin.com" if DEMO_MODE else "https://openapi.blofin.com"
WS_URL = "wss://demo-trading-openapi.blofin.com/ws/public" if DEMO_MODE else "wss://openapi.blofin.com/ws/public"
LOCAL_TZ = pytz.timezone("America/Los_Angeles")

RISK_PER_TRADE = 0.01
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
MAX_DRAWDOWN = 0.10
DEFAULT_BALANCE = 10000.0
MIN_PRICE_POINTS = 20
VWAP_PERIOD = 20
VOLATILITY_THRESHOLD = 0.005
CANDLE_TIMEFRAME = "1m"
CANDLE_FETCH_INTERVAL = 60
CANDLE_LIMIT = 2000
MAX_LEVERAGE_PERCENTAGE = 0.8
TIMEFRAMES = ["1m", "5m", "15m", "1h", "1d"]
SENTIMENT_WEIGHT = 0.1
MAX_SYMBOLS = 10
CORRELATION_THRESHOLD = 0.7
COST_AVERAGE_DIP = 0.02
COST_AVERAGE_LIMIT = 2
TRAILING_STOP_MULTIPLIER = 1.5
DB_PATH = os.path.join("/opt/render/project/src/db", "market_data.db") if os.getenv("RENDER") else os.path.join(os.path.dirname(__file__), "..", "market_data.db")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
API_KEY = os.getenv("API_KEY", "your_secure_api_key")