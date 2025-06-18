import os
import pytz

# Core configurations
DB_PATH = os.getenv("DB_PATH", os.path.join("/opt/render/project/src/db", "market_data.db") if os.getenv("RENDER") else os.path.join(os.path.dirname(__file__), "..", "db", "market_data.db"))
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
SYMBOLS = os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,XRP-USDT").split(",")

# API credentials
API_KEY = os.getenv("API_KEY", "your_secure_api_key")
API_SECRET = os.getenv("API_SECRET", "your_secure_api_secret")
API_PASSPHRASE = os.getenv("API_PASSPHRASE", "your_secure_passphrase")
DEMO_API_KEY = os.getenv("DEMO_API_KEY", "your_demo_api_key")
DEMO_API_SECRET = os.getenv("DEMO_API_SECRET", "your_demo_api_secret")
DEMO_API_PASSPHRASE = os.getenv("DEMO_API_PASSPHRASE", "your_demo_passphrase")

# Trading configurations
DEMO_MODE = os.getenv("DEMO_MODE", "True").lower() == "true"
BASE_URL = "https://demo-trading-openapi.blofin.com" if DEMO_MODE else "https://openapi.blofin.com"
WS_URL = "wss://demo-trading-openapi.blofin.com/ws/public" if DEMO_MODE else "wss://openapi.blofin.com/ws/public"
DEFAULT_BALANCE = float(os.getenv("DEFAULT_BALANCE", 49999.42))
INITIAL_BID_USD = float(os.getenv("INITIAL_BID_USD", 100.0))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))
LOGTAIL_TOKEN = os.getenv("LOGTAIL_TOKEN", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://your-webhook-url.com")
LOCAL_TZ = pytz.timezone("America/Los_Angeles")

# Technical analysis parameters
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