import os
import pytz

class Config:
    # Core configurations
    SYMBOLS = os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,XRP-USDT").split(",")
    DB_PATH = os.getenv("DB_PATH", "postgresql://market_data_19fb_user:yxlMERTZ36Wm15LbYxir74tKhsVCxQOd@dpg-d19825nfte5s73c40jl0-a/market_data_19fb")
    DB_PATH_LOCAL = os.path.join(os.path.dirname(__file__), "..", "db", "market_data.db")
    REDIS_HOST = os.getenv("REDIS_HOST", "red-d195gnjuibrs73breos0")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    MAIN_BOT_URL = os.getenv("MAIN_BOT_URL", "https://novara-tradingbot.onrender.com")
    LOCAL_TZ = pytz.timezone("America/Los_Angeles")
    RENDER = os.getenv("RENDER", "true").lower() == "true"

    # API credentials
    API_KEY = os.getenv("DEMO_API_KEY" if os.getenv("DEMO_MODE", "true").lower() == "true" else "API_KEY", "")
    API_SECRET = os.getenv("DEMO_API_SECRET" if os.getenv("DEMO_MODE", "true").lower() == "true" else "API_SECRET", "")
    API_PASSPHRASE = os.getenv("DEMO_API_PASSPHRASE" if os.getenv("DEMO_MODE", "true").lower() == "true" else "API_PASSPHRASE", "")
    DEMO_API_KEY = os.getenv("DEMO_API_KEY", "")
    DEMO_API_SECRET = os.getenv("DEMO_API_SECRET", "")
    DEMO_API_PASSPHRASE = os.getenv("DEMO_API_PASSPHRASE", "")
    XAI_API_KEY = os.getenv("XAI_API_KEY", "")
    OKX_API_KEY = os.getenv("OKX_API_KEY", "")
    OKX_API_SECRET = os.getenv("OKX_API_SECRET", "")
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    ONCHAIN_API_KEY = os.getenv("ONCHAIN_API_KEY", "")
    CLOUD_DB_URL = os.getenv("CLOUD_DB_URL", "")

    # Trading configurations
    DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
    BASE_URL = "https://demo-trading-openapi.blofin.com" if DEMO_MODE else "https://openapi.blofin.com"
    WS_URL = "wss://demo-trading-openapi.blofin.com/ws/public" if DEMO_MODE else "wss://openapi.blofin.com/ws/public"
    DEFAULT_BALANCE = float(os.getenv("DEFAULT_BALANCE", 49999.42))
    INITIAL_BID_USD = float(os.getenv("INITIAL_BID_USD", 100.0))
    RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))
    LOGTAIL_TOKEN = os.getenv("LOGTAIL_TOKEN", "7pWvyBP5PB8b1h1wjrSXSowz")
    WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

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
    SENTIMENT_WEIGHT = 0.15
    MAX_SYMBOLS = 10
    CORRELATION_THRESHOLD = 0.7
    COST_AVERAGE_DIP = 0.02
    COST_AVERAGE_LIMIT = 2
    TRAILING_STOP_MULTIPLIER = 1.5