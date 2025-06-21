import os

class Config:
    SYMBOLS = os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,XRP-USDT").split(",")
    TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    VOLATILITY_THRESHOLD = 0.005
    RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))
    MAX_LEVERAGE_PERCENTAGE = 0.5
    TRAILING_STOP_MULTIPLIER = 1.5
    DEFAULT_BALANCE = float(os.getenv("DEFAULT_BALANCE", 49999.42))
    INITIAL_BID_USD = float(os.getenv("INITIAL_BID_USD", 100.0))
    DB_PATH = os.getenv("DB_PATH", "postgresql://market_data_19fb_user:yxlMERTZ36Wm15LbYxir74tKhsVCxQOd@dpg-d19825nfte5s73c40jl0-a/market_data_19fb")
    REDIS_URL = f"redis://{os.getenv('REDIS_HOST', 'red-d195gnjuibrs73breos0')}:{os.getenv('REDIS_PORT', '6379')}"
    MAIN_BOT_URL = os.getenv("MAIN_BOT_URL", "https://novara-tradingbot.onrender.com")
    SENTIMENT_WEIGHT = 0.2
    ML_LOOKBACK = 100
    RENDER = os.getenv("RENDER", "true").lower() == "true"
    DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
    LOGTAIL_TOKEN = os.getenv("LOGTAIL_TOKEN", "7pWvyBP5PB8b1h1wjrSXSowz")
    XAI_API_KEY = os.getenv("XAI_API_KEY", "")
    OKX_API_KEY = os.getenv("OKX_API_KEY", "")
    OKX_API_SECRET = os.getenv("OKX_API_SECRET", "")
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    ONCHAIN_API_KEY = os.getenv("ONCHAIN_API_KEY", "")
    CLOUD_DB_URL = os.getenv("CLOUD_DB_URL", "")
    API_KEY = os.getenv("DEMO_API_KEY" if DEMO_MODE else "API_KEY", "")
    API_SECRET = os.getenv("DEMO_API_SECRET" if DEMO_MODE else "API_SECRET", "")
    API_PASSPHRASE = os.getenv("DEMO_API_PASSPHRASE" if DEMO_MODE else "API_PASSPHRASE", "")