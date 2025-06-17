import sqlite3
import os
from logging import getLogger
from features.config import DB_PATH

logger = getLogger(__name__)

class Database:
    def initialize_db(self):
        try:
            if os.getenv("RENDER"):
                os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
                os.chmod(DB_PATH, 0o666) if os.path.exists(DB_PATH) else None
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT,
                    timestamp INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    symbol TEXT,
                    entry_time INTEGER,
                    exit_time INTEGER,
                    entry_price REAL,
                    exit_price REAL,
                    size REAL,
                    leverage REAL,
                    profit_loss REAL,
                    features_used TEXT,
                    timeframes TEXT,
                    side TEXT
                )
            """)
            conn.commit()
            conn.close()
            logger.debug(f"Database initialized at {DB_PATH}")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")

    def save_candles(self, symbol: str, candles: list):
        try:
            table_name = symbol.replace("-", "_") + "_candles"
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    timestamp INTEGER PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL
                )
            """)
            for candle in candles:
                cursor.execute(f"""
                    INSERT OR REPLACE INTO {table_name} (timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (candle["timestamp"], candle["open"], candle["high"], candle["low"], candle["close"], candle["volume"]))
            conn.commit()
            conn.close()
            logger.debug(f"Saved {len(candles)} candles for {symbol}")
        except sqlite3.Error as e:
            logger.error(f"Failed to save candles for {symbol}: {e}")

    def load_candles(self, symbol: str, candle_history: dict):
        try:
            table_name = symbol.replace("-", "_") + "_candles"
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT 100")
            rows = cursor.fetchall()
            conn.close()
            if rows:
                candle_history[symbol] = [{
                    "timestamp": row[0],
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5]
                } for row in rows]
                logger.debug(f"Loaded {len(rows)} candles for {symbol} from database")
        except sqlite3.Error as e:
            logger.error(f"Failed to load candles for {symbol}: {e}")