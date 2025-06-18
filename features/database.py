import os
import psycopg2
from logging import getLogger
from features.config import DB_PATH  # Now a PostgreSQL connection string

logger = getLogger(__name__)

class Database:
    def get_connection(self):
        return psycopg2.connect(DB_PATH)

    def initialize_db(self):
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS candles (
                            symbol TEXT,
                            timestamp BIGINT,
                            open DOUBLE PRECISION,
                            high DOUBLE PRECISION,
                            low DOUBLE PRECISION,
                            close DOUBLE PRECISION,
                            volume DOUBLE PRECISION,
                            PRIMARY KEY (symbol, timestamp)
                        )
                    """)
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS trades (
                            symbol TEXT,
                            entry_time BIGINT,
                            exit_time BIGINT,
                            entry_price DOUBLE PRECISION,
                            exit_price DOUBLE PRECISION,
                            size DOUBLE PRECISION,
                            leverage DOUBLE PRECISION,
                            profit_loss DOUBLE PRECISION,
                            features_used TEXT,
                            timeframes TEXT,
                            side TEXT
                        )
                    """)
                    for symbol in ["BTC-USDT", "ETH-USDT", "XRP-USDT"]:
                        table_name = symbol.replace("-", "_") + "_candles"
                        cursor.execute(f"""
                            CREATE TABLE IF NOT EXISTS {table_name} (
                                timestamp BIGINT PRIMARY KEY,
                                open DOUBLE PRECISION,
                                high DOUBLE PRECISION,
                                low DOUBLE PRECISION,
                                close DOUBLE PRECISION,
                                volume DOUBLE PRECISION
                            )
                        """)
                conn.commit()
                logger.debug(f"PostgreSQL database initialized")
        except psycopg2.Error as e:
            logger.error(f"Failed to initialize database: {e}")

    def save_candles(self, symbol: str, candles: list):
        try:
            table_name = symbol.replace("-", "_") + "_candles"
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    for candle in candles:
                        cursor.execute(f"""
                            INSERT INTO {table_name} (timestamp, open, high, low, close, volume)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (timestamp) DO UPDATE
                            SET open = EXCLUDED.open,
                                high = EXCLUDED.high,
                                low = EXCLUDED.low,
                                close = EXCLUDED.close,
                                volume = EXCLUDED.volume
                        """, (candle["timestamp"], candle["open"], candle["high"], candle["low"], candle["close"], candle["volume"]))
                conn.commit()
                logger.debug(f"Saved {len(candles)} candles for {symbol}")
        except psycopg2.Error as e:
            logger.error(f"Failed to save candles for {symbol}: {e}")

    def load_candles(self, symbol: str, candle_history: dict):
        try:
            table_name = symbol.replace("-", "_") + "_candles"
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT 100")
                    rows = cursor.fetchall()
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
        except psycopg2.Error as e:
            logger.error(f"Failed to load candles for {symbol}: {e}")