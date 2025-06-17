# SQLite interactions (save/load candles, trades)
import sqlite3
import pandas as pd
import os
from logging import getLogger

logger = getLogger(__name__)

class Database:
    def initialize_db(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
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
            logger.info("Initialized trades table in database")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")

    def save_candles(self, symbol: str, candles: list):
        try:
            conn = sqlite3.connect(DB_PATH)
            os.chmod(DB_PATH, 0o666) if os.path.exists(DB_PATH) else None
            df = pd.DataFrame(candles)
            if not df.empty:
                table_name = symbol.replace("-", "_") + "_candles"
                df.to_sql(table_name, conn, if_exists='append', index=False)
                logger.debug(f"Saved {len(df)} candles for {symbol} to {DB_PATH} (table: {table_name})")
            else:
                logger.warning(f"No candles to save for {symbol}")
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Failed to save candles for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving candles for {symbol}: {e}")

    def load_candles(self, symbol: str, candle_history: dict):
        try:
            conn = sqlite3.connect(DB_PATH)
            table_name = symbol.replace("-", "_") + "_candles"
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            candle_history[symbol] = df.to_dict('records')
            logger.info(f"Loaded {len(candle_history[symbol])} candles for {symbol} from {DB_PATH} (table: {table_name})")
            conn.close()
        except sqlite3.Error as e:
            logger.debug(f"No stored candles found for {symbol} or database error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading candles for {symbol}: {e}")