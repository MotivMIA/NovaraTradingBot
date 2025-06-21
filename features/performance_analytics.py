import pandas as pd
import sqlite3
import plotly.graph_objects as go
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceAnalytics:
    def __init__(self):
        self.conn = None

    def initialize(self, db_path: str):
        try:
            self.conn = sqlite3.connect(db_path)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    side TEXT,
                    price REAL,
                    size REAL,
                    leverage REAL,
                    timestamp INTEGER
                )
            """)
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error initializing performance analytics: {e}")

    def log_trade(self, symbol: str, side: str, price: float, size: float, leverage: float):
        try:
            timestamp = int(pd.Timestamp.now().timestamp())
            self.conn.execute("""
                INSERT INTO trades (symbol, side, price, size, leverage, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, side, price, size, leverage, timestamp))
            self.conn.commit()
            logger.info(f"Trade logged: {side} {symbol} at ${price:.2f}")
        except Exception as e:
            logger.error(f"Error logging trade for {symbol}: {e}")

    def generate_report(self):
        try:
            df = pd.read_sql("SELECT * FROM trades", self.conn)
            if df.empty:
                return None

            total_trades = len(df)
            win_rate = len(df[df["price"] * (df["size"] if df["side"] == "buy" else -df["size"]) > 0]) / total_trades
            returns = (df["price"] * df["size"] * df["leverage"]).pct_change().fillna(0)
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df["price"] * df["size"] * df["leverage"], mode="lines"))
            fig.update_layout(title="Portfolio Performance", xaxis_title="Time", yaxis_title="Value")
            return {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "plot": fig.to_html()
            }
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None