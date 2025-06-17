# Trade logging, performance reports, plots
import pandas as pd
import sqlite3
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from logging import getLogger
from datetime import datetime
import os
from features.notifications import Notifications
from features.config import DB_PATH

logger = getLogger(__name__)

class PerformanceAnalytics:
    def log_trade(self, symbol: str, entry_time: int, exit_time: int, entry_price: float, exit_price: float, size: float, leverage: float, features_used: dict, timeframes: list, side: str, bot):
        try:
            profit_loss = (exit_price - entry_price) * size if side == "buy" else (entry_price - exit_price) * size
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (symbol, entry_time, exit_time, entry_price, exit_price, size, leverage, profit_loss, features_used, timeframes, side)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, entry_time, exit_time, entry_price, exit_price, size, leverage, profit_loss, json.dumps(features_used), json.dumps(timeframes), side))
            conn.commit()
            conn.close()
            logger.info(f"Logged trade for {symbol}: P/L ${profit_loss:.2f}")
            bot.notifications.send_webhook_alert(f"Trade closed for {symbol}: P/L ${profit_loss:.2f}, Leverage: {leverage:.2f}x")
        except sqlite3.Error as e:
            logger.error(f"Failed to log trade for {symbol}: {e}")

    def analyze_performance(self, bot):
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT * FROM trades", conn)
            conn.close()
            if df.empty:
                logger.info("No trades to analyze")
                return
            
            win_rate = len(df[df["profit_loss"] > 0]) / len(df)
            avg_pl = df["profit_loss"].mean()
            sharpe_ratio = df["profit_loss"].mean() / df["profit_loss"].std() * np.sqrt(252) if df["profit_loss"].std() != 0 else 0
            feature_impact = {}
            for feature in ["atr", "sentiment", "multi_timeframe"]:
                feature_impact[feature] = df[df["features_used"].str.contains(feature)]["profit_loss"].mean()
            
            report = {
                "win_rate": win_rate,
                "avg_profit_loss": avg_pl,
                "sharpe_ratio": sharpe_ratio,
                "feature_impact": feature_impact
            }
            logger.info(f"Performance report: {json.dumps(report, indent=2)}")
            bot.notifications.send_webhook_alert(f"Performance Report: Win Rate {win_rate:.2%}, Sharpe {sharpe_ratio:.2f}")
            return report
        except Exception as e:
            logger.error(f"Failed to analyze performance: {e}")
            return None

    def generate_performance_plots(self, bot):
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql("SELECT * FROM trades", conn)
            conn.close()
            if df.empty:
                logger.info("No trades to plot")
                return
            
            fig = make_subplots(rows=3, cols=1, subplot_titles=("Profit/Loss Over Time", "Win/Loss Distribution", "Leverage vs. Return"))
            
            fig.add_trace(go.Scatter(x=pd.to_datetime(df["exit_time"], unit="s"), y=df["profit_loss"].cumsum(), mode="lines", name="Cumulative P/L"), row=1, col=1)
            fig.add_trace(go.Histogram(x=df["profit_loss"], name="P/L Distribution"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df["leverage"], y=df["profit_loss"], mode="markers", name="Leverage vs. P/L"), row=3, col=1)
            
            plot_dir = os.path.join(os.path.dirname(__file__), "../plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_file = os.path.join(plot_dir, f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            fig.write_html(plot_file)
            logger.info(f"Generated performance plot: {plot_file}")
            bot.notifications.send_webhook_alert(f"New performance plot generated: {plot_file}")
        except Exception as e:
            logger.error(f"Failed to generate performance plots: {e}")