import streamlit as st
import plotly.express as px
import pandas as pd
import psycopg2
import logging
from performance_analytics import PerformanceAnalytics
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dashboard:
    def __init__(self):
        self.config = Config()
        self.performance = PerformanceAnalytics()
        self.performance.initialize(self.config.DB_PATH)

    def run(self):
        st.title("NovaraTradingBot Dashboard")
        report = self.performance.generate_report()
        if report:
            st.write(f"Total Trades: {report['total_trades']}")
            st.write(f"Win Rate: {report['win_rate']:.2%}")
            st.write(f"Sharpe Ratio: {report['sharpe_ratio']:.2f}")
            st.markdown("### Portfolio Performance")
            st.components.v1.html(report["plot"], height=500)

            st.markdown("### Recent Trades")
            conn = psycopg2.connect(self.config.DB_PATH)
            df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10", conn)
            conn.close()
            st.dataframe(df)

            st.markdown("### Signal Metrics")
            signals = pd.DataFrame([
                {"Bot": "TrendBot", "Patterns": "Ichimoku, EMA", "Confidence": "0.7-0.9"},
                {"Bot": "VolumeBot", "Patterns": "ATR, ADX", "Confidence": "0.6-0.8"},
                {"Bot": "PredictBot", "Patterns": "ML, X/News Sentiment", "Confidence": "0.6-0.9"},
                {"Bot": "ArbitrageBot", "Patterns": "Cross-Exchange", "Confidence": "0.5-0.9"},
                {"Bot": "PatternBot", "Patterns": "Doji, Engulfing", "Confidence": "0.6-0.8"},
                {"Bot": "NewsBot", "Patterns": "News Sentiment", "Confidence": "0.5-0.9"}
            ])
            st.dataframe(signals)
        else:
            st.write("No performance data available.")

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()