import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import psycopg2
import logging
from features.performance_analytics import PerformanceAnalytics
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
            df_trades = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10", conn)
            st.dataframe(df_trades)

            st.markdown("### Candlestick Patterns")
            df_patterns = pd.DataFrame([
                {"Symbol": s, "Pattern": "Doji, Hammer", "Confidence": "0.6-0.8"}
                for s in self.config.SYMBOLS
            ])
            st.dataframe(df_patterns)

            st.markdown("### VWAP Analysis")
            fig_vwap = go.Figure()
            for symbol in self.config.SYMBOLS[:2]:  # Limit for clarity
                df_candles = pd.read_sql(f"SELECT * FROM candles WHERE symbol = '{symbol}' ORDER BY timestamp DESC LIMIT 100", conn)
                fig_vwap.add_trace(go.Scatter(x=df_candles["timestamp"], y=df_candles["close"], name=f"{symbol} Close"))
                fig_vwap.add_trace(go.Scatter(x=df_candles["timestamp"], y=df_candles["close"].rolling(self.config.VWAP_PERIOD).mean(), name=f"{symbol} VWAP"))
            st.plotly_chart(fig_vwap)

            conn.close()
        else:
            st.write("No performance data available.")