from features.config import SYMBOLS
import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from logging import getLogger
from features.config import DB_PATH, DEFAULT_BALANCE

logger = getLogger(__name__)

st.set_page_config(page_title="NovaraTradingBot Dashboard", layout="wide")

st.title("NovaraTradingBot Dashboard")

@st.cache_data
def load_trades():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM trades", conn)
        conn.close()
        logger.info("Loaded trades for dashboard")
        return df
    except Exception as e:
        logger.error(f"Failed to load trades: {e}")
        return pd.DataFrame()

@st.cache_data
def load_candles(symbol: str):
    try:
        table_name = symbol.replace("-", "_") + "_candles"
        conn = sqlite3.connect(DB_PATH)
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            logger.warning(f"Table {table_name} does not exist")
            conn.close()
            return pd.DataFrame()
        df = pd.read_sql(f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT 100", conn)
        conn.close()
        logger.info(f"Loaded candles for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Failed to load candles for {symbol}: {e}")
        return pd.DataFrame()

st.header("Account Overview")
balance = st.session_state.get("balance", DEFAULT_BALANCE)
st.metric("Account Balance", f"${balance:.2f}")

st.header("Trade History")
trades = load_trades()
if not trades.empty:
    st.dataframe(trades[["symbol", "entry_time", "profit_loss", "side", "leverage"]])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(trades["entry_time"], unit="s"),
        y=trades["profit_loss"].cumsum(),
        mode="lines",
        name="Cumulative P/L"
    ))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No trades available")

st.header("Market Data")
symbol = st.selectbox("Select Symbol", SYMBOLS)
candles = load_candles(symbol)
if not candles.empty:
    fig = go.Figure(data=[
        go.Candlestick(
            x=pd.to_datetime(candles["timestamp"], unit="ms"),
            open=candles["open"],
            high=candles["high"],
            low=candles["low"],
            close=candles["close"]
        )
    ])
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write(f"No candle data available for {symbol}")

st.header("Performance Metrics")
if not trades.empty:
    win_rate = len(trades[trades["profit_loss"] > 0]) / len(trades)
    sharpe_ratio = trades["profit_loss"].mean() / trades["profit_loss"].std() * np.sqrt(252) if trades["profit_loss"].std() != 0 else 0
    st.metric("Win Rate", f"{win_rate:.2%}")
    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

if __name__ == "__main__":
    logger.info("Starting Streamlit dashboard")
    st.write("Run with `streamlit run dashboard.py`")