import pytest
import pandas as pd
import numpy as np
from features.indicators import Indicators
from features.config import VWAP_PERIOD

@pytest.fixture
def mock_candles():
    # Generate 25 candles to exceed VWAP_PERIOD (20)
    candles = []
    for i in range(25):
        candles.append({
            "timestamp": i + 1,
            "open": 100 + i,
            "high": 110 + i,
            "low": 90 + i,
            "close": 105 + i,
            "volume": 1000 + i * 100
        })
    return candles

def test_calculate_vwap(mock_candles):
    indicators = Indicators()
    vwap = indicators.calculate_vwap("BTC-USDT", mock_candles)
    assert vwap is not None
    # Approximate VWAP calculation for verification
    df = pd.DataFrame(mock_candles)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    window = min(VWAP_PERIOD, len(df))
    expected_vwap = (typical_price * df["volume"]).rolling(window=window).sum() / df["volume"].rolling(window=window).sum()
    expected_vwap = expected_vwap.iloc[-1] if not pd.isna(expected_vwap.iloc[-1]) else df["close"].iloc[-1]
    assert abs(vwap - expected_vwap) < 0.01

def test_calculate_atr(mock_candles):
    indicators = Indicators()
    period = 2
    atr = indicators.calculate_atr("BTC-USDT", mock_candles, period=period)
    assert atr is not None
    # Approximate ATR calculation for verification
    df = pd.DataFrame(mock_candles)
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.DataFrame({"high_low": high_low, "high_close": high_close, "low_close": low_close}).max(axis=1)
    expected_atr = tr.rolling(window=period).mean().iloc[-1] if len(tr) >= period else tr.mean()
    assert abs(atr - expected_atr) < 0.01