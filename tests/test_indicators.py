import pytest
import pandas as pd
from features.indicators import Indicators
from features.config import VWAP_PERIOD

@pytest.fixture
def mock_candles():
    # Create enough candles to satisfy VWAP_PERIOD
    candles = []
    for i in range(VWAP_PERIOD):
        candles.append({
            "timestamp": i+1,
            "open": 100 + i,
            "high": 110 + i,
            "low": 90 + i,
            "close": 105 + i,
            "volume": 1000 + i*100
        })
    return candles

def test_calculate_vwap(mock_candles):
    indicators = Indicators()
    candle_history = {"BTC-USDT": mock_candles}
    vwap = indicators.calculate_vwap("BTC-USDT", candle_history)
    assert vwap is not None
    # Approximate VWAP calculation for verification
    df = pd.DataFrame(mock_candles)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    expected_vwap = (typical_price * df["volume"]).sum() / df["volume"].sum()
    assert abs(vwap - expected_vwap) < 0.01

def test_calculate_atr(mock_candles):
    indicators = Indicators()
    candle_history = {"BTC-USDT": mock_candles}
    atr = indicators.calculate_atr("BTC-USDT", candle_history, period=2)
    assert atr is not None
    assert atr > 0