import pytest
import pandas as pd
from features.indicators import Indicators

@pytest.fixture
def mock_candles():
    return [
        {"timestamp": 1, "open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000},
        {"timestamp": 2, "open": 105, "high": 115, "low": 95, "close": 110, "volume": 1200},
        {"timestamp": 3, "open": 110, "high": 120, "low": 100, "close": 115, "volume": 1100}
    ]

def test_calculate_vwap(mock_candles):
    indicators = Indicators()
    candle_history = {"BTC-USDT": mock_candles}
    vwap = indicators.calculate_vwap("BTC-USDT", candle_history)
    assert vwap is not None
    assert abs(vwap - 108.33) < 0.01  # Typical price: (110+90+105)/3 * 1000 + ...

def test_calculate_atr(mock_candles):
    indicators = Indicators()
    candle_history = {"BTC-USDT": mock_candles}
    atr = indicators.calculate_atr("BTC-USDT", candle_history, period=2)
    assert atr is not None
    assert atr > 0