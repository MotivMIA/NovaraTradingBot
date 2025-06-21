import pytest
import pandas as pd
from features.indicators import Indicators
from config import Config

@pytest.fixture
def indicators():
    return Indicators()

@pytest.fixture
def sample_candles():
    return [
        {"timestamp": i, "open": 100 + i, "high": 105 + i, "low": 95 + i, "close": 102 + i, "volume": 1000}
        for i in range(100)
    ]

def test_calculate_indicators(indicators, sample_candles):
    config = Config()
    result = indicators.calculate_indicators("BTC-USDT", sample_candles)
    assert "rsi" in result
    assert "macd" in result
    assert "vwap" in result
    assert result["rsi"] > config.RSI_OVERSOLD
    assert result["rsi"] < config.RSI_OVERBOUGHT