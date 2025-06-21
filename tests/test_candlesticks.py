import pytest
import pandas as pd
from features.candlesticks import CandlestickPatterns
from config import Config

@pytest.fixture
def candlesticks():
    return CandlestickPatterns()

@pytest.fixture
def sample_candles():
    return [
        {"timestamp": 1, "open": 100, "high": 110, "low": 90, "close": 100, "volume": 1000},
        {"timestamp": 2, "open": 100, "high": 100.5, "low": 99.5, "close": 100, "volume": 1000}
    ]

def test_detect_doji(candlesticks, sample_candles):
    result = candlesticks.detect_patterns("BTC-USDT", sample_candles)
    assert result is not None
    assert "doji" in result["patterns"]
    assert result["confidence"] > 0