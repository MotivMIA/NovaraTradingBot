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
        {"timestamp": 1, "open": 110, "high": 112, "low": 108, "close": 109, "volume": 1000},  # Bearish
        {"timestamp": 2, "open": 109, "high": 109.5, "low": 108.5, "close": 109, "volume": 1000},  # Doji-like
        {"timestamp": 3, "open": 109, "high": 111, "low": 108, "close": 110.5, "volume": 1000}  # Bullish
    ]

def test_detect_doji(candlesticks, sample_candles):
    result = candlesticks.detect_patterns("BTC-USDT", sample_candles)
    assert result is not None
    assert "doji" in result["patterns"]
    assert result["confidence"] > 0

def test_detect_morning_star(candlesticks, sample_candles):
    result = candlesticks.detect_patterns("BTC-USDT", sample_candles)
    assert result is not None
    assert "morning_star" in result["patterns"]
    assert result["confidence"] > 0