import pytest
import logging
from unittest.mock import Mock
from features.trading_logic import TradingLogic
from features.indicators import Indicators
from features.config import MIN_PRICE_POINTS, TIMEFRAMES

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def mock_bot():
    class MockBot:
        def __init__(self):
            self.price_history = {"BTC-USDT": [100, 101, 102, 103, 104]}
            self.candle_history = {
                "BTC-USDT": [
                    {"timestamp": 1, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
                    {"timestamp": 2, "open": 100, "high": 102, "low": 100, "close": 101, "volume": 1200},
                    {"timestamp": 3, "open": 101, "high": 103, "low": 101, "close": 102, "volume": 1100},
                    {"timestamp": 4, "open": 102, "high": 104, "low": 102, "close": 103, "volume": 1300},
                    {"timestamp": 5, "open": 103, "high": 105, "low": 103, "close": 104, "volume": 1400},
                ]
            }
            self.trading_logic = TradingLogic()
            self.indicators = Indicators()
            self.ml = Mock()
            self.ml.is_model_trained = {"BTC-USDT": False}
            self.ml.predict_signal = Mock(return_value=(None, 0.0))
            self.config = type("Config", (), {"MIN_PRICE_POINTS": MIN_PRICE_POINTS})

    return MockBot()

def test_generate_signal(mock_bot):
    signal_info = mock_bot.trading_logic.generate_signal("BTC-USDT", 105, mock_bot)
    assert signal_info is None or isinstance(signal_info, tuple), f"Expected None or tuple, got {signal_info}"