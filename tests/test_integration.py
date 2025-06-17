import pytest
from features.indicators import Indicators
from features.trading_logic import TradingLogic
from features.machine_learning import MachineLearning
from features.sentiment_analysis import SentimentAnalysis

@pytest.fixture
def mock_bot():
    class MockBot:
        def __init__(self):
            self.candle_history = {"BTC-USDT": [
                {"timestamp": 1, "open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000},
                {"timestamp": 2, "open": 105, "high": 115, "low": 95, "close": 110, "volume": 1200}
            ]}
            self.price_history = {"BTC-USDT": [100, 105]}
            self.volume_history = {"BTC-USDT": [1000, 1200]}  # Added volume_history
            self.indicators = Indicators()
            self.trading_logic = TradingLogic()
            self.ml = MachineLearning()
            self.sentiment = SentimentAnalysis()
            self.sentiment_cache = {"BTC-USDT": {"score": 0.0, "timestamp": 0}}
            self.api_utils = type('MockAPI', (), {'get_candles': lambda *args, **kwargs: None})()
            self.is_model_trained = {"BTC-USDT": False}
            self.last_model_train = {"BTC-USDT": 0}
    return MockBot()

def test_generate_signal(mock_bot):
    signal_info = mock_bot.trading_logic.generate_signal("BTC-USDT", 105, mock_bot)
    assert signal_info is None or isinstance(signal_info, tuple)