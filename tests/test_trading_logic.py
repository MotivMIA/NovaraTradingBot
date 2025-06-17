import pytest
from features.trading_logic import TradingLogic

@pytest.fixture
def mock_bot():
    class MockBot:
        def __init__(self):
            self.candle_history = {"BTC-USDT": [
                {"close": 100, "high": 110, "low": 90, "volume": 1000},
                {"close": 105, "high": 115, "low": 95, "volume": 1200}
            ]}
            self.price_history = {"BTC-USDT": [100, 105]}
            self.indicators = type('MockIndicators', (), {
                'calculate_indicators': lambda *args, **kwargs: {'price': 105, 'rsi': 50},
                'calculate_atr': lambda *args, **kwargs: 5.0
            })()
            self.ml = type('MockML', (), {'predict_ml_signal': lambda *args, **kwargs: None})()
            self.sentiment = type('MockSentiment', (), {'get_x_sentiment': lambda *args, **kwargs: 0.0})()
            self.sentiment_cache = {"BTC-USDT": {"score": 0.0, "timestamp": 0}}
            self.api_utils = type('MockAPI', (), {'get_candles': lambda *args, **kwargs: None})()
    return MockBot()

def test_check_volatility(mock_bot):
    trading_logic = TradingLogic()
    is_volatile, volatility = trading_logic.check_volatility("BTC-USDT", 105, mock_bot.price_history)
    assert is_volatile
    assert abs(volatility - 0.05) < 0.01  # (105-100)/100 = 0.05