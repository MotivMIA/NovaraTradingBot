import pytest
from features.indicators import Indicators
from features.trading_logic import TradingLogic

@pytest.fixture
def mock_bot():
    class MockBot:
        def __init__(self):
            self.candle_history = {"BTC-USDT": [
                {"timestamp": 1, "open": 100, "high": 110, "low": 90, "close": 105, "volume": 1000},
                {"timestamp": 2, "open": 105, "high": 115, "low": 95, "close": 110, "volume": 1200}
            ]}
            self.price_history = {"BTC-USDT": [100, 105]}
            self.indicators = Indicators()
            self.trading_logic = TradingLogic()
            self.api_utils = type('MockAPI', (), {'get_candles': lambda *args, **kwargs: None})()
    return MockBot()

def test_generate_signal(mock_bot):
    signal_info = mock_bot.trading_logic.generate_signal("BTC-USDT", 105, mock_bot)
    assert signal_info is None or isinstance(signal_info, tuple)