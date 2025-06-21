import logging
from typing import Tuple, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeanReversionStrategy:
    def generate_signal(self, symbol: str, price: float, indicators: dict) -> Tuple[Optional[str], float, List[str]]:
        try:
            if indicators["rsi"] < 30 and price < indicators["bb_lower"]:
                return "buy", 0.7, ["rsi_oversold", "bb_lower"]
            elif indicators["rsi"] > 70 and price > indicators["bb_upper"]:
                return "sell", 0.7, ["rsi_overbought", "bb_upper"]
            return None, 0.0, []
        except Exception as e:
            logger.error(f"Error generating mean reversion signal for {symbol}: {e}")
            return None, 0.0, []

class MomentumStrategy:
    def generate_signal(self, symbol: str, price: float, indicators: dict) -> Tuple[Optional[str], float, List[str]]:
        try:
            if indicators["ema_fast"] > indicators["ema_slow"] and indicators["macd"] > indicators["signal_line"]:
                return "buy", 0.7, ["ema_crossover", "macd_bullish"]
            elif indicators["ema_fast"] < indicators["ema_slow"] and indicators["macd"] < indicators["signal_line"]:
                return "sell", 0.7, ["ema_crossunder", "macd_bearish"]
            return None, 0.0, []
        except Exception as e:
            logger.error(f"Error generating momentum signal for {symbol}: {e}")
            return None, 0.0, []