from typing import Optional, Tuple, List, Dict
from logging import getLogger
from features.config import RSI_OVERSOLD, RSI_OVERBOUGHT, EMA_FAST, EMA_SLOW

logger = getLogger(__name__)

class Strategy:
    def __init__(self, name: str):
        self.name = name
    
    def generate_signal(self, symbol: str, price: float, bot, indicators: Dict) -> Optional[Tuple[str, float, str, str]]:
        pass

class MeanReversionStrategy(Strategy):
    def generate_signal(self, symbol: str, price: float, bot, indicators: Dict) -> Optional[Tuple[str, float, str, str]]:
        try:
            for tf, ind in indicators.items():
                rsi = ind.get("rsi", 50.0)
                vwap = ind.get("vwap", price)
                if rsi < RSI_OVERSOLD and price < vwap:
                    logger.debug(f"Mean reversion buy signal for {symbol} on {tf}")
                    return "buy", 0.8, "mean_reversion_oversold", tf
                elif rsi > RSI_OVERBOUGHT and price > vwap:
                    logger.debug(f"Mean reversion sell signal for {symbol} on {tf}")
                    return "sell", 0.8, "mean_reversion_overbought", tf
            return None
        except Exception as e:
            logger.error(f"Mean reversion signal failed for {symbol}: {e}")
            return None

class MomentumStrategy(Strategy):
    def generate_signal(self, symbol: str, price: float, bot, indicators: Dict) -> Optional[Tuple[str, float, str, str]]:
        try:
            for tf, ind in indicators.items():
                ema_fast = ind.get("ema_fast", price)
                ema_slow = ind.get("ema_slow", price)
                if ema_fast > ema_slow and price > ema_fast:
                    logger.debug(f"Momentum buy signal for {symbol} on {tf}")
                    return "buy", 0.75, "momentum_uptrend", tf
                elif ema_fast < ema_slow and price < ema_fast:
                    logger.debug(f"Momentum sell signal for {symbol} on {tf}")
                    return "sell", 0.75, "momentum_downtrend", tf
            return None
        except Exception as e:
            logger.error(f"Momentum signal failed for {symbol}: {e}")
            return None