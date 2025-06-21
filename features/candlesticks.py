import pandas as pd
import logging
from typing import List, Dict, Optional
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CandlestickPatterns:
    def __init__(self):
        self.config = Config()

    def detect_patterns(self, symbol: str, candles: List[Dict]) -> Optional[Dict]:
        try:
            df = pd.DataFrame(candles)
            if len(df) < 3:  # Need at least 3 candles for morning star
                return None

            patterns = []
            confidence = 0.0

            # Doji: Small body relative to range
            last_candle = df.iloc[-1]
            body = abs(last_candle["close"] - last_candle["open"])
            range_candle = last_candle["high"] - last_candle["low"]
            if range_candle > 0 and body / range_candle < 0.1:
                patterns.append("doji")
                confidence += 0.4

            # Bullish Engulfing
            prev_candle = df.iloc[-2]
            if (prev_candle["close"] < prev_candle["open"] and
                last_candle["close"] > last_candle["open"] and
                last_candle["open"] < prev_candle["close"] and
                last_candle["close"] > prev_candle["open"]):
                patterns.append("bullish_engulfing")
                confidence += 0.5

            # Bearish Engulfing
            if (prev_candle["close"] > prev_candle["open"] and
                last_candle["close"] < last_candle["open"] and
                last_candle["open"] > prev_candle["close"] and
                last_candle["close"] < prev_candle["open"]):
                patterns.append("bearish_engulfing")
                confidence += 0.5

            # Hammer: Small body, long lower wick, after downtrend
            if (len(df) >= 3 and
                df.iloc[-3:-1]["close"].mean() > df.iloc[-3:-1]["open"].mean() and
                body / range_candle < 0.3 and
                (last_candle["open"] - last_candle["low"]) / range_candle > 0.6):
                patterns.append("hammer")
                confidence += 0.45

            # Shooting Star: Small body, long upper wick, after uptrend
            if (len(df) >= 3 and
                df.iloc[-3:-1]["close"].mean() < df.iloc[-3:-1]["open"].mean() and
                body / range_candle < 0.3 and
                (last_candle["high"] - last_candle["open"]) / range_candle > 0.6):
                patterns.append("shooting_star")
                confidence += 0.45

            # Morning Star: Three-candle bullish reversal after downtrend
            if (len(df) >= 3 and
                df.iloc[-3]["close"] < df.iloc[-3]["open"] and  # Bearish first candle
                df.iloc[-2]["close"] < df.iloc[-2]["open"] and  # Small body (doji-like)
                abs(df.iloc[-2]["close"] - df.iloc[-2]["open"]) / (df.iloc[-2]["high"] - df.iloc[-2]["low"]) < 0.2 and
                last_candle["close"] > last_candle["open"] and  # Bullish third candle
                last_candle["close"] > df.iloc[-3]["open"] / 2 + df.iloc[-3]["close"] / 2):  # Closes above midpoint
                patterns.append("morning_star")
                confidence += 0.5

            if patterns:
                logger.info(f"Candlestick patterns for {symbol}: {patterns}")
                return {"patterns": patterns, "confidence": min(confidence, 0.8)}
            return None
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns for {symbol}: {e}")
            return None