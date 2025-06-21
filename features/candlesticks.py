import pandas as pd
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CandlestickPatterns:
    def detect_patterns(self, symbol: str, candles: List[Dict]) -> Optional[Dict]:
        try:
            df = pd.DataFrame(candles)
            if len(df) < 2:
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

            if patterns:
                logger.info(f"Candlestick patterns for {symbol}: {patterns}")
                return {"patterns": patterns, "confidence": min(confidence, 0.8)}
            return None
        except Exception as e:
            logger.error(f"Error detecting candlestick patterns for {symbol}: {e}")
            return None