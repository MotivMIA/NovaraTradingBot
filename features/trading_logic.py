import logging
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingLogic:
    def generate_signal(self, symbol: str, price: float, bot) -> Optional[Tuple[str, float, List[str], List[str]]]:
        try:
            candles = bot.candle_history[symbol]
            if len(candles) < 50:
                return None

            indicators = bot.indicators.calculate_indicators(symbol, candles)
            patterns = []
            confidence = 0.0
            timeframes = ["1h"]

            if hasattr(bot, "strategy"):
                signal, strat_confidence, strat_patterns = bot.strategy.generate_signal(symbol, price, indicators)
                confidence += strat_confidence * 0.4
                patterns.extend(strat_patterns)

            if hasattr(bot, "ml"):
                ml_signal, ml_confidence = bot.ml.predict_signal(symbol, price, bot)
                if ml_signal == signal:
                    confidence += ml_confidence * 0.3
                    patterns.append("ml_confirmed")

            if hasattr(bot, "sentiment"):
                sentiment_score = (bot.sentiment.get_x_sentiment(symbol) * bot.config.SENTIMENT_WEIGHT +
                                  bot.sentiment.get_news_sentiment(symbol) * 0.1 +
                                  bot.sentiment.get_onchain_sentiment(symbol) * 0.05)
                if sentiment_score > 0.3:
                    patterns.append("bullish_sentiment")
                    confidence += 0.2
                elif sentiment_score < -0.3:
                    patterns.append("bearish_sentiment")
                    confidence += 0.2

            if bot.bot_name in ["TrendBot", "PatternBot"]:
                if (indicators["tenkan"] > indicators["kijun"] and price > indicators["senkou_a"] and
                    indicators["adx"] > 25):
                    signal = "buy"
                    confidence += 0.3
                    patterns.append("ichimoku_bullish")
                elif (indicators["tenkan"] < indicators["kijun"] and price < indicators["senkou_b"] and
                      indicators["adx"] > 25):
                    signal = "sell"
                    confidence += 0.3
                    patterns.append("ichimoku_bearish")

            if bot.bot_name == "VolumeBot":
                price_change = abs(price - candles[-2]["close"]) / candles[-2]["close"]
                if price_change > bot.config.VOLATILITY_THRESHOLD and indicators["adx"] > 20:
                    signal = "buy" if price > candles[-2]["close"] else "sell"
                    confidence += 0.3
                    patterns.append("volatility_breakout")

            if bot.bot_name == "NewsBot":
                news_score = bot.sentiment.get_news_sentiment(symbol)
                if news_score > 0.5:
                    signal = "buy"
                    confidence += 0.4
                    patterns.append("news_bullish")
                elif news_score < -0.5:
                    signal = "sell"
                    confidence += 0.4
                    patterns.append("news_bearish")

            if bot.bot_name == "PatternBot" and hasattr(bot, "candlesticks"):
                candle_patterns = bot.candlesticks.detect_patterns(symbol, candles)
                if candle_patterns:
                    if "bullish_engulfing" in candle_patterns["patterns"] or "doji" in candle_patterns["patterns"]:
                        signal = "buy"
                        confidence += candle_patterns["confidence"]
                        patterns.extend(candle_patterns["patterns"])
                    elif "bearish_engulfing" in candle_patterns["patterns"]:
                        signal = "sell"
                        confidence += candle_patterns["confidence"]
                        patterns.extend(candle_patterns["patterns"])

            if confidence > 0.5 and signal:
                logger.info(f"Signal generated for {symbol}: {signal} with confidence {confidence:.2f}")
                return signal, min(confidence, 0.9), patterns, timeframes
            return None
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def process_trade(self, symbol: str, signal: str, price: float, size: float, leverage: float, bot):
        try:
            atr = bot.indicators.calculate_indicators(symbol, bot.candle_history[symbol])["atr"]
            stop_loss = price * (0.99 if signal == "buy" else 1.01)
            take_profit = price * (1.02 if signal == "buy" else 0.98)
            trailing_stop = atr * bot.config.TRAILING_STOP_MULTIPLIER
            return {
                "symbol": symbol,
                "signal": signal,
                "price": price,
                "size": size,
                "leverage": leverage,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop": trailing_stop
            }
        except Exception as e:
            logger.error(f"Error processing trade for {symbol}: {e}")
            return None