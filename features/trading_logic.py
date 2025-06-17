import numpy as np
import pandas as pd
from logging import getLogger
from typing import Optional, Tuple, List
from features.indicators import Indicators
from features.strategies import MeanReversionStrategy, MomentumStrategy
from features.config import VOLATILITY_THRESHOLD, RSI_OVERSOLD, RSI_OVERBOUGHT, MIN_PRICE_POINTS, TIMEFRAMES, SIZE_PRECISION

logger = getLogger(__name__)

class TradingLogic:
    def __init__(self):
        self.indicators = Indicators()
        self.strategies = [
            MeanReversionStrategy("mean_reversion"),
            MomentumStrategy("momentum")
        ]

    def generate_signal(self, symbol: str, price: float, bot) -> Optional[Tuple[str, float, List[str], List[str]]]:
        logger.debug(f"Generating signal for {symbol} at price ${price:.2f}")
        logger.debug(f"Price history length: {len(bot.price_history[symbol])}, Candle history length: {len(bot.candle_history[symbol])}")
        if len(bot.price_history[symbol]) < MIN_PRICE_POINTS or len(bot.candle_history[symbol]) < MIN_PRICE_POINTS:
            logger.warning(f"Insufficient data for {symbol}: {len(bot.price_history[symbol])} points")
            return None
        
        indicators = {}
        for tf in TIMEFRAMES:
            candles = bot.candle_history[symbol][-50:] if len(bot.candle_history[symbol]) >= 50 else bot.candle_history[symbol]
            if candles:
                ind = self.indicators.calculate_indicators(symbol, candles)
                if ind:
                    indicators[tf] = ind
                    logger.debug(f"{symbol} ({tf}) indicators - RSI: {ind.get('rsi', 0):.2f}, VWAP: {ind.get('vwap', 0):.2f}")
        
        volatility_passed, price_change = self.check_volatility(symbol, price, bot.price_history)
        if not volatility_passed:
            logger.debug(f"Volatility check failed for {symbol}: {price_change:.4f}")
            return None
        
        signals = []
        confidences = []
        patterns = []
        timeframes_used = []
        
        for strategy in self.strategies:
            signal_info = strategy.generate_signal(symbol, price, bot, indicators)
            if signal_info:
                signal, confidence, pattern, timeframe = signal_info
                signals.append(signal)
                confidences.append(confidence)
                patterns.append(pattern)
                timeframes_used.append(timeframe)
        
        if bot.is_model_trained.get(symbol, False):
            ml_signal, ml_confidence = bot.ml.predict_signal(symbol, {"price": price, "volume_change": 0.0, "price_lag1": price, "price_lag2": price}, bot)
            if ml_signal:
                signals.append(ml_signal)
                confidences.append(ml_confidence)
                patterns.append("ml")
                timeframes_used.append("ml")
        
        sentiment_score = bot.sentiment.get_x_sentiment(symbol, bot.sentiment_cache)
        if sentiment_score > 0.5:
            signals.append("buy")
            confidences.append(0.6)
            patterns.append("positive_sentiment")
            timeframes_used.append("sentiment")
        elif sentiment_score < -0.5:
            signals.append("sell")
            confidences.append(0.6)
            patterns.append("negative_sentiment")
            timeframes_used.append("sentiment")
        
        if not signals:
            logger.debug(f"No valid signal for {symbol}: no signals generated")
            return None
        
        final_signal = max(set(signals), key=signals.count)
        final_confidence = float(np.mean(confidences)) if confidences else 0.3
        if final_confidence < 0.5:
            logger.debug(f"No valid signal for {symbol}: confidence {final_confidence:.2f}")
            return None
        
        logger.info(f"Generated signal for {symbol}: {final_signal} with confidence {final_confidence:.2f}, patterns: {patterns}")
        return final_signal, final_confidence, patterns, timeframes_used

    def check_volatility(self, symbol: str, price: float, price_history: dict) -> Tuple[bool, float]:
        try:
            prices = price_history.get(symbol, [])[-50:]
            if len(prices) < 2:
                logger.debug(f"Insufficient price data for volatility check: {len(prices)}")
                return False, 0.0
            price_change = abs(price - prices[-2]) / prices[-2] if prices[-2] > 0 else 0.0
            if price_change >= VOLATILITY_THRESHOLD:
                logger.debug(f"Volatility check passed for {symbol}: {price_change:.4f}")
                return True, price_change
            logger.debug(f"Volatility check failed for {symbol}: {price_change:.4f}")
            return False, price_change
        except Exception as e:
            logger.error(f"Volatility check failed for {symbol}: {e}")
            return False, 0.0

    async def process_trade(self, symbol: str, price: float, signal: str, price_change: float, confidence: float, patterns: List[str], timeframes: List[str], bot):
        try:
            if not bot.portfolio.check_daily_loss(bot):
                logger.error(f"Trade aborted for {symbol}: Daily loss limit exceeded")
                return
            risk_amount = bot.account_balance * RISK_PER_TRADE
            margin, leverage = bot.portfolio.calculate_margin(symbol, risk_amount, confidence, price_change, patterns, bot)
            size = round(margin / price, SIZE_PRECISION) if price > 0 else 0.0
            if size <= 0:
                logger.error(f"Invalid trade size for {symbol}: {size:.4f}")
                return
            atr = self.indicators.calculate_atr(symbol, bot.candle_history)
            stop_loss = price * (0.99 if signal == "buy" else 1.01)
            take_profit = price * (1.02 if signal == "buy" else 0.98)
            logger.info(f"Processing {signal} trade for {symbol}: size={size:.4f}, leverage={leverage:.2f}x at ${price:.2f}, stop-loss=${stop_loss:.2f}, take-profit=${take_profit:.2f}")
            order_response = await bot.exchange.place_order(symbol, signal, size, price)
            if order_response:
                bot.analytics.log_trade(symbol, int(time.time()), 0, price, 0, size, leverage, {}, timeframes, signal, bot)
                bot.notifications.send_webhook_alert(f"{signal.upper()} trade for {symbol}: {size:.4f} at ${price:.2f}, leverage {leverage:.2f}x")
                bot.open_orders[symbol] = {
                    "side": signal,
                    "size": size,
                    "entry_price": price,
                    "leverage": leverage,
                    "cost_average_count": 0,
                    "trailing_stop": price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                }
        except Exception as e:
            logger.error(f"Failed to process trade for {symbol}: {e}")

    def manage_cost_averaging(self, symbol: str, price: float, bot):
        try:
            position = bot.open_orders.get(symbol, {})
            if not position:
                return
            entry_price = position.get("entry_price", price)
            size = position.get("size", 0.0)
            dip_threshold = entry_price * (1 - COST_AVERAGE_DIP)
            if price <= dip_threshold and position.get("cost_average_count", 0) < COST_AVERAGE_LIMIT:
                new_size = size * 0.5
                logger.info(f"Cost averaging for {symbol}: adding {new_size:.4f} at ${price:.2f}")
                bot.notifications.send_webhook_alert(f"Cost-averaged {symbol}: {new_size:.4f} at ${price:.2f}")
                position["cost_average_count"] += 1
                position["size"] += new_size
                position["entry_price"] = (entry_price * size + price * new_size) / (size + new_size)
        except Exception as e:
            logger.error(f"Failed to manage cost averaging for {symbol}: {e}")

    def manage_trailing_stop(self, symbol: str, price: float, bot):
        try:
            position = bot.open_orders.get(symbol, {})
            if not position:
                return
            atr = self.indicators.calculate_atr(symbol, bot.candle_history)
            trailing_stop = position.get("trailing_stop", price)
            if position["side"] == "buy":
                new_stop = price - atr * TRAILING_STOP_MULTIPLIER
                position["trailing_stop"] = max(new_stop, trailing_stop)
            else:
                new_stop = price + atr * TRAILING_STOP_MULTIPLIER
                position["trailing_stop"] = min(new_stop, trailing_stop)
            logger.debug(f"Updated trailing stop for {symbol}: ${position['trailing_stop']:.2f}")
        except Exception as e:
            logger.error(f"Failed to manage trailing stop for {symbol}: {e}")