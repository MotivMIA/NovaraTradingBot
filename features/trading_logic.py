import numpy as np
from logging import getLogger
from features.indicators import Indicators
from features.config import VOLATILITY_THRESHOLD, RSI_OVERSOLD, RSI_OVERBOUGHT, MIN_PRICE_POINTS, TIMEFRAMES

logger = getLogger(__name__)

class TradingLogic:
    def __init__(self):
        self.indicators = Indicators()

    def generate_signal(self, symbol: str, current_price: float, bot):
        logger.debug(f"Generating signal for {symbol} at price ${current_price:.2f}")
        logger.debug(f"Price history length: {len(bot.price_history[symbol])}, Candle history length: {len(bot.candle_history[symbol])}")
        if len(bot.price_history[symbol]) < MIN_PRICE_POINTS or len(bot.candle_history[symbol]) < MIN_PRICE_POINTS:
            logger.warning(f"Insufficient data for {symbol}: {len(bot.price_history[symbol])} points")
            return None
        
        indicators = {}
        for tf in TIMEFRAMES:
            candles = bot.candle_history[symbol][-50:] if len(bot.candle_history[symbol]) >= 50 else bot.candle_history[symbol]
            if candles:
                indicators[tf] = bot.indicators.calculate_indicators(symbol, candles)
                logger.debug(f"{symbol} ({tf}) indicators - RSI: {indicators[tf].get('rsi', 0):.2f}")
        
        volatility_passed, price_change = self.check_volatility(symbol, current_price, bot.price_history)
        if not volatility_passed:
            logger.debug(f"Volatility check failed for {symbol}: {price_change:.4f}")
            return None
        
        signals = []
        confidences = []
        patterns = []
        used_timeframes = []
        
        for tf, ind in indicators.items():
            rsi = ind.get("rsi", 50.0)
            vwap = ind.get("vwap", current_price)
            if rsi < RSI_OVERSOLD and current_price < vwap:
                signals.append("buy")
                confidences.append(0.7)
                patterns.append("oversold")
                used_timeframes.append(tf)
            elif rsi > RSI_OVERBOUGHT and current_price > vwap:
                signals.append("sell")
                confidences.append(0.7)
                patterns.append("overbought")
                used_timeframes.append(tf)
        
        if bot.ml.is_model_trained[symbol]:
            ml_signal, ml_confidence = bot.ml.predict_signal(symbol, current_price, bot)
            if ml_signal:
                signals.append(ml_signal)
                confidences.append(ml_confidence)
                patterns.append("ml")
                used_timeframes.append("ml")
        
        if not signals:
            logger.debug(f"No valid signal for {symbol}: confidence too low")
            return None
        
        final_signal = max(set(signals), key=signals.count)
        final_confidence = np.mean(confidences) if confidences else 0.3
        if final_confidence < 0.5:
            logger.debug(f"No valid signal for {symbol}: confidence {final_confidence:.2f}")
            return None
        
        return final_signal, final_confidence, patterns, used_timeframes

    def check_volatility(self, symbol: str, current_price: float, price_history: dict) -> tuple[bool, float]:
        try:
            prices = price_history[symbol][-10:]
            if len(prices) < 2:
                logger.debug(f"Insufficient price data for volatility check: {len(prices)}")
                return False, 0.0
            price_change = abs(current_price - prices[-2]) / prices[-2] if prices[-2] != 0 else 0.0
            if price_change >= VOLATILITY_THRESHOLD:
                logger.debug(f"Volatility check passed for {symbol}: {price_change:.4f}")
                return True, price_change
            logger.debug(f"Volatility check failed for {symbol}: {price_change:.4f}")
            return False, price_change
        except Exception as e:
            logger.error(f"Volatility check failed for {symbol}: {e}")
            return False, 0.0

    async def process_trade(self, symbol: str, price: float, signal: str, price_change: float, confidence: float, patterns: list, timeframes: list, bot):
        try:
            risk_amount = bot.account_balance * RISK_PER_TRADE
            margin, leverage = bot.portfolio.calculate_margin(symbol, risk_amount, confidence, price_change, patterns, bot)
            size = margin / price
            logger.info(f"Processing {signal} trade for {symbol}: size {size:.4f}, leverage {leverage:.2f}x")
            bot.analytics.log_trade(symbol, int(time.time()), 0, price, 0, size, leverage, {}, timeframes, signal, bot)
        except Exception as e:
            logger.error(f"Failed to process trade for {symbol}: {e}")

    def manage_cost_averaging(self, symbol: str, current_price: float, bot):
        try:
            position = bot.open_orders.get(symbol, {})
            if not position:
                return
            entry_price = position.get("entry_price", current_price)
            size = position.get("size", 0.0)
            dip_threshold = entry_price * (1 - COST_AVERAGE_DIP)
            if current_price <= dip_threshold and position.get("cost_average_count", 0) < COST_AVERAGE_LIMIT:
                new_size = size * 0.5
                logger.info(f"Cost averaging for {symbol}: adding {new_size:.4f} at ${current_price:.2f}")
                bot.notifications.send_webhook_alert(f"Cost-averaged {symbol}: {new_size:.4f} at ${current_price:.2f}")
                position["cost_average_count"] = position.get("cost_average_count", 0) + 1
        except Exception as e:
            logger.error(f"Failed to manage cost averaging for {symbol}: {e}")

    def manage_trailing_stop(self, symbol: str, current_price: float, bot):
        try:
            position = bot.open_orders.get(symbol, {})
            if not position:
                return
            atr = bot.indicators.calculate_atr(symbol, bot.candle_history)
            trailing_stop = position.get("trailing_stop", current_price)
            if position["side"] == "buy":
                new_stop = current_price - atr * TRAILING_STOP_MULTIPLIER
                position["trailing_stop"] = max(new_stop, trailing_stop)
            else:
                new_stop = current_price + atr * TRAILING_STOP_MULTIPLIER
                position["trailing_stop"] = min(new_stop, trailing_stop)
            logger.debug(f"Updated trailing stop for {symbol}: ${position['trailing_stop']:.2f}")
        except Exception as e:
            logger.error(f"Failed to manage trailing stop for {symbol}: {e}")