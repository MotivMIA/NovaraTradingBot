# Signal generation, order placement, cost averaging, trailing stop-loss
import pandas as pd
import numpy as np
from logging import getLogger
from features.indicators import Indicators
from features.machine_learning import MachineLearning
from features.sentiment_analysis import SentimentAnalysis

logger = getLogger(__name__)

class TradingLogic:
    def check_volatility(self, symbol: str, price: float, price_history: dict) -> tuple[bool, float]:
        volatility = 0.0
        if len(price_history[symbol]) >= 2:
            volatility = abs(price - price_history[symbol][-2]) / price_history[symbol][-2]
            if volatility < VOLATILITY_THRESHOLD:
                logger.debug(f"Low volatility for {symbol}: {volatility:.4f}")
                return False, volatility
        logger.debug(f"Volatility check passed for {symbol}: {volatility:.4f}")
        return True, volatility

    def analyze_indicators(self, symbol, indicators):
        vwap = indicators.get("vwap")
        rsi = indicators.get("rsi")
        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        ema_fast = indicators.get("ema_fast")
        ema_slow = indicators.get("ema_slow")
        bb_upper = indicators.get("bb_upper")
        bb_lower = indicators.get("bb_lower")
        price = indicators.get("price")
        
        confidence = 0.0
        signal = None
        return signal, confidence, vwap, rsi, macd, macd_signal, ema_fast, ema_slow, bb_upper, bb_lower, price

    def analyze_patterns_and_indicators(self, patterns, symbol, price, vwap, confidence, signal, price_history):
        if patterns:
            previous_price = price_history[symbol][-2] if len(price_history[symbol]) >= 2 else price
            if patterns.get("bullish_engulfing") or patterns.get("bullish_pin") or (patterns.get("inside_bar") and price > previous_price):
                signal = "buy"
                confidence += 0.4
            elif patterns.get("bearish_engulfing") or patterns.get("bearish_pin") or (patterns.get("inside_bar") and price < previous_price):
                signal = "sell"
                confidence += 0.4
            elif patterns.get("doji") and vwap is not None:
                if price > vwap:
                    signal = "buy"
                    confidence += 0.3
                else:
                    signal = "sell"
                    confidence += 0.3
        return signal, confidence

    def analyze_vwap(self, vwap, price, signal, confidence):
        if vwap is not None:
            if price > vwap:
                if signal == "buy":
                    confidence += 0.3
                elif not signal:
                    signal = "buy"
                    confidence += 0.3
            elif price < vwap:
                if signal == "sell":
                    confidence += 0.3
                elif not signal:
                    signal = "sell"
                    confidence += 0.3
        return signal, confidence

    def check_indicators(self, symbol: str, macd: float, macd_signal: float, ema_fast: float, ema_slow: float, rsi: float, bb_upper: float, bb_lower: float, price: float, signal: str, confidence: float, candle_history: dict):
        if len(candle_history[symbol]) >= RSI_PERIOD:
            if macd is not None and macd_signal is not None and ema_fast is not None and ema_slow is not None:
                if macd > macd_signal and ema_fast > ema_slow:
                    if signal == "buy":
                        confidence += 0.2
                    elif not signal:
                        signal = "buy"
                        confidence += 0.2
                elif macd < macd_signal and ema_fast < ema_slow:
                    if signal == "sell":
                        confidence += 0.2
                    elif not signal:
                        signal = "sell"
                        confidence += 0.2
            
            if rsi is not None and bb_upper is not None and bb_lower is not None:
                if rsi < RSI_OVERSOLD and price <= bb_lower:
                    if signal == "buy":
                        confidence += 0.2
                    elif not signal:
                        signal = "buy"
                        confidence += 0.2
                elif rsi > RSI_OVERBOUGHT and price >= bb_upper:
                    if signal == "sell":
                        confidence += 0.2
                    elif not signal:
                        signal = "sell"
                        confidence += 0.2
        return signal, confidence

    def evaluate_signal(self, symbol, signal, confidence):
        if not signal or confidence < 0.5:
            logger.debug(f"No valid signal for {symbol}: confidence {confidence:.2f}")
            return None
        logger.info(f"Generated signal for {symbol}: {signal} with confidence {confidence:.2f}")
        return signal, confidence

    def generate_signal(self, symbol: str, current_price: float, bot) -> tuple[str, float, dict, list] | None:
        logger.debug(f"Generating signal for {symbol} at current_price: {current_price}")
        if len(bot.price_history[symbol]) < MIN_PRICE_POINTS or len(bot.candle_history[symbol]) < MIN_PRICE_POINTS:
            logger.warning(f"Skipping signal for {symbol}: insufficient data (price: {len(bot.price_history[symbol])}, candles: {len(bot.candle_history[symbol])})")
            return None
        
        indicators = bot.indicators.calculate_indicators(symbol, bot.candle_history, bot.api_utils.get_candles)
        patterns = {}  # Placeholder for price action patterns
        if not indicators:
            return None
        
        signal, confidence, vwap, rsi, macd, macd_signal, ema_fast, ema_slow, bb_upper, bb_lower, price = self.analyze_indicators(symbol, indicators)
        
        is_volatile, volatility = self.check_volatility(symbol, current_price, bot.price_history)
        if not is_volatile:
            return None
        
        timeframes_used = ["1m"]
        timeframe_weights = {"1m": 0.4, "5m": 0.2, "15m": 0.15, "1h": 0.15, "1d": 0.1}
        for tf in TIMEFRAMES[1:]:
            tf_indicators = bot.indicators.calculate_indicators(symbol, bot.candle_history, bot.api_utils.get_candles, timeframe=tf)
            if tf_indicators:
                tf_rsi = tf_indicators.get("rsi", 50.0)
                if tf_rsi < RSI_OVERSOLD and signal == "buy":
                    confidence += timeframe_weights[tf]
                    timeframes_used.append(tf)
                elif tf_rsi > RSI_OVERBOUGHT and signal == "sell":
                    confidence += timeframe_weights[tf]
                    timeframes_used.append(tf)
        
        signal, confidence = self.analyze_patterns_and_indicators(patterns, symbol, current_price, vwap, confidence, signal, bot.price_history)
        
        signal, confidence = self.analyze_vwap(vwap, current_price, signal, confidence)
        
        signal, confidence = self.check_indicators(symbol, macd, macd_signal, ema_fast, ema_slow, rsi, bb_upper, bb_lower, current_price, signal, confidence, bot.candle_history)
        
        ml_signal = bot.ml.predict_ml_signal(symbol, indicators, bot)
        if ml_signal:
            ml_side, ml_confidence = ml_signal
            if signal == ml_side:
                confidence += ml_confidence * 0.2
            elif not signal:
                signal = ml_side
                confidence += ml_confidence * 0.2
        
        sentiment_score = bot.sentiment.get_x_sentiment(symbol, bot.sentiment_cache)
        if signal == "buy" and sentiment_score > 0.3:
            confidence += SENTIMENT_WEIGHT
        elif signal == "sell" and sentiment_score < -0.3:
            confidence += SENTIMENT_WEIGHT
        
        signal_info = self.evaluate_signal(symbol, signal, confidence)
        if signal_info:
            return signal_info[0], signal_info[1], patterns or {}, timeframes_used
        return None

    def place_order(self, symbol: str, price: float, size_usd: float, side: str, confidence: float, patterns: dict, volatility: float, bot, max_retries: int = 3) -> str | None:
        logger.info(f"Attempting to place {side} order for {symbol}: ${size_usd:.2f} at ${price}")
        path = "/api/v1/trade/order"
        inst_info = bot.api_utils.get_instrument_info(symbol)
        if not inst_info:
            logger.error(f"Failed to get instrument info for {symbol}")
            return None
        
        min_size = inst_info["minSize"]
        lot_size = inst_info["lotSize"]
        contract_value = inst_info["contractValue"]
        
        atr = bot.indicators.calculate_atr(symbol, bot.candle_history)
        risk_multiplier = 1.0 if atr == 0 else max(0.5, min(2.0, 100 / atr))
        adjusted_size_usd = size_usd * risk_multiplier
        logger.debug(f"Adjusted size for {symbol}: ${adjusted_size_usd:.2f} based on ATR ${atr:.2f}")
        
        margin_amount, leverage = bot.portfolio.calculate_margin(symbol, adjusted_size_usd, confidence, volatility, patterns, bot)
        size = (margin_amount / price) / contract_value
        size = max(round(size / lot_size) * lot_size, min_size)
        logger.info(f"Calculated order size for {symbol}: {size} contracts at ${price} with {leverage:.2f}x leverage")
        
        stop_loss = price * (0.99 if side == "buy" else 1.01)
        take_profit = price * (1.02 if side == "buy" else 0.98)
        
        order_request = {
            "instId": symbol,
            "instType": "SWAP",
            "marginMode": "cross",
            "leverage": str(leverage),
            "positionSide": "net",
            "side": side,
            "orderType": "limit",
            "price": str(round(price, SIZE_PRECISION)),
            "size": str(size)
        }
        
        for attempt in range(max_retries):
            try:
                headers, _, _ = bot.api_utils.sign_request("POST", path, body=order_request)
                response = requests.post(f"{BASE_URL}{path}", headers=headers, json=order_request, timeout=5)
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Order response for {symbol}: {data}")
                if data.get("code") == "0" and data.get("data"):
                    order_id = data["data"][0]["orderId"]
                    logger.info(f"Placed {side} order for {symbol}: {size} contracts at ${price}, SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}, Leverage: {leverage:.2f}x")
                    bot.notifications.send_webhook_alert(f"Placed {side} order for {symbol}: ${size_usd:.2f} at ${price}, Leverage: {leverage:.2f}x")
                    bot.open_orders.setdefault(symbol, {})[order_id] = {
                        "entry_price": price,
                        "size": size,
                        "leverage": leverage,
                        "side": side,
                        "confidence": confidence,
                        "patterns": patterns,
                        "volatility": volatility,
                        "cost_average_count": 0,
                        "entry_time": int(time.time())
                    }
                    return order_id
                logger.error(f"Order failed for {symbol}: {data}")
                if data.get("code") == "152406":
                    logger.error("IP whitelisting issue, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                return None
            except requests.RequestException as e:
                logger.error(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    return None
        return None

    def manage_cost_averaging(self, symbol: str, current_price: float, bot):
        if symbol not in bot.open_orders:
            return
        
        for order_id, order in list(bot.open_orders[symbol].items()):
            if order["confidence"] < 0.7:
                continue
            entry_price = order["entry_price"]
            size = order["size"]
            count = order.get("cost_average_count", 0)
            if count >= COST_AVERAGE_LIMIT:
                continue
            dip_threshold = entry_price * (1 - COST_AVERAGE_DIP if order["side"] == "buy" else 1 + COST_AVERAGE_DIP)
            if (order["side"] == "buy" and current_price <= dip_threshold) or (order["side"] == "sell" and current_price >= dip_threshold):
                new_size = size * 0.5
                new_leverage = order["leverage"] * 0.5
                new_order_id = self.place_order(
                    symbol, current_price, new_size * current_price, order["side"],
                    order["confidence"], order["patterns"], order["volatility"], bot
                )
                if new_order_id:
                    bot.open_orders[symbol][new_order_id] = {
                        "entry_price": current_price,
                        "size": new_size,
                        "leverage": new_leverage,
                        "side": order["side"],
                        "confidence": order["confidence"],
                        "patterns": order["patterns"],
                        "volatility": order["volatility"],
                        "cost_average_count": count + 1,
                        "entry_time": int(time.time())
                    }
                    logger.info(f"Cost-averaged {order['side']} order for {symbol} at ${current_price}")

    def manage_trailing_stop(self, symbol: str, current_price: float, bot):
        if symbol not in bot.open_orders:
            return
        
        atr = bot.indicators.calculate_atr(symbol, bot.candle_history)
        for order_id, order in list(bot.open_orders[symbol].items()):
            if order["side"] == "buy":
                new_stop = max(order.get("trailing_stop", order["entry_price"] * 0.99), current_price - atr * TRAILING_STOP_MULTIPLIER)
                if new_stop > order.get("trailing_stop", 0):
                    order["trailing_stop"] = new_stop
                    logger.debug(f"Updated trailing stop for {symbol} buy order {order_id} to ${new_stop:.2f}")
                if current_price <= new_stop:
                    bot.analytics.log_trade(
                        symbol, order["entry_time"], int(time.time()), order["entry_price"],
                        current_price, order["size"], order["leverage"],
                        {"atr": atr, "sentiment": bot.sentiment.get_x_sentiment(symbol, bot.sentiment_cache), "multi_timeframe": True},
                        TIMEFRAMES, order["side"], bot
                    )
                    del bot.open_orders[symbol][order_id]
            else:
                new_stop = min(order.get("trailing_stop", order["entry_price"] * 1.01), current_price + atr * TRAILING_STOP_MULTIPLIER)
                if new_stop < order.get("trailing_stop", float("inf")):
                    order["trailing_stop"] = new_stop
                    logger.debug(f"Updated trailing stop for {symbol} sell order {order_id} to ${new_stop:.2f}")
                if current_price >= new_stop:
                    bot.analytics.log_trade(
                        symbol, order["entry_time"], int(time.time()), order["entry_price"],
                        current_price, order["size"], order["leverage"],
                        {"atr": atr, "sentiment": bot.sentiment.get_x_sentiment(symbol, bot.sentiment_cache), "multi_timeframe": True},
                        TIMEFRAMES, order["side"], bot
                    )
                    del bot.open_orders[symbol][order_id]

    async def process_trade(self, symbol: str, price: float, side: str, price_change: float, confidence: float, patterns: dict, timeframes: list, bot):
        if not bot.account_balance:
            bot.account_balance = bot.api_utils.get_account_balance(bot)
            if not bot.account_balance:
                logger.error("Failed to get account balance")
                return
            if not bot.initial_balance:
                bot.initial_balance = bot.account_balance
        
        if bot.account_balance < bot.initial_balance * (1 - MAX_DRAWDOWN):
            logger.error(f"Max drawdown reached for {symbol}: {bot.account_balance:.2f}/{bot.initial_balance:.2f}")
            return
        
        allocations = bot.portfolio.calculate_portfolio_allocation(bot)
        risk_amount = bot.account_balance * allocations.get(symbol, RISK_PER_TRADE / len(bot.symbols))
        logger.debug(f"Dynamic risk for {symbol}: ${risk_amount:.2f}")
        
        order_id = self.place_order(symbol, price, risk_amount, side, confidence, patterns, price_change, bot)
        if order_id:
            logger.info(f"Trade executed for {symbol}: {side} ${risk_amount:.2f} at ${price}")
            bot.account_balance -= risk_amount * 0.01