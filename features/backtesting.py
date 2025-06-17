# Backtesting logic
import sqlite3
import pandas as pd
import sqlite3
import numpy as np
from logging import getLogger
import time
from features.config import VOLATILITY_THRESHOLD, RISK_PER_TRADE, TRAILING_STOP_MULTIPLIER, COST_AVERAGE_DIP, DEFAULT_BALANCE

logger = getLogger(__name__)

class Backtesting:
    async def backtest_strategy(self, symbol: str, bot, lookback_days: int = 7):
        try:
            start_time = int(time.time() - lookback_days * 86400)
            conn = sqlite3.connect(DB_PATH)
            table_name = symbol.replace("-", "_") + "_candles"
            df = pd.read_sql(f"SELECT * FROM {table_name} WHERE timestamp >= {start_time}", conn)
            conn.close()
            if df.empty:
                logger.info(f"No historical data for backtesting {symbol}")
                return
            
            simulated_balance = bot.account_balance or DEFAULT_BALANCE
            open_orders = {}
            for i in range(len(df) - 1):
                price = df["close"].iloc[i]
                signal_info = bot.trading_logic.generate_signal(symbol, price, bot)
                if signal_info:
                    signal, confidence, patterns, timeframes = signal_info
                    price_change = abs(price - df["close"].iloc[i-1]) / df["close"].iloc[i-1] if i > 0 else VOLATILITY_THRESHOLD
                    risk_amount = simulated_balance * RISK_PER_TRADE
                    margin_amount, leverage = bot.portfolio.calculate_margin(symbol, risk_amount, confidence, price_change, patterns, bot)
                    size = margin_amount / price
                    atr = bot.indicators.calculate_atr(symbol, bot.candle_history)
                    stop_loss = price * (0.99 if signal == "buy" else 1.01)
                    open_orders[i] = {
                        "entry_price": price,
                        "size": size,
                        "leverage": leverage,
                        "side": signal,
                        "trailing_stop": stop_loss,
                        "entry_time": df["timestamp"].iloc[i]
                    }
                
                for idx, order in list(open_orders.items()):
                    new_stop = (order["entry_price"] - atr * TRAILING_STOP_MULTIPLIER if order["side"] == "buy" else
                                order["entry_price"] + atr * TRAILING_STOP_MULTIPLIER)
                    order["trailing_stop"] = max(new_stop, order["trailing_stop"]) if order["side"] == "buy" else min(new_stop, order["trailing_stop"])
                    next_price = df["close"].iloc[i+1]
                    if (order["side"] == "buy" and next_price <= order["trailing_stop"]) or (order["side"] == "sell" and next_price >= order["trailing_stop"]):
                        pl = (next_price - order["entry_price"]) * order["size"] if order["side"] == "buy" else (order["entry_price"] - next_price) * order["size"]
                        simulated_balance += pl
                        logger.debug(f"Backtest trade for {symbol}: {order['side']} at ${order['entry_price']}, P/L ${pl:.2f}")
                        del open_orders[idx]
                    elif (order["side"] == "buy" and next_price <= order["entry_price"] * (1 - COST_AVERAGE_DIP)) or (
                            order["side"] == "sell" and next_price >= order["entry_price"] * (1 + COST_AVERAGE_DIP)):
                        new_size = order["size"] * 0.5
                        open_orders[i + 0.5] = {
                            "entry_price": next_price,
                            "size": new_size,
                            "leverage": order["leverage"] * 0.5,
                            "side": order["side"],
                            "trailing_stop": next_price * (0.99 if order["side"] == "buy" else 1.01),
                            "entry_time": df["timestamp"].iloc[i+1]
                        }
            
            logger.info(f"Backtest result for {symbol}: Final balance ${simulated_balance:.2f}")
        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}")