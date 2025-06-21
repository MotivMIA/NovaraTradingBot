import pandas as pd
import numpy as np
import sqlite3
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Backtesting:
    async def backtest_strategy(self, symbol: str, bot, lookback_days: int = 7):
        try:
            start_time = int(pd.Timestamp.now().timestamp() - lookback_days * 86400)
            conn = sqlite3.connect(bot.config.DB_PATH)
            table_name = symbol.replace("-", "_") + "_candles"
            df = pd.read_sql(f"SELECT * FROM {table_name} WHERE timestamp >= {start_time}", conn)
            conn.close()

            if df.empty:
                logger.info(f"No historical data for backtesting {symbol}")
                return None

            results = []
            for _ in range(100):  # Monte Carlo
                df_sim = df.copy()
                df_sim["close"] = df["close"] * (1 + np.random.normal(0, df["close"].pct_change().std(), len(df)))
                simulated_balance = bot.account_balance or bot.config.DEFAULT_BALANCE
                open_orders = {}
                for i in range(len(df_sim) - 1):
                    price = df_sim["close"].iloc[i]
                    signal_info = bot.trading_logic.generate_signal(symbol, price, bot)
                    if signal_info:
                        signal, confidence, patterns, timeframes = signal_info
                        price_change = abs(price - df_sim["close"].iloc[i-1]) / df_sim["close"].iloc[i-1] if i > 0 else bot.config.VOLATILITY_THRESHOLD
                        risk_amount = simulated_balance * bot.config.RISK_PER_TRADE
                        margin_amount, leverage = bot.portfolio.calculate_margin(symbol, risk_amount, confidence, price_change, patterns, bot)
                        size = margin_amount / price
                        atr = bot.indicators.calculate_indicators(symbol, bot.candle_history[symbol])["atr"]
                        stop_loss = price * (0.99 if signal == "buy" else 1.01)
                        open_orders[i] = {
                            "entry_price": price,
                            "size": size,
                            "leverage": leverage,
                            "side": signal,
                            "trailing_stop": stop_loss,
                            "entry_time": df_sim["timestamp"].iloc[i]
                        }
                    for idx, order in list(open_orders.items()):
                        new_stop = (order["entry_price"] - atr * bot.config.TRAILING_STOP_MULTIPLIER if order["side"] == "buy" else
                                    order["entry_price"] + atr * bot.config.TRAILING_STOP_MULTIPLIER)
                        order["trailing_stop"] = max(new_stop, order["trailing_stop"]) if order["side"] == "buy" else min(new_stop, order["trailing_stop"])
                        next_price = df_sim["close"].iloc[i+1]
                        if (order["side"] == "buy" and next_price <= order["trailing_stop"]) or (order["side"] == "sell" and next_price >= order["trailing_stop"]):
                            pl = (next_price - order["entry_price"]) * order["size"] if order["side"] == "buy" else (order["entry_price"] - next_price) * order["size"]
                            simulated_balance += pl
                            del open_orders[idx]
                results.append(simulated_balance)
            final_balance = np.percentile(results, 50)
            logger.info(f"Backtest result for {symbol}: Median balance ${final_balance:.2f}")
            return {"final_balance": final_balance}
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}")
            return None