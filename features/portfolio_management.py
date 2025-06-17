import pandas as pd
import numpy as np
import requests
from logging import getLogger
from features.config import BASE_URL, MAX_SYMBOLS, CORRELATION_THRESHOLD, RISK_PER_TRADE, MAX_LEVERAGE, DEFAULT_BALANCE, VOLATILITY_THRESHOLD

logger = getLogger(__name__)

class PortfolioManagement:
    def select_top_symbols(self, bot) -> list[str]:
        try:
            path = "/api/v1/market/tickers"
            response = requests.get(f"{BASE_URL}{path}", timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("code") != "0" or not data.get("data"):
                logger.error(f"Failed to fetch market tickers: {data}")
                return bot.symbols
            
            df = pd.DataFrame(data["data"])
            df["volume"] = df["vol24h"].astype(float)
            df["volatility"] = (df["high24h"].astype(float) - df["low24h"].astype(float)) / df["open24h"].astype(float)
            df = df[df["instId"].str.endswith("-USDT")]
            df = df[df["volume"] > df["volume"].quantile(0.75)]
            df = df[df["volatility"] > df["volatility"].quantile(0.5)]
            
            top_symbols = df.sort_values(by="volume", ascending=False)["instId"].head(MAX_SYMBOLS).tolist()
            correlation_matrix = self.calculate_correlations(top_symbols, bot)
            if correlation_matrix is None:
                return top_symbols
            
            selected = []
            for symbol in top_symbols:
                if len(selected) >= MAX_SYMBOLS:
                    break
                if all(correlation_matrix.loc[symbol, s] < CORRELATION_THRESHOLD for s in selected):
                    selected.append(symbol)
            
            logger.info(f"Selected top symbols: {selected}")
            return selected or bot.symbols
        except Exception as e:
            logger.error(f"Failed to select top symbols: {e}")
            return bot.symbols

    def calculate_correlations(self, symbols: list[str], bot) -> pd.DataFrame | None:
        try:
            price_data = {}
            for symbol in symbols:
                candles = bot.api_utils.get_candles(symbol, limit=100)
                if candles and len(candles) >= 50:
                    price_data[symbol] = [candle["close"] for candle in candles]
                else:
                    logger.warning(f"Insufficient candle data for {symbol}")
                    return None
            
            df = pd.DataFrame(price_data)
            if df.empty or len(df) < 50:
                logger.warning("Insufficient data for correlation calculation")
                return None
            
            return df.pct_change().corr()
        except Exception as e:
            logger.error(f"Failed to calculate correlations: {e}")
            return None

    def calculate_margin(self, symbol: str, risk_amount: float, confidence: float, price_change: float, patterns: list[str], bot) -> tuple[float, float]:
        try:
            max_leverage = bot.api_utils.get_max_leverage(symbol, bot)
            base_margin = risk_amount / max_leverage
            confidence_factor = min(confidence, 1.0)
            volatility_factor = min(price_change / VOLATILITY_THRESHOLD, 2.0)
            pattern_factor = 1.0 + (0.1 * len(patterns))
            
            adjusted_margin = base_margin * confidence_factor * volatility_factor * pattern_factor
            leverage = min(risk_amount / adjusted_margin, max_leverage)
            if leverage > MAX_LEVERAGE:
                leverage = MAX_LEVERAGE
                adjusted_margin = risk_amount / leverage
            
            logger.debug(f"Margin for {symbol}: ${adjusted_margin:.2f}, Leverage: {leverage:.2f}x")
            return adjusted_margin, leverage
        except Exception as e:
            logger.error(f"Failed to calculate margin for {symbol}: {e}")
            return risk_amount, 1.0

    def monitor_drawdown(self, bot) -> bool:
        try:
            current_balance = bot.account_balance or DEFAULT_BALANCE
            initial_balance = bot.initial_balance or DEFAULT_BALANCE
            drawdown = (initial_balance - current_balance) / initial_balance
            logger.debug(f"Current drawdown: {drawdown:.2%}")
            if drawdown > bot.config.MAX_DRAWDOWN:
                logger.error(f"Max drawdown exceeded: {drawdown:.2%} > {bot.config.MAX_DRAWDOWN:.2%}")
                bot.notifications.send_webhook_alert(f"Max drawdown exceeded: {drawdown:.2%}")
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to monitor drawdown: {e}")
            return True

    def rebalance_portfolio(self, bot):
        try:
            total_balance = bot.account_balance or DEFAULT_BALANCE
            risk_per_symbol = total_balance * RISK_PER_TRADE / len(bot.symbols)
            for symbol in bot.symbols:
                position = bot.open_orders.get(symbol, {})
                if not position:
                    continue
                current_size = position.get("size", 0.0)
                current_price = bot.price_history[symbol][-1] if bot.price_history[symbol] else 0.0
                if not current_price:
                    continue
                target_size = risk_per_symbol / current_price
                size_diff = target_size - current_size
                if abs(size_diff) / target_size > 0.1:
                    logger.info(f"Rebalancing {symbol}: Adjusting size from {current_size:.4f} to {target_size:.4f}")
                    bot.notifications.send_webhook_alert(f"Rebalancing {symbol}: Size {size_diff:.4f}")
        except Exception as e:
            logger.error(f"Failed to rebalance portfolio: {e}")