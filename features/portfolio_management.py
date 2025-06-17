# Dynamic symbol selection, portfolio allocation
import pandas as pd
import requests
from logging import getLogger
from features.config import MAX_SYMBOLS, CORRELATION_THRESHOLD, RISK_PER_TRADE, MAX_LEVERAGE

logger = getLogger(__name__)

class PortfolioManagement:
    def select_top_symbols(self, bot) -> list:
        path = "/api/v1/market/tickers"
        try:
            response = requests.get(f"{BASE_URL}{path}", timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("code") != "0" or not data.get("data"):
                logger.error(f"Failed to fetch tickers: {data}")
                return ["BTC-USDT", "ETH-USDT", "XRP-USDT"]
            
            df = pd.DataFrame(data["data"])
            df["vol24h"] = df["vol24h"].astype(float)
            df["price_change"] = (df["last"].astype(float) - df["open24h"].astype(float)) / df["open24h"].astype(float)
            df["atr"] = 0.0
            for symbol in df["instId"]:
                candles = bot.api_utils.get_candles(symbol, limit=50)
                if candles:
                    df_candles = pd.DataFrame(candles)
                    df_candles["tr"] = df_candles[["high", "low", "close"]].apply(
                        lambda x: max(x["high"] - x["low"], abs(x["high"] - x["close"].shift(1)), abs(x["low"] - x["close"].shift(1))), axis=1
                    )
                    df.loc[df["instId"] == symbol, "atr"] = df_candles["tr"].rolling(window=14).mean().iloc[-1]
            
            df["score"] = 0.5 * df["price_change"] + 0.3 * df["vol24h"] / df["vol24h"].max() - 0.2 * df["atr"] / df["atr"].max()
            top_symbols = df[df["vol24h"] > 1_000_000]["instId"].nlargest(MAX_SYMBOLS).tolist()
            logger.info(f"Selected top symbols: {top_symbols}")
            bot.notifications.send_webhook_alert(f"Updated top symbols: {top_symbols}")
            return top_symbols
        except Exception as e:
            logger.error(f"Failed to select top symbols: {e}")
            return ["BTC-USDT", "ETH-USDT", "XRP-USDT"]

    def calculate_portfolio_allocation(self, bot) -> dict:
        allocations = {symbol: RISK_PER_TRADE / len(bot.symbols) for symbol in bot.symbols}
        try:
            correlations = {}
            for i, symbol1 in enumerate(bot.symbols):
                for symbol2 in bot.symbols[i+1:]:
                    df1 = pd.DataFrame(bot.candle_history.get(symbol1, []))["close"]
                    df2 = pd.DataFrame(bot.candle_history.get(symbol2, []))["close"]
                    if len(df1) > 14 and len(df2) > 14:
                        corr = df1.tail(14).corr(df2.tail(14))
                        if corr > CORRELATION_THRESHOLD:
                            correlations[(symbol1, symbol2)] = corr
            
            atrs = {symbol: bot.indicators.calculate_atr(symbol, bot.candle_history) for symbol in bot.symbols}
            total_inverse_atr = sum(1 / atr if atr > 0 else 1 for atr in atrs.values())
            for symbol in bot.symbols:
                atr = atrs[symbol] if atrs[symbol] > 0 else 1
                allocations[symbol] = (1 / atr / total_inverse_atr) * RISK_PER_TRADE
            
            for (s1, s2), corr in correlations.items():
                total_alloc = allocations[s1] + allocations[s2]
                allocations[s1] = total_alloc * 0.6
                allocations[s2] = total_alloc * 0.4
            
            logger.debug(f"Portfolio allocations: {allocations}")
            return allocations
        except Exception as e:
            logger.error(f"Failed to calculate portfolio allocations: {e}")
            return allocations

    def calculate_margin(self, symbol: str, size_usd: float, confidence: float, volatility: float, patterns: dict, bot) -> tuple[float, float]:
        max_leverage = bot.leverage_info.get(symbol) or bot.api_utils.get_max_leverage(symbol, bot)
        risk_amount = size_usd
        
        leverage = 1.0
        leverage_percentage = 0.0
        if confidence > 0.7 or patterns.get("bullish_engulfing") or patterns.get("bearish_engulfing") or patterns.get("bullish_pin") or patterns.get("bearish_pin"):
            leverage_percentage = 0.5 + (confidence - 0.7) * 1.5
            leverage = max_leverage * min(leverage_percentage, 0.8)
        elif confidence > 0.5 or patterns.get("doji") or patterns.get("inside_bar"):
            leverage = min(max_leverage, 3.0)
            leverage_percentage = leverage / max_leverage if max_leverage > 0 else 1.0
        if volatility > 2 * VOLATILITY_THRESHOLD:
            leverage = min(leverage, 2.0)
            leverage_percentage = leverage / max_leverage if max_leverage > 0 else 1.0
        
        if confidence <= 0.7:
            leverage = min(leverage, MAX_LEVERAGE)
        
        margin_amount = risk_amount * leverage * 0.8
        logger.debug(f"Calculated margin for {symbol}: ${margin_amount:.2f} at {leverage:.2f}x leverage ({leverage_percentage*100:.1f}% of {max_leverage}x)")
        return margin_amount, leverage