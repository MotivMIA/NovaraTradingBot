# Dynamic symbol selection, portfolio allocation
import pandas as pd
import requests
from logging import getLogger

logger = getLogger(__name__)

class PortfolioManagement:
    def select_top_symbols(self, candle_history: dict, api_utils: APIUtils) -> list:
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
                candles = api_utils.get_candles(symbol, limit=50)
                if candles:
                    df_candles = pd.DataFrame(candles)
                    df_candles["tr"] = df_candles[["high", "low", "close"]].apply(
                        lambda x: max(x["high"] - x["low"], abs(x["high"] - x["close"].shift(1)), abs(x["low"] - x["close"].shift(1))), axis=1
                    )
                    df.loc[df["instId"] == symbol, "atr"] = df_candles["tr"].rolling(window=14).mean().iloc[-1]
            
            df["score"] = 0.5 * df["price_change"] + 0.3 * df["vol24h"] / df["vol24h"].max() - 0.2 * df["atr"] / df["atr"].max()
            top_symbols = df[df["vol24h"] > 1_000_000]["instId"].nlargest(MAX_SYMBOLS).tolist()
            logger.info(f"Selected top symbols: {top_symbols}")
            api_utils.notifications.send_webhook_alert(f"Updated top symbols: {top_symbols}")
            return top_symbols
        except Exception as e:
            logger.error(f"Failed to select top symbols: {e}")
            return ["BTC-USDT", "ETH-USDT", "XRP-USDT"]

    def calculate_portfolio_allocation(self, symbols: list, candle_history: dict) -> dict:
        allocations = {symbol: RISK_PER_TRADE / len(symbols) for symbol in symbols}
        try:
            correlations = {}
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    df1 = pd.DataFrame(candle_history.get(symbol1, []))["close"]
                    df2 = pd.DataFrame(candle_history.get(symbol2, []))["close"]
                    if len(df1) > 14 and len(df2) > 14:
                        corr = df1.tail(14).corr(df2.tail(14))
                        if corr > CORRELATION_THRESHOLD:
                            correlations[(symbol1, symbol2)] = corr
            
            atrs = {symbol: self.calculate_atr(symbol, candle_history) for symbol in symbols}
            total_inverse_atr = sum(1 / atr if atr > 0 else 1 for atr in atrs.values())
            for symbol in symbols:
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