import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Indicators:
    def calculate_indicators(self, symbol: str, candles: list):
        try:
            df = pd.DataFrame(candles)
            df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
            df["rsi"] = self.calculate_rsi(df["close"])
            df["macd"], df["signal_line"] = self.calculate_macd(df["close"])
            df["ema_fast"] = df["close"].ewm(span=12, adjust=False).mean()
            df["ema_slow"] = df["close"].ewm(span=26, adjust=False).mean()
            df["bb_upper"], df["bb_lower"] = self.calculate_bollinger_bands(df["close"])
            df["atr"] = self.calculate_atr(df)
            df["ichimoku"] = self.calculate_ichimoku(df)
            df["adx"] = self.calculate_adx(df)
            indicators = {
                "vwap": df["vwap"].iloc[-1],
                "rsi": df["rsi"].iloc[-1],
                "macd": df["macd"].iloc[-1],
                "signal_line": df["signal_line"].iloc[-1],
                "ema_fast": df["ema_fast"].iloc[-1],
                "ema_slow": df["ema_slow"].iloc[-1],
                "bb_upper": df["bb_upper"].iloc[-1],
                "bb_lower": df["bb_lower"].iloc[-1],
                "atr": df["atr"].iloc[-1],
                "tenkan": df["ichimoku"].iloc[-1]["tenkan"],
                "kijun": df["ichimoku"].iloc[-1]["kijun"],
                "senkou_a": df["ichimoku"].iloc[-1]["senkou_a"],
                "senkou_b": df["ichimoku"].iloc[-1]["senkou_b"],
                "adx": df["adx"].iloc[-1]
            }
            return indicators
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return {}

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def calculate_bollinger_bands(self, prices, period=20, std=2):
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, lower

    def calculate_atr(self, df, period=14):
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calculate_ichimoku(self, df):
        tenkan = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
        kijun = (df["high"].rolling(26).max() + df["low"].rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = (df["high"].rolling(52).max() + df["low"].rolling(52).min()) / 2
        return pd.DataFrame({
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b
        })

    def calculate_adx(self, df, period=14):
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        dx = 100 * ((df["high"] - df["high"].shift()).abs() - (df["low"] - df["low"].shift()).abs()) / tr
        return dx.rolling(period).mean()