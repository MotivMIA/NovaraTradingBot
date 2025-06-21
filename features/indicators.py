import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Indicators:
    def __init__(self):
        self.config = Config()

    def calculate_indicators(self, symbol: str, candles: List[Dict]) -> Dict:
        try:
            df = pd.DataFrame(candles)
            if len(df) < self.config.MIN_PRICE_POINTS:
                return {}

            # RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.config.RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.RSI_PERIOD).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

            # MACD
            ema_fast = df["close"].ewm(span=self.config.MACD_FAST, adjust=False).mean()
            ema_slow = df["close"].ewm(span=self.config.MACD_SLOW, adjust=False).mean()
            df["macd"] = ema_fast - ema_slow
            df["macd_signal"] = df["macd"].ewm(span=self.config.MACD_SIGNAL, adjust=False).mean()

            # Bollinger Bands
            df["bb_mid"] = df["close"].rolling(window=self.config.BB_PERIOD).mean()
            df["bb_std"] = df["close"].rolling(window=self.config.BB_PERIOD).std()
            df["bb_upper"] = df["bb_mid"] + self.config.BB_STD * df["bb_std"]
            df["bb_lower"] = df["bb_mid"] - self.config.BB_STD * df["bb_std"]

            # Ichimoku Cloud
            high_9 = df["high"].rolling(window=9).max()
            low_9 = df["low"].rolling(window=9).min()
            df["tenkan"] = (high_9 + low_9) / 2
            high_26 = df["high"].rolling(window=26).max()
            low_26 = df["low"].rolling(window=26).min()
            df["kijun"] = (high_26 + low_26) / 2
            df["senkou_a"] = ((df["tenkan"] + df["kijun"]) / 2).shift(26)
            df["senkou_b"] = ((df["high"].rolling(window=52).max() + df["low"].rolling(window=52).min()) / 2).shift(26)

            # ADX
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs()
            ], axis=1).max(axis=1)
            dm_plus = (df["high"] - df["high"].shift()).where((df["high"] - df["high"].shift()) > (df["low"].shift() - df["low"]), 0)
            dm_minus = (df["low"].shift() - df["low"]).where((df["low"].shift() - df["low"]) > (df["high"] - df["high"].shift()), 0)
            tr_smooth = tr.rolling(window=14).mean()
            dm_plus_smooth = dm_plus.rolling(window=14).mean()
            dm_minus_smooth = dm_minus.rolling(window=14).mean()
            di_plus = (dm_plus_smooth / tr_smooth) * 100
            di_minus = (dm_minus_smooth / tr_smooth) * 100
            dx = ((di_plus - di_minus).abs() / (di_plus + di_minus)) * 100
            df["adx"] = dx.rolling(window=14).mean()

            # VWAP
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            vwap_volume = (typical_price * df["volume"]).cumsum()
            volume_cumsum = df["volume"].cumsum()
            df["vwap"] = vwap_volume / volume_cumsum

            return {
                "rsi": df["rsi"].iloc[-1],
                "macd": df["macd"].iloc[-1],
                "macd_signal": df["macd_signal"].iloc[-1],
                "bb_upper": df["bb_upper"].iloc[-1],
                "bb_mid": df["bb_mid"].iloc[-1],
                "bb_lower": df["bb_lower"].iloc[-1],
                "tenkan": df["tenkan"].iloc[-1],
                "kijun": df["kijun"].iloc[-1],
                "senkou_a": df["senkou_a"].iloc[-1],
                "senkou_b": df["senkou_b"].iloc[-1],
                "adx": df["adx"].iloc[-1],
                "vwap": df["vwap"].iloc[-1]
            }
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return {}