import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional
from logging import getLogger
from features.config import VWAP_PERIOD, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, EMA_FAST, EMA_SLOW, BB_PERIOD, BB_STD, MIN_PRICE_POINTS

logger = getLogger(__name__)

class Indicators:
    def calculate_indicators(self, symbol: str, candles: List[Dict], get_candles_func: Optional[Callable[[str, int, str], List[Dict]]] = None) -> Dict:
        try:
            if len(candles) < MIN_PRICE_POINTS:
                logger.warning(f"Insufficient candle data for {symbol}: {len(candles)} points")
                return {}

            df = pd.DataFrame(candles)
            required_columns = ["close", "high", "low", "volume"]
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns for {symbol}: {df.columns}")
                return {}
            if df[required_columns].isna().any().any():
                logger.error(f"NaN values in candle data for {symbol}")
                return {}

            df["close"] = df["close"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["volume"] = df["volume"].astype(float)

            indicators = {}

            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            window = min(VWAP_PERIOD, len(df))
            vwap = (typical_price * df["volume"]).rolling(window=window).sum() / df["volume"].rolling(window=window).sum()
            indicators["vwap"] = vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else df["close"].iloc[-1]

            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
            rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else np.inf
            indicators["rsi"] = 100 - (100 / (1 + rs)) if not np.isinf(rs) else 50.0

            ema_fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
            ema_slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
            indicators["macd"] = macd.iloc[-1]
            indicators["macd_signal"] = signal.iloc[-1]

            indicators["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean().iloc[-1]
            indicators["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean().iloc[-1]

            sma = df["close"].rolling(window=BB_PERIOD).mean()
            std = df["close"].rolling(window=BB_PERIOD).std()
            indicators["bb_upper"] = (sma + BB_STD * std).iloc[-1]
            indicators["bb_lower"] = (sma - BB_STD * std).iloc[-1]

            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            tr = pd.DataFrame({"high_low": high_low, "high_close": high_close, "low_close": low_close}).max(axis=1)
            indicators["atr"] = tr.rolling(window=14).mean().iloc[-1] if len(tr) >= 14 else tr.mean()

            logger.debug(f"Calculated indicators for {symbol}: {indicators}")
            return indicators
        except Exception as e:
            logger.error(f"Failed to calculate indicators for {symbol}: {e}")
            return {}

    def calculate_vwap(self, symbol: str, candles: List[Dict] | Dict[str, List[Dict]]) -> float:
        try:
            if isinstance(candles, dict):
                candles = candles.get(symbol, [])
            if not candles or len(candles) < MIN_PRICE_POINTS:
                logger.warning(f"Insufficient candle data for VWAP: {len(candles)} points")
                return 0.0

            df = pd.DataFrame(candles)
            required_columns = ["high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns for VWAP: {df.columns}")
                return 0.0
            if df[required_columns].isna().any().any():
                logger.error(f"NaN values in VWAP input for {symbol}")
                return 0.0

            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)

            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            window = min(VWAP_PERIOD, len(df))
            vwap = (typical_price * df["volume"]).rolling(window=window).sum() / df["volume"].rolling(window=window).sum()
            result = vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else df["close"].iloc[-1]
            logger.debug(f"VWAP for {symbol}: ${result:.2f}")
            return result
        except Exception as e:
            logger.error(f"Failed to calculate VWAP for {symbol}: {e}")
            return 0.0

    def calculate_atr(self, symbol: str, candles: List[Dict] | Dict[str, List[Dict]], period: int = 14) -> float:
        try:
            if isinstance(candles, dict):
                candles = candles.get(symbol, [])
            if not candles or len(candles) < MIN_PRICE_POINTS:
                logger.warning(f"Insufficient candle data for ATR: {len(candles)} points")
                return 0.0

            df = pd.DataFrame(candles)
            required_columns = ["high", "low", "close"]
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns for ATR: {df.columns}")
                return 0.0
            if df[required_columns].isna().any().any():
                logger.error(f"NaN values in ATR input for {symbol}")
                return 0.0

            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)

            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift())
            low_close = np.abs(df["low"] - df["close"].shift())
            tr = pd.DataFrame({"high_low": high_low, "high_close": high_close, "low_close": low_close}).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1] if len(tr) >= period else tr.mean()
            logger.debug(f"ATR for {symbol}: ${atr:.2f}")
            return atr
        except Exception as e:
            logger.error(f"Failed to calculate ATR for {symbol}: {e}")
            return 0.0