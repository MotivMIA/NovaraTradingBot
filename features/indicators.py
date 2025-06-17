# VWAP, ATR, RSI, MACD, Bollinger Bands
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from logging import getLogger
from features.config import VWAP_PERIOD, RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, EMA_FAST, EMA_SLOW, BB_PERIOD, BB_STD, MIN_PRICE_POINTS

logger = getLogger(__name__)

class Indicators:
    def calculate_vwap(self, symbol: str, candle_history: dict) -> float | None:
        candles = candle_history.get(symbol, [])
        if len(candles) < VWAP_PERIOD:
            logger.debug(f"Insufficient candles for VWAP calculation for {symbol}: {len(candles)} < {VWAP_PERIOD}")
            return None
        df = pd.DataFrame(candles[-VWAP_PERIOD:])
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical_price * df["volume"]).sum() / df["volume"].sum()
        logger.debug(f"VWAP for {symbol}: ${vwap:.2f}")
        return vwap if not np.isnan(vwap) else None

    def calculate_atr(self, symbol: str, candle_history: dict, period: int = 14) -> float:
        candles = candle_history.get(symbol, [])
        if len(candles) < period + 1:
            return 0.0
        df = pd.DataFrame(candles[-period-1:])
        df["high_low"] = df["high"] - df["low"]
        df["high_prev_close"] = abs(df["high"] - df["close"].shift(1))
        df["low_prev_close"] = abs(df["low"] - df["close"].shift(1))
        df["tr"] = df[["high_low", "high_prev_close", "low_prev_close"]].max(axis=1)
        atr = df["tr"].rolling(window=period).mean().iloc[-1]
        logger.debug(f"ATR for {symbol}: ${atr:.2f}")
        return atr if not np.isnan(atr) else 0.0

    def calculate_indicators(self, symbol: str, candle_history: dict, get_candles_func, timeframe: str = "1m") -> dict | None:
        candles = get_candles_func(symbol, timeframe=timeframe) or candle_history.get(symbol, [])
        if len(candles) < MIN_PRICE_POINTS:
            logger.warning(f"Insufficient candle data for {symbol} ({timeframe}): {len(candles)} candles")
            return {"price": candles[-1]["close"] if candles else None}
        
        df = pd.DataFrame(candles)
        indicators = {"price": df["close"].iloc[-1]}
        vwap = self.calculate_vwap(symbol, candle_history)
        if vwap:
            indicators["vwap"] = vwap
        
        if len(candles) >= RSI_PERIOD:
            indicators["rsi"] = RSIIndicator(df["close"], window=RSI_PERIOD).rsi().iloc[-1]
        if len(candles) >= max(MACD_SLOW + MACD_SIGNAL, EMA_SLOW, BB_PERIOD):
            macd = MACD(df["close"], window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
            indicators["macd"] = macd.macd().iloc[-1]
            indicators["macd_signal"] = macd.macd_signal().iloc[-1]
            indicators["ema_fast"] = EMAIndicator(df["close"], window=EMA_FAST).ema_indicator().iloc[-1]
            indicators["ema_slow"] = EMAIndicator(df["close"], window=EMA_SLOW).ema_indicator().iloc[-1]
            bb = BollingerBands(df["close"], window=BB_PERIOD, window_dev=BB_STD)
            indicators["bb_upper"] = bb.bollinger_hband().iloc[-1]
            indicators["bb_lower"] = bb.bollinger_lband().iloc[-1]
        indicators["price_lag1"] = df["close"].iloc[-2] if len(df) > 1 else np.nan
        indicators["price_lag2"] = df["close"].iloc[-3] if len(df) > 2 else np.nan
        indicators["volume_change"] = df["volume"].pct_change().iloc[-1] if len(df) > 1 else 0.0
        
        logger.debug(f"{symbol} ({timeframe}) indicators - RSI: {indicators.get('rsi', 'N/A')}")
        return indicators