# ML model training and prediction
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from logging import getLogger
from features.indicators import Indicators
from features.sentiment_analysis import SentimentAnalysis

logger = getLogger(__name__)

class MachineLearning:
    def __init__(self):
        self.ml_model = LogisticRegression()
        self.rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.ensemble_model = None
        self.scaler = StandardScaler()

    def train_ml_model(self, symbol: str, bot):
        current_time = time.time()
        if current_time - bot.last_model_train.get(symbol, 0) < 300:
            return
        
        prices = bot.price_history.get(symbol, [])
        volumes = bot.volume_history.get(symbol, [])
        if len(prices) < ML_LOOKBACK + RSI_PERIOD + MACD_SLOW + MACD_SIGNAL:
            logger.warning(f"Insufficient data for {symbol}: {len(prices)} points")
            return
        
        df = pd.DataFrame({"price": prices, "volume": volumes})
        for tf in TIMEFRAMES:
            tf_indicators = bot.indicators.calculate_indicators(symbol, bot.candle_history, bot.api_utils.get_candles, timeframe=tf)
            if tf_indicators:
                df[f"rsi_{tf}"] = tf_indicators.get("rsi", 50.0)
                df[f"macd_{tf}"] = tf_indicators.get("macd", 0.0)
                df[f"macd_signal_{tf}"] = tf_indicators.get("macd_signal", 0.0)
                df[f"bb_upper_{tf}"] = tf_indicators.get("bb_upper", df["price"].iloc[-1] * 1.02)
                df[f"bb_lower_{tf}"] = tf_indicators.get("bb_lower", df["price"].iloc[-1] * 0.98)
        
        df["atr"] = pd.DataFrame(bot.candle_history[symbol])[["high", "low", "close"]].apply(
            lambda x: max(x["high"] - x["low"], abs(x["high"] - x["close"].shift(1)), abs(x["low"] - x["close"].shift(1))), axis=1
        ).rolling(window=14).mean()
        df["volume_change"] = df["volume"].pct_change()
        df["sentiment"] = bot.sentiment.get_x_sentiment(symbol, bot.sentiment_cache)
        df["price_lag1"] = df["price"].shift(1)
        df["price_lag2"] = df["price"].shift(2)
        df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
        
        df = df.dropna()
        if len(df) < 10:
            logger.warning(f"Insufficient valid data for {symbol}: {len(df)} rows")
            return
        
        features = [
            f"rsi_{tf}" for tf in TIMEFRAMES
        ] + [
            f"macd_{tf}" for tf in TIMEFRAMES
        ] + [
            f"macd_signal_{tf}" for tf in TIMEFRAMES
        ] + [
            f"bb_upper_{tf}" for tf in TIMEFRAMES
        ] + [
            f"bb_lower_{tf}" for tf in TIMEFRAMES
        ] + [
            "atr", "volume_change", "sentiment", "price_lag1", "price_lag2"
        ]
        X = df[features].values
        y = df["target"].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.partial_fit(X_scaled, y, classes=[0, 1])
        self.rf_model.partial_fit(X_scaled, y, classes=[0, 1])
        self.ensemble_model = VotingClassifier(
            estimators=[('lr', self.ml_model), ('rf', self.rf_model)], voting='soft'
        )
        self.ensemble_model.fit(X_scaled, y)
        bot.is_model_trained[symbol] = True
        bot.last_model_train[symbol] = current_time
        logger.info(f"ML ensemble trained for {symbol} with {len(df)} data points")

    def predict_ml_signal(self, symbol: str, indicators: dict, bot) -> tuple[str, float] | None:
        if not bot.is_model_trained.get(symbol, False):
            self.train_ml_model(symbol, bot)
            if not bot.is_model_trained[symbol]:
                return None
        
        features = []
        for tf in TIMEFRAMES:
            tf_indicators = bot.indicators.calculate_indicators(symbol, bot.candle_history, bot.api_utils.get_candles, timeframe=tf)
            features.extend([
                tf_indicators.get("rsi", 50.0),
                tf_indicators.get("macd", 0.0),
                tf_indicators.get("macd_signal", 0.0),
                tf_indicators.get("bb_upper", indicators["price"] * 1.02),
                tf_indicators.get("bb_lower", indicators["price"] * 0.98)
            ])
        features.extend([
            bot.indicators.calculate_atr(symbol, bot.candle_history),
            indicators.get("volume_change", 0.0),
            bot.sentiment.get_x_sentiment(symbol, bot.sentiment_cache),
            indicators.get("price_lag1", indicators["price"]),
            indicators.get("price_lag2", indicators["price"])
        ])
        
        if any(np.isnan(f) for f in features):
            logger.warning(f"Invalid features for {symbol}: {features}")
            return None
        
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        prediction = self.ensemble_model.predict_proba(X_scaled)[0]
        confidence = max(prediction)
        
        if confidence < 0.6:
            return None
        
        return ("buy" if prediction[1] > prediction[0] else "sell", confidence)