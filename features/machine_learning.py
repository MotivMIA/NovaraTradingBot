import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MachineLearning:
    def __init__(self):
        self.config = Config()
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_model = XGBClassifier(n_estimators=50, random_state=42, use_label_encoder=False, eval_metric="logloss")
        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_features(self, candles: list, sentiment_score: float, indicators: dict) -> Optional[np.ndarray]:
        try:
            df = pd.DataFrame(candles).tail(self.config.ML_LOOKBACK)
            if len(df) < self.config.ML_LOOKBACK:
                return None

            features = pd.DataFrame()
            features["returns"] = df["close"].pct_change()
            features["volatility"] = features["returns"].rolling(window=20).std()
            features["volume_change"] = df["volume"].pct_change()
            features["sentiment"] = sentiment_score
            for key, value in indicators.items():
                features[key] = value if isinstance(value, (int, float)) else np.nan

            features = features.dropna()
            if len(features) == 0:
                return None

            return self.scaler.fit_transform(features)
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    def train_model(self, features: np.ndarray, target: np.ndarray):
        try:
            self.rf_model.fit(features, target)
            self.xgb_model.fit(features, target)
            self.is_trained = True
            logger.info("ML models trained successfully")
        except Exception as e:
            logger.error(f"Error training ML models: {e}")

    def predict_signal(self, symbol: str, price: float, bot) -> Tuple[Optional[str], float]:
        try:
            candles = bot.candle_history[symbol]
            indicators = bot.indicators.calculate_indicators(symbol, candles)
            sentiment_score = bot.sentiment.get_x_sentiment(symbol) if hasattr(bot, "sentiment") else 0.0
            features = self.prepare_features(candles, sentiment_score, indicators)
            if features is None or not self.is_trained:
                return None, 0.0

            rf_pred = self.rf_model.predict_proba(features[-1].reshape(1, -1))
            xgb_pred = self.xgb_model.predict_proba(features[-1].reshape(1, -1))
            confidence = (rf_pred[0][1] + xgb_pred[0][1]) / 2
            signal = "buy" if confidence > 0.6 else "sell" if confidence < 0.4 else None
            return signal, confidence
        except Exception as e:
            logger.error(f"Error predicting signal for {symbol}: {e}")
            return None, 0.0