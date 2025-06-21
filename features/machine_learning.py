import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MachineLearning:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()

    async def train_ml_model(self, symbol: str, bot):
        try:
            candles = bot.candle_history[symbol]
            if len(candles) < 50:
                return

            df = pd.DataFrame(candles)
            indicators = bot.indicators.calculate_indicators(symbol, candles)
            x_sentiment = bot.sentiment.get_x_sentiment(symbol)
            news_sentiment = bot.sentiment.get_news_sentiment(symbol)

            features = pd.DataFrame({
                "rsi": df["close"].apply(lambda x: bot.indicators.calculate_rsi(df["close"]).iloc[-1]),
                "macd": df["close"].apply(lambda x: bot.indicators.calculate_macd(df["close"])[0].iloc[-1]),
                "x_sentiment": x_sentiment,
                "news_sentiment": news_sentiment
            })

            target = (df["close"].shift(-1) > df["close"]).astype(int)
            X = features[:-1]
            y = target[:-1]

            X_scaled = self.scaler.fit_transform(X)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_scaled, y)
            self.models[symbol] = model
            logger.info(f"ML model trained for {symbol}")
        except Exception as e:
            logger.error(f"Error training ML model for {symbol}: {e}")

    def predict_signal(self, symbol: str, price: float, bot):
        try:
            if symbol not in self.models:
                return None, 0.0

            indicators = bot.indicators.calculate_indicators(symbol, bot.candle_history[symbol])
            x_sentiment = bot.sentiment.get_x_sentiment(symbol)
            news_sentiment = bot.sentiment.get_news_sentiment(symbol)

            features = np.array([[
                indicators["rsi"],
                indicators["macd"],
                x_sentiment,
                news_sentiment
            ]])
            X_scaled = self.scaler.transform(features)
            prediction = self.models[symbol].predict_proba(X_scaled)[0]
            signal = "buy" if prediction[1] > prediction[0] else "sell"
            confidence = max(prediction)
            return signal, confidence
        except Exception as e:
            logger.error(f"Error predicting signal for {symbol}: {e}")
            return None, 0.0

    def explain_signal(self, symbol: str, features: dict):
        logger.info(f"Signal explanation placeholder for {symbol}")
        return {"features": features, "explanation": "TBD"}