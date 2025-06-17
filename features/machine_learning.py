import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from logging import getLogger
from features.indicators import Indicators
from features.sentiment_analysis import SentimentAnalysis
from features.config import ML_LOOKBACK, RSI_PERIOD, MACD_SLOW, MACD_SIGNAL, TIMEFRAMES

logger = getLogger(__name__)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

class MachineLearning:
    def __init__(self):
        self.ml_model = LogisticRegression()
        self.rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.lstm_model = LSTMModel(input_size=5*len(TIMEFRAMES)+5, hidden_size=64, num_layers=2)
        self.lstm_optimizer = torch.optim.Adam(self.lstm_model.parameters())
        self.lstm_criterion = nn.CrossEntropyLoss()

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
            tf_indicators = bot.indicators.calculate_indicators(symbol, bot.candle_history[symbol])
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
        
        features = [f"rsi_{tf}" for tf in TIMEFRAMES] + [f"macd_{tf}" for tf in TIMEFRAMES] + \
                   [f"macd_signal_{tf}" for tf in TIMEFRAMES] + [f"bb_upper_{tf}" for tf in TIMEFRAMES] + \
                   [f"bb_lower_{tf}" for tf in TIMEFRAMES] + ["atr", "volume_change", "sentiment", "price_lag1", "price_lag2"]
        X = df[features].values
        y = df["target"].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.partial_fit(X_scaled, y, classes=[0, 1])
        self.rf_model.partial_fit(X_scaled, y, classes=[0, 1])
        self.ensemble_model = VotingClassifier(
            estimators=[('lr', self.ml_model), ('rf', self.rf_model)], voting='soft'
        )
        self.ensemble_model.fit(X_scaled, y)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor(y, dtype=torch.long)
        for _ in range(10):
            self.lstm_model.train()
            self.lstm_optimizer.zero_grad()
            output = self.lstm_model(X_tensor)
            loss = self.lstm_criterion(output, y_tensor)
            loss.backward()
            self.lstm_optimizer.step()
        
        bot.is_model_trained[symbol] = True
        bot.last_model_train[symbol] = current_time
        logger.info(f"ML models trained for {symbol} with {len(df)} data points")

    def predict_signal(self, symbol: str, indicators: dict, bot) -> Tuple[str, float] | None:
        if not bot.is_model_trained.get(symbol, False):
            self.train_ml_model(symbol, bot)
            if not bot.is_model_trained[symbol]:
                return None
        
        features = []
        for tf in TIMEFRAMES:
            tf_indicators = bot.indicators.calculate_indicators(symbol, bot.candle_history[symbol])
            features.extend([
                tf_indicators.get("rsi", 50.0),
                tf_indicators.get("macd", 0.0),
                tf_indicators.get("macd_signal", 0.0),
                tf_indicators.get("bb_upper", indicators["price"] * 1.02),
                tf_indicators.get("bb_lower", indicators["price"] * 0.98)
            ])
        features.extend([
            bot.indicators.calculate_atr(symbol, bot.candle_history[symbol]),
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
        ensemble_pred = self.ensemble_model.predict_proba(X_scaled)[0]
        ensemble_confidence = max(ensemble_pred)
        
        self.lstm_model.eval()
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            lstm_pred = torch.softmax(self.lstm_model(X_tensor), dim=1)[0].numpy()
        lstm_confidence = max(lstm_pred)
        
        final_confidence = (ensemble_confidence + lstm_confidence) / 2
        if final_confidence < 0.6:
            return None
        
        final_signal = "buy" if (ensemble_pred[1] + lstm_pred[1]) > (ensemble_pred[0] + lstm_pred[0]) else "sell"
        logger.debug(f"ML signal for {symbol}: {final_signal} with confidence {final_confidence:.2f}")
        return final_signal, final_confidence