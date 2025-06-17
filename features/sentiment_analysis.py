# NLTK-based X sentiment analysis
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from logging import getLogger

logger = getLogger(__name__)

class SentimentAnalysis:
    def __init__(self):
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sid = SentimentIntensityAnalyzer()
            logger.info("NLTK vader_lexicon initialized")
        except Exception as e:
            logger.error(f"Failed to initialize NLTK vader_lexicon: {e}")
            self.sid = None

    def get_x_sentiment(self, symbol: str, sentiment_cache: dict) -> float:
        if not self.sid:
            return 0.0
        current_time = time.time()
        if current_time - sentiment_cache.get(symbol, {}).get("timestamp", 0) < 300:
            return sentiment_cache[symbol]["score"]
        
        mock_posts = [
            f"{symbol.replace('-USDT', '')} to the moon! 🚀",
            f"Bearish on {symbol.replace('-USDT', '')} this week.",
            f"Buying {symbol.replace('-USDT', '')} at dip!"
        ]
        scores = [self.sid.polarity_scores(post)["compound"] for post in mock_posts]
        sentiment_score = sum(scores) / len(scores) if scores else 0.0
        sentiment_cache[symbol] = {"score": sentiment_score, "timestamp": current_time}
        logger.debug(f"X sentiment for {symbol}: {sentiment_score:.2f}")
        return sentiment_score