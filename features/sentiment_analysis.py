import logging
import time
import feedparser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalysis:
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        self.sentiment_cache: Dict[str, Tuple[float, float]] = {}
        self.cache_duration = 300  # 5 minutes

    def get_x_sentiment(self, symbol: str) -> float:
        try:
            cache_key = f"{symbol}_x"
            if cache_key in self.sentiment_cache:
                sentiment, timestamp = self.sentiment_cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    return sentiment

            # Assumes xAI API integration (per https://docs.x.ai/docs/overview)
            posts = self.fetch_x_posts(symbol.replace("-USDT", ""), limit=100)
            scores = [self.sid.polarity_scores(post["text"])["compound"] for post in posts]
            sentiment = sum(scores) / len(scores) if scores else 0.0
            self.sentiment_cache[cache_key] = (sentiment, time.time())
            logger.info(f"X sentiment for {symbol}: {sentiment:.2f}")
            return sentiment
        except Exception as e:
            logger.error(f"X sentiment fetch failed for {symbol}: {e}")
            return 0.0

    def get_news_sentiment(self, symbol: str) -> float:
        try:
            cache_key = f"{symbol}_news"
            if cache_key in self.sentiment_cache:
                sentiment, timestamp = self.sentiment_cache[cache_key]
                if time.time() - timestamp < self.cache_duration:
                    return sentiment

            feed = feedparser.parse(f"https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml&q={symbol.replace('-USDT', '')}")
            scores = [self.sid.polarity_scores(entry["title"])["compound"] for entry in feed.entries[:50]]
            sentiment = sum(scores) / len(scores) if scores else 0.0
            self.sentiment_cache[cache_key] = (sentiment, time.time())
            logger.info(f"News sentiment for {symbol}: {sentiment:.2f}")
            return sentiment
        except Exception as e:
            logger.error(f"News sentiment fetch failed for {symbol}: {e}")
            return 0.0

    def get_onchain_sentiment(self, symbol: str) -> float:
        logger.info(f"On-chain sentiment placeholder for {symbol}")
        return 0.0

    def fetch_x_posts(self, query: str, limit: int = 100) -> list:
        # Mock implementation; replace with xAI API call
        try:
            import xai_sdk
            client = xai_sdk.Client(api_key=os.getenv("XAI_API_KEY"))
            posts = client.search.posts(query=query, limit=limit)
            return [{"text": post["content"]} for post in posts]
        except Exception as e:
            logger.error(f"X API fetch failed: {e}")
            return []