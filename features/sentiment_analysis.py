import logging
import nltk
from typing import Dict, Optional
from config import Config
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import feedparser
import random
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalysis:
    def __init__(self):
        self.config = Config()
        nltk.download('vader_lexicon', quiet=True)
        self.analyzer = SentimentIntensityAnalyzer()

    def fetch_x_posts(self, symbol: str) -> list:
        try:
            if not self.config.XAI_API_KEY or self.config.XAI_API_KEY == "your_xai_api_key":
                # Mock data for testing until X API is implemented
                logger.warning(f"X API key not set for {symbol}. Using mock data.")
                sentiments = [
                    f"{symbol} is bullish! 🚀",
                    f"Sell {symbol} now, market crash incoming! 📉",
                    f"{symbol} holding strong, good for HODL.",
                    f"New whale activity on {symbol}, watch out!"
                ]
                return [
                    {
                        "text": random.choice(sentiments),
                        "created_at": datetime.now(timezone.utc).isoformat()
                    }
                    for _ in range(random.randint(5, 10))
                ]
            
            # Template for real X API integration
            # from xai import XClient
            # client = XClient(api_key=self.config.XAI_API_KEY)
            # posts = client.search_posts(query=f"{symbol} crypto", max_results=100)
            # return [{"text": post["content"], "created_at": post["created_at"]} for post in posts]
            logger.error(f"X API not implemented for {symbol}. Configure XAI_API_KEY and uncomment client code.")
            return []
        except Exception as e:
            logger.error(f"Error fetching X posts for {symbol}: {e}")
            return []

    def fetch_news(self, symbol: str) -> list:
        try:
            feed = feedparser.parse("https://www.coindesk.com/arc/outboundfeeds/rss/")
            news = [entry for entry in feed.entries if symbol in entry.title or symbol in entry.description]
            return [{"text": entry.title + " " + entry.description, "published": entry.published} for entry in news]
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []

    def get_x_sentiment(self, symbol: str) -> float:
        try:
            posts = self.fetch_x_posts(symbol)
            if not posts:
                return 0.0
            scores = [self.analyzer.polarity_scores(post["text"])["compound"] for post in posts]
            sentiment = sum(scores) / len(scores) * self.config.SENTIMENT_WEIGHT
            logger.info(f"X sentiment for {symbol}: {sentiment}")
            return sentiment
        except Exception as e:
            logger.error(f"Error analyzing X sentiment for {symbol}: {e}")
            return 0.0

    def get_news_sentiment(self, symbol: str) -> float:
        try:
            news = self.fetch_news(symbol)
            if not news:
                return 0.0
            scores = [self.analyzer.polarity_scores(item["text"])["compound"] for item in news]
            sentiment = sum(scores) / len(scores) * self.config.SENTIMENT_WEIGHT
            logger.info(f"News sentiment for {symbol}: {sentiment}")
            return sentiment
        except Exception as e:
            logger.error(f"Error analyzing news sentiment for {symbol}: {e}")
            return 0.0

    def get_combined_sentiment(self, symbol: str) -> Dict:
        x_sentiment = self.get_x_sentiment(symbol)
        news_sentiment = self.get_news_sentiment(symbol)
        combined = (x_sentiment + news_sentiment) / 2
        return {"x_sentiment": x_sentiment, "news_sentiment": news_sentiment, "combined": combined}
</xArtifact>

**Changes**:
- Enhanced mock `fetch_x_posts` with varied sentiments and random post counts (5-10) for robust testing.
- Added check for `XAI_API_KEY` to log a warning if unset.
- Provided commented template for real X API integration (`client.search_posts`).
- Updated error logging to guide implementation.

**Action**:
- Test locally:
  ```bash
  python -m features.sentiment_analysis
  ```
- Provide the exact X API method (e.g., `client.search_posts`) and `XAI_API_KEY` when ready to replace the mock.
- Add `xai` to `requirements.txt` once the X API client is available:
  ```
  xai==0.1.0  # Hypothetical, replace with actual version
  ```

---

### Step 4: Preparing for Render Deployment

To launch all 7 services on Render (`main`, `trendbot`, `volumebot`, `predictbot`, `arbitragebot`, `patternbot`, `newsbot`) and ensure the bot is actively trading, we need to:
1. Set up each service with the updated `.env`.
2. Verify dependencies (`requirements.txt`, artifact ID: 54d845e1-2487-48bb-b45d-94ae032ea081, version: e377d4f7-3ca5-4898-bb8e-595c893d802e).
3. Test API endpoints and webhooks.
4. Monitor via `dashboard.py` and Logtail.

#### Render Service Configuration
Each service runs a Python script (`python bots/main/main.py`, etc.). Here’s a template for Render’s `render.yaml`:
<xaiArtifact artifact_id="97c17a9a-7771-427c-99c8-17b2b645ad9b" artifact_version_id="eaed1feb-b2a8-45fd-a5b6-c16ad2ca9a81" title="render.yaml" contentType="text/yaml">
services:
  - type: web
    name: novara-main-bot
    env: python
    plan: starter
    region: oregon
    buildCommand: pip install -r requirements.txt
    startCommand: python bots/main/main.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: BLOFIN_API_KEY
        value: 50f02f2dc2d4442cb4de50b7f87d37f2
      - key: BLOFIN_API_SECRET
        sync: false
      - key: BLOFIN_API_PASSPHRASE
        value: zWz1sFh6O25173wnkTQ1
      - key: DEMO_BLOFIN_API_KEY
        value: 1068db9f2fd8486dad50c5e304b0a150
      - key: DEMO_BLOFIN_API_SECRET
        value: deb4daaf7c234351bcb0b053bb846c56
      - key: DEMO_BLOFIN_API_PASSPHRASE
        value: 0NS81dhSL8qIs2Gx4O9x
      - key: DEMO_MODE
        value: True
      - key: XAI_API_KEY
        sync: false
      - key: OKX_API_KEY
        sync: false
      - key: OKX_API_SECRET
        sync: false
      - key: BINANCE_API_KEY
        sync: false
      - key: BINANCE_API_SECRET
        sync: false
      - key: MAIN_BOT_URL
        value: https://novara-tradingbot.onrender.com
      - key: ONCHAIN_API_KEY
        value: ""
      - key: CLOUD_DB_URL
        value: ""
      - key: LOGTAIL_TOKEN
        value: 7pWvyBP5PB8b1h1wjrSXSowz
      - key: WEBHOOK_URL
        sync: false
      - key: DB_PATH
        value: postgresql://market_data_19fb_user:yxlMERTZ36Wm15LbYxir74tKhsVCxQOd@dpg-d19825nfte5s73c40jl0-a/market_data_19fb
      - key: REDIS_HOST
        value: red-d195gnjuibrs73breos0
      - key: REDIS_PORT
        value: 6379
      - key: INITIAL_BID_USD
        value: 100.0
      - key: RISK_PER_TRADE
        value: 0.01
      - key: SYMBOLS
        value: BTC-USDT,ETH-USDT,XRP-USDT
      - key: DEFAULT_BALANCE
        value: 49999.42
      - key: RENDER
        value: true

  - type: web
    name: novara-trend-bot
    env: python
    plan: starter
    region: oregon
    buildCommand: pip install -r requirements.txt
    startCommand: python bots/trendbot/main.py
    envVars:
      - fromService: novara-main-bot

  - type: web
    name: novara-volume-bot
    env: python
    plan: starter
    region: oregon
    buildCommand: pip install -r requirements.txt
    startCommand: python bots/volumebot/main.py
    envVars:
      - fromService: novara-main-bot

  - type: web
    name: novara-predict-bot
    env: python
    plan: starter
    region: oregon
    buildCommand: pip install -r requirements.txt
    startCommand: python bots/predictbot/main.py
    envVars:
      - fromService: novara-main-bot

  - type: web
    name: novara-arbitrage-bot
    env: python
    plan: starter
    region: oregon
    buildCommand: pip install -r requirements.txt
    startCommand: python bots/arbitragebot/main.py
    envVars:
      - fromService: novara-main-bot

  - type: web
    name: novara-pattern-bot
    env: python
    plan: starter
    region: oregon
    buildCommand: pip install -r requirements.txt
    startCommand: python bots/patternbot/main.py
    envVars:
      - fromService: novara-main-bot

  - type: web
    name: novara-news-bot
    env: python
    plan: starter
    region: oregon
    buildCommand: pip install -r requirements.txt
    startCommand: python bots/newsbot/main.py
    envVars:
      - fromService: novara-main-bot