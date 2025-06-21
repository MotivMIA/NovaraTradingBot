# API Documentation

## Overview
NovaraTradingBot exposes FastAPI endpoints via the Main Bot (`api_endpoints.py`) for signal aggregation, health checks, and monitoring. Bots (TrendBot, VolumeBot, PredictBot, ArbitrageBot, PatternBot, NewsBot) send signals to the Main Bot’s `/receive` endpoint.

## Endpoints
- **Base URL**: `https://novara-tradingbot.onrender.com`

### GET /health
- **Description**: Checks Main Bot status.
- **Response**: `{"status": "healthy"}`
- **Example**: `curl https://novara-tradingbot.onrender.com/health`

### GET /balance
- **Description**: Retrieves account balance (placeholder).
- **Response**: `{"balance": 50000}`
- **Example**: `curl https://novara-tradingbot.onrender.com/balance`

### GET /trades
- **Description**: Lists executed trades (placeholder).
- **Response**: `[]`
- **Example**: `curl https://novara-tradingbot.onrender.com/trades`

### GET /analytics
- **Description**: Returns performance metrics (placeholder).
- **Response**: `{}`
- **Example**: `curl https://novara-tradingbot.onrender.com/analytics`

### POST /receive
- **Description**: Receives signals from bots for aggregation and trading.
- **Request Body**:
  ```json
  {
    "symbol": "BTC-USDT",
    "signal": "buy",
    "confidence": 0.8,
    "patterns": ["ichimoku_bullish", "doji"],
    "timeframes": ["1h"],
    "bot_name": "PatternBot"
  }
  ```
- **Response**: `{"status": "received"}`
- **Example**:
  ```bash
  curl -X POST https://novara-tradingbot.onrender.com/receive -H "Content-Type: application/json" -d '{"symbol":"BTC-USDT","signal":"buy","confidence":0.8,"patterns":["ichimoku_bullish","doji"],"timeframes":["1h"],"bot_name":"PatternBot"}'
  ```

## External APIs
- **BloFin API**: `https://api.blofin.com/v1/market/candles` for candles and trading (`api_utils.py`).
- **X API**: Paid access for sentiment analysis (https://docs.x.ai/docs/overview, `sentiment_analysis.py`).
- **OKX/Binance**: Via CCXT for arbitrage (`exchange_interface.py`).
- **CoinDesk RSS**: `https://www.coindesk.com/arc/outboundfeeds/rss/` for news (`sentiment_analysis.py`).

## Authentication
- **BloFin**: `API_KEY`, `API_SECRET`, `API_PASSPHRASE` in `.env` (`api_utils.py`).
- **X API**: `XAI_API_KEY` in `.env` (`sentiment_analysis.py`).
- **OKX/Binance**: `OKX_API_KEY`, `BINANCE_API_KEY` in `.env` (`exchange_interface.py`).

## Error Handling
- Logs errors to Logtail (`config.py:LOGTAIL_TOKEN`).
- Returns HTTP 500 for failed signal processing.