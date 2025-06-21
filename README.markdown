# NovaraTradingBot

A modular, multi-bot trading system for cryptocurrency markets, designed to maximize profits through specialized trading strategies.

## Overview
NovaraTradingBot is a Python-based trading system with a 6-bot architecture:
- **Main Bot**: Aggregates signals, executes trades, hosts API endpoints.
- **TrendBot**: Momentum signals with Ichimoku Cloud, ADX, EMA, VWAP.
- **VolumeBot**: Volatility breakouts with ATR, ADX, VWAP.
- **PredictBot**: Machine learning with X and news sentiment, Random Forest, XGBoost.
- **ArbitrageBot**: Cross-exchange arbitrage (BloFin, OKX, Binance).
- **PatternBot**: Mean-reversion with candlestick patterns (doji, engulfing, hammer, shooting star).
- **NewsBot**: News-driven signals from CoinDesk and X posts.

## Goals
- **Short-Term (1-3 Months)**: Add Ichimoku/ADX, X sentiment, dynamic risk, Monte Carlo backtesting, candlestick patterns, VWAP, XGBoost (15-25% return boost).
- **Mid-Term (6-12 Months)**: News feeds, multi-exchange arbitrage, Markowitz optimization (20-35% returns).
- **Long-Term**: On-chain data, explainable AI, cloud sync, AI/GPU training, WebSocket support, OKX/gmgn.ai integration (5-10% signal boost).
- **Profit Target**: 35-60% return on $50,000 ($17,500-$30,000 in 12 months).

## Setup
1. Clone: `git clone https://github.com/MotivMIA/NovaraTradingBot`.
2. Install: `pip install -r requirements.txt` (Note: Use `PyPortfolioOpt` with capital letters).
3. Configure `.env`:
   - **BloFin API**: Generate keys at https://www.blofin.com/account/api. Set `API_KEY`, `API_SECRET`, `API_PASSPHRASE`, `DEMO_API_KEY`, `DEMO_API_SECRET`, `DEMO_API_PASSPHRASE`.
   - `XAI_API_KEY`, `OKX_API_KEY`, `OKX_API_SECRET`, `BINANCE_API_KEY`, `BINANCE_API_SECRET` for additional exchanges.
   - `DB_PATH`, `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `LOGTAIL_TOKEN`, `WEBHOOK_URL` (see Webhook Setup).
4. Deploy on Render: 7 services (`python bots/main/main.py`, `python bots/trendbot/main.py`, etc.).
5. Run locally:
   - Main Bot: `python bots/main/main.py`.
   - Bots: `python bots/trendbot/main.py`, etc.
   - Dashboard: `streamlit run dashboard.py`.
6. Test signals:
   ```bash
   curl -X POST https://novara-tradingbot.onrender.com/receive -d '{"symbol":"BTC-USDT","signal":"buy","confidence":0.8,"patterns":["hammer"],"timeframes":["1h"],"bot_name":"PatternBot"}'
   ```
7. Test BloFin signal webhook:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"action":"buy","marketPosition":"long","prevMarketPosition":"flat","instrument":"BTCUSD","timestamp":"2025-06-21T01:33:00Z","maxLag":"60","investmentType":"base","amount":"10","id":"test123"}' https://demo-trading-api.blofin.com/uapi/v1/algo/signal/trigger
   ```

## Webhook Setup
1. Create a Discord server or use an existing one.
2. Go to Channel > Edit Channel > Integrations > Create Webhook.
3. Copy the Webhook URL.
4. Add to `.env` as `WEBHOOK_URL=https://discord.com/api/webhooks/...`.
5. Test:
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"content":"Test NovaraTradingBot alert"}' $WEBHOOK_URL
   ```

## Dependencies
- `pandas`, `numpy`, `ccxt`, `feedparser`, `PyPortfolioOpt`, `nltk`, `fastapi`, `uvicorn`, `streamlit`, `psycopg2-binary`, `redis`, `optuna`, `scikit-learn`, `xgboost`, `plotly`, `pytest` (see `requirements.txt`).

## Infrastructure
- **Render**: 7 services (~$49/month, https://render.com).
- **PostgreSQL**: Cloud database (`DB_PATH=postgresql://market_data_19fb_user:yxlMERTZ36Wm15LbYxir74tKhsVCxQOd@dpg-d19825nfte5s73c40jl0-a/market_data_19fb`).
- **Redis**: Optional event bus (`REDIS_HOST=red-d195gnjuibrs73breos0`, `REDIS_PORT=6379`).
- **Streamlit**: Dashboard with VWAP, candlestick visuals (`dashboard.py`).
- **APIs**:
  - BloFin: https://api.blofin.com/ (REST), https://demo-trading-api.blofin.com/uapi/v1/algo/signal/trigger (signal webhook, uses `API_KEY`, `API_SECRET`).
  - X API: https://docs.x.ai/docs/overview (paid, for sentiment).
  - OKX/Binance: Via CCXT (`features/exchange_interface.py`).
  - CoinDesk RSS: https://www.coindesk.com/arc/outboundfeeds/rss/.
- **Logtail**: Logging (`LOGTAIL_TOKEN=7pWvyBP5PB8b1h1wjrSXSowz`, https://logtail.com).

## Testing
- **Backtesting**: `python -m backtesting` (Monte Carlo simulations).
- **Unit Tests**: `pytest tests/` (covers indicators, candlesticks, XGBoost).
- **Demo Mode**: Set `DEMO_MODE=True` in `.env` for BloFin demo API.
- **Monitoring**: Run `streamlit run dashboard.py` for trade analytics, VWAP, candlestick patterns (doji, hammer, shooting star); check Logtail for logs.

## Documentation
- `docs/bot_architecture.md`: System architecture and signal flow.
- `docs/feature_development.md`: Feature roadmap and development process.
- `docs/api_documentation.md`: API endpoints and authentication.
- `docs/project_outline.md`: Project goals and setup details.

## Contributing
- Submit issues/PRs: https://github.com/MotivMIA/NovaraTradingBot.
- Join discussions: [Discord Server](https://discord.gg/ZcM6r4Pd).
- Share documents: [Google Drive Folder](https://drive.google.com/drive/folders/1SG64pWFwuPpE89yTMg1tDPzOf_BLAwAs).
- Ensure PRs include tests (`pytest tests/`) and clear descriptions.

## License
MIT License