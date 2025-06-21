# NovaraTradingBot

A modular, multi-bot trading system for cryptocurrency markets, designed to maximize profits through specialized trading strategies.

## Overview
NovaraTradingBot is a Python-based trading system with a 6-bot architecture:
- **Main Bot**: Aggregates signals, executes trades, hosts API endpoints.
- **TrendBot**: Momentum signals with Ichimoku Cloud, ADX, EMA.
- **VolumeBot**: Volatility breakouts with ATR, ADX.
- **PredictBot**: Machine learning with X and news sentiment.
- **ArbitrageBot**: Cross-exchange arbitrage (BloFin, OKX, Binance).
- **PatternBot**: Mean-reversion with candlestick patterns (doji, engulfing).
- **NewsBot**: News-driven signals from CoinDesk and X posts.

## Goals
- **Short-Term (1-3 Months)**: Add Ichimoku/ADX, X sentiment, dynamic risk, Monte Carlo backtesting, candlestick patterns (15-25% return boost).
- **Mid-Term (6-12 Months)**: News feeds, multi-exchange arbitrage, Markowitz optimization (20-35% returns).
- **Long-Term**: On-chain data, explainable AI, cloud sync, AI/GPU training (5-10% signal boost).
- **Profit Target**: 35-60% return on $50,000 ($17,500-$30,000 in 12 months).

## Setup
1. Clone: `git clone https://github.com/MotivMIA/NovaraTradingBot`.
2. Install: `pip install -r requirements.txt`.
3. Configure `.env`:
   - `API_KEY`, `API_SECRET`, `API_PASSPHRASE` (BloFin).
   - `DEMO_API_KEY`, `DEMO_API_SECRET`, `DEMO_API_PASSPHRASE`.
   - `XAI_API_KEY`, `OKX_API_KEY`, `BINANCE_API_KEY`.
   - `DB_PATH`, `REDIS_HOST`, `LOGTAIL_TOKEN`.
4. Deploy on Render: 7 services (`bots/main/main.py`, `bots/trendbot/main.py`, etc.).
5. Run locally:
   - Main Bot: `python bots/main/main.py`.
   - Bots: `python bots/trendbot/main.py`, etc.
   - Dashboard: `streamlit run dashboard.py`.
6. Test signals:
   ```bash
   curl -X POST https://novara-tradingbot.onrender.com/receive -d '{"symbol":"BTC-USDT","signal":"buy","confidence":0.8,"patterns":["doji"],"timeframes":["1h"],"bot_name":"PatternBot"}'
   ```

## Dependencies
- `pandas`, `numpy`, `ccxt`, `feedparser`, `pypfopt`, `nltk`, `fastapi`, `streamlit`, `psycopg2-binary`, `redis`, `optuna`, etc. (see `requirements.txt`).

## Infrastructure
- **Render**: 7 services (~$49/month).
- **PostgreSQL**: Cloud (`DB_PATH=postgresql://market_data_19fb_user:...`).
- **Redis**: Event bus (`REDIS_HOST=red-d195gnjuibrs73breos0`).
- **Streamlit**: Dashboard (`dashboard.py`).
- **APIs**: BloFin, X (paid), OKX, Binance, CoinDesk RSS.

## Testing
- Backtest: `python -m backtesting` (Monte Carlo).
- Demo: `DEMO_MODE=True` in `.env`.
- Monitor: `dashboard.py`, Logtail (`LOGTAIL_TOKEN`).

## Contributing
Submit issues/PRs to https://github.com/MotivMIA/NovaraTradingBot.

## License
MIT License