# NovaraTradingBot Project Outline

## Overview
NovaraTradingBot is a Python-based cryptocurrency trading system transitioning from a single-bot (`api.py`) to a distributed 6-bot architecture to maximize profits (35-60% return improvement on $50,000 balance). The system leverages BloFin API, paid X API, OKX/Binance keys, and Render for deployment.

## Architecture
- **Main Bot**: Aggregates signals via `/receive` (https://novara-tradingbot.onrender.com/receive), executes trades on BloFin, logs performance (`performance_analytics.py`), and sends Discord alerts (`notifications.py`).
- **TrendBot**: Momentum signals using Ichimoku Cloud, ADX, EMA (`indicators.py`, `strategies.py`).
- **VolumeBot**: Volatility breakouts with ATR, ADX, volume spikes (`indicators.py`, `trading_logic.py`).
- **PredictBot**: Random Forest ML with X and news sentiment (`machine_learning.py`, `sentiment_analysis.py`).
- **ArbitrageBot**: Cross-exchange arbitrage (BloFin, OKX, Binance) via CCXT (`exchange_interface.py`).
- **PatternBot**: Mean-reversion with candlestick patterns (doji, engulfing) and dynamic risk (`candlesticks.py`, `portfolio_management.py`).
- **NewsBot**: News-driven signals from CoinDesk RSS and X posts (`sentiment_analysis.py`).

## Goals
### Short-Term (1-3 Months, 15-25% Return Boost)
- Enhance TrendBot/VolumeBot/PatternBot with Ichimoku Cloud and ADX (`indicators.py`, 5-10% signal accuracy).
- Integrate paid X API for real-time sentiment in PredictBot (`sentiment_analysis.py`, 5-8% profit boost).
- Implement dynamic risk management (`portfolio_management.py`, 5-10% return increase).
- Add Monte Carlo backtesting (`backtesting.py`, 10% strategy confidence).
- Introduce candlestick pattern detection (doji, engulfing) for PatternBot (`candlesticks.py`, 5-8% accuracy boost).

### Mid-Term (6-12 Months, 20-35% Additional Returns)
- Add CoinDesk RSS news sentiment for NewsBot (`sentiment_analysis.py`, 10-15% profit boost).
- Enable multi-exchange arbitrage in ArbitrageBot (`exchange_interface.py`, 5-10% returns).
- Implement Markowitz portfolio optimization (`portfolio_management.py`, 5-10% risk-adjusted returns).

### Long-Term (1-2 Years, Scalability Setup)
- Integrate on-chain data (e.g., Covalent/The Graph) for sentiment analysis (`sentiment_analysis.py`, 5-10% signal boost).
- Add explainable AI (e.g., SHAP) for ML signal transparency (`machine_learning.py`).
- Enable cloud database sync (AWS/GCP) for scalability (`database.py`).
- Prepare for AI/GPU acceleration:
  - Train advanced ML models (e.g., LSTMs) on GPUs for PredictBot (`machine_learning.py`).
  - Deploy AI-driven signal generation on cloud GPU instances (e.g., AWS EC2 g4dn, ~$0.5/hour).

## Profit Target
- **Total**: 35-60% return improvement ($17,500-$30,000 profit on $50,000 in 12 months).
- **Metrics**: Win rate >60%, Sharpe ratio >1.5 (`performance_analytics.py`).

## Infrastructure
- **Render**: 7 services (Main Bot + 6 bots, ~$7/month each, paid tier, no constraints).
- **PostgreSQL**: Stores candles/trades (`database.py`, `DB_PATH=postgresql://market_data_19fb_user:yxlMERTZ36Wm15LbYxir74tKhsVCxQOd@dpg-d19825nfte5s73c40jl0-a/market_data_19fb`).
- **Redis**: Event bus for bot communication (`event_bus.py`, `REDIS_HOST=red-d195gnjuibrs73breos0`).
- **Streamlit**: Dashboard for monitoring trades, patterns, sentiment (`dashboard.py`).
- **APIs**:
  - BloFin: Trading and candles (https://api.blofin.com, `api_utils.py`).
  - X API: Paid sentiment (https://docs.x.ai/docs/overview, `sentiment_analysis.py`).
  - OKX/Binance: Arbitrage via CCXT (`exchange_interface.py`).
  - CoinDesk RSS: News sentiment (`sentiment_analysis.py`).

## Deployment
- **Repo Structure**: Single repo (`MotivMIA/NovaraTradingBot`) with bot directories (`bots/trendbot/`, `bots/volumebot/`, etc.).
- **Steps**:
  1. Clone: `git clone https://github.com/MotivMIA/NovaraTradingBot`.
  2. Install: `pip install -r requirements.txt`.
  3. Configure: Set `.env` with `API_KEY`, `XAI_API_KEY`, `OKX_API_KEY`, `BINANCE_API_KEY`.
  4. Deploy: Create 7 Render services with `python bots/main/main.py`, `python bots/trendbot/main.py`, etc.
  5. Test: `curl -X POST https://novara-tradingbot.onrender.com/receive -d '{"symbol":"BTC-USDT","signal":"buy","confidence":0.8,"patterns":["doji"],"timeframes":["1h"],"bot_name":"PatternBot"}'`.
  6. Monitor: Run `streamlit run dashboard.py`, check Logtail (`LOGTAIL_TOKEN=7pWvyBP5PB8b1h1wjrSXSowz`).

## Cost Control
- **Current**:
  - Render: ~$49/month for 7 services.
  - X API: Unknown cost (per https://x.ai/api).
  - Free libraries: `ccxt`, `feedparser`, `pypfopt`, `nltk`, `pandas`.
- **Future**:
  - On-chain APIs (e.g., Covalent, ~$50/month) after $1,000/month profits.
  - GPU instances (e.g., AWS g4dn, ~$0.5/hour) for AI training after $5,000/month profits.

## Testing
- **Backtesting**: Run `python -m backtesting` with Monte Carlo (`backtesting.py`).
- **Demo Mode**: Set `DEMO_MODE=True` in `.env` for BloFin demo API.
- **Unit Tests**: Add `tests/` directory with `pytest` scripts (TBD).
- **Signal Validation**: Test via `/receive` endpoint (see Deployment).
- **Monitoring**: Use `dashboard.py` for metrics, Logtail for logs.

## Repo Structure
- **Recommended**: Single repo (`MotivMIA/NovaraTradingBot`) with:
  - `bots/main/main.py`: Main Bot.
  - `bots/trendbot/main.py`, `bots/volumebot/main.py`, etc.: Bot implementations.
  - `features/`: `indicators.py`, `candlesticks.py`, `trading_logic.py`, etc.
  - `docs/`: `README.md`, `bot_architecture.md`, `feature_development.md`, `api_documentation.md`.
  - Root: `requirements.txt`, `.env` (git-ignored), `dashboard.py`.
- **Rationale**: Simplifies dependency management, CI/CD, and deployment vs. separate repos.

## Next Steps
1. Review 28 artifacts (1 main, 6 bots, 11 features, 4 docs, 2 supporting, 1 dashboard).
2. Commit to `MotivMIA/NovaraTradingBot` with recommended structure.
3. Install dependencies: `pip install -r requirements.txt`.
4. Test locally: Run `python bots/main/main.py`, `streamlit run dashboard.py`.
5. Deploy on Render: 7 services with `.env` settings.
6. Monitor profits via `dashboard.py`, aim for 15-25% short-term gains.