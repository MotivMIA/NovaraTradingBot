# Bot Architecture

## Overview
NovaraTradingBot uses a 6-bot architecture:
- **Main Bot**: Aggregates signals, executes trades via BloFin, hosts FastAPI endpoints (`bots/main/main.py`).
- **TrendBot**: Momentum with Ichimoku, ADX, VWAP (`bots/trendbot/main.py`).
- **VolumeBot**: Volatility breakouts with ATR, ADX, VWAP (`bots/volumebot/main.py`).
- **PredictBot**: ML with X/news sentiment, XGBoost (`bots/predictbot/main.py`).
- **ArbitrageBot**: Cross-exchange arbitrage (BloFin, OKX, Binance) (`bots/arbitragebot/main.py`).
- **PatternBot**: Mean-reversion with candlestick patterns (doji, hammer) (`bots/patternbot/main.py`).
- **NewsBot**: News-driven signals (`bots/newsbot/main.py`).

## Signal Flow
1. Bots fetch candles (`api_utils.py`), generate signals (`trading_logic.py`).
2. Signals sent to Main Bot’s `/receive` (https://novara-tradingbot.onrender.com/receive).
3. Main Bot scores signals (+1 buy, -1 sell, trades if |score| ≥ 3, confidence > 0.5).
4. Trades executed (`exchange_interface.py`), logged (`performance_analytics.py`), alerted (`notifications.py`).

## Infrastructure
- **Render**: 7 services (~$49/month).
- **PostgreSQL**: Candle/trade storage (`features/database.py`, `DB_PATH`).
- **Redis**: Optional event bus (`features/event_bus.py`, `REDIS_HOST`).
- **Streamlit**: Dashboard with VWAP, candlestick visuals (`dashboard.py`).
- **APIs**: BloFin, X (paid), OKX, Binance, CoinDesk RSS.

## Deployment
- **Repo**: `MotivMIA/NovaraTradingBot` with `bots/`, `features/`, `docs/`.
- **Render**: Deploy each bot (`python bots/main/main.py`, etc.) with `.env`.
- **Config**: Set `API_KEY`, `XAI_API_KEY`, `OKX_API_KEY`, `BINANCE_API_KEY` in `.env`.

## Testing
- Backtest: `python -m backtesting`.
- Demo: `DEMO_MODE=True`.
- Unit Tests: `pytest tests/`.
- Signal Test:
  ```bash
  curl -X POST https://novara-tradingbot.onrender.com/receive -d '{"symbol":"BTC-USDT","signal":"buy","confidence":0.8,"patterns":["hammer"],"timeframes":["1h"],"bot_name":"PatternBot"}'
  ```