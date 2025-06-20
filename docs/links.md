# NovaraTradingBot Links

## Overview
This document centralizes all critical URLs for the NovaraTradingBot project, now structured as a multi-bot architecture with specialized bots feeding insights to a main orchestrator bot. It ensures seamless coordination, accurate configurations, and maximized trading profits.

## Quick Reference
| Category           | Description                     | URL                                           |
|--------------------|---------------------------------|-----------------------------------------------|
| Main Bot           | Trade orchestrator              | [Main Bot](https://novara-tradingbot.onrender.com/) |
| Trend & Momentum   | RSI, MACD, trends               | [Trend Bot](https://trend-bot.onrender.com/) |
| Volatility & Volume| ATR, Bollinger, volume          | [Volume Bot](https://volume-bot.onrender.com/) |
| Prediction & Sentiment | ML, X sentiment             | [Predict Bot](https://predict-bot.onrender.com/) |
| Arbitrage & Order Book | Price diffs, liquidity       | [Arbitrage Bot](https://arbitrage-bot.onrender.com/) |
| Patterns & Risk    | Candlesticks, risk ratios       | [Pattern Bot](https://pattern-bot.onrender.com/) |
| Discord Webhook    | Notifications                   | [Webhook](https://discord.com/api/webhooks/1385514247626297365/...) |
| BloFin Signal      | Trading signals (demo)          | [Webhook](https://demo-trading-api.blofin.com/uapi/v1/algo/signal/trigger/) |
| BloFin API         | Market data                     | [Endpoint](https://api.blofin.com/v1/market/candles/) |

## Project Repositories
- **Main Repository**: https://github.com/MotivMIA/NovaraTradingBot
  Hosts the main bot’s codebase.
- **Trend & Momentum Bot**: https://github.com/MotivMIA/TrendBot
  *Create repo for trend and momentum analysis.*
- **Volatility & Volume Bot**: https://github.com/MotivMIA/VolumeBot
  *Create repo for volatility and volume analysis.*
- **Prediction & Sentiment Bot**: https://github.com/MotivMIA/PredictBot
  *Create repo for prediction and sentiment analysis.*
- **Arbitrage & Order Book Bot**: https://github.com/MotivMIA/ArbitrageBot
  *Create repo for arbitrage and order book analysis.*
- **Patterns & Risk Bot**: https://github.com/MotivMIA/PatternBot
  *Create repo for pattern recognition and risk assessment.*
- **Documentation Folder**: https://github.com/MotivMIA/NovaraTradingBot/tree/main/docs
  *Ensure `/docs` exists.*
- **Webhook Setup Guide**: https://github.com/MotivMIA/NovaraTradingBot/blob/main/docs/webhook_setup.md
- **BloFin API Configuration**: https://github.com/MotivMIA/NovaraTradingBot/blob/main/docs/blofin_api_config.md

## Webhooks
- **Discord Webhook**: https://discord.com/api/webhooks/1385514247626297365/cBZ6GkIeM4pxe1JJLoVsl_dSP2m8VaCk2d4iCFtBsQHofpMzJDcIEwkY1kFg7QuDK-r6
  Sends notifications to Discord.
  *Test: `curl -X POST -H "Content-Type: application/json" -d '{"content":"Test"}' <URL>`*
- **BloFin Signal Webhook**: https://demo-trading-api.blofin.com/uapi/v1/algo/signal/trigger
  Triggers trades in BloFin’s Signal Bot (demo).
  *Test: `curl -X POST -H "Content-Type: application/json" -d '{"action":"buy","marketPosition":"long","prevMarketPosition":"flat","instrument":"BTCUSD","signalToken":"...","timestamp":"2025-06-20T02:16:00Z","maxLag":"60","investmentType":"base","amount":"10","id":"test123"}' <URL>`*
  *Update to production: https://api.blofin.com/uapi/v1/algo/signal/trigger*

## APIs
- **Main Bot Receive Endpoint**: https://novara-tradingbot.onrender.com/receive
  Receives analysis from specialized bots.
  *Test: `curl -X POST -H "Content-Type: application/json" -d '{"botId":"trend-bot","analysis":{"rsi":75,"signal":"sell"}}' <URL>`*
- **BloFin API Endpoint**: https://api.blofin.com/v1/market/candles
  Fetches market data.
  *Test: `curl -H "X-API-KEY: your_key" -H "X-API-SECRET: your_secret" <URL>?symbol=BTCUSD`*
- **BloFin API Documentation**: https://docs.blofin.com/index.html#overview
- **BloFin Support**: https://support.blofin.com/hc/en-us

## Deployment
- **Render Project Dashboard**: https://dashboard.render.com/project/prj-d17eu27diees73e5ap40
- **Main Bot URL**: https://novara-tradingbot.onrender.com
  *Confirm public URL in Render.*
- **Trend & Momentum Bot URL**: https://trend-bot.onrender.com
  *Create Render service.*
- **Volatility & Volume Bot URL**: https://volume-bot.onrender.com
  *Create Render service.*
- **Prediction & Sentiment Bot URL**: https://predict-bot.onrender.com
  *Create Render service.*
- **Arbitrage & Order Book Bot URL**: https://arbitrage-bot.onrender.com
  *Create Render service.*
- **Patterns & Risk Bot URL**: https://pattern-bot.onrender.com
  *Create Render service.*
- **Render Documentation**: https://docs.render.com

## Communication
- **Discord Server**: https://discord.gg/ZcM6r4Pd
  *Verify invite.*
- **Google Drive Folder**: https://drive.google.com/drive/folders/1SG64pWFwuPpE89yTMg1tDPzOf_BqH2as?usp=drive_link
- **GitHub Issues**: https://github.com/MotivMIA/NovaraTradingBot/issues

## Additional Resources
- **Docker Hub Repository**: https://hub.docker.com/r/MotivMIA/novaratradingbot
  *Update for each bot if using Docker.*
- **TradingView**: https://www.tradingview.com
  *Optional for future webhook integration.*

## Usage
- **Maintenance**: Update URLs as new bots are deployed. Push to GitHub:
  ```bash
  git add docs/links.md
  git commit -m "Update links.md with new bots"
  git push origin main
  ```
- **Verification Checklist**:
  - [ ] Test Discord and BloFin webhooks.
  - [ ] Verify main bot receives data from all specialized bots.
  - [ ] Confirm BloFin API returns valid data.
  - [ ] Ensure all Render URLs are accessible.
  - [ ] Check Discord/Google Drive access.

## Notes
- **Last Updated**: June 20, 2025, 02:16 AM PDT