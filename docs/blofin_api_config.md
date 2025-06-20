# BloFin API Configuration for NovaraTradingBot

## Overview
This document configures the BloFin API for NovaraTradingBot’s multi-bot architecture, enabling specialized bots to fetch market data and the main bot to execute trades.

## Setup Instructions
1. **Create API Keys**:
   - Log in: https://www.blofin.com
   - Account > API Management > Create API Key (Read/Trade permissions).
   - Copy API Key and Secret Key.
2. **Add to Environment** (all bots):
   - In Render, add:
     ```
     BLOFIN_API_KEY=your_api_key
     BLOFIN_API_SECRET=your_secret_key
     ```
3. **IP Whitelist**:
   - Add Render IPs to BloFin’s API settings or use a VPN.

## Multi-Bot API Access
- **Purpose**: Specialized bots (e.g., trend-bot) fetch data independently.
- **Setup**:
  1. Ensure each bot has `BLOFIN_API_KEY` and `BLOFIN_API_SECRET`.
  2. Implement rate limiting (e.g., 1 request/second per bot).
  3. Code (trend-bot example):
     ```javascript
     const fetch = require('node-fetch');
     async function fetchCandles(symbol = 'BTCUSD') {
       try {
         const response = await fetch(`https://api.blofin.com/v1/market/candles?symbol=${symbol}`, {
           headers: {
             'X-API-KEY': process.env.BLOFIN_API_KEY,
             'X-API-SECRET': process.env.BLOFIN_API_SECRET
           }
         });
         if (!response.ok) throw new Error(`HTTP ${response.status}`);
         const data = await response.json();
         return data;
       } catch (error) {
         console.error('API request failed:', error);
         return null;
       }
     }
     ```
  4. Send results to main bot (from `webhook_setup.md`).

## Sample Request
Parallel candlestick fetch (trend-bot):
```javascript
const fetch = require('node-fetch');
async function fetchMultipleCandles(symbols = ['BTCUSD', 'ETHUSD']) {
  const promises = symbols.map(symbol =>
    fetch(`https://api.blofin.com/v1/market/candles?symbol=${symbol}`, {
      headers: {
        'X-API-KEY': process.env.BLOFIN_API_KEY,
        'X-API-SECRET': process.env.BLOFIN_API_SECRET
      }
    }).then(res => res.json())
  );
  return Promise.all(promises);
}
```
Test:
```bash
curl -H "X-API-KEY: your_key" -H "X-API-SECRET: your_secret" https://api.blofin.com/v1/market/candles?symbol=BTCUSD
```

## Common Errors
| Error                     | Cause                            | Solution                              |
|---------------------------|----------------------------------|---------------------------------------|
| 403 Forbidden             | IP not whitelisted               | Add Render IPs or use VPN.            |
| 429 Too Many Requests     | Multiple bots overloading API    | Implement rate limiting per bot.      |
| 401 Unauthorized          | Invalid keys                     | Regenerate keys; check Render env.    |

## Notes
- **API Endpoint**: https://api.blofin.com/v1/market/candles
- **Documentation**: https://docs.blofin.com/index.html#overview
- **Support**: https://support.blofin.com/hc/en-us
- **Last Updated**: June 20, 2025, 02:16 AM PDT