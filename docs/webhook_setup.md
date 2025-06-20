# Webhook Setup for NovaraTradingBot

## Discord Webhook
- **URL**: https://discord.com/api/webhooks/1385514247626297365/cBZ6GkIeM4pxe1JJLoVsl_dSP2m8VaCk2d4iCFtBsQHofpMzJDcIEwkY1kFg7QuDK-r6
- **Setup**:
  1. Add to `.env.local`: `WEBHOOK_URL=https://discord.com/api/webhooks/1385514247626297365/...`
  2. Use in code:
     ```javascript
     const fetch = require('node-fetch');
     async function sendNotification(message) {
       await fetch(process.env.WEBHOOK_URL, {
         method: 'POST',
         headers: { 'Content-Type': application/json' },
         body: JSON.stringify({ content: message })
       });
     }
     ```
  3. Test: `curl -X POST -H "Content-Type: application/json" -d '{"content":"Test"}' <URL>`

## BloFin Signal Webhook
- **URL**: https://demo-trading-api.blofin.com/uapi/v1/algo/signal/trigger
- **Setup**:
  1. Add to Render environment:
     ```
     BLOFIN_SIGNAL_WEBHOOK=https://demo-trading-api.blofin.com/uapi/v1/algo/signal/trigger
     BLOFIN_SIGNAL_TOKEN=+50PtBb5CLeGah94UCNpAhgtnSdgXDbZqzVIkWq3To9egvRxc5TSpn/tzsDtgOPyRodeY1Dm2EpPLgqQplb9ew==
     ```
  2. Use in code:
     ```javascript
     const fetch = require('node-fetch');
     async function sendBloFinSignal(action, marketPosition, prevMarketPosition, instrument, amount, orderId) {
       const payload = {
         action,
         marketPosition,
         prevMarketPosition,
         instrument,
         signalToken: process.env.BLOFIN_SIGNAL_TOKEN,
         timestamp: new Date().toISOString(),
         maxLag: "60",
         investmentType: "base",
         amount: amount.toString(),
         id: orderId
       };
       await fetch(process.env.BLOFIN_SIGNAL_WEBHOOK, {
         method: 'POST',
         headers: { 'Content-Type': application/json' },
         body: JSON.stringify(payload)
       });
     }
     ```
  3. Test: `curl -X POST -H "Content-Type: application/json" -d '{"action":"buy","marketPosition":"long","prevMarketPosition":"flat","instrument":"BTCUSD","signalToken":"...","timestamp":"2025-06-20T00:36:00Z","maxLag":"60","investmentType":"base","amount":"10","id":"test123"}' <URL>`