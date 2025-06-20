# Webhook Setup for NovaraTradingBot

## Overview
This guide configures webhooks and communication for NovaraTradingBot’s multi-bot architecture, including:
- Discord notifications
- BloFin trading signals
- Inter-bot data exchange between specialized bots and the main orchestrator

---

## Discord Webhook

### URL
`https://discord.com/api/webhooks/1385514247626297365/cBZ6GkIeM4pxe1JJLoVsl_dSP2m8VaCk2d4iCFtBsQHofpMzJDcIEwkY1kFg7QuDK-r6`

### Purpose
Sends notifications (e.g., trade confirmations).

### Setup
Add to Render environment:
```plaintext
WEBHOOK_URL=https://discord.com/api/webhooks/1385514247626297365/cBZ6GkIeM4pxe1JJLoVsl_dSP2m8VaCk2d4iCFtBsQHofpMzJDcIEwkY1kFg7QuDK-r6
```

### Code (Node.js)
```javascript
const fetch = require('node-fetch');
async function sendNotification(message) {
  try {
    const response = await fetch(process.env.WEBHOOK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: message })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    console.error('Notification failed:', error);
  }
}
```

### Test
```bash
curl -X POST -H "Content-Type: application/json" -d '{"content":"Test"}' <URL>
```

---

## BloFin Signal Webhook

### URL
`https://demo-trading-api.blofin.com/uapi/v1/algo/signal/trigger`

### Purpose
Triggers trades in BloFin’s Signal Bot (demo).

### Setup
Add to Render environment (main bot):
```plaintext
BLOFIN_SIGNAL_WEBHOOK=https://demo-trading-api.blofin.com/uapi/v1/algo/signal/trigger
BLOFIN_SIGNAL_TOKEN=+50PtBb5CLeGah94UCNpAhgtnSdgXDbZqzVIkWq3To9egvRxc5TSpn/tzsDtgOPyRodeY1Dm2EpPLgqQplb9ew==
```

### Code (main bot, with scoring)
```javascript
const fetch = require('node-fetch');
async function sendBloFinSignal(analysis) {
  const score = analysis.reduce((sum, { signal }) => sum + (signal === 'buy' ? 1 : signal === 'sell' ? -1 : 0), 0);
  if (Math.abs(score) < 3) return; // Require consensus
  const action = score > 0 ? 'buy' : 'sell';
  const payload = {
    action,
    marketPosition: action === 'buy' ? 'long' : 'flat',
    prevMarketPosition: action === 'buy' ? 'flat' : 'long',
    instrument: 'BTCUSD',
    signalToken: process.env.BLOFIN_SIGNAL_TOKEN,
    timestamp: new Date().toISOString(),
    maxLag: '60',
    investmentType: 'base',
    amount: '10',
    id: `order-${Date.now()}`
  };
  try {
    const response = await fetch(process.env.BLOFIN_SIGNAL_WEBHOOK, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    await sendNotification(`Trade sent: ${action} BTCUSD`);
  } catch (error) {
    console.error('Signal failed:', error);
    await sendNotification(`Trade failed: ${error.message}`);
  }
}
```

### Test
```bash
curl -X POST -H "Content-Type: application/json" -d '{"action":"buy","marketPosition":"long","prevMarketPosition":"flat","instrument":"BTCUSD","signalToken":"...","timestamp":"2025-06-20T02:16:00Z","maxLag":"60","investmentType":"base","amount":"10","id":"test123"}' <URL>
```

---

## Inter-Bot Communication

### Purpose
Specialized bots (e.g., trend-bot) send analysis to the main bot.

### Setup
Add to Render environment (specialized bots):
```plaintext
MAIN_BOT_URL=https://novara-tradingbot.onrender.com
```

### Code (specialized bot, e.g., trend-bot)
```javascript
const fetch = require('node-fetch');
async function sendAnalysis(analysis) {
  try {
    const response = await fetch(`${process.env.MAIN_BOT_URL}/receive`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ botId: 'trend-bot', analysis })
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
  } catch (error) {
    console.error('Analysis send failed:', error);
  }
}
sendAnalysis({ rsi: 75, signal: 'sell' });
```

### Test
```bash
curl -X POST -H "Content-Type: application/json" -d '{"botId":"trend-bot","analysis":{"rsi":75,"signal":"sell"}}' https://novara-tradingbot.onrender.com/receive
```

---

## Best Practices

- **Rate Limiting**: Cap webhook calls at 1/second.
- **Error Logging**: Log failures to Render logs and Discord.
- **Security**: Use environment variables for URLs/tokens.
- **Monitoring**: Alert on repeated failures via Discord.

---

## Troubleshooting

| Issue                  | Cause                     | Solution                          |
|------------------------|---------------------------|-----------------------------------|
| Discord 400 Bad Request | Invalid URL/payload       | Regenerate webhook; check JSON.  |
| BloFin No Events        | Wrong signalToken/URL     | Verify token; test with curl.    |
| Inter-Bot Failure       | Main bot down or wrong URL | Check main bot logs; verify URL. |

---

### Notes
**Last Updated**: June 20, 2025, 02:16 AM PDT

