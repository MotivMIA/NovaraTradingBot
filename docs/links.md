## NovaraTradingBot Links

This document serves as the central hub for all critical URLs related to the NovaraTradingBot project, including code repositories, webhook configurations, API endpoints, deployment details, and communication channels. It ensures seamless collaboration, accurate configurations, and error-free operations for the trading bot.

---

### Project Repositories

- **Main Repository**: [NovaraTradingBot GitHub Repository](https://github.com/MotivMIA/NovaraTradingBot)  
    The core codebase for NovaraTradingBot, hosting source files and configurations.

- **Documentation Folder**: [Docs Folder](https://github.com/MotivMIA/NovaraTradingBot/tree/main/docs)  
    Create a `/docs` folder in the repository to store documentation files.  
    Run: `mkdir docs && git add docs && git commit -m "Create docs folder" && git push origin main`.

- **Webhook Setup Guide**: [Webhook Setup Documentation](https://github.com/MotivMIA/NovaraTradingBot/blob/main/docs/webhook_setup.md)  
    Create `webhook_setup.md` in `/docs` with instructions for Discord and BloFin webhooks (see Notes for sample content).

- **BloFin API Configuration**: [BloFin API Config Documentation](https://github.com/MotivMIA/NovaraTradingBot/blob/main/docs/blofin_api_config.md)  
    Optional: Create `blofin_api_config.md` to document API keys, endpoints, and authentication details.

---

### Webhooks

- **Discord Webhook**: [Discord Webhook URL](https://discord.com/api/webhooks/1385514247626297365/cBZ6GkIeM4pxe1JJLoVsl_dSP2m8VaCk2d4iCFtBsQHofpMzJDcIEwkY1kFg7QuDK-r6)  
    Sends real-time notifications (e.g., trade executions) to a Discord channel.  
    Test with:  
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"content":"Test from NovaraTradingBot"}' <URL>
    ```  
    If it fails (e.g., 400 Bad Request), regenerate in Discord: Channel > Edit > Integrations > Webhooks.

- **BloFin Signal Webhook**: [BloFin Signal Webhook URL](https://demo-trading-api.blofin.com/uapi/v1/algo/signal/trigger)  
    Triggers automated trades in BloFin’s Signal Bot (demo environment).  
    Integrated into bot code due to lack of TradingView account (see `webhook_setup.md`).  
    For production, update to: `https://api.blofin.com/uapi/v1/algo/signal/trigger`.  
    Test with:  
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{"action":"buy","marketPosition":"long","prevMarketPosition":"flat","instrument":"BTCUSD","signalToken":"+50PtBb5CLeGah94UCNpAhgtnSdgXDbZqzVIkWq3To9egvRxc5TSpn/tzsDtgOPyRodeY1Dm2EpPLgqQplb9ew==","timestamp":"2025-06-20T01:53:00Z","maxLag":"60","investmentType":"base","amount":"10","id":"test123"}' <URL>
    ```

---

### APIs

- **BloFin API Endpoint**: [BloFin Market Data API](https://api.blofin.com/v1/market/candles)  
    Retrieves market data (e.g., BTC/USD candlestick data) for trading algorithms.  
    Test with:  
    ```bash
    curl -H "X-API-KEY: your_key" -H "X-API-SECRET: your_secret" <URL>?symbol=BTCUSD
    ```  
    If IP restricted, use a VPN or contact BloFin support.

- **BloFin API Documentation**: [BloFin API Docs](https://docs.blofin.com/index.html#overview)  
    Comprehensive guide for API setup, endpoints, and authentication.

- **BloFin Support**: [BloFin Support Center](https://support.blofin.com/hc/en-us)  
    Resource for resolving API issues, account queries, or IP restrictions.

---

### Deployment

- **Render Project Dashboard**: [Render Dashboard](https://dashboard.render.com/project/prj-d17eu27diees73e5ap40)  
    Centralized management for the NovaraTradingBot service, including environment variables and logs.

- **Deployed Bot URL**: [NovaraTradingBot Render URL](https://novara-tradingbot.onrender.com)  
    Placeholder: Replace with the actual public URL from Render (Dashboard > novara-tradingbot > Open).  
    If no public URL, confirm if the service is a private worker or misconfigured (check Settings > Type, Port).

- **Render Documentation**: [Render Docs](https://docs.render.com)  
    Reference for deployment configurations, environment variables, and troubleshooting.

---

### Communication

- **Discord Server**: [NovaraTradingBot Discord](https://discord.gg/ZcM6r4Pd)  
    Primary hub for team discussions, bot notifications, and collaboration.  
    Ensure invite is active; regenerate if expired (Server > Invite People).

- **Google Drive Folder**: [Shared Google Drive Folder](https://drive.google.com/drive/folders/1SG64pWFwuPpE89yTMg1tDPzOf_BqH2as?usp=drive_link)  
    Shared repository for project documents, such as trading logs and configuration files.

- **GitHub Issues**: [NovaraTradingBot GitHub Issues](https://github.com/MotivMIA/NovaraTradingBot/issues)  
    Platform for tracking bugs, feature requests, and project tasks (e.g., webhook errors, API failures).

---

### Additional Resources

- **Docker Hub Repository**: [NovaraTradingBot Docker Hub](https://hub.docker.com/r/MotivMIA/novaratradingbot)  
    Create a Docker Hub repository under MotivMIA if using Docker images. Update if using a different account.

- **TradingView**: [TradingView](https://www.tradingview.com)  
    Optional: Requires Pro/Pro+/Premium for webhook alerts. Currently using bot code for BloFin signals.

---