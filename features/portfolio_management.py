import pandas as pd
import numpy as np
import logging
from pypfopt import EfficientFrontier, risk_models, expected_returns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioManagement:
    def select_symbols(self, bot, max_symbols: int = 10) -> list:
        try:
            correlations = self.calculate_correlations(bot)
            vols = {s: np.std([c["close"] for c in bot.candle_history[s]]) for s in bot.symbols}
            vols = sorted(vols.items(), key=lambda x: x[1], reverse=True)
            selected = []
            for symbol, _ in vols[:max_symbols]:
                if all(correlations.get((symbol, s), 0) < 0.7 for s in selected):
                    selected.append(symbol)
            return selected[:max_symbols]
        except Exception as e:
            logger.error(f"Error selecting symbols: {e}")
            return bot.symbols[:max_symbols]

    def calculate_margin(self, symbol: str, risk_amount: float, confidence: float, price_change: float, patterns: list, bot):
        try:
            atr = bot.indicators.calculate_indicators(symbol, bot.candle_history[symbol])["atr"]
            risk_factor = min(1.5, confidence * (price_change / bot.config.VOLATILITY_THRESHOLD) / (atr / bot.candle_history[symbol][-1]["close"]))
            dynamic_risk = bot.config.RISK_PER_TRADE * risk_factor
            effective_leverage = min(self.get_effective_leverage(symbol, bot), bot.config.MAX_LEVERAGE_PERCENTAGE * bot.api_utils.get_max_leverage(symbol, bot))
            adjusted_margin = bot.account_balance * dynamic_risk / effective_leverage
            return adjusted_margin, effective_leverage
        except Exception as e:
            logger.error(f"Error calculating margin for {symbol}: {e}")
            return risk_amount, 1.0

    def optimize_portfolio(self, bot):
        try:
            prices = {s: [c["close"] for c in bot.candle_history[s]] for s in bot.symbols}
            df = pd.DataFrame(prices).pct_change()
            mu = expected_returns.mean_historical_return(df)
            S = risk_models.sample_cov(df)
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            logger.info(f"Portfolio weights: {cleaned_weights}")
            return {s: w for s, w in cleaned_weights.items()}
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return {s: 1/len(bot.symbols) for s in bot.symbols}

    def calculate_correlations(self, bot):
        try:
            prices = {s: [c["close"] for c in bot.candle_history[s]] for s in bot.symbols}
            df = pd.DataFrame(prices).pct_change()
            corr_matrix = df.corr()
            correlations = {}
            for i in range(len(corr_matrix)):
                for j in range(i + 1, len(corr_matrix)):
                    correlations[(corr_matrix.index[i], corr_matrix.index[j])] = corr_matrix.iloc[i, j]
            return correlations
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return {}

    def get_effective_leverage(self, symbol: str, bot):
        return 1.0  # Placeholder