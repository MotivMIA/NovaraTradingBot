import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
from config import Config
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioManagement:
    def __init__(self):
        self.config = Config()

    def optimize_portfolio(self, returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> Optional[Dict]:
        try:
            mu = expected_returns.mean_historical_return(returns)
            S = risk_models.sample_cov(returns)
            ef = EfficientFrontier(mu, S)
            weights = ef.max_sharpe(risk_free_rate=self.config.RISK_PER_TRADE)
            cleaned_weights = ef.clean_weights()
            logger.info(f"Optimized allocations: {cleaned_weights}")
            return {"allocations": cleaned_weights, "expected_return": ef.portfolio_performance()[0]}
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            return None

    def adjust_position(self, symbol: str, signal: str, price: float, portfolio: Dict) -> Optional[Dict]:
        try:
            current_position = portfolio.get(symbol, 0)
            if signal == "buy":
                new_position = current_position + self.config.INITIAL_BID_USD / price
            elif signal == "sell":
                new_position = current_position - self.config.INITIAL_BID_USD / price
            else:
                return portfolio

            portfolio[symbol] = max(new_position, 0)
            logger.info(f"Adjusted position for {symbol}: {portfolio[symbol]}")
            return portfolio
        except Exception as e:
            logger.error(f"Error adjusting position for {symbol}: {e}")
            return None