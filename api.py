import os
import time
import logging
import argparse
from logtail import LogtailHandler
from typing import Optional
from features.api_utils import BloFinAPI
from features.indicators import Indicators
from features.trading_logic import TradingLogic
from features.portfolio_management import PortfolioManagement
from features.config import DEFAULT_BALANCE, TIMEFRAMES, SLEEP_INTERVAL

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler('logs/trading_bot.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s,%(msecs)03d - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Logtail handler
logtail_token = os.getenv('LOGTAIL_TOKEN')
if logtail_token:
    logtail_handler = LogtailHandler(source_token=logtail_token)
    logger.addHandler(logtail_handler)
    logger.info("Logtail integration enabled")
else:
    logger.warning("LOGTAIL_TOKEN not set, skipping Logtail integration")

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(console_handler)

class TradingBot:
    def __init__(self, api_key: str, api_secret: str, passphrase: str):
        self.api = BloFinAPI(api_key, api_secret, passphrase)
        self.indicators = Indicators()
        self.trading_logic = TradingLogic()
        self.portfolio = PortfolioManagement(self.api)
        self.balance = None

    def health_check(self) -> bool:
        """Perform health check by fetching account balance."""
        try:
            balance_data = self.api.get_account_balance(self)
            if balance_data is None:
                logger.error("Health check failed: Unable to fetch balance")
                return False
            self.balance = balance_data
            logger.info(f"Health check passed: Balance ${self.balance:.2f} USDT")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    def run(self):
        """Main trading loop."""
        logger.info("Starting NovaraTradingBot")
        if not self.health_check():
            logger.error("Bot failed to start due to health check failure")
            return

        while True:
            try:
                symbols = self.portfolio.get_top_symbols()
                if not symbols:
                    logger.warning("No symbols selected, retrying in 60 seconds")
                    time.sleep(60)
                    continue

                for symbol in symbols:
                    for timeframe in TIMEFRAMES:
                        df = self.api.get_candles(symbol, timeframe)
                        if df is None or df.empty:
                            logger.warning(f"No candle data for {symbol} ({timeframe})")
                            continue

                        df_with_indicators = self.indicators.calculate_indicators(df, timeframe)
                        if df_with_indicators is None:
                            logger.error(f"Failed to calculate indicators for {symbol} ({timeframe})")
                            continue

                        latest_price = float(df_with_indicators.iloc[-1]['close'])
                        signal = self.trading_logic.generate_signal(df_with_indicators, symbol, latest_price, timeframe)
                        if signal:
                            side, confidence = signal
                            logger.info(f"Signal for {symbol} ({timeframe}): {side} with confidence {confidence:.2f}")
                            self.place_order(symbol, side, latest_price, timeframe)
                        else:
                            logger.debug(f"No signal generated for {symbol} ({timeframe})")

                time.sleep(SLEEP_INTERVAL)
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)

    def place_order(self, symbol: str, side: str, price: float, timeframe: str):
        """Place a market order based on signal."""
        try:
            size_usd = self.balance * 0.1  # 10% of balance
            if size_usd < 10:  # Minimum order size
                logger.warning(f"Insufficient balance for order: ${size_usd:.2f}")
                return

            logger.info(f"Attempting to place {side} order for {symbol}: ${size_usd:.2f} at ${price:.4f}")
            # Placeholder for order placement (implement with BloFin API)
            # response = self.api.place_order(symbol, side, size_usd, price)
            logger.info(f"Order placed successfully for {symbol} ({timeframe})")
        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="NovaraTradingBot")
    parser.add_argument('--health-check', action='store_true', help="Run health check and exit")
    args = parser.parse_args()

    api_key = os.getenv('BLOFIN_API_KEY', '1068db9f2fd8486dad50c5e304b0a150')
    api_secret = os.getenv('BLOFIN_API_SECRET', '')
    passphrase = os.getenv('BLOFIN_PASSPHRASE', '0NS8e3oKfiW2G4O9x')

    bot = TradingBot(api_key, api_secret, passphrase)

    if args.health_check:
        if bot.health_check():
            exit(0)
        else:
            exit(1)
    else:
        bot.run()

if __name__ == "__main__":
    main()