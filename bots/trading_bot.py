from api import API
import logging

# Setup logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class TradingBot:
    def __init__(self):
        self.api = API()
        self.active = True

    def get_signals(self):
        """Fetch signals from test indicators."""
        try:
            data = self.api.get_market_data()
            # Example logic: buy if price above threshold, sell if below
            price = data.get('price', 0)
            signal = "buy" if price > 100 else "sell" if price < 90 else "hold"
            logging.info(f"Generated signal: {signal}, price: {price}")
            return signal
        except Exception as e:
            logging.error(f"Error fetching signals: {e}")
            return "hold"

    def execute_trade(self, signal):
        """Execute trade based on signal."""
        try:
            if signal == "buy":
                self.api.place_order("buy", amount=1)
                logging.info("Placed buy order")
            elif signal == "sell":
                self.api.place_order("sell", amount=1)
                logging.info("Placed sell order")
            else:
                logging.info("No trade executed: hold signal")
        except Exception as e:
            logging.error(f"Error executing trade: {e}")

    def run(self):
        """Main loop to run the bot."""
        while self.active:
            try:
                signal = self.get_signals()
                self.execute_trade(signal)
                # Pause to avoid overwhelming the API
                import time
                time.sleep(60)  # Wait 1 minute between checks
            except KeyboardInterrupt:
                logging.info("Shutting down trading bot")
                self.active = False
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()