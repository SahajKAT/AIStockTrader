# Importing necessary modules and classes from lumibot and alpaca_trade_api, as well as datetime utilities and a custom sentiment analysis function.
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime, timedelta
from alpaca_trade_api import REST
from finbert_utils import estimate_sentiment

# Alpaca API credentials and the base URL for the paper trading API endpoint.
API_KEY = "PKECKEBF1W0K500MT3VA"
API_SECRET = "n5ScjgkSJHTtBdgEEiI0lZSOOh11QoqbXjolMzSK"
BASE_URl = "https://paper-api.alpaca.markets/v2"

# Configuration settings for Alpaca API; including API keys and specifying it as a paper trading account.
ALPACA_CONFIG = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True,
}

# Defining a trading strategy class that inherits from Strategy class provided by lumibot.
class MLTrader(Strategy):
    # Initialization method with default parameters.
    def initialize(self, symbol: str = "SPY", cash_at_risk: float = .5):
        self.symbol = symbol  # Symbol to trade.
        self.sleeptime = "24H"  # Interval between trading iterations.
        self.last_trade = None  # Tracks the last trading action to manage trading strategy.
        self.cash_at_risk = cash_at_risk  # Fraction of cash to risk on each trade.
        # Initializes Alpaca REST API client for accessing market data and trading.
        self.api = REST(base_url=BASE_URl, key_id=API_KEY, secret_key=API_SECRET)

    # Calculates the quantity of shares to trade based on cash at risk and last price of the symbol.
    def position_sizing(self):
        cash = self.get_cash()  # Retrieves the current cash balance of the account.
        last_price = self.get_last_price(self.symbol)  # Fetches the latest trading price for the symbol.
        quantity = round(cash * self.cash_at_risk / last_price, 0)  # Calculates the quantity to trade.
        return cash, last_price, quantity

    # Determines the relevant dates for fetching news data.
    def get_dates(self):
        today = self.get_datetime()  # Current date and time.
        three_days_prior = today - timedelta(days=3)  # Date three days before the current date.
        return today.strftime("%Y-%m-%d"), three_days_prior.strftime("%Y-%m-%d")

    # Retrieves news headlines for the symbol and estimates sentiment.
    def get_sentiment(self):
        today, three_days_prior = self.get_dates()  # Fetching the formatted dates.
        # Retrieving news headlines for the symbol within the specified date range.
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        # Extracting headlines from the news data.
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        # Analyzing sentiment of the news headlines.
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    # Defines the logic for trading iterations based on sentiment analysis.
    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()  # Determines cash, last price, and quantity for trading.
        probability, sentiment = self.get_sentiment()  # Analyzes sentiment of recent news.

        # Executes trading logic based on the sentiment analysis results.
        if cash > last_price:  # Checks if there is enough cash to place an order.
            if sentiment == "positive" and probability > .999:  # Condition for a positive sentiment.
                if self.last_trade == "sell":  # Ensures not selling immediately after buying.
                    self.sell_all()  # Sells all positions before buying again.
                # Creates a buy order with specific conditions for taking profits and stopping losses.
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * 0.95
                )
                self.submit_order(order)  # Submits the buy order to the broker.
                self.last_trade = "buy"  # Updates the last trade action.
            elif sentiment == "negative" and probability > .999:  # Condition for a negative sentiment.
                if self.last_trade == "buy":  # Ensures not buying immediately after selling.
                    self.sell_all()  # Sells all positions before selling again.
                # Creates a sell order with specific conditions for taking profits and stopping losses.
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.8,
                    stop_loss_price=last_price * 1.05
                )
                self.submit_order(order)  # Submits the sell order to the broker.
                self.last_trade = "sell"  # Updates the last trade action.

# Setup for backtesting the trading strategy, including start and end dates.
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)

# Instantiating the Alpaca broker with the given configuration.
broker = Alpaca(ALPACA_CONFIG)

# Creating an instance of the MLTrader strategy with specific parameters.
strategy = MLTrader(name='mlstrat', broker=broker,
                    parameters={"symbol": "SPY", "cash_at_risk": .5})

# Running the backtest of the strategy with Yahoo Finance data and specified dates.
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "SPY", "cash_at_risk": .5}
)
