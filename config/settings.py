# config/settings.py
"""
Application-wide constants and default parameters.
"""
import pytz
import numpy as np

# Default Tickers for yfinance
DEFAULT_TICKERS = {
    "Gold (XAU/USD)": "GC=F",
    "S&P 500 Index": "^GSPC",
    "NASDAQ Composite": "^IXIC",
    "EUR/USD": "EURUSD=X",
    "Bitcoin (BTC/USD)": "BTC-USD"
}

# Default Backtesting Parameters
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_RISK_PER_TRADE_PERCENT = 0.5
DEFAULT_STOP_LOSS_POINTS = 15.0
DEFAULT_RRR = 3.0

# Strategy Time Settings
STRATEGY_TIME_FRAME = "15m" # Crucial for WFO period calculations
NY_TIMEZONE_STR = "America/New_York"
NY_TIMEZONE = pytz.timezone(NY_TIMEZONE_STR)
ENTRY_WINDOW_START_HOUR = 9
ENTRY_WINDOW_START_MINUTE = 30
ENTRY_WINDOW_END_HOUR = 11
ENTRY_WINDOW_END_MINUTE = 0

# Data Fetching
MAX_INTRADAY_DAYS = 60

# Plotting
PLOTLY_TEMPLATE = "plotly_white"

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# UI
APP_TITLE = "Gap Guardian Strategy Backtester"

# Metric Colors
POSITIVE_METRIC_COLOR = "#28a745"
NEGATIVE_METRIC_COLOR = "#dc3545"
NEUTRAL_METRIC_COLOR = "#FAFAFA"

# --- Optimization Settings ---
OPTIMIZATION_ALGORITHMS = ["Grid Search", "Random Search"]
DEFAULT_OPTIMIZATION_ALGORITHM = "Grid Search"

OPTIMIZATION_METRICS = ["Total P&L", "Profit Factor", "Win Rate", "Sharpe Ratio (Annualized)", "Sortino Ratio (Annualized)", "Max Drawdown (%)"]
DEFAULT_OPTIMIZATION_METRIC = "Sharpe Ratio (Annualized)" # Changed default to a risk-adjusted metric

DEFAULT_SL_POINTS_OPTIMIZATION_RANGE = {"min": 5.0, "max": 30.0, "steps": 5} # Reduced steps for faster demo
DEFAULT_RRR_OPTIMIZATION_RANGE = {"min": 1.0, "max": 3.0, "steps": 5}      # Reduced steps for faster demo

# Random Search Settings
DEFAULT_RANDOM_SEARCH_ITERATIONS = 20 # Number of random parameter sets to try

# Walk-Forward Optimization (WFO) Settings
# Note: These are in CALENDAR DAYS. Actual trading days will be less.
# For a "15m" strategy, ensure these periods are not too short relative to data availability.
DEFAULT_WFO_IN_SAMPLE_DAYS = 90      # e.g., ~3 months of data for training/optimization
DEFAULT_WFO_OUT_OF_SAMPLE_DAYS = 30  # e.g., ~1 month of data for testing
DEFAULT_WFO_STEP_DAYS = 30           # How often to re-optimize (roll forward by OOS period)
MIN_TRADES_FOR_METRICS = 5           # Minimum trades in a period to consider metrics valid

# Risk-Free Rate for Sharpe/Sortino calculation (annualized)
RISK_FREE_RATE = 0.01
TRADING_DAYS_PER_YEAR = 252
