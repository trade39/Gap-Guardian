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
DEFAULT_STOP_LOSS_POINTS = 15.0 # Default SL for a common asset like S&P E-mini
DEFAULT_RRR = 3.0

# --- Timeframe Settings ---
AVAILABLE_TIMEFRAMES = {
    "1 Minute": "1m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Hour": "1h",
    "Daily": "1d",
    "Weekly": "1wk",
}
DEFAULT_STRATEGY_TIMEFRAME = "15m"

# --- Default Strategy Entry Window (NY Time) ---
# These are the fallback values if not optimizing entry times.
DEFAULT_ENTRY_WINDOW_START_HOUR = 9
DEFAULT_ENTRY_WINDOW_START_MINUTE = 30
DEFAULT_ENTRY_WINDOW_END_HOUR = 11
DEFAULT_ENTRY_WINDOW_END_MINUTE = 0 # End of window, signals can form on bars starting before this

NY_TIMEZONE_STR = "America/New_York"
NY_TIMEZONE = pytz.timezone(NY_TIMEZONE_STR)

# Data Fetching Limits
MAX_SHORT_INTRADAY_DAYS = 60 # For intervals < 1h (e.g., 1m, 5m, 15m, 30m)
MAX_HOURLY_INTRADAY_DAYS = 730 # For 1h interval

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
DEFAULT_OPTIMIZATION_METRIC = "Sharpe Ratio (Annualized)"

# Parameter Optimization Ranges
# For Stop Loss Points
DEFAULT_SL_POINTS_OPTIMIZATION_RANGE = {"min": 5.0, "max": 30.0, "steps": 5}
# For Risk-Reward Ratio
DEFAULT_RRR_OPTIMIZATION_RANGE = {"min": 1.0, "max": 3.0, "steps": 5}
# For Entry Window Start Hour (NY Time)
DEFAULT_ENTRY_START_HOUR_OPTIMIZATION_RANGE = {"min": 8, "max": 10, "steps": 3} # e.g., 8, 9, 10
# For Entry Window Start Minute (NY Time) - discrete values often make more sense
DEFAULT_ENTRY_START_MINUTE_OPTIMIZATION_VALUES = [0, 15, 30, 45] # Discrete list of minutes
# For Entry Window End Hour (NY Time)
DEFAULT_ENTRY_END_HOUR_OPTIMIZATION_RANGE = {"min": 10, "max": 12, "steps": 3} # e.g., 10, 11, 12
# For Entry Window End Minute (NY Time) - often fixed, e.g., 00
DEFAULT_ENTRY_END_MINUTE_OPTIMIZATION_VALUES = [0] # Usually end on the hour

# Random Search Settings
DEFAULT_RANDOM_SEARCH_ITERATIONS = 25 # Increased slightly

# Walk-Forward Optimization (WFO) Settings
DEFAULT_WFO_IN_SAMPLE_DAYS = 90
DEFAULT_WFO_OUT_OF_SAMPLE_DAYS = 30
DEFAULT_WFO_STEP_DAYS = 30
MIN_TRADES_FOR_METRICS = 3 # Reduced slightly for smaller WFO segments

# Risk-Free Rate for Sharpe/Sortino calculation (annualized)
RISK_FREE_RATE = 0.01
TRADING_DAYS_PER_YEAR = 252
