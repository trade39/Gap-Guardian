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

# --- Timeframe Settings ---
# Valid yfinance intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
AVAILABLE_TIMEFRAMES = {
    "1 Minute": "1m",
    "5 Minutes": "5m",
    "15 Minutes": "15m",
    "30 Minutes": "30m",
    "1 Hour": "1h", # or "60m" - "1h" is generally preferred by yfinance for >60d history
    "4 Hours": "4h", # Note: yfinance does not have a direct "4h". Typically use 1h and resample or use daily. For direct fetch, this might be an issue. Let's use 1h as max intraday for now.
    "Daily": "1d",
    "Weekly": "1wk"
}
# Let's refine AVAILABLE_TIMEFRAMES for common use and yfinance compatibility
AVAILABLE_TIMEFRAMES = {
    "1 Minute": "1m",   # Very short history from yfinance (usually 7 days)
    "5 Minutes": "5m",  # Short history (usually 60 days)
    "15 Minutes": "15m",# Short history (usually 60 days)
    "30 Minutes": "30m",# Short history (usually 60 days)
    "1 Hour": "1h",     # Longer history (up to 730 days)
    "Daily": "1d",      # Very long history
    "Weekly": "1wk",    # Very long history
}
DEFAULT_STRATEGY_TIMEFRAME = "15m" # Strategy's native/default timeframe

# Strategy Time Window (remains based on NY time, regardless of data timeframe)
NY_TIMEZONE_STR = "America/New_York"
NY_TIMEZONE = pytz.timezone(NY_TIMEZONE_STR)
ENTRY_WINDOW_START_HOUR = 9
ENTRY_WINDOW_START_MINUTE = 30
ENTRY_WINDOW_END_HOUR = 11
ENTRY_WINDOW_END_MINUTE = 0

# Data Fetching
# MAX_INTRADAY_DAYS is for intervals like 1m, 5m, 15m, 30m.
# 1h data often has a longer history (e.g., 730 days).
# We'll handle this dynamically in data_loader.
MAX_SHORT_INTRADAY_DAYS = 60 # For intervals < 1h
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

DEFAULT_SL_POINTS_OPTIMIZATION_RANGE = {"min": 5.0, "max": 30.0, "steps": 5}
DEFAULT_RRR_OPTIMIZATION_RANGE = {"min": 1.0, "max": 3.0, "steps": 5}
# Entry Window Time Optimization Ranges (Hour as integer, Minute as integer)
DEFAULT_ENTRY_START_HOUR_OPTIMIZATION_RANGE = {"min": 8, "max": 10, "steps": 3} # e.g., 8, 9, 10
DEFAULT_ENTRY_START_MINUTE_OPTIMIZATION_RANGE = {"min": 0, "max": 45, "steps": 4} # e.g., 0, 15, 30, 45
DEFAULT_ENTRY_END_HOUR_OPTIMIZATION_RANGE = {"min": 10, "max": 12, "steps": 3} # e.g., 10, 11, 12
# Minute for end time is less common to optimize, often fixed (e.g., 00 for on the hour)

# Random Search Settings
DEFAULT_RANDOM_SEARCH_ITERATIONS = 20

# Walk-Forward Optimization (WFO) Settings
DEFAULT_WFO_IN_SAMPLE_DAYS = 90
DEFAULT_WFO_OUT_OF_SAMPLE_DAYS = 30
DEFAULT_WFO_STEP_DAYS = 30
MIN_TRADES_FOR_METRICS = 5

# Risk-Free Rate for Sharpe/Sortino calculation (annualized)
RISK_FREE_RATE = 0.01
TRADING_DAYS_PER_YEAR = 252
