# config/settings.py
"""
Application-wide constants and default parameters.
"""
import pytz
import numpy as np # Added for optimization ranges

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
STRATEGY_TIME_FRAME = "15m"
NY_TIMEZONE_STR = "America/New_York"
NY_TIMEZONE = pytz.timezone(NY_TIMEZONE_STR)
ENTRY_WINDOW_START_HOUR = 9
ENTRY_WINDOW_START_MINUTE = 30
ENTRY_WINDOW_END_HOUR = 11
ENTRY_WINDOW_END_MINUTE = 0

# Data Fetching
MAX_INTRADAY_DAYS = 60 # Max 60 days for 15m interval from yfinance

# Plotting
PLOTLY_TEMPLATE = "plotly_white"

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# UI
APP_TITLE = "Gap Guardian Strategy Backtester"

# Metric Colors
POSITIVE_METRIC_COLOR = "#28a745"  # Green
NEGATIVE_METRIC_COLOR = "#dc3545"  # Red
NEUTRAL_METRIC_COLOR = "#FAFAFA"   # Light gray/White (should match theme's textColor)

# --- Optimization Settings ---
OPTIMIZATION_METRICS = ["Total P&L", "Profit Factor", "Win Rate", "Sharpe Ratio (Annualized)", "Sortino Ratio (Annualized)", "Max Drawdown (%)"] # Added Sharpe & Sortino
DEFAULT_OPTIMIZATION_METRIC = "Total P&L"

# Default ranges for optimization parameters (min, max, number of steps/values)
# For Stop Loss Points (e.g., 5 to 30, in 5 steps)
DEFAULT_SL_POINTS_OPTIMIZATION_RANGE = {"min": 5.0, "max": 30.0, "steps": 6} # e.g., 5, 10, 15, 20, 25, 30
# For Risk-Reward Ratio (e.g., 1.0 to 4.0, in 7 steps)
DEFAULT_RRR_OPTIMIZATION_RANGE = {"min": 1.0, "max": 4.0, "steps": 7} # e.g., 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0

# Risk-Free Rate for Sharpe/Sortino calculation (annualized)
RISK_FREE_RATE = 0.01 # Example: 1%

# Trading days per year for annualizing Sharpe/Sortino
TRADING_DAYS_PER_YEAR = 252

