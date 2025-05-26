# config/settings.py
"""
Application-wide constants and default parameters.
"""
import pytz

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
MAX_INTRADAY_DAYS = 60

# Plotting
PLOTLY_TEMPLATE = "plotly_white" # "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# UI
APP_TITLE = "Gap Guardian Strategy Backtester"

# Metric Colors (Ensure these contrast well with your dark theme's secondaryBackgroundColor)
POSITIVE_METRIC_COLOR = "#28a745"  # Green
NEGATIVE_METRIC_COLOR = "#dc3545"  # Red
NEUTRAL_METRIC_COLOR = "#FAFAFA"   # Light gray/White (should match theme's textColor)

