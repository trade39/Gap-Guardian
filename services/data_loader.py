# services/data_loader.py
"""
Handles fetching and preparing market data using yfinance.
"""
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from config.settings import MAX_INTRADAY_DAYS, STRATEGY_TIME_FRAME, NY_TIMEZONE_STR
from utils.logger import get_logger

logger = get_logger(__name__)

@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_historical_data(ticker: str, start_date: datetime, end_date: datetime, interval: str = STRATEGY_TIME_FRAME) -> pd.DataFrame:
    """
    Fetches historical market data from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., "GC=F", "^GSPC").
        start_date (datetime.date): The start date for the data.
        end_date (datetime.date): The end date for the data.
        interval (str): Data interval (e.g., "15m", "1h", "1d").

    Returns:
        pd.DataFrame: DataFrame with OHLCV data, indexed by Datetime.
                      Returns an empty DataFrame if data fetching fails or no data is found.
    """
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date} with interval {interval}")
    
    # For intraday data, yfinance has limitations on the date range (e.g., 60 days for <1h intervals)
    # Adjust start_date if it's too far in the past for the requested interval
    if interval not in ["1d", "1wk", "1mo"]:
        max_start_date = datetime.now() - timedelta(days=MAX_INTRADAY_DAYS)
        if start_date < max_start_date:
            logger.warning(f"Requested start date {start_date} is too old for {interval} data. Adjusting to {max_start_date}.")
            start_date = max_start_date
    
    if start_date >= end_date:
        logger.error(f"Start date {start_date} must be before end date {end_date}.")
        st.error(f"Start date {start_date.strftime('%Y-%m-%d')} must be before end date {end_date.strftime('%Y-%m-%d')}.")
        return pd.DataFrame()

    try:
        # yf.download expects end_date to be exclusive for daily, but inclusive for intraday if it's just a date.
        # To be safe, add one day to end_date for daily, and ensure time component is handled for intraday.
        # However, for intraday, yfinance handles start/end well if they are datetime objects.
        # For simplicity, we'll use date objects and let yfinance handle it.
        # yfinance typically returns data up to, but not including, the end_date if it's just a date.
        # So, to include the end_date, we might need to fetch end_date + 1 day.
        
        data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), interval=interval, progress=False)
        
        if data.empty:
            logger.warning(f"No data found for {ticker} in the specified range and interval.")
            st.warning(f"No data found for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} at {interval} interval.")
            return pd.DataFrame()

        # Ensure DateTimeIndex is timezone-aware (yfinance often returns UTC or exchange local)
        if data.index.tz is None:
            logger.info(f"Localizing naive DatetimeIndex from yfinance to UTC for {ticker}.")
            data = data.tz_localize('UTC')
        
        # Convert to New York time as strategy logic depends on it
        logger.info(f"Converting data for {ticker} to {NY_TIMEZONE_STR} timezone.")
        data = data.tz_convert(NY_TIMEZONE_STR)
        
        # Filter out data outside the requested end_date (yf might give extra day)
        data = data[data.index.date <= end_date]

        # Drop rows with NaN values, common in crypto or less liquid assets
        data.dropna(inplace=True)
        
        logger.info(f"Successfully fetched and processed {len(data)} rows for {ticker}.")
        return data

    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}", exc_info=True)
        st.error(f"Failed to fetch data for {ticker}. Error: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example usage:
    # Make sure to run this from the root directory of the project for imports to work
    # Or adjust PYTHONPATH
    from datetime import date
    
    # Test with an index
    sp500_data = fetch_historical_data("^GSPC", date(2024, 1, 1), date(2024, 3, 1), "15m")
    if not sp500_data.empty:
        print("\nS&P 500 Data (first 5 rows):")
        print(sp500_data.head())
        print(f"Timezone: {sp500_data.index.tz}")

    # Test with a commodity future
    gold_data = fetch_historical_data("GC=F", date(2024, 4, 1), date(2024, 5, 1), "15m")
    if not gold_data.empty:
        print("\nGold Futures Data (first 5 rows):")
        print(gold_data.head())
        print(f"Timezone: {gold_data.index.tz}")

    # Test with Forex
    eurusd_data = fetch_historical_data("EURUSD=X", date(2024, 5, 1), date.today(), "15m")
    if not eurusd_data.empty:
        print("\nEURUSD Data (first 5 rows):")
        print(eurusd_data.head())
        print(f"Timezone: {eurusd_data.index.tz}")
    else:
        print("\nNo EURUSD data found or error occurred.")
