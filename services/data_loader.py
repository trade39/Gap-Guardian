# services/data_loader.py
"""
Handles fetching and preparing market data using yfinance.
"""
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta, date # Ensure date is imported
from config.settings import MAX_INTRADAY_DAYS, STRATEGY_TIME_FRAME, NY_TIMEZONE_STR, NY_TIMEZONE
from utils.logger import get_logger

logger = get_logger(__name__)

@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_historical_data(ticker: str, start_date_input: date, end_date_input: date, interval: str = STRATEGY_TIME_FRAME) -> pd.DataFrame:
    """
    Fetches historical market data from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol (e.g., "GC=F", "^GSPC").
        start_date_input (datetime.date): The start date for the data (from st.date_input).
        end_date_input (datetime.date): The end date for the data (from st.date_input).
        interval (str): Data interval (e.g., "15m", "1h", "1d").

    Returns:
        pd.DataFrame: DataFrame with OHLCV data, indexed by Datetime.
                      Returns an empty DataFrame if data fetching fails or no data is found.
    """
    logger.info(f"Fetching data for {ticker} from {start_date_input} to {end_date_input} with interval {interval}")
    
    # Make a mutable copy of start_date_input for potential modification
    current_start_date = start_date_input

    # For intraday data, yfinance has limitations on the date range (e.g., 60 days for <1h intervals)
    # Adjust start_date if it's too far in the past for the requested interval
    if interval not in ["1d", "1wk", "1mo"]:
        # max_start_date_dt is datetime.datetime, convert its date part for comparison
        max_start_date_limit_dt = datetime.now(NY_TIMEZONE) - timedelta(days=MAX_INTRADAY_DAYS)
        max_start_date_limit_date = max_start_date_limit_dt.date() # Convert to datetime.date

        if current_start_date < max_start_date_limit_date:
            logger.warning(f"Requested start date {current_start_date} is too old for {interval} data. Adjusting to {max_start_date_limit_date}.")
            current_start_date = max_start_date_limit_date # Assign the adjusted date
    
    if current_start_date >= end_date_input:
        logger.error(f"Start date {current_start_date} must be before end date {end_date_input}.")
        st.error(f"Start date {current_start_date.strftime('%Y-%m-%d')} must be before end date {end_date_input.strftime('%Y-%m-%d')}.")
        return pd.DataFrame()

    try:
        # yfinance download expects start and end.
        # For daily data, end_date is often exclusive. Add timedelta(days=1) to include the end_date_input.
        # For intraday data, yfinance usually handles date objects well as start/end of day.
        fetch_end_date = end_date_input + timedelta(days=1) if interval in ["1d", "1wk", "1mo"] else end_date_input
        
        data = yf.download(
            ticker, 
            start=current_start_date, # Use the potentially adjusted start_date
            end=fetch_end_date, # Use the original end_date_input or adjusted for daily
            interval=interval, 
            progress=False,
            # auto_adjust=True, # Consider if you want auto-adjusted prices
            # prepost=False # Typically False for strategy backtesting unless specifically needed
        )
        
        if data.empty:
            logger.warning(f"No data found for {ticker} in the period {current_start_date} to {end_date_input} with interval {interval}.")
            st.warning(f"No data found for {ticker} from {current_start_date.strftime('%Y-%m-%d')} to {end_date_input.strftime('%Y-%m-%d')} at {interval} interval.")
            return pd.DataFrame()

        # Ensure DateTimeIndex is timezone-aware (yfinance often returns UTC or exchange local)
        if data.index.tz is None:
            try:
                # Attempt to localize to UTC first, as it's a common naive timezone from yf
                data = data.tz_localize('UTC')
                logger.info(f"Localized naive DatetimeIndex from yfinance to UTC for {ticker}.")
            except Exception as tz_localize_err: # Catch AmbiguousTimeError or NonExistentTimeError
                logger.error(f"Could not localize naive DatetimeIndex to UTC for {ticker}: {tz_localize_err}. Data might be unusable.")
                st.error(f"Error localizing timezone for {ticker}. Data might be incomplete or incorrect.")
                return pd.DataFrame() # Or handle more gracefully
        
        # Convert to New York time as strategy logic depends on it
        try:
            data = data.tz_convert(NY_TIMEZONE_STR)
            logger.info(f"Converted data for {ticker} to {NY_TIMEZONE_STR} timezone.")
        except Exception as tz_convert_err:
            logger.error(f"Could not convert data timezone to {NY_TIMEZONE_STR} for {ticker}: {tz_convert_err}")
            st.error(f"Error converting data to New York time for {ticker}.")
            return pd.DataFrame()

        # Filter out data outside the originally requested end_date_input (yf might give extra day for intraday too)
        # Ensure comparison is between date part of index and the date object
        data = data[data.index.date <= end_date_input]

        # Drop rows with NaN values, common in crypto or less liquid assets
        # This should be done carefully; sometimes NaNs are legitimate (e.g., missing volume for an index)
        # For OHLC, a NaN in Open, High, Low, or Close usually means the bar is invalid.
        data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        
        if data.empty:
            logger.warning(f"Data became empty after processing (timezone conversion, filtering, NaN drop) for {ticker}.")
            st.warning(f"No valid data remained after processing for {ticker} for the selected period.")
            return pd.DataFrame()
            
        logger.info(f"Successfully fetched and processed {len(data)} rows for {ticker}.")
        return data

    except Exception as e:
        logger.error(f"Error fetching or processing data for {ticker}: {e}", exc_info=True)
        st.error(f"Failed to fetch or process data for {ticker}. Error: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example usage:
    from datetime import date
    
    # Test with an index
    sp500_data = fetch_historical_data("^GSPC", date(2024, 3, 1), date(2024, 3, 5), "15m")
    if not sp500_data.empty:
        print("\nS&P 500 Data (first 5 rows):")
        print(sp500_data.head())
        print(f"Timezone: {sp500_data.index.tz}")
    else:
        print("\nNo S&P 500 data or error.")

    # Test with a commodity future
    gold_data = fetch_historical_data("GC=F", date(2024, 4, 1), date(2024, 4, 5), "15m")
    if not gold_data.empty:
        print("\nGold Futures Data (first 5 rows):")
        print(gold_data.head())
        print(f"Timezone: {gold_data.index.tz}")
    else:
        print("\nNo Gold data or error.")
        
    # Test with a date range that will trigger the adjustment
    print("\nTesting date adjustment for intraday:")
    too_old_start_date = date.today() - timedelta(days=90)
    recent_end_date = date.today() - timedelta(days=80) # Ensure range is valid after adjustment
    if too_old_start_date >= recent_end_date: # Adjust if test setup is wrong
        recent_end_date = too_old_start_date + timedelta(days=5)

    adjusted_data = fetch_historical_data("MSFT", too_old_start_date, recent_end_date, "15m")
    if not adjusted_data.empty:
        print(f"\nMSFT Data (adjusted start date) (first 5 rows):")
        print(adjusted_data.head())
        print(f"Timezone: {adjusted_data.index.tz}")
        print(f"Original requested start: {too_old_start_date}, Actual data start: {adjusted_data.index.date.min()}")
    else:
        print(f"\nNo MSFT data after adjustment or error. Original start: {too_old_start_date}, End: {recent_end_date}")

