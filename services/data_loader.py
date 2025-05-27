# services/data_loader.py
"""
Handles fetching and preparing market data using yfinance.
"""
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta, date
from config.settings import (
    MAX_SHORT_INTRADAY_DAYS, MAX_HOURLY_INTRADAY_DAYS,
    NY_TIMEZONE_STR, NY_TIMEZONE
)
from utils.logger import get_logger

logger = get_logger(__name__)

# yfinance interval categories for history limits
YFINANCE_SHORT_INTRADAY_INTERVALS = ["1m", "2m", "5m", "15m", "30m"] # Typically up to 60 days
YFINANCE_HOURLY_INTERVALS = ["60m", "1h", "90m"] # "1h", "90m" often up to 730 days. "60m" can be like shorter ones.
                                                # Let's treat "60m" and "1h" as potentially longer.
                                                # "90m" is also available.

# @st.cache_data(ttl=3600) # Consider re-enabling after thorough testing
def fetch_historical_data(ticker: str, start_date_input: date, end_date_input: date, interval: str) -> pd.DataFrame:
    logger.info(f"Attempting to fetch_historical_data for {ticker} from {start_date_input} to {end_date_input} (interval: {interval}).")

    current_start_date = start_date_input

    if current_start_date >= end_date_input:
        logger.error(f"Initial validation failed: Start date {current_start_date} must be before end date {end_date_input} for {ticker}.")
        return pd.DataFrame()

    # Determine max history days based on interval
    max_history_days = None
    if interval in YFINANCE_SHORT_INTRADAY_INTERVALS:
        max_history_days = MAX_SHORT_INTRADAY_DAYS
    elif interval in YFINANCE_HOURLY_INTERVALS: # e.g., "1h"
        max_history_days = MAX_HOURLY_INTRADAY_DAYS
    # For daily ("1d"), weekly ("1wk"), monthly ("1mo"), max_history_days is None (effectively unlimited for practical purposes)

    if max_history_days is not None:
        # Using NY_TIMEZONE for datetime.now() to be consistent if strategy times are NY based.
        max_permissible_start_datetime = datetime.now(NY_TIMEZONE) - timedelta(days=max_history_days)
        max_permissible_start_date = max_permissible_start_datetime.date()

        if current_start_date < max_permissible_start_date:
            logger.warning(
                f"Requested start date {current_start_date} for {ticker} ({interval}) is too old. "
                f"Adjusting to {max_permissible_start_date} due to ~{max_history_days}-day history limit for this interval."
            )
            current_start_date = max_permissible_start_date

    if current_start_date >= end_date_input:
        logger.error(f"Date range validation failed after adjustment for {ticker} ({interval}): "
                       f"Adjusted start date {current_start_date} is not before end date {end_date_input}.")
        st.warning(f"The selected date range for {ticker} ({interval}) is invalid after adjusting for data limits. "
                   f"Start date set to {current_start_date.strftime('%Y-%m-%d')}, End date: {end_date_input.strftime('%Y-%m-%d')}. "
                   f"Please select a valid range.")
        return pd.DataFrame()

    fetch_api_end_date = end_date_input + timedelta(days=1)

    try:
        logger.info(f"Proceeding to yf.download for {ticker}: API Start={current_start_date}, API End={fetch_api_end_date} (Interval: {interval}).")
        data = yf.download(
            tickers=ticker, start=current_start_date, end=fetch_api_end_date,
            interval=interval, progress=False, auto_adjust=False, actions=False,
        )

        if data.empty:
            logger.warning(f"yf.download returned empty DataFrame for {ticker} (API Period: {current_start_date} to {fetch_api_end_date}, Interval: {interval}).")
            st.warning(f"No data found for {ticker} from {current_start_date.strftime('%Y-%m-%d')} to {end_date_input.strftime('%Y-%m-%d')} at {interval} interval.")
            return pd.DataFrame()

        logger.info(f"Downloaded data for {ticker}. Rows: {len(data)}. Initial yf columns: {data.columns.tolist()}")

        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        
        std_rename_map = {
            col_str: expected_name for col_str in data.columns
            for key, expected_name in {'open':'Open', 'high':'High', 'low':'Low', 'close':'Close',
                                       'adjclose':'Adj Close', 'adj close':'Adj Close', 'volume':'Volume'}.items()
            if str(col_str).lower().replace(' ','').replace('.','') == key
        }
        if std_rename_map: data.rename(columns=std_rename_map, inplace=True)
        logger.info(f"Columns after standardization for {ticker}: {data.columns.tolist()}")

        req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in req_cols if col not in data.columns]
        if missing:
            logger.error(f"Data for {ticker} missing required columns: {missing}. Available: {data.columns.tolist()}")
            st.error(f"Fetched data for {ticker} incomplete (missing: {', '.join(missing)}).")
            return pd.DataFrame()

        if data.index.tz is None:
            try: data = data.tz_localize('UTC') # Default assumption for naive daily/older data
            except Exception as tz_err:
                logger.error(f"Could not localize naive DatetimeIndex to UTC for {ticker}: {tz_err}", exc_info=True)
                st.error(f"Error localizing timezone for {ticker}.")
                return pd.DataFrame()
        
        try: data = data.tz_convert(NY_TIMEZONE_STR)
        except Exception as tz_conv_err:
            logger.error(f"Could not convert timezone to {NY_TIMEZONE_STR} for {ticker}: {tz_conv_err}", exc_info=True)
            st.error(f"Error converting data to New York time for {ticker}.")
            return pd.DataFrame()

        # Final filter to ensure data is strictly within user's requested end_date_input (inclusive at date level)
        # Convert end_date_input to a timezone-aware datetime at the end of that day for comparison
        end_datetime_inclusive = pd.Timestamp(end_date_input, tz=NY_TIMEZONE_STR) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        data = data[data.index <= end_datetime_inclusive]
        # Also ensure start date is respected (yf might give slightly earlier data for some intervals)
        start_datetime_inclusive = pd.Timestamp(current_start_date, tz=NY_TIMEZONE_STR)
        data = data[data.index >= start_datetime_inclusive]


        if data.empty:
            logger.warning(f"Data for {ticker} became empty after final date/time filtering (User range: {start_date_input} to {end_date_input}).")
            st.warning(f"No data available for {ticker} within the precise range after filtering.")
            return pd.DataFrame()
        
        data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        if data.empty:
            logger.warning(f"Data for {ticker} became empty after dropna.")
            st.warning(f"No valid data after cleaning (NaN drop) for {ticker}.")
            return pd.DataFrame()
            
        logger.info(f"Successfully processed {len(data)} rows for {ticker} (Interval: {interval}, Range: {data.index.min()} to {data.index.max()}).")
        return data

    except Exception as e:
        logger.error(f"General error in fetch_historical_data for {ticker} ({interval}): {e}", exc_info=True)
        st.error(f"Failed to fetch/process data for {ticker}. Details: {e}")
        return pd.DataFrame()

