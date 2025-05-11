# services/data_loader.py
"""
Handles fetching and preparing market data using yfinance.
"""
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta, date 
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
    
    current_start_date = start_date_input

    if interval not in ["1d", "1wk", "1mo"]:
        max_start_date_limit_dt = datetime.now(NY_TIMEZONE) - timedelta(days=MAX_INTRADAY_DAYS)
        max_start_date_limit_date = max_start_date_limit_dt.date()

        if current_start_date < max_start_date_limit_date:
            logger.warning(f"Requested start date {current_start_date} is too old for {interval} data. Adjusting to {max_start_date_limit_date}.")
            current_start_date = max_start_date_limit_date
    
    if current_start_date >= end_date_input:
        logger.error(f"Start date {current_start_date} must be before end date {end_date_input}.")
        st.error(f"Start date {current_start_date.strftime('%Y-%m-%d')} must be before end date {end_date_input.strftime('%Y-%m-%d')}.")
        return pd.DataFrame()

    try:
        # For daily data, yf.download's end_date is exclusive. Add 1 day.
        # For intraday, yf.download's end_date is inclusive if it's just a date.
        # To be safe and consistent, pass date objects and let yf handle time part for intraday.
        # For daily, add timedelta(days=1) to ensure the end_date_input day is included.
        fetch_end_date = end_date_input + timedelta(days=1) if interval in ["1d", "1wk", "1mo"] else end_date_input
        
        data = yf.download(
            tickers=ticker, 
            start=current_start_date,
            end=fetch_end_date, # Use end_date_input directly for intraday, +1 day for daily
            interval=interval, 
            progress=False,
            auto_adjust=False, # Explicitly set to False to get raw OHLC and 'Adj Close'
            actions=False # No need for dividend/split actions columns for this strategy
        )
        
        logger.debug(f"Columns received from yfinance for {ticker}: {data.columns.tolist()}")

        if data.empty:
            logger.warning(f"No data found for {ticker} in the period {current_start_date} to {end_date_input} with interval {interval}.")
            st.warning(f"No data found for {ticker} from {current_start_date.strftime('%Y-%m-%d')} to {end_date_input.strftime('%Y-%m-%d')} at {interval} interval.")
            return pd.DataFrame()
        
        # Validate essential columns before proceeding
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume'] # 'Adj Close' is also available if auto_adjust=False
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            logger.error(f"Data for {ticker} is missing required columns: {missing_cols}. Available columns: {data.columns.tolist()}")
            st.error(f"Fetched data for {ticker} is incomplete (missing: {', '.join(missing_cols)}). Cannot proceed.")
            return pd.DataFrame()

        if data.index.tz is None:
            try:
                data = data.tz_localize('UTC')
                logger.info(f"Localized naive DatetimeIndex from yfinance to UTC for {ticker}.")
            except Exception as tz_localize_err:
                logger.error(f"Could not localize naive DatetimeIndex to UTC for {ticker}: {tz_localize_err}. Data might be unusable.")
                st.error(f"Error localizing timezone for {ticker}. Data might be incomplete or incorrect.")
                return pd.DataFrame()
        
        try:
            data = data.tz_convert(NY_TIMEZONE_STR)
            logger.info(f"Converted data for {ticker} to {NY_TIMEZONE_STR} timezone.")
        except Exception as tz_convert_err:
            logger.error(f"Could not convert data timezone to {NY_TIMEZONE_STR} for {ticker}: {tz_convert_err}")
            st.error(f"Error converting data to New York time for {ticker}.")
            return pd.DataFrame()

        # Filter out data outside the originally requested end_date_input
        data = data[data.index.date <= end_date_input]
        
        # Drop rows if essential OHLC data is NaN
        data.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        
        if data.empty:
            logger.warning(f"Data became empty after processing (timezone conversion, filtering, NaN drop) for {ticker}.")
            st.warning(f"No valid data remained after processing for {ticker} for the selected period.")
            return pd.DataFrame()
            
        logger.info(f"Successfully fetched and processed {len(data)} rows for {ticker}.")
        return data

    except KeyError as ke: # Specifically catch KeyError if subset columns are still an issue
        logger.error(f"KeyError during data processing for {ticker}: {ke}. This might indicate unexpected column names from yfinance even with auto_adjust=False.", exc_info=True)
        st.error(f"A data column error occurred for {ticker}: {ke}. Please check logs.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching or processing data for {ticker}: {e}", exc_info=True)
        # The error message from the exception 'e' might be a list of column names if it's a KeyError from dropna
        # This was the case in the user's log: Error: ['Open', 'High', 'Low', 'Close']
        st.error(f"Failed to fetch or process data for {ticker}. Error details: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    from datetime import date
    
    # Test with an index
    print("Fetching S&P 500 data...")
    sp500_data = fetch_historical_data("^GSPC", date(2024, 3, 1), date(2024, 3, 5), "15m")
    if not sp500_data.empty:
        print("\nS&P 500 Data (first 5 rows):")
        print(sp500_data.head())
        print(f"Columns: {sp500_data.columns.tolist()}")
        print(f"Timezone: {sp500_data.index.tz}")
    else:
        print("\nNo S&P 500 data or error.")

    # Test with a commodity future
    print("\nFetching Gold Futures data...")
    gold_data = fetch_historical_data("GC=F", date(2024, 4, 1), date(2024, 4, 5), "15m")
    if not gold_data.empty:
        print("\nGold Futures Data (first 5 rows):")
        print(gold_data.head())
        print(f"Columns: {gold_data.columns.tolist()}")
        print(f"Timezone: {gold_data.index.tz}")
    else:
        print("\nNo Gold data or error.")
        
    print("\nTesting date adjustment for intraday (MSFT)...")
    too_old_start_date = date.today() - timedelta(days=90)
    recent_end_date = too_old_start_date + timedelta(days=5) # Ensure a valid range

    adjusted_data = fetch_historical_data("MSFT", too_old_start_date, recent_end_date, "15m")
    if not adjusted_data.empty:
        print(f"\nMSFT Data (adjusted start date) (first 5 rows):")
        print(adjusted_data.head())
        print(f"Columns: {adjusted_data.columns.tolist()}")
        print(f"Timezone: {adjusted_data.index.tz}")
        print(f"Original requested start: {too_old_start_date}, Actual data start: {adjusted_data.index.date.min()}")
    else:
        print(f"\nNo MSFT data after adjustment or error. Original start: {too_old_start_date}, End: {recent_end_date}")

