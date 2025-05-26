# services/data_loader.py
"""
Handles fetching and preparing market data using yfinance.
"""
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta, date, time # Added time
from config.settings import MAX_INTRADAY_DAYS, STRATEGY_TIME_FRAME, NY_TIMEZONE_STR, NY_TIMEZONE
from utils.logger import get_logger

logger = get_logger(__name__)

# For debugging, you can temporarily comment out @st.cache_data
# to ensure fresh data fetches on every run.
# @st.cache_data(ttl=3600)
def fetch_historical_data(ticker: str, start_date_input: date, end_date_input: date, interval: str = STRATEGY_TIME_FRAME) -> pd.DataFrame:
    """
    Fetches historical market data from Yahoo Finance.
    Ensures that the data fetched is inclusive of the end_date_input.
    """
    logger.info(f"Attempting to fetch_historical_data for {ticker} from {start_date_input} to {end_date_input} (interval: {interval}).")

    current_start_date = start_date_input

    # Validate that start_date_input is before end_date_input initially (Streamlit's UI should also enforce this)
    if current_start_date >= end_date_input:
        logger.error(f"Initial validation failed: Start date {current_start_date} must be before end date {end_date_input} for {ticker}.")
        # This error should ideally be caught by app.py's UI validation first.
        # st.error(f"Data fetching error: Start date ({current_start_date.strftime('%Y-%m-%d')}) must be before end date ({end_date_input.strftime('%Y-%m-%d')}).")
        return pd.DataFrame()

    # Adjust start date for intraday data limitations
    if interval not in ["1d", "1wk", "1mo"]: # Intraday intervals
        # Calculate the earliest possible start date for intraday data
        # Ensure datetime.now() is timezone-aware if comparing with timezone-aware dates, though MAX_INTRADAY_DAYS is a simple subtraction.
        # Using a fixed reference like NY_TIMEZONE for datetime.now() is good practice if strategy times are NY based.
        max_permissible_start_datetime = datetime.now(NY_TIMEZONE) - timedelta(days=MAX_INTRADAY_DAYS)
        max_permissible_start_date = max_permissible_start_datetime.date()

        if current_start_date < max_permissible_start_date:
            logger.warning(f"Requested start date {current_start_date} for {ticker} ({interval}) is too old. "
                           f"Adjusting to {max_permissible_start_date} due to {MAX_INTRADAY_DAYS}-day intraday limit.")
            current_start_date = max_permissible_start_date

    # After potential adjustment, re-check if start_date is still valid relative to end_date
    if current_start_date >= end_date_input:
        logger.error(f"Date range validation failed after adjustment for {ticker}: "
                       f"Adjusted start date {current_start_date} is not before end date {end_date_input}.")
        st.warning(f"The selected date range for {ticker} is invalid after adjusting for intraday data limits. "
                   f"Start date set to {current_start_date.strftime('%Y-%m-%d')}, End date: {end_date_input.strftime('%Y-%m-%d')}. "
                   f"Please select a valid range where start date is before end date.")
        return pd.DataFrame()

    # For yf.download, the 'end' parameter is typically exclusive.
    # To make it inclusive of end_date_input, we request data up to the day AFTER end_date_input.
    fetch_api_end_date = end_date_input + timedelta(days=1)

    try:
        logger.info(f"Proceeding to yf.download for {ticker}: "
                    f"API Start={current_start_date}, API End={fetch_api_end_date} (Interval: {interval}). "
                    f"(User requested end date: {end_date_input})")

        data = yf.download(
            tickers=ticker,
            start=current_start_date, # This is a date object
            end=fetch_api_end_date,   # This is also a date object (day after user's end_date)
            interval=interval,
            progress=False,
            auto_adjust=False, # Keep False to get OHLC, True gives adjusted prices
            actions=False,
        )

        if data.empty:
            logger.warning(f"yf.download returned an empty DataFrame for {ticker} "
                           f"(API Period: {current_start_date} to {fetch_api_end_date}, Interval: {interval}).")
            st.warning(f"No data found for {ticker} from {current_start_date.strftime('%Y-%m-%d')} "
                       f"to {end_date_input.strftime('%Y-%m-%d')} at {interval} interval.")
            return pd.DataFrame()

        logger.info(f"Successfully downloaded data for {ticker}. Rows: {len(data)}. Initial columns from yf: {data.columns.tolist()}")

        # --- Handle MultiIndex Columns & Standardize ---
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        standardized_rename_map = {
            col_str: expected_name
            for col_str in data.columns
            for key, expected_name in {
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
                'adjclose': 'Adj Close', 'adj close': 'Adj Close', 'volume': 'Volume'
            }.items() if str(col_str).lower().replace(' ', '').replace('.', '') == key
        }
        if standardized_rename_map:
            data.rename(columns=standardized_rename_map, inplace=True)
        logger.info(f"Columns after standardization for {ticker}: {data.columns.tolist()}")

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            logger.error(f"Data for {ticker} is missing required standardized columns: {missing_cols}. Available: {data.columns.tolist()}")
            st.error(f"Fetched data for {ticker} is incomplete (missing: {', '.join(missing_cols)}).")
            return pd.DataFrame()

        # --- Timezone Handling ---
        if data.index.tz is None: # yfinance usually returns timezone-aware for intraday, naive for daily
            try:
                # For daily data, yfinance might return naive datetimes representing market's local time.
                # Assuming UTC if naive, then converting to NY. This might need adjustment based on asset's exchange.
                # However, for major US tickers, yf often returns NY time for daily if tz_localize isn't needed.
                # A safer bet for daily data might be to localize to the exchange's timezone if known, or assume NY for US assets.
                # For simplicity, if it's naive, we'll attempt UTC then convert.
                data = data.tz_localize('UTC') # Or specific exchange timezone if known and naive
                logger.info(f"Localized naive DatetimeIndex from yfinance to UTC for {ticker}.")
            except Exception as tz_localize_err: # Catch specific errors like AmbiguousTimeError if needed
                logger.error(f"Could not localize naive DatetimeIndex to UTC for {ticker}: {tz_localize_err}. Data might be unusable.", exc_info=True)
                st.error(f"Error localizing timezone for {ticker}. Data might be incomplete or incorrect.")
                return pd.DataFrame()
        
        try: # Convert to NY timezone
            data = data.tz_convert(NY_TIMEZONE_STR)
            logger.info(f"Converted data for {ticker} to {NY_TIMEZONE_STR} timezone.")
        except Exception as tz_convert_err:
            logger.error(f"Could not convert data timezone to {NY_TIMEZONE_STR} for {ticker}: {tz_convert_err}", exc_info=True)
            st.error(f"Error converting data to New York time for {ticker}.")
            return pd.DataFrame()

        # --- Final Filtering to User's Requested End Date (Inclusive) ---
        # Since we fetched up to fetch_api_end_date (end_date_input + 1 day),
        # we now filter to ensure we only include data up to and including end_date_input.
        # This handles cases where yf.download might give partial data for the next day.
        # For intraday data, data.index.normalize() converts datetime to date at midnight.
        data = data[data.index.normalize() <= pd.to_datetime(end_date_input).tz_localize(NY_TIMEZONE_STR)] # Ensure comparison is timezone aware

        if data.empty:
            logger.warning(f"Data for {ticker} became empty after final date filtering (<= {end_date_input}).")
            st.warning(f"No data available for {ticker} within the precise range up to {end_date_input.strftime('%Y-%m-%d')}.")
            return pd.DataFrame()
        
        logger.info(f"Before dropna for {ticker} - Rows: {len(data)}")
        
        dropna_subset = ['Open', 'High', 'Low', 'Close'] # Volume can sometimes be NaN for indices
        data.dropna(subset=dropna_subset, inplace=True)

        if data.empty:
            logger.warning(f"Data for {ticker} became empty after dropna.")
            st.warning(f"No valid data remained after cleaning (NaN drop) for {ticker} for the selected period.")
            return pd.DataFrame()
            
        logger.info(f"Successfully fetched and processed {len(data)} rows for {ticker} (Range: {data.index.min()} to {data.index.max()}).")
        return data

    except Exception as e:
        logger.error(f"General error in fetch_historical_data for {ticker}: {e}", exc_info=True)
        st.error(f"Failed to fetch or process data for {ticker}. Details: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Example for testing
    test_end_date = date.today() - timedelta(days=1) # Ensure we are not asking for today's incomplete data
    test_start_date = test_end_date - timedelta(days=5)

    tickers_to_test = ["GC=F", "^GSPC", "BTC-USD", "MSFT", "EURUSD=X", "AAPL"]
    intervals_to_test = ["15m", "1h", "1d"]

    for test_ticker in tickers_to_test:
        for test_interval in intervals_to_test:
            # Adjust start date for intraday test to be within yf limits
            current_test_start = test_start_date
            if test_interval not in ["1d", "1wk", "1mo"]:
                max_start = date.today() - timedelta(days=MAX_INTRADAY_DAYS -1) # -1 to give some room
                if current_test_start < max_start:
                    current_test_start = max_start
            
            # Ensure start is still before end for the test
            if current_test_start >= test_end_date:
                print(f"\nSkipping {test_ticker} ({test_interval}): Adjusted start date {current_test_start} is not before end date {test_end_date}")
                continue

            print(f"\n--- Testing: {test_ticker} ({test_interval}) from {current_test_start} to {test_end_date} ---")
            df = fetch_historical_data(test_ticker, current_test_start, test_end_date, test_interval)
            if not df.empty:
                print(f"Data ({len(df)} rows). Min Date: {df.index.min()}, Max Date: {df.index.max()}")
                print(df.head(2))
                print(df.tail(2))
            else:
                print(f"No data or error for {test_ticker} ({test_interval}).")
            print(f"--- Finished: {test_ticker} ({test_interval}) ---\n")

