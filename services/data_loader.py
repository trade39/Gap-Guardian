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

# For debugging, you can temporarily comment out @st.cache_data
# to ensure fresh data fetches on every run.
# @st.cache_data(ttl=3600) 
def fetch_historical_data(ticker: str, start_date_input: date, end_date_input: date, interval: str = STRATEGY_TIME_FRAME) -> pd.DataFrame:
    """
    Fetches historical market data from Yahoo Finance.
    """
    logger.info(f"Attempting to fetch_historical_data for {ticker} from {start_date_input} to {end_date_input} (interval: {interval}).")
    
    current_start_date = start_date_input

    if interval not in ["1d", "1wk", "1mo"]:
        max_start_date_limit_dt = datetime.now(NY_TIMEZONE) - timedelta(days=MAX_INTRADAY_DAYS)
        max_start_date_limit_date = max_start_date_limit_dt.date()
        if current_start_date < max_start_date_limit_date:
            logger.warning(f"Requested start date {current_start_date} for {ticker} ({interval}) is too old. Adjusting to {max_start_date_limit_date}.")
            current_start_date = max_start_date_limit_date
    
    if current_start_date >= end_date_input:
        logger.error(f"Invalid date range for {ticker}: Start date {current_start_date} must be before end date {end_date_input}.")
        st.error(f"Start date ({current_start_date.strftime('%Y-%m-%d')}) must be before end date ({end_date_input.strftime('%Y-%m-%d')}).")
        return pd.DataFrame()

    try:
        logger.info(f"Proceeding to yf.download for {ticker} with start={current_start_date}, end={end_date_input} (interval: {interval}).")
        
        fetch_end_date_param = end_date_input + timedelta(days=1) if interval in ["1d", "1wk", "1mo"] else end_date_input

        data = yf.download(
            tickers=ticker, 
            start=current_start_date,
            end=fetch_end_date_param,
            interval=interval, 
            progress=False,
            auto_adjust=False, 
            actions=False,
            # group_by='column' # Keep this or remove; handling MultiIndex explicitly now
        )
        
        if data.empty:
            logger.warning(f"yf.download returned an empty DataFrame for {ticker} (Period: {current_start_date} to {fetch_end_date_param}, Interval: {interval}).")
            st.warning(f"No data found for {ticker} from {current_start_date.strftime('%Y-%m-%d')} to {end_date_input.strftime('%Y-%m-%d')} at {interval} interval.")
            return pd.DataFrame()

        logger.info(f"Successfully downloaded data for {ticker}. Rows: {len(data)}. Initial columns from yf: {data.columns.tolist()}")

        # --- Handle MultiIndex Columns from yfinance ---
        # If columns are MultiIndex (e.g., [('Open', 'TICKER'), ...]), flatten them.
        if isinstance(data.columns, pd.MultiIndex):
            logger.info(f"Data for {ticker} has MultiIndex columns. Flattening to first level.")
            # Example: from [('Open', 'GC=F'), ('Close', 'GC=F')] to ['Open', 'Close']
            data.columns = data.columns.get_level_values(0)
            logger.info(f"Columns after potential MultiIndex flattening for {ticker}: {data.columns.tolist()}")
        
        # --- Standardize Column Names (after potential flattening) ---
        # Now, data.columns should be a simple Index of strings.
        standardized_rename_map = {}
        expected_final_names = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 
            'adjclose': 'Adj Close', # Handles 'Adj Close'
            'adj close': 'Adj Close', # Handles 'adj close' if it somehow appears
            'volume': 'Volume'
        }
        
        for col_name_str in data.columns:
            # Ensure col_name_str is a string (it should be after flattening or if it was already simple)
            col_name_str = str(col_name_str) 
            processed_col_key = col_name_str.lower().replace(' ', '').replace('.', '') # Process the string name
            
            if processed_col_key in expected_final_names:
                # Map from the current string name to the desired final standardized name
                standardized_rename_map[col_name_str] = expected_final_names[processed_col_key]
        
        if standardized_rename_map: # Only rename if there's something to map
            data.rename(columns=standardized_rename_map, inplace=True)
        logger.info(f"Columns after string standardization for {ticker}: {data.columns.tolist()}")


        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            logger.error(f"Data for {ticker} is missing required standardized columns: {missing_cols}. Available columns: {data.columns.tolist()}")
            st.error(f"Fetched data for {ticker} is incomplete (missing: {', '.join(missing_cols)}). Cannot proceed.")
            return pd.DataFrame()

        if data.index.tz is None:
            try:
                data = data.tz_localize('UTC')
                logger.info(f"Localized naive DatetimeIndex from yfinance to UTC for {ticker}.")
            except Exception as tz_localize_err:
                logger.error(f"Could not localize naive DatetimeIndex to UTC for {ticker}: {tz_localize_err}. Data might be unusable.", exc_info=True)
                st.error(f"Error localizing timezone for {ticker}. Data might be incomplete or incorrect.")
                return pd.DataFrame()
        
        try:
            data = data.tz_convert(NY_TIMEZONE_STR)
            logger.info(f"Converted data for {ticker} to {NY_TIMEZONE_STR} timezone.")
        except Exception as tz_convert_err:
            logger.error(f"Could not convert data timezone to {NY_TIMEZONE_STR} for {ticker}: {tz_convert_err}", exc_info=True)
            st.error(f"Error converting data to New York time for {ticker}.")
            return pd.DataFrame()

        data = data[data.index.date <= end_date_input]
        if data.empty:
            logger.warning(f"Data for {ticker} became empty after date filtering (<= {end_date_input}).")
            return pd.DataFrame()
        
        logger.info(f"Before dropna for {ticker} - Columns: {data.columns.tolist()}")
        # Optional: Further debugging if needed
        # logger.info(f"Before dropna for {ticker} - Data head (first 3 rows):\n{data.head(3).to_string()}")
        # import io
        # buffer = io.StringIO()
        # data.info(buf=buffer)
        # logger.info(buffer.getvalue())

        dropna_subset = ['Open', 'High', 'Low', 'Close']
        try:
            data.dropna(subset=dropna_subset, inplace=True)
        except KeyError as ke_dropna:
            logger.error(f"KeyError during dropna for {ticker} with subset {dropna_subset}. Columns were: {data.columns.tolist()}", exc_info=True)
            st.error(f"A data column error occurred during NaN processing for {ticker}: {ke_dropna}. Check logs for column details.")
            return pd.DataFrame()

        if data.empty:
            logger.warning(f"Data for {ticker} became empty after dropna.")
            st.warning(f"No valid data remained after cleaning (NaN drop) for {ticker} for the selected period.")
            return pd.DataFrame()
            
        logger.info(f"Successfully fetched and processed {len(data)} rows for {ticker}.")
        return data

    except Exception as e:
        logger.error(f"General error in fetch_historical_data for {ticker}: {e}", exc_info=True)
        st.error(f"Failed to fetch or process data for {ticker}. Details: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    from datetime import date, timedelta
    
    test_end_date = date.today()
    # Ensure start_date is reasonably before end_date for testing
    test_start_date = test_end_date - timedelta(days=min(30, settings.MAX_INTRADAY_DAYS - 2)) # Test with a smaller, valid range

    tickers_to_test = ["GC=F", "^GSPC", "BTC-USD", "MSFT", "EURUSD=X"]
    for test_ticker in tickers_to_test:
        print(f"\n--- Testing: {test_ticker} ---")
        df = fetch_historical_data(test_ticker, test_start_date, test_end_date, "15m")
        if not df.empty:
            print(f"\n{test_ticker} Data (first 3 rows):")
            print(df.head(3))
            print(f"Columns: {df.columns.tolist()}")
            print(f"Index type: {type(df.index)}, Index TZ: {df.index.tz}")
            print(f"Data types:\n{df.dtypes}")
        else:
            print(f"\nNo {test_ticker} data or error for period {test_start_date} to {test_end_date}.")
        print(f"--- Finished: {test_ticker} ---\n")
