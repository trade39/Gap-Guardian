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
        
        # Determine fetch_end_date based on interval
        # For daily intervals, yf.download's end_date is often exclusive, so add 1 day.
        # For intraday intervals, yf.download usually treats the date as inclusive up to the end of that day.
        fetch_end_date_param = end_date_input + timedelta(days=1) if interval in ["1d", "1wk", "1mo"] else end_date_input

        data = yf.download(
            tickers=ticker, 
            start=current_start_date,
            end=fetch_end_date_param,
            interval=interval, 
            progress=False,
            auto_adjust=False, 
            actions=False 
        )
        
        if data.empty:
            logger.warning(f"yf.download returned an empty DataFrame for {ticker} (Period: {current_start_date} to {fetch_end_date_param}, Interval: {interval}).")
            st.warning(f"No data found for {ticker} from {current_start_date.strftime('%Y-%m-%d')} to {end_date_input.strftime('%Y-%m-%d')} at {interval} interval.")
            return pd.DataFrame()

        logger.info(f"Successfully downloaded data for {ticker}. Rows: {len(data)}. Initial columns: {data.columns.tolist()}")

        # Standardize column names to catch variations like "Adj Close" vs "Adj. Close" etc.
        # Convert all to lower, remove spaces and dots, then map to expected capitalized form.
        # This is a more robust way than direct checking if yfinance changes minor things.
        standardized_columns = {}
        expected_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 
            'adjclose': 'Adj Close', 'adj close': 'Adj Close', # common variations for adj close
            'volume': 'Volume'
        }
        for col in data.columns:
            processed_col = col.lower().replace(' ', '').replace('.', '')
            if processed_col in expected_map:
                standardized_columns[col] = expected_map[processed_col]
        
        data.rename(columns=standardized_columns, inplace=True)
        logger.info(f"Columns after standardization for {ticker}: {data.columns.tolist()}")


        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            logger.error(f"Data for {ticker} is missing required standardized columns: {missing_cols}. Available columns after standardization: {data.columns.tolist()}")
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

        # Filter out data outside the originally requested end_date_input
        # This is important because yf.download with end_date_param might fetch up to the start of that day.
        data = data[data.index.date <= end_date_input]
        if data.empty:
            logger.warning(f"Data for {ticker} became empty after date filtering (<= {end_date_input}).")
            return pd.DataFrame()
        
        # --- Enhanced Debugging before dropna ---
        logger.info(f"Before dropna for {ticker} - Columns: {data.columns.tolist()}")
        logger.info(f"Before dropna for {ticker} - Data head (first 3 rows):\n{data.head(3).to_string()}")
        logger.info(f"Before dropna for {ticker} - Data info:")
        # Streaming data.info() to logger is a bit tricky, capture with StringIO
        import io
        buffer = io.StringIO()
        data.info(buf=buffer)
        logger.info(buffer.getvalue())
        # --- End Enhanced Debugging ---

        # Drop rows if essential OHLC data is NaN
        dropna_subset = ['Open', 'High', 'Low', 'Close']
        try:
            data.dropna(subset=dropna_subset, inplace=True)
        except KeyError as ke_dropna:
            logger.error(f"KeyError during dropna for {ticker} with subset {dropna_subset}. Columns were: {data.columns.tolist()}", exc_info=True)
            st.error(f"A data column error occurred during NaN processing for {ticker}: {ke_dropna}. Check logs for column details.")
            return pd.DataFrame()

        if data.empty:
            logger.warning(f"Data for {ticker} became empty after dropna. Original rows before dropna: (check previous logs for count before this step)")
            st.warning(f"No valid data remained after cleaning (NaN drop) for {ticker} for the selected period.")
            return pd.DataFrame()
            
        logger.info(f"Successfully fetched and processed {len(data)} rows for {ticker}.")
        return data

    except Exception as e:
        logger.error(f"General error in fetch_historical_data for {ticker}: {e}", exc_info=True)
        st.error(f"Failed to fetch or process data for {ticker}. Details: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Test with a known problematic ticker if possible, or a standard one
    print("Fetching Gold Futures data (GC=F)...")
    # Use a very recent, short period for testing to minimize data issues
    test_end_date = date.today()
    test_start_date = test_end_date - timedelta(days=7) # Fetch last 7 days

    gold_data = fetch_historical_data("GC=F", test_start_date, test_end_date, "15m")
    if not gold_data.empty:
        print("\nGold Futures Data (first 5 rows):")
        print(gold_data.head())
        print(f"Columns: {gold_data.columns.tolist()}")
        print(f"Timezone: {gold_data.index.tz}")
    else:
        print(f"\nNo Gold data or error for period {test_start_date} to {test_end_date}.")

    print("\nFetching S&P 500 data (^GSPC)...")
    sp500_data = fetch_historical_data("^GSPC", test_start_date, test_end_date, "15m")
    if not sp500_data.empty:
        print("\nS&P 500 Data (first 5 rows):")
        print(sp500_data.head())
        print(f"Columns: {sp500_data.columns.tolist()}")
    else:
        print(f"\nNo S&P 500 data or error for period {test_start_date} to {test_end_date}.")

