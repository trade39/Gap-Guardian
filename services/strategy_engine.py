# services/strategy_engine.py
"""
Dispatcher for generating trading signals based on the selected strategy.
Imports and calls strategy-specific signal generation functions.
Uses absolute import for the 'strategies' submodule.
"""
import pandas as pd
from datetime import time as dt_time

from config import settings # For NY_TIMEZONE and potentially other shared settings
from utils.logger import get_logger

# Import strategy-specific generation functions using an absolute path
# from the project root, assuming 'services' is a top-level package directory.
# THIS IS THE CRITICAL LINE THAT NEEDS TO BE ABSOLUTE:
from services.strategies import (
    generate_gap_guardian_signals,
    generate_unicorn_signals,
    generate_silver_bullet_signals
)

logger = get_logger(__name__)

# --- Main Signal Generation Dispatcher ---
def generate_signals(
    data: pd.DataFrame,
    strategy_name: str,
    stop_loss_points: float,
    rrr: float,
    entry_start_time: dt_time | None = None, # Required for Gap Guardian
    entry_end_time: dt_time | None = None    # Required for Gap Guardian
) -> pd.DataFrame:
    """
    Generates trading signals by dispatching to the appropriate strategy function.

    Args:
        data (pd.DataFrame): Price data (OHLCV) indexed by Datetime.
        strategy_name (str): The name of the strategy to use.
        stop_loss_points (float): Stop loss distance in price points.
        rrr (float): Risk-Reward Ratio.
        entry_start_time (dt_time | None): Start of the entry window (NY time), for Gap Guardian.
        entry_end_time (dt_time | None): End of the entry window (NY time), for Gap Guardian.

    Returns:
        pd.DataFrame: DataFrame containing trade signals, with 'SignalTime' as index.
    """
    if data.empty:
        logger.warning(f"Strategy Engine: Input data for '{strategy_name}' is empty.")
        return pd.DataFrame()

    # Ensure data is in NY timezone, as most strategies are time-sensitive to this market.
    if data.index.tz is None:
        logger.warning(f"Strategy Engine: Data timezone is naive for '{strategy_name}', localizing to NY timezone.")
        try:
            data = data.tz_localize(settings.NY_TIMEZONE_STR)
        except Exception as e: # Handles cases like AmbiguousTimeError during DST changes
            logger.error(f"Strategy Engine: Failed to localize naive data to NY timezone for '{strategy_name}': {e}. Attempting to infer and convert.", exc_info=True)
            # Fallback or error, depending on strictness. For now, let's attempt conversion if already localized to something else.
            try:
                 data = data.tz_convert(settings.NY_TIMEZONE_STR)
            except Exception as e_conv:
                logger.error(f"Strategy Engine: Failed to convert data to NY timezone after localization attempt for '{strategy_name}': {e_conv}.", exc_info=True)
                return pd.DataFrame()

    elif data.index.tz.zone != settings.NY_TIMEZONE.zone: # Compare zone strings for robustness
        logger.info(f"Strategy Engine: Data timezone is {data.index.tz}, converting to NY timezone for '{strategy_name}'.")
        try:
            data = data.tz_convert(settings.NY_TIMEZONE_STR)
        except Exception as e:
            logger.error(f"Strategy Engine: Failed to convert data to NY timezone for '{strategy_name}': {e}. Current tz: {data.index.tz}", exc_info=True)
            return pd.DataFrame()
    
    logger.info(f"Strategy Engine: Generating signals for strategy: '{strategy_name}' with SL points: {stop_loss_points}, RRR: {rrr}.")

    signals_df = pd.DataFrame()

    if strategy_name == "Gap Guardian":
        if entry_start_time is None or entry_end_time is None:
            logger.error(f"Strategy Engine: Gap Guardian strategy requires 'entry_start_time' and 'entry_end_time'.")
            return pd.DataFrame()
        signals_df = generate_gap_guardian_signals(data.copy(), stop_loss_points, rrr, entry_start_time, entry_end_time)
    elif strategy_name == "Unicorn":
        signals_df = generate_unicorn_signals(data.copy(), stop_loss_points, rrr)
    elif strategy_name == "Silver Bullet":
        signals_df = generate_silver_bullet_signals(data.copy(), stop_loss_points, rrr)
    else:
        logger.error(f"Strategy Engine: Unknown strategy specified: '{strategy_name}'.")
        return pd.DataFrame()

    # Common post-processing for all signals_df
    if not signals_df.empty:
        if 'SignalTime' not in signals_df.columns:
            logger.error(f"Strategy Engine: '{strategy_name}' did not return 'SignalTime' column.")
            return pd.DataFrame()

        try:
            signals_df['SignalTime'] = pd.to_datetime(signals_df['SignalTime'])
            # Ensure SignalTime is also in NY timezone if it got converted to naive or UTC by mistake
            if signals_df['SignalTime'].dt.tz is None:
                signals_df['SignalTime'] = signals_df['SignalTime'].dt.tz_localize(settings.NY_TIMEZONE_STR)
            elif signals_df['SignalTime'].dt.tz.zone != settings.NY_TIMEZONE.zone:
                 signals_df['SignalTime'] = signals_df['SignalTime'].dt.tz_convert(settings.NY_TIMEZONE_STR)

            signals_df.set_index('SignalTime', inplace=True, drop=False) # Keep SignalTime as a column too
            signals_df.sort_index(inplace=True)
            logger.info(f"Strategy Engine: Successfully generated {len(signals_df)} signals for '{strategy_name}'.")
        except Exception as e:
            logger.error(f"Strategy Engine: Error processing signals DataFrame for '{strategy_name}': {e}", exc_info=True)
            return pd.DataFrame()
    else:
        logger.info(f"Strategy Engine: No signals generated by '{strategy_name}'.")
        
    return signals_df

if __name__ == '__main__':
    # This section can be used for basic testing of the dispatcher
    # Note: For this __main__ block to run correctly with the absolute import 'from services.strategies',
    # you would need to run this script from the project root directory like:
    # python -m services.strategy_engine
    # Or ensure the project root is in PYTHONPATH.
    
    # To make this runnable standalone for testing, we might need to adjust sys.path
    import sys
    import os
    # Add project root to sys.path if this script is run directly
    # This assumes the script is in Gap-Guardian-Strategy-Backtester-main/services/
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from services.data_loader import fetch_historical_data 
    from datetime import date as dt_date
    # Re-import config and utils if they were not found due to path issues before sys.path modification
    from config import settings as main_settings # Use an alias to avoid conflict if settings was already imported
    from utils.logger import get_logger as main_get_logger

    logger_main_test = main_get_logger(__name__ + "_standalone_test")


    sample_ticker = "GC=F" # Gold
    start_d = dt_date.today() - pd.Timedelta(days=20) 
    end_d = dt_date.today()
    
    test_sl = 2.0 
    test_rrr = 1.5
    test_tf = "15m"

    logger_main_test.info(f"--- Standalone Test: Strategy Dispatcher for {sample_ticker} on {test_tf} ---")
    
    try:
        price_data_test = fetch_historical_data(sample_ticker, start_d, end_d, test_tf)
        
        if not price_data_test.empty:
            logger_main_test.info(f"Price data ({len(price_data_test)} rows) from {price_data_test.index.min()} to {price_data_test.index.max()}")
            
            for strategy_to_test in main_settings.AVAILABLE_STRATEGIES:
                logger_main_test.info(f"-- Testing: {strategy_to_test} --")
                params_for_generation = {
                    'data': price_data_test.copy(), 
                    'strategy_name': strategy_to_test,
                    'stop_loss_points': test_sl,
                    'rrr': test_rrr
                }
                if strategy_to_test == "Gap Guardian":
                    params_for_generation['entry_start_time'] = dt_time(main_settings.DEFAULT_ENTRY_WINDOW_START_HOUR, main_settings.DEFAULT_ENTRY_WINDOW_START_MINUTE)
                    params_for_generation['entry_end_time'] = dt_time(main_settings.DEFAULT_ENTRY_WINDOW_END_HOUR, main_settings.DEFAULT_ENTRY_WINDOW_END_MINUTE)

                generated_signals = generate_signals(**params_for_generation)
                
                if not generated_signals.empty:
                    logger_main_test.info(f"Generated Signals ({len(generated_signals)}):\n{generated_signals.head()}")
                else:
                    logger_main_test.info(f"No signals generated for {strategy_to_test}.")
        else:
            logger_main_test.warning(f"Could not fetch price data for {test_tf} to test dispatcher.")
    except Exception as e:
        logger_main_test.error(f"Error during standalone test of strategy_engine: {e}", exc_info=True)
        logger_main_test.info("Ensure you run this test from the project root using 'python -m services.strategy_engine' or that PYTHONPATH is set correctly if issues persist.")
