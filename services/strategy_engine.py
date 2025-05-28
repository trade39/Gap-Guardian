# services/strategy_engine.py
"""
Dispatcher for generating trading signals based on the selected strategy.
Imports and calls strategy-specific signal generation functions.
"""
import sys
import os
import pandas as pd
from datetime import time as dt_time

# --- sys.path diagnostics for strategy_engine.py ---
STRATEGY_ENGINE_FILE_PATH = os.path.abspath(__file__)
# Assuming strategy_engine.py is in 'project_root/services/'
# So, project_root is one level up from the 'services' directory.
# Or, if app.py already set sys.path correctly, this module should just use it.
print(f"--- [DEBUG strategy_engine.py] ---")
print(f"STRATEGY_ENGINE_FILE_PATH: {STRATEGY_ENGINE_FILE_PATH}")
print(f"sys.path as seen by strategy_engine.py (before any local modification): {sys.path}")
# We expect the project root (e.g., '/mount/src/gap-guardian') to be in sys.path, added by app.py.
# Let's check if 'utils' and 'config' are accessible from the current sys.path
project_root_candidate = None
for path_entry in sys.path:
    if os.path.isdir(os.path.join(path_entry, 'utils')) and \
       os.path.isdir(os.path.join(path_entry, 'config')) and \
       os.path.isdir(os.path.join(path_entry, 'services')):
        project_root_candidate = path_entry
        break
print(f"Attempted to find project root in sys.path: {project_root_candidate if project_root_candidate else 'Not found based on subdirs.'}")
print(f"--- [END DEBUG strategy_engine.py] ---")
# --- end of sys.path diagnostics ---


# These imports depend on the project root being in sys.path
try:
    from config import settings # For NY_TIMEZONE and potentially other shared settings
    from utils.logger import get_logger
except ImportError as e:
    print(f"CRITICAL ERROR in strategy_engine.py: Could not import 'config' or 'utils'. Sys.path: {sys.path}. Error: {e}")
    # If these fail, the module cannot proceed. Raising an error here might give a clearer traceback.
    raise ImportError(f"strategy_engine.py: Failed to import 'config' or 'utils'. Ensure project root is in sys.path. Original error: {e}") from e

# THIS IS THE CRITICAL IMPORT (line 16 in your traceback)
# It relies on 'services.strategies' being a discoverable package,
# and its __init__.py being able to successfully import from individual strategy files.
try:
    from services.strategies import (
        generate_gap_guardian_signals,
        generate_unicorn_signals,
        generate_silver_bullet_signals
    )
except ImportError as e:
    print(f"ERROR in strategy_engine.py: Failed during 'from services.strategies import ...'. This often means a strategy file (e.g., gap_guardian.py) within 'services/strategies/' failed to load its own dependencies (like utils/config). Sys.path: {sys.path}. Error: {e}")
    raise # Re-raise the error to see the original traceback clearly.

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
        logger.warning(f"Strategy Engine: Data timezone is naive for '{strategy_name}', localizing to NY timezone ({settings.NY_TIMEZONE_STR}).")
        try:
            data = data.tz_localize(settings.NY_TIMEZONE_STR) # Use NY_TIMEZONE_STR from settings
        except Exception as e: # Handles cases like AmbiguousTimeError during DST changes
            logger.error(f"Strategy Engine: Failed to localize naive data to NY timezone for '{strategy_name}': {e}. Attempting to infer and convert.", exc_info=True)
            try:
                 data = data.tz_convert(settings.NY_TIMEZONE_STR)
            except Exception as e_conv:
                logger.error(f"Strategy Engine: Failed to convert data to NY timezone after localization attempt for '{strategy_name}': {e_conv}.", exc_info=True)
                return pd.DataFrame()

    elif data.index.tz.zone != settings.NY_TIMEZONE.zone: # Compare zone strings for robustness
        logger.info(f"Strategy Engine: Data timezone is {data.index.tz}, converting to NY timezone ({settings.NY_TIMEZONE_STR}) for '{strategy_name}'.")
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

# Example of how to run this for testing (if needed, but typically run via app.py)
if __name__ == '__main__':
    # This block is for direct testing of strategy_engine.py.
    # To make it work, you'd typically run: python -m services.strategy_engine
    # from the project root directory.
    
    # For robust testing, ensure PROJECT_ROOT is correctly identified and added if not already.
    # This might involve more complex path logic if running from various locations.
    # The sys.path modification in app.py is the primary mechanism for the app itself.
    
    print("Running strategy_engine.py as main script (for testing purposes).")
    
    # Example: Add project root to sys.path if not already there, for standalone testing
    # This assumes this script is in 'project_root/services/strategy_engine.py'
    module_path = os.path.abspath(__file__)
    project_root_for_test = os.path.dirname(os.path.dirname(module_path)) # Up two levels
    if project_root_for_test not in sys.path:
        sys.path.insert(0, project_root_for_test)
        print(f"[TEST MODE] Added to sys.path for testing: {project_root_for_test}")
        # Re-attempt imports that might have failed if sys.path wasn't set
        from config import settings as test_settings
        from utils.logger import get_logger as test_get_logger
        from services.data_loader import fetch_historical_data
        from services.strategies import generate_gap_guardian_signals as gg_test
    else:
        print(f"[TEST MODE] Project root {project_root_for_test} already in sys.path.")
        from config import settings as test_settings
        from utils.logger import get_logger as test_get_logger
        from services.data_loader import fetch_historical_data
        from services.strategies import generate_gap_guardian_signals as gg_test


    logger_main_test = test_get_logger(__name__ + "_standalone_test")

    sample_ticker = "GC=F" # Gold
    start_d = pd.Timestamp.today().normalize() - pd.Timedelta(days=20)
    end_d = pd.Timestamp.today().normalize()
    
    test_sl = 2.0 
    test_rrr = 1.5
    test_tf = "15m"

    logger_main_test.info(f"--- Standalone Test: Strategy Dispatcher for {sample_ticker} on {test_tf} ---")
    
    try:
        price_data_test = fetch_historical_data(sample_ticker, start_d.date(), end_d.date(), test_tf)
        
        if not price_data_test.empty:
            logger_main_test.info(f"Price data ({len(price_data_test)} rows) from {price_data_test.index.min()} to {price_data_test.index.max()}")
            
            for strategy_to_test in test_settings.AVAILABLE_STRATEGIES:
                logger_main_test.info(f"-- Testing: {strategy_to_test} --")
                params_for_generation = {
                    'data': price_data_test.copy(), 
                    'strategy_name': strategy_to_test,
                    'stop_loss_points': test_sl,
                    'rrr': test_rrr
                }
                if strategy_to_test == "Gap Guardian":
                    params_for_generation['entry_start_time'] = dt_time(test_settings.DEFAULT_ENTRY_WINDOW_START_HOUR, test_settings.DEFAULT_ENTRY_WINDOW_START_MINUTE)
                    params_for_generation['entry_end_time'] = dt_time(test_settings.DEFAULT_ENTRY_WINDOW_END_HOUR, test_settings.DEFAULT_ENTRY_WINDOW_END_MINUTE)

                generated_signals = generate_signals(**params_for_generation)
                
                if not generated_signals.empty:
                    logger_main_test.info(f"Generated Signals ({len(generated_signals)}):\n{generated_signals.head()}")
                else:
                    logger_main_test.info(f"No signals generated for {strategy_to_test}.")
        else:
            logger_main_test.warning(f"Could not fetch price data for {test_tf} to test dispatcher.")
    except Exception as e:
        logger_main_test.error(f"Error during standalone test of strategy_engine: {e}", exc_info=True)

