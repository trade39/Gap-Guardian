# services/optimizer.py
"""
Performs parameter optimization for trading strategies using Grid Search, Random Search,
and Walk-Forward Optimization (WFO).
"""
import sys
import os
import pandas as pd
import numpy as np
import itertools
import random
from datetime import timedelta, time as dt_time

# --- sys.path diagnostics for optimizer.py ---
OPTIMIZER_FILE_PATH = os.path.abspath(__file__)
print(f"--- [DEBUG optimizer.py @ {os.path.basename(OPTIMIZER_FILE_PATH)}] ---")
print(f"    OPTIMIZER_FILE_PATH: {OPTIMIZER_FILE_PATH}")
print(f"    sys.path as seen by optimizer.py: {sys.path}")
# We expect the project root (e.g., '/mount/src/gap-guardian') to be in sys.path, added by app.py.
project_root_candidate_opt = None
for path_entry in sys.path:
    if os.path.isdir(os.path.join(path_entry, 'utils')) and \
       os.path.isdir(os.path.join(path_entry, 'config')) and \
       os.path.isdir(os.path.join(path_entry, 'services')):
        project_root_candidate_opt = path_entry
        break
print(f"    Attempted to find project root in sys.path: {project_root_candidate_opt if project_root_candidate_opt else 'Not found based on subdirs.'}")
print(f"--- [END DEBUG optimizer.py @ {os.path.basename(OPTIMIZER_FILE_PATH)}] ---")
# --- end of sys.path diagnostics ---

print(f"--- [DEBUG optimizer.py @ {os.path.basename(OPTIMIZER_FILE_PATH)}] Attempting to import services, utils, config ---")
try:
    # The following import is line 13 in your traceback for optimizer.py
    from services import strategy_engine, backtester # This might be a circular dependency if strategy_engine also tries to import optimizer indirectly.
    from utils.logger import get_logger
    from config import settings
    print(f"--- [DEBUG optimizer.py @ {os.path.basename(OPTIMIZER_FILE_PATH)}] Successfully imported services, utils, config ---")
except ImportError as e:
    print(f"--- [CRITICAL ERROR optimizer.py @ {os.path.basename(OPTIMIZER_FILE_PATH)}] Failed to import. Error: {e} ---")
    print(f"    Current sys.path during this error: {sys.path}")
    raise


logger = get_logger(__name__)

def _calculate_daily_returns(equity_series: pd.Series) -> pd.Series:
    """
    Calculates daily percentage returns from an equity series.
    Ensures the equity_series has a DatetimeIndex before resampling.
    """
    if equity_series.empty:
        logger.debug("_calculate_daily_returns: Input equity_series is empty.")
        return pd.Series(dtype=float)

    if not isinstance(equity_series.index, pd.DatetimeIndex):
        logger.warning(f"_calculate_daily_returns: equity_series.index is not a DatetimeIndex (type: {type(equity_series.index)}). Attempting conversion.")
        try:
            equity_series.index = pd.to_datetime(equity_series.index)
            if equity_series.index.tz is None: # If still naive, attempt to localize
                logger.warning("_calculate_daily_returns: Index converted to DatetimeIndex but is timezone-naive. Attempting to localize to NY as a default.")
                equity_series.index = equity_series.index.tz_localize(settings.NY_TIMEZONE_STR)
        except Exception as e:
            logger.error(f"_calculate_daily_returns: Failed to convert/localize equity_series.index. Error: {e}", exc_info=True)
            return pd.Series(dtype=float) # Return empty if conversion fails

    # Ensure timezone consistency if already localized
    if equity_series.index.tz is not None and equity_series.index.tz.zone != settings.NY_TIMEZONE.zone:
        try:
            equity_series.index = equity_series.index.tz_convert(settings.NY_TIMEZONE_STR)
        except Exception as e:
            logger.error(f"_calculate_daily_returns: Failed to convert equity_series.index to NY timezone. Error: {e}", exc_info=True)
            return pd.Series(dtype=float)


    if equity_series.nunique() <= 1: # Handles cases with flat equity or single point
        logger.debug("_calculate_daily_returns: Equity series has zero or one unique value, or is too short.")
        if len(equity_series) <= 1: return pd.Series(dtype=float)
        # If multiple identical values, returns will be zero.
        # Create an index for returns starting from the second day.
        # Ensure the index is daily for pct_change to work as expected after resample.
        daily_index_for_zero_returns = pd.date_range(start=equity_series.index.min().normalize(), end=equity_series.index.max().normalize(), freq='B') # Business day freq
        if len(daily_index_for_zero_returns) > 1:
             return pd.Series(0.0, index=daily_index_for_zero_returns[1:], dtype=float)
        return pd.Series(dtype=float)


    try:
        # Resample to daily, taking the last known equity for that day
        daily_equity = equity_series.resample('D').last()
    except TypeError as e: # Should not happen if index is DatetimeIndex
        logger.error(f"_calculate_daily_returns: TypeError during resample. Index type: {type(equity_series.index)}, Index head: {equity_series.index[:5]}. Error: {e}", exc_info=True)
        return pd.Series(dtype=float)

    if daily_equity.empty:
        logger.debug("_calculate_daily_returns: daily_equity is empty after resample.")
        return pd.Series(dtype=float)

    # Forward fill NaNs that might appear from resampling (e.g., weekends, holidays)
    daily_equity = daily_equity.ffill()
    
    # Drop any leading NaNs if ffill didn't cover the start (e.g. if series starts with NaN)
    first_valid_idx = daily_equity.first_valid_index()
    if first_valid_idx is not None:
        daily_equity = daily_equity.loc[first_valid_idx:]
    else: # All NaN after ffill
        logger.debug("_calculate_daily_returns: daily_equity is all NaN after resample and ffill.")
        return pd.Series(dtype=float)
    
    if daily_equity.empty or len(daily_equity) < 2 : # Need at least two points for pct_change
        logger.debug("_calculate_daily_returns: daily_equity became too short after ffill/loc for pct_change.")
        return pd.Series(dtype=float)

    daily_returns = daily_equity.pct_change().fillna(0) # Fill first NaN from pct_change with 0
    
    return daily_returns.iloc[1:] # Remove the first row which is always 0 after fillna(0) if original had data

def calculate_sharpe_ratio(returns_series, risk_free_rate=settings.RISK_FREE_RATE, trading_days_per_year=settings.TRADING_DAYS_PER_YEAR):
    """Calculates the annualized Sharpe ratio."""
    if returns_series.empty or len(returns_series) < max(2, settings.MIN_TRADES_FOR_METRICS) or returns_series.std() == 0:
        return np.nan
    excess_returns = returns_series - (risk_free_rate / trading_days_per_year)
    sharpe = excess_returns.mean() / excess_returns.std()
    return sharpe * np.sqrt(trading_days_per_year)

def calculate_sortino_ratio(returns_series, risk_free_rate=settings.RISK_FREE_RATE, trading_days_per_year=settings.TRADING_DAYS_PER_YEAR):
    """Calculates the annualized Sortino ratio."""
    if returns_series.empty or len(returns_series) < max(2, settings.MIN_TRADES_FOR_METRICS):
        return np.nan
    
    target_return = risk_free_rate / trading_days_per_year
    excess_returns = returns_series - target_return
    downside_returns = excess_returns[excess_returns < 0]

    if downside_returns.empty or downside_returns.std() == 0:
        # If no downside returns, Sortino is undefined or infinite if mean excess return is positive
        return np.inf if excess_returns.mean() > 0 else 0.0 if excess_returns.mean() == 0 else np.nan
        
    downside_std = downside_returns.std()
    if downside_std == 0 : # Should be caught by downside_returns.empty check but as a safeguard
        return np.inf if excess_returns.mean() > 0 else 0.0 if excess_returns.mean() == 0 else np.nan

    sortino = excess_returns.mean() / downside_std
    return sortino * np.sqrt(trading_days_per_year)

def _run_single_backtest_for_optimization(
    params: dict, # Expects 'strategy_name', 'SL Points', 'RRR', and strategy-specifics like 'EntryStartTime'
    price_data: pd.DataFrame, 
    initial_capital: float,
    risk_per_trade_percent: float, 
    data_interval_str: str
) -> dict:
    """
    Helper function to run a single backtest iteration for optimization purposes.
    Returns a dictionary of performance metrics and parameters.
    """
    strategy_name_opt = params['strategy_name']
    sl = params['SL Points'] # Should be float
    rrr = params['RRR']   # Should be float
    
    signal_gen_params_for_engine = {
        'strategy_name': strategy_name_opt,
        'stop_loss_points': float(sl), # Ensure float
        'rrr': float(rrr)             # Ensure float
    }
    if strategy_name_opt == "Gap Guardian":
        # Ensure EntryStartTime and EntryEndTime are datetime.time objects
        est_val = params.get('EntryStartTime')
        eet_val = params.get('EntryEndTime')
        signal_gen_params_for_engine['entry_start_time'] = est_val if isinstance(est_val, dt_time) else dt_time(int(est_val.hour), int(est_val.minute)) if hasattr(est_val, 'hour') else settings.DEFAULT_ENTRY_WINDOW_START_HOUR # Fallback
        signal_gen_params_for_engine['entry_end_time'] = eet_val if isinstance(eet_val, dt_time) else dt_time(int(eet_val.hour), int(eet_val.minute)) if hasattr(eet_val, 'hour') else settings.DEFAULT_ENTRY_WINDOW_END_HOUR


    signals_df = strategy_engine.generate_signals(price_data.copy(), **signal_gen_params_for_engine)
    trades_df, equity_s, perf_metrics = backtester.run_backtest(
        price_data.copy(), signals_df, initial_capital, 
        risk_per_trade_percent, float(sl), data_interval_str # Pass SL as float
    )
    
    # Ensure equity_s index is a DatetimeIndex and localized for _calculate_daily_returns
    if not isinstance(equity_s.index, pd.DatetimeIndex):
        logger.warning(f"_run_single_backtest_for_optimization: equity_s.index is not DatetimeIndex (type: {type(equity_s.index)}). Attempting conversion.")
        try:
            equity_s.index = pd.to_datetime(equity_s.index)
        except Exception as e_conv_idx:
            logger.error(f"Failed to convert equity_s index to DatetimeIndex: {e_conv_idx}", exc_info=True)
            equity_s = pd.Series(dtype=float, index=pd.to_datetime([])) # Empty series with DatetimeIndex

    if equity_s.index.tz is None and not equity_s.empty:
        try:
            equity_s.index = equity_s.index.tz_localize(settings.NY_TIMEZONE_STR)
        except Exception as e_loc_idx: # Handle AmbiguousTimeError, etc.
            logger.error(f"Failed to localize equity_s index to NY: {e_loc_idx}. Trying UTC.", exc_info=True)
            try:
                equity_s.index = equity_s.index.tz_localize('UTC').tz_convert(settings.NY_TIMEZONE_STR)
            except Exception as e_utc_conv:
                logger.error(f"Failed to convert equity_s index from UTC to NY: {e_utc_conv}", exc_info=True)
                # Fallback to naive if all fails, but metrics might be skewed
                equity_s.index = pd.to_datetime(equity_s.index).tz_localize(None)


    daily_ret = _calculate_daily_returns(equity_s.copy()) # Pass a copy to avoid modifying original

    result = {
        'SL Points': float(sl), 'RRR': float(rrr),
        'EntryStartHour': params.get('EntryStartTime', dt_time(0,0)).hour if hasattr(params.get('EntryStartTime'), 'hour') else np.nan,
        'EntryStartMinute': params.get('EntryStartTime', dt_time(0,0)).minute if hasattr(params.get('EntryStartTime'), 'minute') else np.nan,
        'EntryEndHour': params.get('EntryEndTime', dt_time(0,0)).hour if hasattr(params.get('EntryEndTime'), 'hour') else np.nan,
        'EntryEndMinute': params.get('EntryEndTime', dt_time(0,0)).minute if hasattr(params.get('EntryEndTime'), 'minute') else np.nan,
        'Total P&L': perf_metrics.get('Total P&L', np.nan),
        'Profit Factor': perf_metrics.get('Profit Factor', np.nan),
        'Win Rate': perf_metrics.get('Win Rate', np.nan),
        'Max Drawdown (%)': perf_metrics.get('Max Drawdown (%)', np.nan),
        'Total Trades': perf_metrics.get('Total Trades', 0),
        'Sharpe Ratio (Annualized)': calculate_sharpe_ratio(daily_ret.copy()), # Pass copy
        'Sortino Ratio (Annualized)': calculate_sortino_ratio(daily_ret.copy()), # Pass copy
        'Average Trade P&L': perf_metrics.get('Average Trade P&L', np.nan),
        'Average Winning Trade': perf_metrics.get('Average Winning Trade', np.nan),
        'Average Losing Trade': perf_metrics.get('Average Losing Trade', np.nan),
        '_trades_df': trades_df, # For WFO to aggregate trades
        '_equity_series': equity_s # For WFO to chain equity
    }
    return result

def run_grid_search(
    price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    param_value_map: dict, # e.g., {'sl_points': [10,15], 'rrr': [1,2], 'entry_start_hour': [9]}
    data_interval_str: str, 
    optimizer_control_params: dict, # e.g., {'metric_to_optimize': 'Sharpe', 'strategy_name': 'Gap Guardian'}
    progress_callback=None
) -> pd.DataFrame:
    """Performs grid search optimization."""
    param_names = list(param_value_map.keys())
    # Ensure all param values are lists/iterables for itertools.product
    param_values_for_product = []
    for name in param_names:
        values = param_value_map[name]
        if not isinstance(values, (list, tuple, np.ndarray)):
            logger.warning(f"Grid Search: Parameter '{name}' values are not iterable: {values}. Wrapping in a list.")
            param_values_for_product.append([values])
        elif len(values) == 0:
             logger.error(f"Grid Search: Parameter '{name}' has an empty list of values. Cannot proceed.")
             return pd.DataFrame() # Or raise error
        else:
            param_values_for_product.append(values)


    value_combinations = list(itertools.product(*param_values_for_product))
    total_combinations = len(value_combinations)
    if total_combinations == 0:
        logger.warning("Grid Search: No parameter combinations generated. Check param_value_map.")
        return pd.DataFrame()

    strategy_name_gs = optimizer_control_params.get('strategy_name', settings.DEFAULT_STRATEGY)
    logger.info(f"Grid Search ({strategy_name_gs}): Parameters: {param_names}. Combinations: {total_combinations}. Interval: {data_interval_str}")
    results_list = []

    for i, combo_values in enumerate(value_combinations):
        current_strategy_params_from_grid = dict(zip(param_names, combo_values))
        
        # Prepare parameters for the backtest, ensuring correct types
        params_for_bt_run = {
            'SL Points': float(current_strategy_params_from_grid['sl_points']), 
            'RRR': float(current_strategy_params_from_grid['rrr']),
            'strategy_name': strategy_name_gs
        }
        if strategy_name_gs == "Gap Guardian":
            entry_s_h = int(current_strategy_params_from_grid.get('entry_start_hour', settings.DEFAULT_ENTRY_WINDOW_START_HOUR))
            entry_s_m = int(current_strategy_params_from_grid.get('entry_start_minute', settings.DEFAULT_ENTRY_WINDOW_START_MINUTE))
            entry_e_h = int(current_strategy_params_from_grid.get('entry_end_hour', settings.DEFAULT_ENTRY_WINDOW_END_HOUR))
            entry_e_m = int(current_strategy_params_from_grid.get('entry_end_minute', settings.DEFAULT_ENTRY_WINDOW_END_MINUTE))
            params_for_bt_run['EntryStartTime'] = dt_time(entry_s_h, entry_s_m)
            params_for_bt_run['EntryEndTime'] = dt_time(entry_e_h, entry_e_m)
        
        try:
            perf_metrics_dict = _run_single_backtest_for_optimization(params_for_bt_run, price_data, initial_capital, risk_per_trade_percent, data_interval_str)
            # Remove temporary objects before appending to results list
            perf_metrics_dict.pop('_trades_df', None)
            perf_metrics_dict.pop('_equity_series', None)
            results_list.append(perf_metrics_dict)
        except Exception as e:
            logger.error(f"Error during Grid Search iteration ({strategy_name_gs}) with params {params_for_bt_run}: {e}", exc_info=True)
            # Log error with parameters for this iteration
            err_log_params = {k: (v.strftime("%H:%M") if isinstance(v, dt_time) else v) for k,v in params_for_bt_run.items() if k not in ['strategy_name']}
            error_entry = {'Strategy': strategy_name_gs, **err_log_params, 'Total P&L': np.nan, 'Error': str(e)}
            results_list.append(error_entry)

        if progress_callback:
            progress_callback((i + 1) / total_combinations, f"Grid Search ({strategy_name_gs})")
            
    return pd.DataFrame(results_list)


def run_random_search(
    price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    param_config_map: dict, # e.g. {'sl_points': (5.0,30.0), 'rrr': (1.0,3.0), 'entry_start_hour': [8,9,10]} or (8,10) for uniform int
    data_interval_str: str, 
    optimizer_control_params: dict, # e.g. {'iterations': 50, 'metric_to_optimize': 'Sharpe', 'strategy_name': 'Gap Guardian'}
    progress_callback=None
) -> pd.DataFrame:
    """Performs random search optimization."""
    num_iterations = optimizer_control_params.get('iterations', settings.DEFAULT_RANDOM_SEARCH_ITERATIONS)
    if num_iterations <= 0:
        logger.warning("Random Search: Number of iterations is non-positive. No search will be performed.")
        return pd.DataFrame()

    strategy_name_rs = optimizer_control_params.get('strategy_name', settings.DEFAULT_STRATEGY)
    logger.info(f"Random Search ({strategy_name_rs}): Iterations: {num_iterations}. Interval: {data_interval_str}")
    results_list = []

    for i in range(num_iterations):
        current_random_params = {}
        for p_name, p_config_item in param_config_map.items():
            if isinstance(p_config_item, list): # Choose from a list of discrete values
                current_random_params[p_name] = random.choice(p_config_item)
            elif isinstance(p_config_item, tuple) and len(p_config_item) == 2: # Range (min, max)
                p_min, p_max = p_config_item
                if isinstance(p_min, float) or isinstance(p_max, float): # Continuous range
                    val = random.uniform(float(p_min), float(p_max))
                    # Round floats for SL/RRR to reasonable precision
                    if "points" in p_name or "rrr" in p_name: val = round(val, 2) 
                elif isinstance(p_min, int) and isinstance(p_max, int): # Discrete integer range
                    val = random.randint(int(p_min), int(p_max))
                else:
                    logger.warning(f"Random Search ({strategy_name_rs}): Parameter '{p_name}' has tuple config with mixed/unsupported types: {p_config_item}. Skipping.")
                    continue
                current_random_params[p_name] = val
            else:
                logger.warning(f"Random Search ({strategy_name_rs}): Skipping param '{p_name}' due to unexpected config format: {p_config_item}")
                continue
        
        # Prepare parameters for the backtest, ensuring correct types
        params_for_bt_run = {
            'SL Points': float(current_random_params.get('sl_points', settings.DEFAULT_STOP_LOSS_POINTS)), 
            'RRR': float(current_random_params.get('rrr', settings.DEFAULT_RRR)),
            'strategy_name': strategy_name_rs
        }
        if strategy_name_rs == "Gap Guardian":
            entry_s_h = int(current_random_params.get('entry_start_hour', settings.DEFAULT_ENTRY_WINDOW_START_HOUR))
            entry_s_m = int(current_random_params.get('entry_start_minute', settings.DEFAULT_ENTRY_WINDOW_START_MINUTE))
            entry_e_h = int(current_random_params.get('entry_end_hour', settings.DEFAULT_ENTRY_WINDOW_END_HOUR))
            entry_e_m = int(current_random_params.get('entry_end_minute', settings.DEFAULT_ENTRY_WINDOW_END_MINUTE))
            params_for_bt_run['EntryStartTime'] = dt_time(entry_s_h, entry_s_m)
            params_for_bt_run['EntryEndTime'] = dt_time(entry_e_h, entry_e_m)
            
        try:
            perf_metrics_dict = _run_single_backtest_for_optimization(params_for_bt_run, price_data, initial_capital, risk_per_trade_percent, data_interval_str)
            perf_metrics_dict.pop('_trades_df', None)
            perf_metrics_dict.pop('_equity_series', None)
            results_list.append(perf_metrics_dict)
        except Exception as e:
            logger.error(f"Error during Random Search iteration ({strategy_name_rs}) with params {params_for_bt_run}: {e}", exc_info=True)
            err_log_params = {k: (v.strftime("%H:%M") if isinstance(v, dt_time) else v) for k,v in params_for_bt_run.items() if k not in ['strategy_name']}
            error_entry = {'Strategy': strategy_name_rs, **err_log_params, 'Total P&L': np.nan, 'Error': str(e)}
            results_list.append(error_entry)

        if progress_callback:
            progress_callback((i + 1) / num_iterations, f"Random Search ({strategy_name_rs})")
            
    return pd.DataFrame(results_list)


def run_walk_forward_optimization(
    full_price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    wfo_params: dict, # {'in_sample_days', 'out_of_sample_days', 'step_days'}
    opt_algo: str, # "Grid Search" or "Random Search" for inner optimization
    opt_config_and_control: dict, # Combined: param ranges (e.g. sl_points: (5,30)) AND control ({'metric_to_optimize', 'strategy_name', 'iterations'})
    data_interval_str: str, 
    progress_callback=None # For overall WFO progress
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Performs walk-forward optimization."""
    
    if full_price_data.empty or not isinstance(full_price_data.index, pd.DatetimeIndex):
        logger.error("WFO: Price data is empty or has invalid (non-DatetimeIndex) index.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float, index=pd.to_datetime([])), {} # Ensure empty Series has DatetimeIndex

    # Ensure data is timezone-aware (NY) for date operations
    if full_price_data.index.tz is None:
        logger.warning(f"WFO: full_price_data.index is timezone-naive. Localizing to {settings.NY_TIMEZONE_STR}.")
        try:
            full_price_data.index = full_price_data.index.tz_localize(settings.NY_TIMEZONE_STR)
        except Exception as tz_err: # Catch AmbiguousTimeError, NonExistentTimeError
            logger.error(f"WFO: Failed to localize full_price_data index to NY: {tz_err}. Trying UTC then convert.", exc_info=True)
            try:
                full_price_data.index = full_price_data.index.tz_localize('UTC').tz_convert(settings.NY_TIMEZONE_STR)
            except Exception as utc_conv_err:
                 logger.error(f"WFO: Failed to localize/convert full_price_data index via UTC to NY: {utc_conv_err}.", exc_info=True)
                 return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float, index=pd.to_datetime([])), {}
    elif full_price_data.index.tz.zone != settings.NY_TIMEZONE.zone:
        logger.warning(f"WFO: full_price_data.index is {full_price_data.index.tz}. Converting to {settings.NY_TIMEZONE_STR}.")
        try:
            full_price_data.index = full_price_data.index.tz_convert(settings.NY_TIMEZONE_STR)
        except Exception as tz_conv_err:
            logger.error(f"WFO: Failed to convert full_price_data index to NY: {tz_conv_err}.", exc_info=True)
            return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float, index=pd.to_datetime([])), {}


    in_sample_days = int(wfo_params['in_sample_days'])
    out_of_sample_days = int(wfo_params['out_of_sample_days'])
    step_days = int(wfo_params['step_days'])
    
    metric_to_optimize = opt_config_and_control.get('metric_to_optimize', settings.DEFAULT_OPTIMIZATION_METRIC)
    strategy_name_wfo = opt_config_and_control.get('strategy_name', settings.DEFAULT_STRATEGY)
    wfo_optimizer_iterations = opt_config_and_control.get('iterations', settings.DEFAULT_RANDOM_SEARCH_ITERATIONS)

    control_keys_for_inner_opt = ['metric_to_optimize', 'strategy_name', 'iterations']
    inner_opt_param_definitions = {k: v for k, v in opt_config_and_control.items() if k not in control_keys_for_inner_opt}

    start_date_overall, end_date_overall = full_price_data.index.min().date(), full_price_data.index.max().date()
    total_data_duration_days = (end_date_overall - start_date_overall).days + 1
    min_required_duration_for_one_fold = in_sample_days + out_of_sample_days
    
    if total_data_duration_days < min_required_duration_for_one_fold:
        logger.warning(f"WFO ({strategy_name_wfo}): Total data duration ({total_data_duration_days}d) < min required for one fold ({min_required_duration_for_one_fold}d). Cannot proceed.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float, index=pd.to_datetime([])), {}
        
    logger.info(f"Starting WFO ({strategy_name_wfo}): IS={in_sample_days}d, OOS={out_of_sample_days}d, Step={step_days}d. Optimizing for {metric_to_optimize} using {opt_algo}. Interval: {data_interval_str}")
    
    all_oos_trades_list, wfo_log_list = [], []
    all_oos_equity_series_list = [] 
    current_fold_capital = initial_capital # Capital for the start of each OOS period

    # Estimate total folds for progress bar
    num_possible_folds = 0; temp_is_start = start_date_overall
    while temp_is_start + timedelta(days=in_sample_days - 1 + out_of_sample_days) <= end_date_overall : # Ensure full IS+OOS window fits
        num_possible_folds += 1
        temp_is_start += timedelta(days=step_days)
    total_folds_estimate = max(1, num_possible_folds)
    logger.info(f"WFO Estimated total folds: {total_folds_estimate}")


    current_in_sample_start_date = start_date_overall
    fold_num = 0
    
    while True:
        fold_num += 1
        current_in_sample_end_date = current_in_sample_start_date + timedelta(days=in_sample_days - 1)
        current_oos_start_date = current_in_sample_end_date + timedelta(days=1)
        current_oos_end_date = current_oos_start_date + timedelta(days=out_of_sample_days - 1)

        # Check if the current In-Sample period goes beyond available data
        if current_in_sample_end_date > end_date_overall:
            logger.info(f"WFO Fold {fold_num} ({strategy_name_wfo}): In-sample period ends ({current_in_sample_end_date}) after all data ({end_date_overall}). Ending WFO.")
            break
        
        # Adjust OOS end date if it exceeds overall data end date
        if current_oos_end_date > end_date_overall:
            current_oos_end_date = end_date_overall
            logger.info(f"WFO Fold {fold_num} ({strategy_name_wfo}): OOS period adjusted to end at {current_oos_end_date} (data end).")

        # Check if OOS period is valid (at least 1 day)
        if current_oos_start_date > current_oos_end_date:
            logger.info(f"WFO Fold {fold_num} ({strategy_name_wfo}): OOS period is invalid (start {current_oos_start_date} > end {current_oos_end_date}). This might happen if IS period consumes all remaining data. Ending WFO.")
            break
        
        logger.info(f"WFO Fold {fold_num}/{total_folds_estimate} ({strategy_name_wfo}): IS [{current_in_sample_start_date} - {current_in_sample_end_date}], OOS [{current_oos_start_date} - {current_oos_end_date}]")
        
        # Slice data for In-Sample period
        in_sample_data = full_price_data[
            (full_price_data.index.date >= current_in_sample_start_date) & 
            (full_price_data.index.date <= current_in_sample_end_date)
        ]
        
        # Slice data for Out-of-Sample period
        out_of_sample_data = full_price_data[
            (full_price_data.index.date >= current_oos_start_date) & 
            (full_price_data.index.date <= current_oos_end_date)
        ]

        if in_sample_data.empty:
            logger.warning(f"WFO Fold {fold_num} ({strategy_name_wfo}): In-sample data empty for period. Advancing to next potential fold.")
            current_in_sample_start_date += timedelta(days=step_days)
            if progress_callback: progress_callback(min(1.0, fold_num / total_folds_estimate), f"WFO Fold {fold_num} (IS Empty)")
            if current_in_sample_start_date > end_date_overall - timedelta(days=in_sample_days -1) : break # Stop if next IS start is too late
            continue
            
        # Run inner optimization on In-Sample data
        optimization_results_df_is = pd.DataFrame()
        inner_optimizer_control_params = {'metric_to_optimize': metric_to_optimize, 'strategy_name': strategy_name_wfo}

        if opt_algo == "Grid Search":
            optimization_results_df_is = run_grid_search(in_sample_data, initial_capital, risk_per_trade_percent, inner_opt_param_definitions, data_interval_str, inner_optimizer_control_params, None) # No progress for inner
        elif opt_algo == "Random Search":
            inner_optimizer_control_params['iterations'] = wfo_optimizer_iterations
            optimization_results_df_is = run_random_search(in_sample_data, initial_capital, risk_per_trade_percent, inner_opt_param_definitions, data_interval_str, inner_optimizer_control_params, None)
        
        # Determine best parameters from in-sample optimization
        best_params_for_oos_run = { # Default parameters if optimization fails or yields no results
            'SL Points': settings.DEFAULT_STOP_LOSS_POINTS, 
            'RRR': settings.DEFAULT_RRR,
            'strategy_name': strategy_name_wfo
        }
        if strategy_name_wfo == "Gap Guardian":
            best_params_for_oos_run['EntryStartTime'] = dt_time(settings.DEFAULT_ENTRY_WINDOW_START_HOUR, settings.DEFAULT_ENTRY_WINDOW_START_MINUTE)
            best_params_for_oos_run['EntryEndTime'] = dt_time(settings.DEFAULT_ENTRY_WINDOW_END_HOUR, settings.DEFAULT_ENTRY_WINDOW_END_MINUTE)

        in_sample_metric_value = np.nan
        
        if not optimization_results_df_is.empty and metric_to_optimize in optimization_results_df_is.columns:
            valid_is_opt_results = optimization_results_df_is.dropna(subset=[metric_to_optimize])
            if not valid_is_opt_results.empty:
                best_row_is = None
                if metric_to_optimize == "Max Drawdown (%)": # Minimize (closer to 0 is better)
                    # Ensure MDD is negative or zero, then pick the one with largest (closest to zero) value
                    valid_mdd_results = valid_is_opt_results[valid_is_opt_results[metric_to_optimize] <= 0]
                    if not valid_mdd_results.empty:
                        best_row_is = valid_mdd_results.loc[valid_mdd_results[metric_to_optimize].idxmax()]
                    else: # No valid non-positive MDD, pick first row as fallback
                        best_row_is = valid_is_opt_results.iloc[0] 
                else: # Maximize other metrics
                    best_row_is = valid_is_opt_results.loc[valid_is_opt_results[metric_to_optimize].idxmax()]
                
                if best_row_is is not None:
                    best_params_for_oos_run.update({
                        'SL Points': float(best_row_is['SL Points']), 
                        'RRR': float(best_row_is['RRR']),
                    })
                    if strategy_name_wfo == "Gap Guardian":
                         best_params_for_oos_run['EntryStartTime'] = dt_time(int(best_row_is['EntryStartHour']), int(best_row_is['EntryStartMinute']))
                         best_params_for_oos_run['EntryEndTime'] = dt_time(int(best_row_is['EntryEndHour']), int(best_row_is.get('EntryEndMinute', settings.DEFAULT_ENTRY_WINDOW_END_MINUTE)))
                    in_sample_metric_value = best_row_is[metric_to_optimize]
        
        current_fold_log_entry = {
            'Fold': fold_num, 
            'InSampleStart': current_in_sample_start_date, 'InSampleEnd': current_in_sample_end_date,
            'OutOfSampleStart': current_oos_start_date, 'OutOfSampleEnd': current_oos_end_date,
            'BestSL': best_params_for_oos_run['SL Points'], 'BestRRR': best_params_for_oos_run['RRR'],
            'OptimizedMetric': metric_to_optimize, 'InSampleMetricValue': in_sample_metric_value
        }
        if strategy_name_wfo == "Gap Guardian":
            current_fold_log_entry['BestEntryStart'] = best_params_for_oos_run['EntryStartTime'].strftime("%H:%M")
            current_fold_log_entry['BestEntryEnd'] = best_params_for_oos_run['EntryEndTime'].strftime("%H:%M")

        # Run backtest on Out-of-Sample data using best parameters from In-Sample
        if not out_of_sample_data.empty:
            oos_backtest_results = _run_single_backtest_for_optimization(
                best_params_for_oos_run, out_of_sample_data, 
                current_fold_capital, # Use capital from end of previous OOS period
                risk_per_trade_percent, data_interval_str
            )
            oos_trades_this_fold = oos_backtest_results.get('_trades_df', pd.DataFrame())
            oos_equity_this_fold = oos_backtest_results.get('_equity_series', pd.Series(dtype=float))

            if not oos_trades_this_fold.empty:
                all_oos_trades_list.append(oos_trades_this_fold)
            
            if not oos_equity_this_fold.empty and oos_equity_this_fold.notna().any():
                all_oos_equity_series_list.append(oos_equity_this_fold)
                current_fold_capital = oos_equity_this_fold.iloc[-1] # Update capital for next fold
            
            # Log OOS performance metrics for this fold
            for k, v in oos_backtest_results.items():
                if not k.startswith('_') and k not in ['SL Points', 'RRR', 'EntryStartHour', 'EntryStartMinute', 'EntryEndHour', 'EntryEndMinute', 'strategy_name']:
                    current_fold_log_entry[f'OOS_{k.replace(" (%)", "Pct").replace(" ", "")}'] = v
        else:
            logger.info(f"WFO Fold {fold_num} ({strategy_name_wfo}): Out-of-Sample data empty for period. No OOS backtest.")
        
        wfo_log_list.append(current_fold_log_entry)
        
        # Advance to the next In-Sample period start date
        current_in_sample_start_date += timedelta(days=step_days)
        if progress_callback: 
            progress_callback(min(1.0, fold_num / total_folds_estimate), f"WFO Fold {fold_num} ({strategy_name_wfo})")

        # Break condition if next IS period is too short or goes beyond data
        if current_in_sample_start_date + timedelta(days=in_sample_days - 1) > end_date_overall:
            logger.info(f"WFO: Next In-Sample period start ({current_in_sample_start_date}) would extend beyond data. Ending WFO after {fold_num} folds.")
            break
            
    wfo_log_df = pd.DataFrame(wfo_log_list)
    final_oos_trades_aggregated_df = pd.concat(all_oos_trades_list, ignore_index=True) if all_oos_trades_list else pd.DataFrame()

    # Chain OOS equity series for a single plottable equity curve and aggregate metrics
    final_chained_oos_equity = pd.Series(dtype=float, index=pd.to_datetime([]).tz_localize(settings.NY_TIMEZONE_STR)) # Ensure tz-aware empty series
    if all_oos_equity_series_list:
        # Ensure all series are tz-aware (NY) before concat
        processed_equity_list = []
        for es in all_oos_equity_series_list:
            if not es.empty:
                if not isinstance(es.index, pd.DatetimeIndex): es.index = pd.to_datetime(es.index)
                if es.index.tz is None: es.index = es.index.tz_localize(settings.NY_TIMEZONE_STR)
                elif es.index.tz.zone != settings.NY_TIMEZONE.zone: es.index = es.index.tz_convert(settings.NY_TIMEZONE_STR)
                processed_equity_list.append(es)
        
        if processed_equity_list:
            # Create a base equity series starting with initial_capital at the very first OOS timestamp
            first_oos_timestamp = min(es.index.min() for es in processed_equity_list if not es.empty)
            base_equity_point = pd.Series([initial_capital], index=[first_oos_timestamp - pd.Timedelta(microseconds=1)]) # Point just before first trade
            if base_equity_point.index.tz is None: base_equity_point.index = base_equity_point.index.tz_localize(settings.NY_TIMEZONE_STR)
            
            temp_chained_equity = pd.concat([base_equity_point] + processed_equity_list).sort_index()
            temp_chained_equity = temp_chained_equity[~temp_chained_equity.index.duplicated(keep='last')] # Keep last if duplicate timestamps

            # Reindex to the full OOS period covered by full_price_data for continuous plot
            min_oos_date_overall = wfo_log_df['OutOfSampleStart'].min() if not wfo_log_df.empty else None
            max_oos_date_overall = wfo_log_df['OutOfSampleEnd'].max() if not wfo_log_df.empty else None

            if min_oos_date_overall and max_oos_date_overall:
                oos_full_datetime_index = full_price_data[
                    (full_price_data.index.date >= min_oos_date_overall) &
                    (full_price_data.index.date <= max_oos_date_overall)
                ].index
                if not oos_full_datetime_index.empty:
                    final_chained_oos_equity = temp_chained_equity.reindex(oos_full_datetime_index, method='ffill')
                    # Ensure first point is initial capital if reindexing caused NaN at start
                    if not final_chained_oos_equity.empty and pd.isna(final_chained_oos_equity.iloc[0]):
                        final_chained_oos_equity.iloc[0] = initial_capital
                        final_chained_oos_equity = final_chained_oos_equity.ffill() # Re-fill after setting first point
                else: # Fallback if no OOS index from price data (should not happen if WFO ran)
                    final_chained_oos_equity = temp_chained_equity
            else: # Fallback if no WFO log to determine overall OOS range
                final_chained_oos_equity = temp_chained_equity
        
    if final_chained_oos_equity.empty: # If still empty, ensure it's a valid Series for return
         final_chained_oos_equity = pd.Series([initial_capital], index=[pd.Timestamp.now(tz=settings.NY_TIMEZONE_STR)])


    # Calculate overall OOS performance metrics from the aggregated trades and chained equity
    aggregated_oos_performance_summary = {
        'Total Trades': 0, 'Total P&L': 0.0, 'Final Capital': initial_capital, 
        'Sharpe Ratio (Annualized)': np.nan, 'Sortino Ratio (Annualized)': np.nan, 
        'Max Drawdown (%)': 0.0, 'Win Rate': 0.0, 'Profit Factor': 0.0,
        'Average Trade P&L': np.nan, 'Average Winning Trade': np.nan, 'Average Losing Trade': np.nan
    }
    if not final_oos_trades_aggregated_df.empty:
        aggregated_oos_performance_summary['Total Trades'] = len(final_oos_trades_aggregated_df)
        total_pnl_from_oos_trades = final_oos_trades_aggregated_df['P&L'].sum()
        aggregated_oos_performance_summary['Total P&L'] = total_pnl_from_oos_trades
        
        # Final capital from the chained equity curve
        if not final_chained_oos_equity.empty and final_chained_oos_equity.notna().any():
            aggregated_oos_performance_summary['Final Capital'] = final_chained_oos_equity.iloc[-1]
            
            # Max Drawdown from chained equity
            cumulative_max_oos = final_chained_oos_equity.cummax()
            drawdown_oos = (final_chained_oos_equity - cumulative_max_oos) / cumulative_max_oos
            drawdown_oos.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero if equity hits 0
            mdd_oos_val = drawdown_oos.min() * 100
            aggregated_oos_performance_summary['Max Drawdown (%)'] = mdd_oos_val if pd.notna(mdd_oos_val) and not drawdown_oos.empty else 0.0
        else: # Fallback if equity curve is problematic
             aggregated_oos_performance_summary['Final Capital'] = initial_capital + total_pnl_from_oos_trades


        # Sharpe and Sortino from daily returns of chained equity
        daily_oos_returns_aggregated = _calculate_daily_returns(final_chained_oos_equity.copy())
        aggregated_oos_performance_summary['Sharpe Ratio (Annualized)'] = calculate_sharpe_ratio(daily_oos_returns_aggregated)
        aggregated_oos_performance_summary['Sortino Ratio (Annualized)'] = calculate_sortino_ratio(daily_oos_returns_aggregated)

        num_winning_oos = len(final_oos_trades_aggregated_df[final_oos_trades_aggregated_df['P&L'] > 0])
        num_losing_oos = len(final_oos_trades_aggregated_df[final_oos_trades_aggregated_df['P&L'] < 0])
        aggregated_oos_performance_summary['Win Rate'] = (num_winning_oos / aggregated_oos_performance_summary['Total Trades'] * 100) if aggregated_oos_performance_summary['Total Trades'] > 0 else 0.0
        
        if aggregated_oos_performance_summary['Total Trades'] > 0:
            aggregated_oos_performance_summary['Average Trade P&L'] = final_oos_trades_aggregated_df['P&L'].mean()
        if num_winning_oos > 0:
            aggregated_oos_performance_summary['Average Winning Trade'] = final_oos_trades_aggregated_df[final_oos_trades_aggregated_df['P&L'] > 0]['P&L'].mean()
        if num_losing_oos > 0:
            aggregated_oos_performance_summary['Average Losing Trade'] = final_oos_trades_aggregated_df[final_oos_trades_aggregated_df['P&L'] < 0]['P&L'].mean()
        
        gross_profit_oos = final_oos_trades_aggregated_df[final_oos_trades_aggregated_df['P&L'] > 0]['P&L'].sum()
        gross_loss_oos = abs(final_oos_trades_aggregated_df[final_oos_trades_aggregated_df['P&L'] < 0]['P&L'].sum())
        if gross_loss_oos > 0:
            aggregated_oos_performance_summary['Profit Factor'] = gross_profit_oos / gross_loss_oos
        elif gross_profit_oos > 0 and gross_loss_oos == 0: # Only profits, no losses
            aggregated_oos_performance_summary['Profit Factor'] = np.inf
        else: # No profits or no trades
            aggregated_oos_performance_summary['Profit Factor'] = 0.0
            
    logger.info(f"WFO ({strategy_name_wfo}) complete. Processed {fold_num-1 if fold_num > 0 else 0} successful folds. Aggregated OOS P&L: {aggregated_oos_performance_summary['Total P&L']:.2f}")
    return wfo_log_df, final_oos_trades_aggregated_df, final_chained_oos_equity, aggregated_oos_performance_summary
