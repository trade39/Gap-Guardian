# services/optimizer.py
"""
Performs parameter optimization for trading strategies using Grid Search, Random Search,
and Walk-Forward Optimization (WFO).
Corrected Max Drawdown calculation and WFO equity curve generation.
Fixed DatetimeIndex.union_many error.
Corrected resample call in _calculate_daily_returns by ensuring DatetimeIndex.
"""
import pandas as pd
import numpy as np
import itertools
import random
from datetime import timedelta, time as dt_time

from services import strategy_engine, backtester
from utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

def _calculate_daily_returns(equity_series: pd.Series) -> pd.Series:
    """
    Calculates daily percentage returns from an equity series.
    Ensures the equity_series has a DatetimeIndex before resampling.
    """
    if equity_series.empty:
        logger.debug("_calculate_daily_returns: Input equity_series is empty.")
        return pd.Series(dtype=float)

    # Ensure the index is a DatetimeIndex
    if not isinstance(equity_series.index, pd.DatetimeIndex):
        logger.warning(f"_calculate_daily_returns: equity_series.index is not a DatetimeIndex (type: {type(equity_series.index)}). Attempting conversion.")
        try:
            equity_series.index = pd.to_datetime(equity_series.index)
            # If conversion results in naive datetime, try to localize (assuming UTC or NY as appropriate context)
            if equity_series.index.tz is None:
                logger.warning("_calculate_daily_returns: Index converted to DatetimeIndex but is timezone-naive. Attempting to localize to UTC as a default.")
                equity_series.index = equity_series.index.tz_localize('UTC') # Or settings.NY_TIMEZONE_STR if more contextually appropriate
        except Exception as e:
            logger.error(f"_calculate_daily_returns: Failed to convert equity_series.index to DatetimeIndex. Error: {e}", exc_info=True)
            return pd.Series(dtype=float)

    if equity_series.nunique() <= 1:
        logger.debug("_calculate_daily_returns: Equity series has zero or one unique value. Returning empty Series or zero returns.")
        # If only one unique value, pct_change would be all NaN or 0.
        # If it's just one point in time, there are no returns.
        if len(equity_series) <= 1:
            return pd.Series(dtype=float)
        # If multiple points but same value, returns are 0
        return pd.Series(0.0, index=equity_series.index[1:], dtype=float)


    # Resample to daily, taking the last value, then ffill for non-trading days within the series span
    try:
        daily_equity = equity_series.resample(rule='D').last()
    except TypeError as e:
        logger.error(f"_calculate_daily_returns: TypeError during resample. Index type: {type(equity_series.index)}, Index head: {equity_series.index[:5]}. Error: {e}", exc_info=True)
        return pd.Series(dtype=float) # Critical error, cannot proceed

    if daily_equity.empty:
        logger.debug("_calculate_daily_returns: daily_equity is empty after resample.")
        return pd.Series(dtype=float)

    # Forward fill missing daily equity values
    first_valid_idx = daily_equity.first_valid_index()
    last_valid_idx = daily_equity.last_valid_index()

    if first_valid_idx is not None and last_valid_idx is not None and first_valid_idx <= last_valid_idx:
        daily_equity = daily_equity.loc[first_valid_idx:last_valid_idx].ffill()
    elif first_valid_idx is not None: # Only one valid point after resample
        daily_equity = daily_equity.loc[[first_valid_idx]]
    else: # All NaN after resample
        logger.debug("_calculate_daily_returns: daily_equity is all NaN after resample and loc.")
        return pd.Series(dtype=float)
    
    if daily_equity.empty: # Check again after ffill and loc
        logger.debug("_calculate_daily_returns: daily_equity became empty after ffill/loc.")
        return pd.Series(dtype=float)

    daily_returns = daily_equity.pct_change().fillna(0)

    # Handle case where only one equity point remains after resample (pct_change would be empty)
    if daily_returns.empty and not daily_equity.empty and len(daily_equity) == 1:
        logger.debug("_calculate_daily_returns: daily_returns empty, daily_equity has 1 point. Returning Series with 0.0.")
        return pd.Series([0.0], index=[daily_equity.index[0]], dtype=float)

    return daily_returns

def calculate_sharpe_ratio(returns_series, risk_free_rate=settings.RISK_FREE_RATE, trading_days_per_year=settings.TRADING_DAYS_PER_YEAR):
    if returns_series.empty or len(returns_series) < max(2, settings.MIN_TRADES_FOR_METRICS) or returns_series.std() == 0: return np.nan
    excess_returns = returns_series - (risk_free_rate / trading_days_per_year)
    sharpe = excess_returns.mean() / excess_returns.std()
    return sharpe * np.sqrt(trading_days_per_year)

def calculate_sortino_ratio(returns_series, risk_free_rate=settings.RISK_FREE_RATE, trading_days_per_year=settings.TRADING_DAYS_PER_YEAR):
    if returns_series.empty or len(returns_series) < max(2, settings.MIN_TRADES_FOR_METRICS): return np.nan
    target_return = risk_free_rate / trading_days_per_year; excess_returns = returns_series - target_return
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.empty or downside_returns.std() == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0 if excess_returns.mean() == 0 else np.nan # Adjusted for 0 mean
    downside_std = downside_returns.std(); sortino = excess_returns.mean() / downside_std
    return sortino * np.sqrt(trading_days_per_year)

def _run_single_backtest_for_optimization(
    params: dict, price_data: pd.DataFrame, initial_capital: float,
    risk_per_trade_percent: float, data_interval_str: str
) -> dict:
    sl = params['SL Points']; rrr = params['RRR']
    entry_start_time = params['EntryStartTime']
    entry_end_time = params['EntryEndTime']
    signals_df = strategy_engine.generate_signals(price_data.copy(), sl, rrr, entry_start_time, entry_end_time)
    trades_df, equity_s, perf_metrics = backtester.run_backtest(price_data.copy(), signals_df, initial_capital, risk_per_trade_percent, sl, data_interval_str)
    
    # Ensure equity_s has DatetimeIndex before calculating returns
    if not isinstance(equity_s.index, pd.DatetimeIndex):
        logger.warning(f"_run_single_backtest_for_optimization: equity_s.index is not DatetimeIndex (type: {type(equity_s.index)}). Attempting conversion.")
        try:
            equity_s.index = pd.to_datetime(equity_s.index)
            if equity_s.index.tz is None: # If naive, assume UTC or NY
                 equity_s.index = equity_s.index.tz_localize('UTC') # Or settings.NY_TIMEZONE_STR
        except Exception as e:
            logger.error(f"Failed to convert equity_s index: {e}", exc_info=True)
            # Fallback: create an empty series for returns if conversion fails
            equity_s = pd.Series(dtype=float, index=pd.to_datetime([]))


    daily_ret = _calculate_daily_returns(equity_s)
    result = {
        'SL Points': sl, 'RRR': rrr,
        'EntryStartHour': entry_start_time.hour, 'EntryStartMinute': entry_start_time.minute,
        'EntryEndHour': entry_end_time.hour, 'EntryEndMinute': entry_end_time.minute,
        'Total P&L': perf_metrics.get('Total P&L', np.nan),
        'Profit Factor': perf_metrics.get('Profit Factor', np.nan),
        'Win Rate': perf_metrics.get('Win Rate', np.nan),
        'Max Drawdown (%)': perf_metrics.get('Max Drawdown (%)', np.nan), # This should be negative or zero
        'Total Trades': perf_metrics.get('Total Trades', 0),
        'Sharpe Ratio (Annualized)': calculate_sharpe_ratio(daily_ret),
        'Sortino Ratio (Annualized)': calculate_sortino_ratio(daily_ret),
        'Average Trade P&L': perf_metrics.get('Average Trade P&L', np.nan),
        'Average Winning Trade': perf_metrics.get('Average Winning Trade', np.nan),
        'Average Losing Trade': perf_metrics.get('Average Losing Trade', np.nan),
        '_trades_df': trades_df, '_equity_series': equity_s # Keep for WFO internal use
    }
    return result

def run_grid_search(
    price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    param_value_map: dict, data_interval_str: str, progress_callback=None
) -> pd.DataFrame:
    param_names = list(param_value_map.keys())
    value_combinations = list(itertools.product(*(param_value_map[name] for name in param_names)))
    total_combinations = len(value_combinations)
    if total_combinations == 0: return pd.DataFrame()
    logger.info(f"Grid Search: Parameters: {param_names}. Combinations: {total_combinations}. Interval: {data_interval_str}")
    results = []
    for i, combo_values in enumerate(value_combinations):
        current_strategy_params = dict(zip(param_names, combo_values))
        # Ensure types are correct for dt_time constructor
        entry_s_h = int(current_strategy_params.get('entry_start_hour', settings.DEFAULT_ENTRY_WINDOW_START_HOUR))
        entry_s_m = int(current_strategy_params.get('entry_start_minute', settings.DEFAULT_ENTRY_WINDOW_START_MINUTE))
        entry_e_h = int(current_strategy_params.get('entry_end_hour', settings.DEFAULT_ENTRY_WINDOW_END_HOUR))
        entry_e_m = int(current_strategy_params.get('entry_end_minute', settings.DEFAULT_ENTRY_WINDOW_END_MINUTE))
        
        params_for_bt = {'SL Points': float(current_strategy_params['sl_points']), 
                         'RRR': float(current_strategy_params['rrr']),
                         'EntryStartTime': dt_time(entry_s_h, entry_s_m), 
                         'EntryEndTime': dt_time(entry_e_h, entry_e_m)}
        try:
            perf = _run_single_backtest_for_optimization(params_for_bt, price_data, initial_capital, risk_per_trade_percent, data_interval_str)
            perf.pop('_trades_df', None); perf.pop('_equity_series', None); results.append(perf)
        except Exception as e:
            logger.error(f"Error Grid Search params {params_for_bt}: {e}", exc_info=True)
            # Log error with stringified times for easier debug
            err_log_params = {k: (v.strftime("%H:%M") if isinstance(v, dt_time) else v) for k,v in params_for_bt.items()}
            err_log = {**err_log_params, 'Total P&L': np.nan, 'Error': str(e)}
            results.append(err_log)
        if progress_callback: progress_callback((i + 1) / total_combinations, "Grid Search")
    return pd.DataFrame(results)

def run_random_search(
    price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    param_config_map: dict, num_iterations: int, data_interval_str: str, progress_callback=None
) -> pd.DataFrame:
    if num_iterations == 0: return pd.DataFrame()
    logger.info(f"Random Search: Iterations: {num_iterations}. Interval: {data_interval_str}")
    results = []
    for i in range(num_iterations):
        current_strategy_params = {}
        for p_name, p_config_item in param_config_map.items():
            if isinstance(p_config_item, list): current_strategy_params[p_name] = random.choice(p_config_item)
            elif isinstance(p_config_item, tuple) and len(p_config_item) == 2:
                p_min, p_max = p_config_item
                val = random.uniform(float(p_min), float(p_max))
                if "points" in p_name or "rrr" in p_name: val = round(val, 2) # Ensure float, then round
                elif "hour" in p_name or "minute" in p_name: val = int(round(val))
                current_strategy_params[p_name] = val
            else: logger.warning(f"Random Search: Skipping param '{p_name}' due to unexpected config: {p_config_item}"); continue
        
        entry_s_h = int(current_strategy_params.get('entry_start_hour', settings.DEFAULT_ENTRY_WINDOW_START_HOUR))
        entry_s_m = int(current_strategy_params.get('entry_start_minute', settings.DEFAULT_ENTRY_WINDOW_START_MINUTE))
        entry_e_h = int(current_strategy_params.get('entry_end_hour', settings.DEFAULT_ENTRY_WINDOW_END_HOUR))
        entry_e_m = int(current_strategy_params.get('entry_end_minute', settings.DEFAULT_ENTRY_WINDOW_END_MINUTE))

        params_for_bt = {'SL Points': float(current_strategy_params.get('sl_points', settings.DEFAULT_STOP_LOSS_POINTS)), 
                         'RRR': float(current_strategy_params.get('rrr', settings.DEFAULT_RRR)),
                         'EntryStartTime': dt_time(entry_s_h, entry_s_m), 
                         'EntryEndTime': dt_time(entry_e_h, entry_e_m)}
        try:
            perf = _run_single_backtest_for_optimization(params_for_bt, price_data, initial_capital, risk_per_trade_percent, data_interval_str)
            perf.pop('_trades_df', None); perf.pop('_equity_series', None); results.append(perf)
        except Exception as e:
            logger.error(f"Error Random Search params {params_for_bt}: {e}", exc_info=True)
            err_log_params = {k: (v.strftime("%H:%M") if isinstance(v, dt_time) else v) for k,v in params_for_bt.items()}
            err_log = {**err_log_params, 'Total P&L': np.nan, 'Error': str(e)}
            results.append(err_log)
        if progress_callback: progress_callback((i + 1) / num_iterations, "Random Search")
    return pd.DataFrame(results)

def run_walk_forward_optimization(
    full_price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    wfo_params: dict, opt_algo: str, opt_control_config: dict, 
    data_interval_str: str, progress_callback=None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    
    if full_price_data.empty or not isinstance(full_price_data.index, pd.DatetimeIndex):
        logger.error("WFO: Price data is empty or has invalid (non-DatetimeIndex) index.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), {}
    
    # Ensure full_price_data index is timezone-aware (e.g., NY or UTC)
    # This should ideally be handled by data_loader, but double-check here.
    if full_price_data.index.tz is None:
        logger.warning("WFO: full_price_data.index is timezone-naive. Attempting to localize to NY timezone.")
        try:
            full_price_data.index = full_price_data.index.tz_localize(settings.NY_TIMEZONE_STR)
        except Exception as tz_err: # Could be already localized to something else, or ambiguous times
            logger.error(f"WFO: Failed to localize full_price_data index to NY: {tz_err}. Trying UTC.", exc_info=True)
            try:
                full_price_data.index = full_price_data.index.tz_localize('UTC')
            except Exception as utc_err:
                 logger.error(f"WFO: Failed to localize full_price_data index to UTC: {utc_err}. Proceeding with naive index if error continues.", exc_info=True)
                 # If still erroring, the problem might be deeper or data is mixed.
    
    in_sample_days = int(wfo_params['in_sample_days'])
    out_of_sample_days = int(wfo_params['out_of_sample_days'])
    step_days = int(wfo_params['step_days'])
    metric_to_optimize = opt_control_config['metric_to_optimize']
    
    start_date_overall, end_date_overall = full_price_data.index.min().date(), full_price_data.index.max().date()
    total_data_duration_days = (end_date_overall - start_date_overall).days + 1
    min_required_duration = in_sample_days + out_of_sample_days
    
    if total_data_duration_days < min_required_duration:
        logger.warning(f"WFO: Total data duration ({total_data_duration_days}d) < min required ({min_required_duration}d) for IS+OOS. WFO cannot proceed.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), {}
        
    strategy_param_keys_for_opt = ['sl_points', 'rrr', 'entry_start_hour', 'entry_start_minute', 'entry_end_hour', 'entry_end_minute']
    inner_opt_param_definitions = {k: opt_control_config[k] for k in strategy_param_keys_for_opt if k in opt_control_config}
    
    logger.info(f"Starting WFO: IS={in_sample_days}d, OOS={out_of_sample_days}d, Step={step_days}d. Optimizing for {metric_to_optimize} using {opt_algo}. Interval: {data_interval_str}")
    
    all_oos_trades_list, wfo_log = [], []
    all_oos_equity_series_list = [] # To store individual OOS equity series for better chaining

    current_in_sample_start_date = start_date_overall
    fold_num = 0
    
    # Estimate total folds for progress bar
    num_possible_starts = 0; temp_start = start_date_overall
    while temp_start + timedelta(days=in_sample_days -1) <= end_date_overall:
        # Check if there's also room for at least a minimal OOS period
        if temp_start + timedelta(days=in_sample_days -1 + 1) <= end_date_overall : # +1 for OOS start
            num_possible_starts += 1
        temp_start += timedelta(days=step_days)
    total_folds_estimate = max(1, num_possible_starts)

    current_fold_capital = initial_capital # Capital at the start of each OOS period for chaining equity

    while current_in_sample_start_date + timedelta(days=in_sample_days - 1) <= end_date_overall:
        fold_num += 1
        current_in_sample_end_date = current_in_sample_start_date + timedelta(days=in_sample_days - 1)
        current_oos_start_date = current_in_sample_end_date + timedelta(days=1)
        current_oos_end_date = current_oos_start_date + timedelta(days=out_of_sample_days - 1)

        if current_oos_start_date > end_date_overall:
            logger.info(f"WFO Fold {fold_num}: OOS period starts ({current_oos_start_date}) after all data ({end_date_overall}). Ending WFO.")
            break
        if current_oos_end_date > end_date_overall:
            current_oos_end_date = end_date_overall
        
        logger.info(f"WFO Fold {fold_num}/{total_folds_estimate}: IS [{current_in_sample_start_date} - {current_in_sample_end_date}], OOS [{current_oos_start_date} - {current_oos_end_date}]")
        
        in_sample_data = full_price_data[(full_price_data.index.date >= current_in_sample_start_date) & (full_price_data.index.date <= current_in_sample_end_date)]
        out_of_sample_data = pd.DataFrame()
        if current_oos_start_date <= current_oos_end_date:
             out_of_sample_data = full_price_data[(full_price_data.index.date >= current_oos_start_date) & (full_price_data.index.date <= current_oos_end_date)]

        if in_sample_data.empty:
            logger.warning(f"WFO Fold {fold_num}: In-sample data empty for period {current_in_sample_start_date} to {current_in_sample_end_date}. Advancing.")
            current_in_sample_start_date += timedelta(days=step_days)
            if progress_callback: progress_callback(min(1.0, fold_num / total_folds_estimate), f"WFO Fold {fold_num} (IS Empty)")
            continue
            
        optimization_results_df = pd.DataFrame()
        if opt_algo == "Grid Search":
            optimization_results_df = run_grid_search(in_sample_data, initial_capital, risk_per_trade_percent, inner_opt_param_definitions, data_interval_str)
        elif opt_algo == "Random Search":
            num_iterations_wfo = opt_control_config.get('iterations', settings.DEFAULT_RANDOM_SEARCH_ITERATIONS)
            optimization_results_df = run_random_search(in_sample_data, initial_capital, risk_per_trade_percent, inner_opt_param_definitions, num_iterations_wfo, data_interval_str)
        
        best_params_for_bt = {'SL Points': settings.DEFAULT_STOP_LOSS_POINTS, 'RRR': settings.DEFAULT_RRR,
                               'EntryStartTime': dt_time(settings.DEFAULT_ENTRY_WINDOW_START_HOUR, settings.DEFAULT_ENTRY_WINDOW_START_MINUTE),
                               'EntryEndTime': dt_time(settings.DEFAULT_ENTRY_WINDOW_END_HOUR, settings.DEFAULT_ENTRY_WINDOW_END_MINUTE)}
        is_metric_val = np.nan
        
        if not optimization_results_df.empty and metric_to_optimize in optimization_results_df.columns:
            valid_opt_res = optimization_results_df.dropna(subset=[metric_to_optimize])
            if not valid_opt_res.empty:
                # Handle Max Drawdown where lower is better
                if metric_to_optimize == "Max Drawdown (%)":
                    # Max Drawdown is stored as negative percentage, so idxmin gives the 'least negative' / smallest magnitude.
                    # We want the one closest to zero, or the smallest absolute value if all are negative.
                    # If there are positive MDD values (errors), they should be ignored.
                    valid_opt_res_mdd = valid_opt_res[valid_opt_res[metric_to_optimize] <= 0] # Consider only valid MDD
                    if not valid_opt_res_mdd.empty:
                        best_row = valid_opt_res_mdd.loc[valid_opt_res_mdd[metric_to_optimize].idxmax()] # Closest to zero
                    else: # All MDD are > 0 (error) or NaN, pick first valid row as fallback
                        best_row = valid_opt_res.iloc[0]
                else: # Higher is better for other metrics
                    best_row = valid_opt_res.loc[valid_opt_res[metric_to_optimize].idxmax()]
                
                best_params_for_bt.update({
                    'SL Points': float(best_row['SL Points']), 
                    'RRR': float(best_row['RRR']),
                    'EntryStartTime': dt_time(int(best_row['EntryStartHour']), int(best_row['EntryStartMinute'])),
                    'EntryEndTime': dt_time(int(best_row['EntryEndHour']), int(best_row.get('EntryEndMinute', settings.DEFAULT_ENTRY_WINDOW_END_MINUTE)))
                })
                is_metric_val = best_row[metric_to_optimize]
        
        fold_log_entry = {'Fold': fold_num, 'InSampleStart': current_in_sample_start_date, 'InSampleEnd': current_in_sample_end_date,
                          'OutOfSampleStart': current_oos_start_date, 'OutOfSampleEnd': current_oos_end_date,
                          'BestSL': best_params_for_bt['SL Points'], 'BestRRR': best_params_for_bt['RRR'],
                          'BestEntryStart': best_params_for_bt['EntryStartTime'].strftime("%H:%M"),
                          'BestEntryEnd': best_params_for_bt['EntryEndTime'].strftime("%H:%M"),
                          'OptimizedMetric': metric_to_optimize, 'OptimizedMetricValue_InSample': is_metric_val}

        if not out_of_sample_data.empty:
            # Run OOS backtest with the capital level at the START of this OOS period
            oos_perf_dict = _run_single_backtest_for_optimization(best_params_for_bt, out_of_sample_data, current_fold_capital, risk_per_trade_percent, data_interval_str)
            
            oos_trades_df = oos_perf_dict.get('_trades_df', pd.DataFrame())
            oos_equity_s = oos_perf_dict.get('_equity_series', pd.Series(dtype=float))

            if not oos_trades_df.empty:
                all_oos_trades_list.append(oos_trades_df)
            
            if not oos_equity_s.empty:
                all_oos_equity_series_list.append(oos_equity_s)
                current_fold_capital = oos_equity_s.iloc[-1] # Update capital for the next OOS segment
            # else, capital remains unchanged for the next fold if no trades/equity change in this OOS

            for k, v in oos_perf_dict.items():
                if not k.startswith('_') and k not in ['SL Points', 'RRR', 'EntryStartHour', 'EntryStartMinute', 'EntryEndHour', 'EntryEndMinute']: # Avoid overwriting best params from IS
                    fold_log_entry[f'OOS_{k.replace(" (%)", "Pct").replace(" ", "")}'] = v
        else:
            logger.info(f"WFO Fold {fold_num}: Out-of-Sample data empty. No OOS test for this fold.")
            # Capital for next fold remains the same as the start of this one
        
        wfo_log.append(fold_log_entry)
        current_in_sample_start_date += timedelta(days=step_days)
        if progress_callback: progress_callback(min(1.0, fold_num / total_folds_estimate), f"WFO Fold {fold_num}")

    wfo_log_df = pd.DataFrame(wfo_log)
    final_oos_trades_df = pd.concat(all_oos_trades_list, ignore_index=True) if all_oos_trades_list else pd.DataFrame()

    # Chain OOS equity series
    final_plottable_equity = pd.Series(dtype=float)
    if all_oos_equity_series_list:
        # Ensure all series have DatetimeIndex
        valid_equity_series = []
        for es in all_oos_equity_series_list:
            if not isinstance(es.index, pd.DatetimeIndex):
                try: 
                    es.index = pd.to_datetime(es.index)
                    if es.index.tz is None: es.index = es.index.tz_localize('UTC') # Or NY
                except: continue # Skip if conversion fails
            if not es.empty: valid_equity_series.append(es)
        
        if valid_equity_series:
            # Create a combined index of all OOS periods to reindex and ffill
            combined_oos_index = pd.DatetimeIndex([])
            if not full_price_data.empty and wfo_log:
                 for fold_info in wfo_log:
                    oos_s_fold, oos_e_fold = fold_info.get('OutOfSampleStart'), fold_info.get('OutOfSampleEnd')
                    if oos_s_fold and oos_e_fold:
                        fold_ts_idx = full_price_data[(full_price_data.index.date >= oos_s_fold) & (full_price_data.index.date <= oos_e_fold)].index
                        if not fold_ts_idx.empty:
                            combined_oos_index = combined_oos_index.union(fold_ts_idx)
            
            if not combined_oos_index.empty:
                # Start with initial capital at the very beginning of the first OOS period's data
                start_equity_point = pd.Series([initial_capital], index=[combined_oos_index.min() - pd.Timedelta(microseconds=1)])
                
                # Concatenate all individual OOS equity series. These already reflect chained capital.
                # The important part is to get the *shape* of the curve over time.
                # We will re-base the P&L calculation for metrics later.
                temp_chained_equity = pd.concat([start_equity_point] + valid_equity_series).sort_index()
                temp_chained_equity = temp_chained_equity[~temp_chained_equity.index.duplicated(keep='last')] # Keep last point for ties

                # Reindex to the full span of OOS data and ffill
                final_plottable_equity = temp_chained_equity.reindex(combined_oos_index, method='ffill')
                
                # Ensure the first point is initial capital if it got lost or is NaN
                if not final_plottable_equity.empty and pd.isna(final_plottable_equity.iloc[0]):
                    final_plottable_equity.iloc[0] = initial_capital
                    final_plottable_equity = final_plottable_equity.ffill()
                
                # If still empty, means no OOS data at all, set a default initial capital point
                if final_plottable_equity.empty and not combined_oos_index.empty:
                     final_plottable_equity = pd.Series(initial_capital, index=[combined_oos_index.min()]).ffill()

            else: # No combined_oos_index, likely no OOS periods had data
                final_plottable_equity = pd.Series([initial_capital], index=[full_price_data.index.min() if not full_price_data.empty else pd.Timestamp.now(tz=settings.NY_TIMEZONE_STR)])

        else: # No valid equity series from OOS
            final_plottable_equity = pd.Series([initial_capital], index=[full_price_data.index.min() if not full_price_data.empty else pd.Timestamp.now(tz=settings.NY_TIMEZONE_STR)])
    else: # No OOS equity series at all
        final_plottable_equity = pd.Series([initial_capital], index=[full_price_data.index.min() if not full_price_data.empty else pd.Timestamp.now(tz=settings.NY_TIMEZONE_STR)])


    # Calculate aggregated OOS performance metrics based on the *final_oos_trades_df* and *final_plottable_equity*
    aggregated_oos_perf = {
        'Total Trades': 0, 'Total P&L': 0.0, 'Final Capital': initial_capital, 
        'Sharpe Ratio (Annualized)': np.nan, 'Sortino Ratio (Annualized)': np.nan, 
        'Max Drawdown (%)': 0.0, 'Win Rate': 0.0, 'Profit Factor': 0.0, 
        'Average Trade P&L': np.nan, 'Average Winning Trade': np.nan, 'Average Losing Trade': np.nan
    }

    if not final_oos_trades_df.empty:
        aggregated_oos_perf['Total Trades'] = len(final_oos_trades_df)
        total_pnl_from_trades = final_oos_trades_df['P&L'].sum()
        aggregated_oos_perf['Total P&L'] = total_pnl_from_trades
        
        # Final capital from equity curve if available, otherwise from trades P&L
        if not final_plottable_equity.empty and final_plottable_equity.notna().all():
            aggregated_oos_perf['Final Capital'] = final_plottable_equity.iloc[-1]
        else:
            aggregated_oos_perf['Final Capital'] = initial_capital + total_pnl_from_trades

        # Max Drawdown from the chained equity curve
        if not final_plottable_equity.empty and final_plottable_equity.notna().any():
            # Ensure MDD is calculated on a series that starts at initial_capital if not already
            equity_for_mdd = final_plottable_equity.copy()
            if equity_for_mdd.iloc[0] != initial_capital and not pd.isna(equity_for_mdd.iloc[0]):
                # This implies the equity curve might represent P&L changes rather than absolute capital.
                # For MDD, we need absolute capital levels.
                # The current `final_plottable_equity` *should* be absolute capital.
                pass

            cumulative_max_raw = equity_for_mdd.cummax()
            drawdown_raw = (equity_for_mdd - cumulative_max_raw) / cumulative_max_raw
            drawdown_raw.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero if capital hits 0
            mdd_val = drawdown_raw.min() * 100
            aggregated_oos_perf['Max Drawdown (%)'] = mdd_val if pd.notna(mdd_val) and not drawdown_raw.empty else 0.0
        
        # Sharpe and Sortino from daily returns of the chained equity curve
        daily_oos_returns_chained = _calculate_daily_returns(final_plottable_equity) # This is where the error occurred
        aggregated_oos_perf['Sharpe Ratio (Annualized)'] = calculate_sharpe_ratio(daily_oos_returns_chained)
        aggregated_oos_perf['Sortino Ratio (Annualized)'] = calculate_sortino_ratio(daily_oos_returns_chained)

        num_winning = len(final_oos_trades_df[final_oos_trades_df['P&L'] > 0])
        num_losing = len(final_oos_trades_df[final_oos_trades_df['P&L'] < 0])
        aggregated_oos_perf['Win Rate'] = (num_winning / aggregated_oos_perf['Total Trades'] * 100) if aggregated_oos_perf['Total Trades'] > 0 else 0.0
        
        if aggregated_oos_perf['Total Trades'] > 0: 
            aggregated_oos_perf['Average Trade P&L'] = final_oos_trades_df['P&L'].mean()
        if num_winning > 0: 
            aggregated_oos_perf['Average Winning Trade'] = final_oos_trades_df[final_oos_trades_df['P&L'] > 0]['P&L'].mean()
        if num_losing > 0: 
            aggregated_oos_perf['Average Losing Trade'] = final_oos_trades_df[final_oos_trades_df['P&L'] < 0]['P&L'].mean()
        
        gross_profit = final_oos_trades_df[final_oos_trades_df['P&L'] > 0]['P&L'].sum()
        gross_loss = abs(final_oos_trades_df[final_oos_trades_df['P&L'] < 0]['P&L'].sum()) # Absolute gross loss
        if gross_loss > 0:
            aggregated_oos_perf['Profit Factor'] = gross_profit / gross_loss
        elif gross_profit > 0 and gross_loss == 0: # Only winning trades
            aggregated_oos_perf['Profit Factor'] = np.inf
        else: # No profit and no loss, or only losses and no profit
            aggregated_oos_perf['Profit Factor'] = 0.0
            
    logger.info(f"WFO complete. Processed {fold_num} folds. Aggregated OOS P&L: {aggregated_oos_perf['Total P&L']:.2f}")
    return wfo_log_df, final_oos_trades_df, final_plottable_equity, aggregated_oos_perf
