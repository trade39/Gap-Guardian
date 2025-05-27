# services/optimizer.py
"""
Performs parameter optimization for trading strategies using Grid Search, Random Search,
and Walk-Forward Optimization (WFO).
Corrected handling of parameter dictionaries.
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
    if equity_series.empty: return pd.Series(dtype=float)
    daily_equity = equity_series.resample('D').last().ffill(); daily_returns = daily_equity.pct_change().fillna(0)
    return daily_returns

def calculate_sharpe_ratio(returns_series, risk_free_rate=settings.RISK_FREE_RATE, trading_days_per_year=settings.TRADING_DAYS_PER_YEAR):
    if returns_series.empty or len(returns_series) < settings.MIN_TRADES_FOR_METRICS or returns_series.std() == 0: return np.nan
    excess_returns = returns_series - (risk_free_rate / trading_days_per_year)
    sharpe = excess_returns.mean() / excess_returns.std()
    return sharpe * np.sqrt(trading_days_per_year)

def calculate_sortino_ratio(returns_series, risk_free_rate=settings.RISK_FREE_RATE, trading_days_per_year=settings.TRADING_DAYS_PER_YEAR):
    if returns_series.empty or len(returns_series) < settings.MIN_TRADES_FOR_METRICS: return np.nan
    target_return = risk_free_rate / trading_days_per_year; excess_returns = returns_series - target_return
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.empty or downside_returns.std() == 0:
        return np.inf if excess_returns.mean() > 0 else np.nan
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
    daily_ret = _calculate_daily_returns(equity_s)
    return {
        'SL Points': sl, 'RRR': rrr,
        'EntryStartHour': entry_start_time.hour, 'EntryStartMinute': entry_start_time.minute,
        'EntryEndHour': entry_end_time.hour, 'EntryEndMinute': entry_end_time.minute,
        'Total P&L': perf_metrics.get('Total P&L', 0),
        'Profit Factor': perf_metrics.get('Profit Factor', 0),
        'Win Rate': perf_metrics.get('Win Rate', 0),
        'Max Drawdown (%)': perf_metrics.get('Max Drawdown (%)', 0),
        'Total Trades': perf_metrics.get('Total Trades', 0),
        'Sharpe Ratio (Annualized)': calculate_sharpe_ratio(daily_ret),
        'Sortino Ratio (Annualized)': calculate_sortino_ratio(daily_ret),
        '_trades_df': trades_df, '_equity_series': equity_s
    }

def run_grid_search(
    price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    param_value_map: dict, # Changed name from param_value_lists for clarity
    data_interval_str: str, progress_callback=None
) -> pd.DataFrame:
    # param_value_map contains keys like 'sl_points', 'rrr', 'entry_start_hour', etc.
    # and values are lists of values to test for that parameter.
    param_names = list(param_value_map.keys())
    value_combinations = list(itertools.product(*(param_value_map[name] for name in param_names)))
    total_combinations = len(value_combinations)
    logger.info(f"Starting Grid Search. Parameters: {param_names}. Total combinations: {total_combinations}. Interval: {data_interval_str}")
    results = []

    for i, combo_values in enumerate(value_combinations):
        current_strategy_params = dict(zip(param_names, combo_values)) # These are the raw values from grid
        
        entry_start_h = int(current_strategy_params.get('entry_start_hour', settings.DEFAULT_ENTRY_WINDOW_START_HOUR))
        entry_start_m = int(current_strategy_params.get('entry_start_minute', settings.DEFAULT_ENTRY_WINDOW_START_MINUTE))
        entry_end_h = int(current_strategy_params.get('entry_end_hour', settings.DEFAULT_ENTRY_WINDOW_END_HOUR))
        entry_end_m = int(current_strategy_params.get('entry_end_minute', settings.DEFAULT_ENTRY_WINDOW_END_MINUTE))

        params_for_backtest = {
            'SL Points': current_strategy_params['sl_points'], 'RRR': current_strategy_params['rrr'],
            'EntryStartTime': dt_time(entry_start_h, entry_start_m),
            'EntryEndTime': dt_time(entry_end_h, entry_end_m)
        }
        try:
            perf = _run_single_backtest_for_optimization(params_for_backtest, price_data, initial_capital, risk_per_trade_percent, data_interval_str)
            perf.pop('_trades_df', None); perf.pop('_equity_series', None)
            results.append(perf)
        except Exception as e:
            logger.error(f"Error during Grid Search for params {params_for_backtest}: {e}", exc_info=True)
            error_log_params = {**params_for_backtest, 'Total P&L': np.nan} # Log with all attempted params
            error_log_params['EntryStartTime'] = error_log_params['EntryStartTime'].strftime("%H:%M")
            error_log_params['EntryEndTime'] = error_log_params['EntryEndTime'].strftime("%H:%M")
            results.append(error_log_params)
        if progress_callback: progress_callback((i + 1) / total_combinations, "Grid Search")
    return pd.DataFrame(results)

def run_random_search(
    price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    param_config_map: dict, # Changed name from param_ranges. Keys are param names.
                           # Values are tuples (min,max) for continuous or lists for discrete.
    num_iterations: int, data_interval_str: str, progress_callback=None
) -> pd.DataFrame:
    logger.info(f"Starting Random Search. Iterations: {num_iterations}. Interval: {data_interval_str}")
    results = []
    for i in range(num_iterations):
        current_strategy_params = {} # Store raw sampled values
        for p_name, p_config_item in param_config_map.items():
            if isinstance(p_config_item, list): # Discrete values
                current_strategy_params[p_name] = random.choice(p_config_item)
            elif isinstance(p_config_item, tuple) and len(p_config_item) == 2: # Continuous range (min, max)
                p_min, p_max = p_config_item
                val = random.uniform(p_min, p_max)
                if "points" in p_name or "rrr" in p_name: val = round(val, 2)
                elif "hour" in p_name or "minute" in p_name: val = int(round(val)) # Ensure hours/minutes are integers
                current_strategy_params[p_name] = val
            else:
                logger.warning(f"Random Search: Skipping parameter '{p_name}' due to unexpected config: {p_config_item}")
                continue # Skip this parameter if config is not list or 2-tuple
        
        entry_start_h = int(current_strategy_params.get('entry_start_hour', settings.DEFAULT_ENTRY_WINDOW_START_HOUR))
        entry_start_m = int(current_strategy_params.get('entry_start_minute', settings.DEFAULT_ENTRY_WINDOW_START_MINUTE))
        entry_end_h = int(current_strategy_params.get('entry_end_hour', settings.DEFAULT_ENTRY_WINDOW_END_HOUR))
        entry_end_m = int(current_strategy_params.get('entry_end_minute', settings.DEFAULT_ENTRY_WINDOW_END_MINUTE))

        params_for_backtest = {
            'SL Points': current_strategy_params['sl_points'], 'RRR': current_strategy_params['rrr'],
            'EntryStartTime': dt_time(entry_start_h, entry_start_m),
            'EntryEndTime': dt_time(entry_end_h, entry_end_m)
        }
        try:
            perf = _run_single_backtest_for_optimization(params_for_backtest, price_data, initial_capital, risk_per_trade_percent, data_interval_str)
            perf.pop('_trades_df', None); perf.pop('_equity_series', None)
            results.append(perf)
        except Exception as e:
            logger.error(f"Error during Random Search for params {params_for_backtest}: {e}", exc_info=True)
            error_log_params = {**params_for_backtest, 'Total P&L': np.nan}
            error_log_params['EntryStartTime'] = error_log_params['EntryStartTime'].strftime("%H:%M")
            error_log_params['EntryEndTime'] = error_log_params['EntryEndTime'].strftime("%H:%M")
            results.append(error_log_params)
        if progress_callback: progress_callback((i + 1) / num_iterations, "Random Search")
    return pd.DataFrame(results)

def run_walk_forward_optimization(
    full_price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    wfo_params: dict, opt_algo: str, 
    # opt_params_config now contains only strategy param definitions (ranges/values)
    # and other control params like 'metric_to_optimize', 'iterations'
    opt_control_config: dict, 
    data_interval_str: str, progress_callback=None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    in_sample_days, out_of_sample_days, step_days = wfo_params['in_sample_days'], wfo_params['out_of_sample_days'], wfo_params['step_days']
    metric_to_optimize = opt_control_config['metric_to_optimize']
    
    # Extract strategy parameter configurations for the inner optimizer
    # These are the keys that hold lists of values (for Grid) or (min,max) tuples/lists (for Random)
    strategy_param_keys_for_opt = ['sl_points', 'rrr', 'entry_start_hour', 'entry_start_minute', 'entry_end_hour', 'entry_end_minute']
    
    # This will be passed to run_grid_search or run_random_search
    inner_opt_param_definitions = {
        k: opt_control_config[k] for k in strategy_param_keys_for_opt if k in opt_control_config
    }

    logger.info(f"Starting WFO: In-sample={in_sample_days}d, OOS={out_of_sample_days}d, Step={step_days}d. Optimizing for {metric_to_optimize} using {opt_algo}. Interval: {data_interval_str}")
    # ... (rest of WFO logic as before, but using the correctly structured inner_opt_param_definitions) ...
    all_oos_trades, all_oos_equity_segments, wfo_log = [], [], []
    start_date_overall, end_date_overall = full_price_data.index.min().date(), full_price_data.index.max().date()
    current_in_sample_start_date = start_date_overall; fold_num = 0
    total_oos_periods_estimate = max(1, (end_date_overall - (start_date_overall + timedelta(days=in_sample_days))).days // step_days)

    while True:
        fold_num += 1
        current_in_sample_end_date = current_in_sample_start_date + timedelta(days=in_sample_days - 1)
        current_oos_start_date = current_in_sample_end_date + timedelta(days=1)
        current_oos_end_date = current_oos_start_date + timedelta(days=out_of_sample_days - 1)
        logger.info(f"WFO Fold {fold_num}: IS [{current_in_sample_start_date} - {current_in_sample_end_date}], OOS [{current_oos_start_date} - {current_oos_end_date}]")

        if current_in_sample_end_date > end_date_overall or current_oos_start_date > end_date_overall: break
        if current_oos_end_date > end_date_overall: current_oos_end_date = end_date_overall
        if current_oos_start_date > current_oos_end_date: break

        in_sample_data = full_price_data[(full_price_data.index.date >= current_in_sample_start_date) & (full_price_data.index.date <= current_in_sample_end_date)]
        out_of_sample_data = full_price_data[(full_price_data.index.date >= current_oos_start_date) & (full_price_data.index.date <= current_oos_end_date)]

        if in_sample_data.empty:
            logger.warning(f"WFO Fold {fold_num}: In-sample data empty. Advancing.")
            current_in_sample_start_date += timedelta(days=step_days)
            if progress_callback: progress_callback(fold_num / total_oos_periods_estimate, f"WFO Fold {fold_num} (Skipped IS)")
            continue
        
        optimization_results_df = pd.DataFrame()
        if opt_algo == "Grid Search":
            optimization_results_df = run_grid_search(in_sample_data, initial_capital, risk_per_trade_percent, inner_opt_param_definitions, data_interval_str)
        elif opt_algo == "Random Search":
            num_iterations_wfo = opt_control_config.get('iterations', settings.DEFAULT_RANDOM_SEARCH_ITERATIONS)
            optimization_results_df = run_random_search(in_sample_data, initial_capital, risk_per_trade_percent, inner_opt_param_definitions, num_iterations_wfo, data_interval_str)

        best_params_for_backtest = {
            'SL Points': settings.DEFAULT_STOP_LOSS_POINTS, 'RRR': settings.DEFAULT_RRR,
            'EntryStartTime': dt_time(settings.DEFAULT_ENTRY_WINDOW_START_HOUR, settings.DEFAULT_ENTRY_WINDOW_START_MINUTE),
            'EntryEndTime': dt_time(settings.DEFAULT_ENTRY_WINDOW_END_HOUR, settings.DEFAULT_ENTRY_WINDOW_END_MINUTE)
        }
        is_metric_value = np.nan

        if not optimization_results_df.empty and metric_to_optimize in optimization_results_df.columns:
            valid_opt_res = optimization_results_df.dropna(subset=[metric_to_optimize])
            if not valid_opt_res.empty:
                best_row = valid_opt_res.loc[valid_opt_res[metric_to_optimize].idxmin()] if metric_to_optimize == "Max Drawdown (%)" else valid_opt_res.loc[valid_opt_res[metric_to_optimize].idxmax()]
                best_params_for_backtest.update({
                    'SL Points': best_row['SL Points'], 'RRR': best_row['RRR'],
                    'EntryStartTime': dt_time(int(best_row['EntryStartHour']), int(best_row['EntryStartMinute'])),
                    'EntryEndTime': dt_time(int(best_row['EntryEndHour']), int(best_row.get('EntryEndMinute', settings.DEFAULT_ENTRY_WINDOW_END_MINUTE)))
                })
                is_metric_value = best_row[metric_to_optimize]
                logger.info(f"WFO Fold {fold_num}: Best IS Params -> SL:{best_row['SL Points']:.2f}, RRR:{best_row['RRR']:.1f}, Start:{best_params_for_backtest['EntryStartTime']:%H:%M}, End:{best_params_for_backtest['EntryEndTime']:%H:%M}")
            else: logger.warning(f"WFO Fold {fold_num}: All IS opt results for '{metric_to_optimize}' were NaN. Using defaults.")
        else: logger.warning(f"WFO Fold {fold_num}: IS optimization yielded no results or missing metric '{metric_to_optimize}'. Using defaults.")
        
        fold_log_entry = {'Fold': fold_num, 'InSampleStart': current_in_sample_start_date, 'InSampleEnd': current_in_sample_end_date,
                          'OutOfSampleStart': current_oos_start_date, 'OutOfSampleEnd': current_oos_end_date,
                          'BestSL': best_params_for_backtest['SL Points'], 'BestRRR': best_params_for_backtest['RRR'],
                          'BestEntryStart': best_params_for_backtest['EntryStartTime'].strftime("%H:%M"),
                          'BestEntryEnd': best_params_for_backtest['EntryEndTime'].strftime("%H:%M"),
                          'OptimizedMetric': metric_to_optimize, 'OptimizedMetricValue_InSample': is_metric_value}

        if not out_of_sample_data.empty:
            oos_perf_dict = _run_single_backtest_for_optimization(best_params_for_backtest, out_of_sample_data, initial_capital, risk_per_trade_percent, data_interval_str)
            if not oos_perf_dict['_trades_df'].empty: all_oos_trades.append(oos_perf_dict['_trades_df'])
            if not oos_perf_dict['_equity_series'].empty: all_oos_equity_segments.append(oos_perf_dict['_equity_series'])
            for k, v in oos_perf_dict.items():
                if not k.startswith('_'): fold_log_entry[f'OOS_{k.replace(" (%)", "Pct").replace(" ", "")}'] = v
        wfo_log.append(fold_log_entry)
        current_in_sample_start_date += timedelta(days=step_days)
        if progress_callback: progress_callback(fold_num / total_oos_periods_estimate, f"WFO Fold {fold_num}")

    wfo_log_df = pd.DataFrame(wfo_log)
    final_oos_trades_df = pd.concat(all_oos_trades, ignore_index=True) if all_oos_trades else pd.DataFrame()
    # ... (Chained equity and aggregated OOS performance calculation as before) ...
    chained_oos_equity = pd.Series(dtype=float); current_chained_capital = initial_capital
    if not final_oos_trades_df.empty:
        final_oos_trades_df_sorted = final_oos_trades_df.sort_values(by='EntryTime').reset_index(drop=True)
        # Initialize temp_equity_dates robustly
        if not final_oos_trades_df_sorted.empty and not final_oos_trades_df_sorted['EntryTime'].dropna().empty:
             start_equity_date = final_oos_trades_df_sorted['EntryTime'].dropna().min() - pd.Timedelta(microseconds=1)
        elif not full_price_data.empty :
             start_equity_date = full_price_data.index.min() - pd.Timedelta(microseconds=1)
        else: # Absolute fallback
             start_equity_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1, microseconds=1)

        temp_equity_values = [initial_capital]; temp_equity_dates = [start_equity_date]

        for _, trade in final_oos_trades_df_sorted.iterrows():
            current_chained_capital += trade['P&L']; temp_equity_values.append(current_chained_capital)
            temp_equity_dates.append(trade['ExitTime'])
        if len(temp_equity_dates) > 0: # Check if any equity points were actually added
            chained_oos_equity = pd.Series(temp_equity_values, index=pd.to_datetime(temp_equity_dates)).sort_index()
            if not chained_oos_equity.empty:
                min_oos_date_overall = min((d['OutOfSampleStart'] for d in wfo_log if d.get('OutOfSampleStart')), default=chained_oos_equity.index.min())
                max_oos_date_overall = max((d['OutOfSampleEnd'] for d in wfo_log if d.get('OutOfSampleEnd')), default=chained_oos_equity.index.max())
                if pd.NaT not in [min_oos_date_overall, max_oos_date_overall] and min_oos_date_overall <= max_oos_date_overall:
                    oos_full_date_range = pd.date_range(start=min_oos_date_overall, end=max_oos_date_overall, freq='B')
                    if not oos_full_date_range.empty:
                        # Ensure chained_oos_equity starts at or before oos_full_date_range.min()
                        combined_index = chained_oos_equity.index.union(oos_full_date_range)
                        chained_oos_equity = chained_oos_equity.reindex(combined_index).ffill()
                        chained_oos_equity = chained_oos_equity.loc[oos_full_date_range.min():oos_full_date_range.max()]


    elif not full_price_data.empty: # No OOS trades, but price data exists
        chained_oos_equity = pd.Series([initial_capital], index=[full_price_data.index.min()])
    else: # No OOS trades and no price data (edge case)
        chained_oos_equity = pd.Series([initial_capital], index=[pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1)])


    aggregated_oos_performance = {}
    if not final_oos_trades_df.empty and not chained_oos_equity.empty and chained_oos_equity.notna().any():
        aggregated_oos_performance['Total Trades'] = len(final_oos_trades_df)
        aggregated_oos_performance['Total P&L'] = final_oos_trades_df['P&L'].sum()
        aggregated_oos_performance['Final Capital'] = chained_oos_equity.iloc[-1] if not chained_oos_equity.empty else initial_capital
        daily_oos_returns_chained = _calculate_daily_returns(chained_oos_equity)
        aggregated_oos_performance['Sharpe Ratio (Annualized)'] = calculate_sharpe_ratio(daily_oos_returns_chained)
        aggregated_oos_performance['Sortino Ratio (Annualized)'] = calculate_sortino_ratio(daily_oos_returns_chained)
        cumulative_max = chained_oos_equity.cummax(); drawdown = (chained_oos_equity - cumulative_max) / cumulative_max
        aggregated_oos_performance['Max Drawdown (%)'] = drawdown.min() * 100 if drawdown.notna().any() and not drawdown.empty else 0
        num_winning = len(final_oos_trades_df[final_oos_trades_df['P&L'] > 0])
        aggregated_oos_performance['Win Rate'] = (num_winning / len(final_oos_trades_df) * 100) if len(final_oos_trades_df) > 0 else 0
        gross_profit = final_oos_trades_df[final_oos_trades_df['P&L'] > 0]['P&L'].sum()
        gross_loss = final_oos_trades_df[final_oos_trades_df['P&L'] < 0]['P&L'].sum()
        aggregated_oos_performance['Profit Factor'] = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf if gross_profit > 0 else 0

    logger.info(f"WFO complete. Processed {fold_num-1 if fold_num > 0 else 0} folds.")
    return wfo_log_df, final_oos_trades_df, chained_oos_equity, aggregated_oos_performance

