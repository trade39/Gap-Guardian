# services/optimizer.py
"""
Performs parameter optimization for trading strategies using Grid Search, Random Search,
and Walk-Forward Optimization (WFO).
Corrected Max Drawdown calculation and WFO equity curve generation.
Fixed DatetimeIndex.union_many error.
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
    if equity_series.empty or equity_series.nunique() <= 1:
        return pd.Series(dtype=float)
    daily_equity = equity_series.resample('D').last()
    if not daily_equity.empty:
        first_valid_idx = daily_equity.first_valid_index()
        last_valid_idx = daily_equity.last_valid_index()
        if first_valid_idx is not None and last_valid_idx is not None:
            daily_equity = daily_equity.loc[first_valid_idx:last_valid_idx].ffill()
    daily_returns = daily_equity.pct_change().fillna(0)
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
    result = {
        'SL Points': sl, 'RRR': rrr,
        'EntryStartHour': entry_start_time.hour, 'EntryStartMinute': entry_start_time.minute,
        'EntryEndHour': entry_end_time.hour, 'EntryEndMinute': entry_end_time.minute,
        'Total P&L': perf_metrics.get('Total P&L', np.nan),
        'Profit Factor': perf_metrics.get('Profit Factor', np.nan),
        'Win Rate': perf_metrics.get('Win Rate', np.nan),
        'Max Drawdown (%)': perf_metrics.get('Max Drawdown (%)', np.nan),
        'Total Trades': perf_metrics.get('Total Trades', 0),
        'Sharpe Ratio (Annualized)': calculate_sharpe_ratio(daily_ret),
        'Sortino Ratio (Annualized)': calculate_sortino_ratio(daily_ret),
        'Average Trade P&L': perf_metrics.get('Average Trade P&L', np.nan),
        'Average Winning Trade': perf_metrics.get('Average Winning Trade', np.nan),
        'Average Losing Trade': perf_metrics.get('Average Losing Trade', np.nan),
        '_trades_df': trades_df, '_equity_series': equity_s
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
        entry_s_h = int(current_strategy_params.get('entry_start_hour', settings.DEFAULT_ENTRY_WINDOW_START_HOUR))
        entry_s_m = int(current_strategy_params.get('entry_start_minute', settings.DEFAULT_ENTRY_WINDOW_START_MINUTE))
        entry_e_h = int(current_strategy_params.get('entry_end_hour', settings.DEFAULT_ENTRY_WINDOW_END_HOUR))
        entry_e_m = int(current_strategy_params.get('entry_end_minute', settings.DEFAULT_ENTRY_WINDOW_END_MINUTE))
        params_for_bt = {'SL Points': current_strategy_params['sl_points'], 'RRR': current_strategy_params['rrr'],
                         'EntryStartTime': dt_time(entry_s_h, entry_s_m), 'EntryEndTime': dt_time(entry_e_h, entry_e_m)}
        try:
            perf = _run_single_backtest_for_optimization(params_for_bt, price_data, initial_capital, risk_per_trade_percent, data_interval_str)
            perf.pop('_trades_df', None); perf.pop('_equity_series', None); results.append(perf)
        except Exception as e:
            logger.error(f"Error Grid Search params {params_for_bt}: {e}", exc_info=True)
            err_log = {**params_for_bt, 'Total P&L': np.nan}; err_log['EntryStartTime'] = err_log['EntryStartTime'].strftime("%H:%M"); err_log['EntryEndTime'] = err_log['EntryEndTime'].strftime("%H:%M"); results.append(err_log)
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
                p_min, p_max = p_config_item; val = random.uniform(p_min, p_max)
                if "points" in p_name or "rrr" in p_name: val = round(val, 2)
                elif "hour" in p_name or "minute" in p_name: val = int(round(val))
                current_strategy_params[p_name] = val
            else: logger.warning(f"Random Search: Skipping param '{p_name}' due to config: {p_config_item}"); continue
        entry_s_h = int(current_strategy_params.get('entry_start_hour', settings.DEFAULT_ENTRY_WINDOW_START_HOUR))
        entry_s_m = int(current_strategy_params.get('entry_start_minute', settings.DEFAULT_ENTRY_WINDOW_START_MINUTE))
        entry_e_h = int(current_strategy_params.get('entry_end_hour', settings.DEFAULT_ENTRY_WINDOW_END_HOUR))
        entry_e_m = int(current_strategy_params.get('entry_end_minute', settings.DEFAULT_ENTRY_WINDOW_END_MINUTE))
        params_for_bt = {'SL Points': current_strategy_params['sl_points'], 'RRR': current_strategy_params['rrr'],
                         'EntryStartTime': dt_time(entry_s_h, entry_s_m), 'EntryEndTime': dt_time(entry_e_h, entry_e_m)}
        try:
            perf = _run_single_backtest_for_optimization(params_for_bt, price_data, initial_capital, risk_per_trade_percent, data_interval_str)
            perf.pop('_trades_df', None); perf.pop('_equity_series', None); results.append(perf)
        except Exception as e:
            logger.error(f"Error Random Search params {params_for_bt}: {e}", exc_info=True)
            err_log = {**params_for_bt, 'Total P&L': np.nan}; err_log['EntryStartTime'] = err_log['EntryStartTime'].strftime("%H:%M"); err_log['EntryEndTime'] = err_log['EntryEndTime'].strftime("%H:%M"); results.append(err_log)
        if progress_callback: progress_callback((i + 1) / num_iterations, "Random Search")
    return pd.DataFrame(results)

def run_walk_forward_optimization(
    full_price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    wfo_params: dict, opt_algo: str, opt_control_config: dict, 
    data_interval_str: str, progress_callback=None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    if full_price_data.empty or not isinstance(full_price_data.index, pd.DatetimeIndex):
        logger.error("WFO: Price data is empty or has invalid index.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), {}
    in_sample_days, out_of_sample_days, step_days = wfo_params['in_sample_days'], wfo_params['out_of_sample_days'], wfo_params['step_days']
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
    current_in_sample_start_date = start_date_overall; fold_num = 0
    num_possible_starts = 0; temp_start = start_date_overall
    while temp_start + timedelta(days=in_sample_days - 1) <= end_date_overall:
        if temp_start + timedelta(days=in_sample_days - 1 + 1) <= end_date_overall : num_possible_starts += 1
        temp_start += timedelta(days=step_days)
    total_folds_estimate = max(1, num_possible_starts)

    while current_in_sample_start_date + timedelta(days=in_sample_days - 1) <= end_date_overall:
        fold_num += 1
        current_in_sample_end_date = current_in_sample_start_date + timedelta(days=in_sample_days - 1)
        current_oos_start_date = current_in_sample_end_date + timedelta(days=1)
        current_oos_end_date = current_oos_start_date + timedelta(days=out_of_sample_days - 1)
        if current_oos_start_date > end_date_overall: logger.info(f"WFO Fold {fold_num}: OOS period starts ({current_oos_start_date}) after all data ({end_date_overall}). Ending WFO."); break
        if current_oos_end_date > end_date_overall: current_oos_end_date = end_date_overall
        logger.info(f"WFO Fold {fold_num}/{total_folds_estimate}: IS [{current_in_sample_start_date} - {current_in_sample_end_date}], OOS [{current_oos_start_date} - {current_oos_end_date}]")
        in_sample_data = full_price_data[(full_price_data.index.date >= current_in_sample_start_date) & (full_price_data.index.date <= current_in_sample_end_date)]
        out_of_sample_data = pd.DataFrame()
        if current_oos_start_date <= current_oos_end_date: out_of_sample_data = full_price_data[(full_price_data.index.date >= current_oos_start_date) & (full_price_data.index.date <= current_oos_end_date)]
        if in_sample_data.empty: logger.warning(f"WFO Fold {fold_num}: In-sample data empty. Advancing.")
        else:
            optimization_results_df = pd.DataFrame()
            if opt_algo == "Grid Search": optimization_results_df = run_grid_search(in_sample_data, initial_capital, risk_per_trade_percent, inner_opt_param_definitions, data_interval_str)
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
                    best_row = valid_opt_res.loc[valid_opt_res[metric_to_optimize].idxmin()] if metric_to_optimize == "Max Drawdown (%)" else valid_opt_res.loc[valid_opt_res[metric_to_optimize].idxmax()]
                    best_params_for_bt.update({'SL Points': best_row['SL Points'], 'RRR': best_row['RRR'],
                                               'EntryStartTime': dt_time(int(best_row['EntryStartHour']), int(best_row['EntryStartMinute'])),
                                               'EntryEndTime': dt_time(int(best_row['EntryEndHour']), int(best_row.get('EntryEndMinute', settings.DEFAULT_ENTRY_WINDOW_END_MINUTE)))})
                    is_metric_val = best_row[metric_to_optimize]
            fold_log_entry = {'Fold': fold_num, 'InSampleStart': current_in_sample_start_date, 'InSampleEnd': current_in_sample_end_date,
                              'OutOfSampleStart': current_oos_start_date, 'OutOfSampleEnd': current_oos_end_date,
                              'BestSL': best_params_for_bt['SL Points'], 'BestRRR': best_params_for_bt['RRR'],
                              'BestEntryStart': best_params_for_bt['EntryStartTime'].strftime("%H:%M"),
                              'BestEntryEnd': best_params_for_bt['EntryEndTime'].strftime("%H:%M"),
                              'OptimizedMetric': metric_to_optimize, 'OptimizedMetricValue_InSample': is_metric_val}
            if not out_of_sample_data.empty:
                oos_perf = _run_single_backtest_for_optimization(best_params_for_bt, out_of_sample_data, initial_capital, risk_per_trade_percent, data_interval_str)
                if not oos_perf['_trades_df'].empty: all_oos_trades_list.append(oos_perf['_trades_df'])
                for k, v in oos_perf.items():
                    if not k.startswith('_'): fold_log_entry[f'OOS_{k.replace(" (%)", "Pct").replace(" ", "")}'] = v
            else: logger.info(f"WFO Fold {fold_num}: OOS data empty. No OOS test.")
            wfo_log.append(fold_log_entry)
        current_in_sample_start_date += timedelta(days=step_days)
        if progress_callback: progress_callback(min(1.0, fold_num / total_folds_estimate), f"WFO Fold {fold_num}")

    wfo_log_df = pd.DataFrame(wfo_log)
    final_oos_trades_df = pd.concat(all_oos_trades_list, ignore_index=True) if all_oos_trades_list else pd.DataFrame()
    
    chained_oos_equity_points = {}
    current_chained_capital = initial_capital
    first_oos_period_start_date = min((f['OutOfSampleStart'] for f in wfo_log if f.get('OutOfSampleStart')), default=None)
    first_oos_timestamp = full_price_data.index.min() # Default if no OOS or no price data
    if first_oos_period_start_date and not full_price_data.empty:
        candidate_ts = full_price_data[full_price_data.index.date >= first_oos_period_start_date].index.min()
        if not pd.isna(candidate_ts): first_oos_timestamp = candidate_ts
    elif not full_price_data.empty: first_oos_timestamp = full_price_data.index.min()
    else: first_oos_timestamp = pd.Timestamp.now(tz=settings.NY_TIMEZONE_STR) - timedelta(days=1)
    if pd.isna(first_oos_timestamp): first_oos_timestamp = pd.Timestamp.now(tz=settings.NY_TIMEZONE_STR) - timedelta(days=1)
    
    chained_oos_equity_points[first_oos_timestamp - pd.Timedelta(microseconds=1)] = initial_capital

    if not final_oos_trades_df.empty:
        final_oos_trades_df_sorted_by_exit = final_oos_trades_df.sort_values(by='ExitTime').reset_index(drop=True)
        for _, trade in final_oos_trades_df_sorted_by_exit.iterrows():
            current_chained_capital += trade['P&L']
            chained_oos_equity_points[trade['ExitTime']] = current_chained_capital
    
    raw_chained_equity_series = pd.Series(dtype=float)
    if chained_oos_equity_points:
        raw_chained_equity_series = pd.Series(chained_oos_equity_points).sort_index()
    elif not full_price_data.empty: # No trades but data exists
        raw_chained_equity_series = pd.Series([initial_capital], index=[full_price_data.index.min()])
    else: # No trades, no data
        raw_chained_equity_series = pd.Series([initial_capital], index=[first_oos_timestamp])


    final_plottable_equity = raw_chained_equity_series # This is the point-in-time equity
    if not raw_chained_equity_series.empty and not full_price_data.empty and wfo_log:
        all_oos_fold_timestamps = pd.DatetimeIndex([])
        for fold_info in wfo_log:
            oos_s_fold = fold_info.get('OutOfSampleStart')
            oos_e_fold = fold_info.get('OutOfSampleEnd')
            if oos_s_fold and oos_e_fold:
                fold_ts = full_price_data[
                    (full_price_data.index.date >= oos_s_fold) &
                    (full_price_data.index.date <= oos_e_fold)
                ].index
                if not fold_ts.empty:
                     all_oos_fold_timestamps = all_oos_fold_timestamps.union(fold_ts)
        
        if not all_oos_fold_timestamps.empty:
            # Ensure the raw series covers the start of the first OOS timestamp
            if raw_chained_equity_series.index.min() > all_oos_fold_timestamps.min():
                 start_equity_point = pd.Series([initial_capital], index=[all_oos_fold_timestamps.min() - pd.Timedelta(microseconds=1)])
                 raw_chained_equity_series = pd.concat([start_equity_point, raw_chained_equity_series]).sort_index()

            final_plottable_equity = raw_chained_equity_series.reindex(
                raw_chained_equity_series.index.union(all_oos_fold_timestamps)
            ).ffill()
            # Filter to the actual range of OOS timestamps
            final_plottable_equity = final_plottable_equity.loc[all_oos_fold_timestamps.min():all_oos_fold_timestamps.max()]
            # Ensure first point is initial_capital if it got lost in reindexing
            if not final_plottable_equity.empty and pd.isna(final_plottable_equity.iloc[0]):
                final_plottable_equity.iloc[0] = initial_capital
                final_plottable_equity = final_plottable_equity.ffill()

    aggregated_oos_perf = {
        'Total Trades': 0, 'Total P&L': np.nan, 'Final Capital': initial_capital,
        'Sharpe Ratio (Annualized)': np.nan, 'Sortino Ratio (Annualized)': np.nan,
        'Max Drawdown (%)': 0.0, 'Win Rate': 0.0, 'Profit Factor': 0.0,
        'Average Trade P&L': np.nan, 'Average Winning Trade': np.nan, 'Average Losing Trade': np.nan
    }
    if not final_oos_trades_df.empty:
        aggregated_oos_perf['Total Trades'] = len(final_oos_trades_df)
        aggregated_oos_perf['Total P&L'] = final_oos_trades_df['P&L'].sum()
        
        if not final_plottable_equity.empty and final_plottable_equity.notna().any():
            aggregated_oos_perf['Final Capital'] = final_plottable_equity.iloc[-1]
            # MDD Calculation from the final_plottable_equity (point-in-time)
            cumulative_max_raw = final_plottable_equity.cummax()
            drawdown_raw = (final_plottable_equity - cumulative_max_raw) / cumulative_max_raw
            drawdown_raw.replace([np.inf, -np.inf], np.nan, inplace=True) 
            aggregated_oos_perf['Max Drawdown (%)'] = drawdown_raw.min() * 100 if drawdown_raw.notna().any() and not drawdown_raw.empty else 0.0
            
            daily_oos_returns_chained = _calculate_daily_returns(final_plottable_equity) # Use final_plottable_equity for daily returns
            aggregated_oos_perf['Sharpe Ratio (Annualized)'] = calculate_sharpe_ratio(daily_oos_returns_chained)
            aggregated_oos_perf['Sortino Ratio (Annualized)'] = calculate_sortino_ratio(daily_oos_returns_chained)
        else: 
            aggregated_oos_perf['Final Capital'] = initial_capital + aggregated_oos_perf['Total P&L']

        num_winning = len(final_oos_trades_df[final_oos_trades_df['P&L'] > 0])
        num_losing = len(final_oos_trades_df[final_oos_trades_df['P&L'] < 0])
        aggregated_oos_perf['Win Rate'] = (num_winning / aggregated_oos_perf['Total Trades'] * 100) if aggregated_oos_perf['Total Trades'] > 0 else 0.0
        if aggregated_oos_perf['Total Trades'] > 0: aggregated_oos_perf['Average Trade P&L'] = final_oos_trades_df['P&L'].mean()
        if num_winning > 0: aggregated_oos_perf['Average Winning Trade'] = final_oos_trades_df[final_oos_trades_df['P&L'] > 0]['P&L'].mean()
        if num_losing > 0: aggregated_oos_perf['Average Losing Trade'] = final_oos_trades_df[final_oos_trades_df['P&L'] < 0]['P&L'].mean()
        gp = final_oos_trades_df[final_oos_trades_df['P&L'] > 0]['P&L'].sum(); gl = final_oos_trades_df[final_oos_trades_df['P&L'] < 0]['P&L'].sum()
        aggregated_oos_perf['Profit Factor'] = abs(gp / gl) if gl != 0 else np.inf if gp > 0 else 0.0
        
    logger.info(f"WFO complete. Processed {fold_num} folds.")
    return wfo_log_df, final_oos_trades_df, final_plottable_equity, aggregated_oos_perf

