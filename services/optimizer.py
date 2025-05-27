# services/optimizer.py
"""
Performs parameter optimization for trading strategies using Grid Search, Random Search,
and Walk-Forward Optimization (WFO).
Corrected handling of parameter dictionaries and WFO loop logic.
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
    if equity_series.empty or equity_series.nunique() <= 1: # Handle flat or single-point equity
        return pd.Series(dtype=float)
    daily_equity = equity_series.resample('D').last().ffill()
    daily_returns = daily_equity.pct_change().fillna(0)
    return daily_returns

def calculate_sharpe_ratio(returns_series, risk_free_rate=settings.RISK_FREE_RATE, trading_days_per_year=settings.TRADING_DAYS_PER_YEAR):
    if returns_series.empty or len(returns_series) < max(2, settings.MIN_TRADES_FOR_METRICS) or returns_series.std() == 0: # Need at least 2 points for std
        return np.nan
    excess_returns = returns_series - (risk_free_rate / trading_days_per_year)
    sharpe = excess_returns.mean() / excess_returns.std()
    return sharpe * np.sqrt(trading_days_per_year)

def calculate_sortino_ratio(returns_series, risk_free_rate=settings.RISK_FREE_RATE, trading_days_per_year=settings.TRADING_DAYS_PER_YEAR):
    if returns_series.empty or len(returns_series) < max(2, settings.MIN_TRADES_FOR_METRICS):
        return np.nan
    target_return = risk_free_rate / trading_days_per_year; excess_returns = returns_series - target_return
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.empty or downside_returns.std() == 0:
        return np.inf if excess_returns.mean() > 0 else np.nan # If no downside, but positive mean -> inf
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
    
    # Ensure all keys are present even if metrics are NaN
    result = {
        'SL Points': sl, 'RRR': rrr,
        'EntryStartHour': entry_start_time.hour, 'EntryStartMinute': entry_start_time.minute,
        'EntryEndHour': entry_end_time.hour, 'EntryEndMinute': entry_end_time.minute,
        'Total P&L': perf_metrics.get('Total P&L', np.nan), # Use np.nan for missing numeric metrics
        'Profit Factor': perf_metrics.get('Profit Factor', np.nan),
        'Win Rate': perf_metrics.get('Win Rate', np.nan),
        'Max Drawdown (%)': perf_metrics.get('Max Drawdown (%)', np.nan),
        'Total Trades': perf_metrics.get('Total Trades', 0), # Trades can be 0
        'Sharpe Ratio (Annualized)': calculate_sharpe_ratio(daily_ret),
        'Sortino Ratio (Annualized)': calculate_sortino_ratio(daily_ret),
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
    if total_combinations == 0: return pd.DataFrame() # No combinations to test
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
    all_oos_trades, all_oos_equity_segments, wfo_log = [], [], []
    current_in_sample_start_date = start_date_overall; fold_num = 0
    
    # Estimate total folds for progress bar
    num_possible_starts = 0
    temp_start = start_date_overall
    while temp_start + timedelta(days=in_sample_days - 1) <= end_date_overall:
        # Check if an OOS period can also be formed
        if temp_start + timedelta(days=in_sample_days - 1 + 1) <= end_date_overall : # At least 1 day for OOS start
             num_possible_starts += 1
        temp_start += timedelta(days=step_days)
    total_folds_estimate = max(1, num_possible_starts)


    while current_in_sample_start_date + timedelta(days=in_sample_days - 1) <= end_date_overall:
        fold_num += 1
        current_in_sample_end_date = current_in_sample_start_date + timedelta(days=in_sample_days - 1)
        current_oos_start_date = current_in_sample_end_date + timedelta(days=1)
        current_oos_end_date = current_oos_start_date + timedelta(days=out_of_sample_days - 1)

        if current_oos_start_date > end_date_overall: # No OOS period possible
            logger.info(f"WFO Fold {fold_num}: OOS period starts ({current_oos_start_date}) after all data ({end_date_overall}). Ending WFO.")
            break
        if current_oos_end_date > end_date_overall: current_oos_end_date = end_date_overall # Truncate OOS

        logger.info(f"WFO Fold {fold_num}/{total_folds_estimate}: IS [{current_in_sample_start_date} - {current_in_sample_end_date}], OOS [{current_oos_start_date} - {current_oos_end_date}]")
        
        in_sample_data = full_price_data[(full_price_data.index.date >= current_in_sample_start_date) & (full_price_data.index.date <= current_in_sample_end_date)]
        out_of_sample_data = pd.DataFrame()
        if current_oos_start_date <= current_oos_end_date: # Valid OOS date range
            out_of_sample_data = full_price_data[(full_price_data.index.date >= current_oos_start_date) & (full_price_data.index.date <= current_oos_end_date)]

        if in_sample_data.empty:
            logger.warning(f"WFO Fold {fold_num}: In-sample data empty. Advancing.")
        else:
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
                    best_row = valid_opt_res.loc[valid_opt_res[metric_to_optimize].idxmin()] if metric_to_optimize == "Max Drawdown (%)" else valid_opt_res.loc[valid_opt_res[metric_to_optimize].idxmax()]
                    best_params_for_bt.update({'SL Points': best_row['SL Points'], 'RRR': best_row['RRR'],
                                               'EntryStartTime': dt_time(int(best_row['EntryStartHour']), int(best_row['EntryStartMinute'])),
                                               'EntryEndTime': dt_time(int(best_row['EntryEndHour']), int(best_row.get('EntryEndMinute', settings.DEFAULT_ENTRY_WINDOW_END_MINUTE)))})
                    is_metric_val = best_row[metric_to_optimize]
                    logger.info(f"WFO Fold {fold_num}: Best IS Params -> SL:{best_row['SL Points']:.2f}, RRR:{best_row['RRR']:.1f}, Start:{best_params_for_bt['EntryStartTime']:%H:%M}, End:{best_params_for_bt['EntryEndTime']:%H:%M}")
                else: logger.warning(f"WFO Fold {fold_num}: All IS opt results for '{metric_to_optimize}' were NaN. Using defaults.")
            else: logger.warning(f"WFO Fold {fold_num}: IS optimization yielded no results or missing metric '{metric_to_optimize}'. Using defaults.")
            
            fold_log = {'Fold': fold_num, 'InSampleStart': current_in_sample_start_date, 'InSampleEnd': current_in_sample_end_date,
                        'OutOfSampleStart': current_oos_start_date, 'OutOfSampleEnd': current_oos_end_date,
                        'BestSL': best_params_for_bt['SL Points'], 'BestRRR': best_params_for_bt['RRR'],
                        'BestEntryStart': best_params_for_bt['EntryStartTime'].strftime("%H:%M"),
                        'BestEntryEnd': best_params_for_bt['EntryEndTime'].strftime("%H:%M"),
                        'OptimizedMetric': metric_to_optimize, 'OptimizedMetricValue_InSample': is_metric_val}

            if not out_of_sample_data.empty:
                oos_perf = _run_single_backtest_for_optimization(best_params_for_bt, out_of_sample_data, initial_capital, risk_per_trade_percent, data_interval_str)
                if not oos_perf['_trades_df'].empty: all_oos_trades.append(oos_perf['_trades_df'])
                if not oos_perf['_equity_series'].empty: all_oos_equity_segments.append(oos_perf['_equity_series'])
                for k, v in oos_perf.items():
                    if not k.startswith('_'): fold_log[f'OOS_{k.replace(" (%)", "Pct").replace(" ", "")}'] = v
            else: logger.info(f"WFO Fold {fold_num}: OOS data empty. No OOS test.")
            wfo_log.append(fold_log)
        
        current_in_sample_start_date += timedelta(days=step_days)
        if progress_callback: progress_callback(min(1.0, fold_num / total_folds_estimate), f"WFO Fold {fold_num}")

    wfo_log_df = pd.DataFrame(wfo_log)
    final_oos_trades_df = pd.concat(all_oos_trades, ignore_index=True) if all_oos_trades else pd.DataFrame()
    
    chained_oos_equity = pd.Series(dtype=float); current_chained_capital_val = initial_capital
    if not final_oos_trades_df.empty:
        # ... (Chained equity logic as before, ensure robust start date for temp_equity_dates)
        final_oos_trades_df_sorted = final_oos_trades_df.sort_values(by='EntryTime').reset_index(drop=True)
        start_equity_dt = final_oos_trades_df_sorted['EntryTime'].dropna().min() - pd.Timedelta(microseconds=1) if not final_oos_trades_df_sorted.empty and not final_oos_trades_df_sorted['EntryTime'].dropna().empty else (full_price_data.index.min() - pd.Timedelta(microseconds=1) if not full_price_data.empty else pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1, microseconds=1))
        temp_eq_vals = [initial_capital]; temp_eq_dts = [start_equity_dt]
        for _, trade in final_oos_trades_df_sorted.iterrows():
            current_chained_capital_val += trade['P&L']; temp_eq_vals.append(current_chained_capital_val); temp_eq_dts.append(trade['ExitTime'])
        if len(temp_eq_dts) > 0:
            chained_oos_equity = pd.Series(temp_eq_vals, index=pd.to_datetime(temp_eq_dts)).sort_index()
            if not chained_oos_equity.empty:
                min_oos_dt_overall = min((d['OutOfSampleStart'] for d in wfo_log if d.get('OutOfSampleStart')), default=chained_oos_equity.index.min())
                max_oos_dt_overall = max((d['OutOfSampleEnd'] for d in wfo_log if d.get('OutOfSampleEnd')), default=chained_oos_equity.index.max())
                if pd.NaT not in [min_oos_dt_overall, max_oos_dt_overall] and min_oos_dt_overall <= max_oos_dt_overall:
                    oos_full_dt_range = pd.date_range(start=min_oos_dt_overall, end=max_oos_dt_overall, freq='B')
                    if not oos_full_dt_range.empty:
                        combined_idx = chained_oos_equity.index.union(oos_full_dt_range); chained_oos_equity = chained_oos_equity.reindex(combined_idx).ffill(); chained_oos_equity = chained_oos_equity.loc[oos_full_dt_range.min():oos_full_dt_range.max()]
    elif not full_price_data.empty: chained_oos_equity = pd.Series([initial_capital], index=[full_price_data.index.min()])
    else: chained_oos_equity = pd.Series([initial_capital], index=[pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1)])

    aggregated_oos_perf = {}
    if not final_oos_trades_df.empty and not chained_oos_equity.empty and chained_oos_equity.notna().any():
        # ... (Aggregated OOS performance calculation as before) ...
        aggregated_oos_perf['Total Trades'] = len(final_oos_trades_df); aggregated_oos_perf['Total P&L'] = final_oos_trades_df['P&L'].sum()
        aggregated_oos_perf['Final Capital'] = chained_oos_equity.iloc[-1] if not chained_oos_equity.empty else initial_capital
        daily_oos_ret_chained = _calculate_daily_returns(chained_oos_equity)
        aggregated_oos_perf['Sharpe Ratio (Annualized)'] = calculate_sharpe_ratio(daily_oos_ret_chained)
        aggregated_oos_perf['Sortino Ratio (Annualized)'] = calculate_sortino_ratio(daily_oos_ret_chained)
        cum_max = chained_oos_equity.cummax(); dd = (chained_oos_equity - cum_max) / cum_max
        aggregated_oos_perf['Max Drawdown (%)'] = dd.min() * 100 if dd.notna().any() and not dd.empty else 0
        num_win = len(final_oos_trades_df[final_oos_trades_df['P&L'] > 0])
        aggregated_oos_perf['Win Rate'] = (num_win / len(final_oos_trades_df) * 100) if len(final_oos_trades_df) > 0 else 0
        gp = final_oos_trades_df[final_oos_trades_df['P&L'] > 0]['P&L'].sum(); gl = final_oos_trades_df[final_oos_trades_df['P&L'] < 0]['P&L'].sum()
        aggregated_oos_perf['Profit Factor'] = abs(gp / gl) if gl != 0 else np.inf if gp > 0 else 0
        
    logger.info(f"WFO complete. Processed {fold_num} folds.")
    return wfo_log_df, final_oos_trades_df, chained_oos_equity, aggregated_oos_perf
