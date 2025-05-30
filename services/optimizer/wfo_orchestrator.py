# services/optimizer/wfo_orchestrator.py
"""
Houses the Walk-Forward Optimization (WFO) orchestration logic.
"""
import pandas as pd
import numpy as np
from datetime import timedelta, time as dt_time

from config import settings
from utils.logger import get_logger
from .optimization_utils import _run_single_backtest_for_optimization
from .search_algorithms import run_grid_search, run_random_search
from .metrics_calculator import _calculate_daily_returns, calculate_sharpe_ratio, calculate_sortino_ratio

logger = get_logger(__name__)

def run_walk_forward_optimization(
    full_price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    wfo_params: dict, 
    opt_algo: str, 
    opt_config_and_control: dict,
    data_interval_str: str, 
    progress_callback=None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Performs walk-forward optimization."""
    
    if full_price_data.empty or not isinstance(full_price_data.index, pd.DatetimeIndex):
        logger.error("WFO: Price data empty or invalid index.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float, index=pd.to_datetime([])), {}

    if full_price_data.index.tz is None:
        logger.warning(f"WFO: full_price_data index naive. Localizing to {settings.NY_TIMEZONE_STR}.")
        try:
            full_price_data.index = full_price_data.index.tz_localize(settings.NY_TIMEZONE_STR)
        except Exception as tz_err:
            logger.error(f"WFO: Failed to localize index to NY: {tz_err}. Trying UTC.", exc_info=True)
            try:
                full_price_data.index = full_price_data.index.tz_localize('UTC').tz_convert(settings.NY_TIMEZONE_STR)
            except Exception as utc_conv_err:
                 logger.error(f"WFO: Failed to convert index via UTC to NY: {utc_conv_err}.", exc_info=True)
                 return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float, index=pd.to_datetime([])), {}
    elif full_price_data.index.tz.zone != settings.NY_TIMEZONE.zone:
        logger.warning(f"WFO: full_price_data index is {full_price_data.index.tz}. Converting to NY.")
        try:
            full_price_data.index = full_price_data.index.tz_convert(settings.NY_TIMEZONE_STR)
        except Exception as tz_conv_err:
            logger.error(f"WFO: Failed to convert index to NY: {tz_conv_err}.", exc_info=True)
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
        logger.warning(f"WFO ({strategy_name_wfo}): Data duration ({total_data_duration_days}d) < min for one fold ({min_required_duration_for_one_fold}d).")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float, index=pd.to_datetime([])), {}
        
    logger.info(f"Starting WFO ({strategy_name_wfo}): IS={in_sample_days}d, OOS={out_of_sample_days}d, Step={step_days}d. Optimizing for {metric_to_optimize} using {opt_algo}. Interval: {data_interval_str}")
    
    all_oos_trades_list, wfo_log_list = [], []
    all_oos_equity_series_list = [] 
    current_fold_capital = initial_capital

    num_possible_folds = 0; temp_is_start = start_date_overall
    while temp_is_start + timedelta(days=in_sample_days - 1 + out_of_sample_days) <= end_date_overall:
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

        if current_in_sample_end_date > end_date_overall:
            logger.info(f"WFO Fold {fold_num} ({strategy_name_wfo}): IS period ends ({current_in_sample_end_date}) after data ({end_date_overall}). Ending.")
            break
        
        if current_oos_end_date > end_date_overall:
            current_oos_end_date = end_date_overall
            logger.info(f"WFO Fold {fold_num} ({strategy_name_wfo}): OOS period adjusted to end at {current_oos_end_date}.")

        if current_oos_start_date > current_oos_end_date:
            logger.info(f"WFO Fold {fold_num} ({strategy_name_wfo}): OOS period invalid (start {current_oos_start_date} > end {current_oos_end_date}). Ending.")
            break
        
        logger.info(f"WFO Fold {fold_num}/{total_folds_estimate} ({strategy_name_wfo}): IS [{current_in_sample_start_date} - {current_in_sample_end_date}], OOS [{current_oos_start_date} - {current_oos_end_date}]")
        
        in_sample_data = full_price_data[(full_price_data.index.date >= current_in_sample_start_date) & (full_price_data.index.date <= current_in_sample_end_date)]
        out_of_sample_data = full_price_data[(full_price_data.index.date >= current_oos_start_date) & (full_price_data.index.date <= current_oos_end_date)]

        if in_sample_data.empty:
            logger.warning(f"WFO Fold {fold_num} ({strategy_name_wfo}): IS data empty. Advancing.")
            current_in_sample_start_date += timedelta(days=step_days)
            if progress_callback: progress_callback(min(1.0, fold_num / total_folds_estimate), f"WFO Fold {fold_num} (IS Empty)")
            if current_in_sample_start_date > end_date_overall - timedelta(days=in_sample_days -1) : break
            continue
            
        optimization_results_df_is = pd.DataFrame()
        inner_optimizer_control_params = {'metric_to_optimize': metric_to_optimize, 'strategy_name': strategy_name_wfo}

        if opt_algo == "Grid Search":
            optimization_results_df_is = run_grid_search(in_sample_data, initial_capital, risk_per_trade_percent, inner_opt_param_definitions, data_interval_str, inner_optimizer_control_params, None)
        elif opt_algo == "Random Search":
            inner_optimizer_control_params['iterations'] = wfo_optimizer_iterations
            optimization_results_df_is = run_random_search(in_sample_data, initial_capital, risk_per_trade_percent, inner_opt_param_definitions, data_interval_str, inner_optimizer_control_params, None)
        
        best_params_for_oos_run = {
            'SL Points': settings.DEFAULT_STOP_LOSS_POINTS, 'RRR': settings.DEFAULT_RRR,
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
                if metric_to_optimize == "Max Drawdown (%)":
                    valid_mdd_results = valid_is_opt_results[valid_is_opt_results[metric_to_optimize] <= 0]
                    if not valid_mdd_results.empty: best_row_is = valid_mdd_results.loc[valid_mdd_results[metric_to_optimize].idxmax()]
                    else: best_row_is = valid_is_opt_results.iloc[0] 
                else:
                    best_row_is = valid_is_opt_results.loc[valid_is_opt_results[metric_to_optimize].idxmax()]
                
                if best_row_is is not None:
                    best_params_for_oos_run.update({'SL Points': float(best_row_is['SL Points']), 'RRR': float(best_row_is['RRR'])})
                    if strategy_name_wfo == "Gap Guardian":
                         best_params_for_oos_run['EntryStartTime'] = dt_time(int(best_row_is['EntryStartHour']), int(best_row_is['EntryStartMinute']))
                         best_params_for_oos_run['EntryEndTime'] = dt_time(int(best_row_is['EntryEndHour']), int(best_row_is.get('EntryEndMinute', settings.DEFAULT_ENTRY_WINDOW_END_MINUTE)))
                    in_sample_metric_value = best_row_is[metric_to_optimize]
        
        current_fold_log_entry = {
            'Fold': fold_num, 'InSampleStart': current_in_sample_start_date, 'InSampleEnd': current_in_sample_end_date,
            'OutOfSampleStart': current_oos_start_date, 'OutOfSampleEnd': current_oos_end_date,
            'BestSL': best_params_for_oos_run['SL Points'], 'BestRRR': best_params_for_oos_run['RRR'],
            'OptimizedMetric': metric_to_optimize, 'InSampleMetricValue': in_sample_metric_value
        }
        if strategy_name_wfo == "Gap Guardian":
            current_fold_log_entry['BestEntryStart'] = best_params_for_oos_run['EntryStartTime'].strftime("%H:%M")
            current_fold_log_entry['BestEntryEnd'] = best_params_for_oos_run['EntryEndTime'].strftime("%H:%M")

        if not out_of_sample_data.empty:
            oos_backtest_results = _run_single_backtest_for_optimization(best_params_for_oos_run, out_of_sample_data, current_fold_capital, risk_per_trade_percent, data_interval_str)
            oos_trades_this_fold = oos_backtest_results.get('_trades_df', pd.DataFrame())
            oos_equity_this_fold = oos_backtest_results.get('_equity_series', pd.Series(dtype=float))

            if not oos_trades_this_fold.empty: all_oos_trades_list.append(oos_trades_this_fold)
            if not oos_equity_this_fold.empty and oos_equity_this_fold.notna().any():
                all_oos_equity_series_list.append(oos_equity_this_fold)
                current_fold_capital = oos_equity_this_fold.iloc[-1]
            
            for k, v in oos_backtest_results.items():
                if not k.startswith('_') and k not in ['SL Points', 'RRR', 'EntryStartHour', 'EntryStartMinute', 'EntryEndHour', 'EntryEndMinute', 'strategy_name']:
                    current_fold_log_entry[f'OOS_{k.replace(" (%)", "Pct").replace(" ", "")}'] = v
        else:
            logger.info(f"WFO Fold {fold_num} ({strategy_name_wfo}): OOS data empty. No OOS backtest.")
        
        wfo_log_list.append(current_fold_log_entry)
        current_in_sample_start_date += timedelta(days=step_days)
        if progress_callback: progress_callback(min(1.0, fold_num / total_folds_estimate), f"WFO Fold {fold_num} ({strategy_name_wfo})")
        if current_in_sample_start_date + timedelta(days=in_sample_days - 1) > end_date_overall:
            logger.info(f"WFO: Next IS period start ({current_in_sample_start_date}) too late. Ending after {fold_num} folds.")
            break
            
    wfo_log_df = pd.DataFrame(wfo_log_list)
    final_oos_trades_aggregated_df = pd.concat(all_oos_trades_list, ignore_index=True) if all_oos_trades_list else pd.DataFrame()

    final_chained_oos_equity = pd.Series(dtype=float, index=pd.to_datetime([]).tz_localize(settings.NY_TIMEZONE_STR))
    if all_oos_equity_series_list:
        processed_equity_list = []
        for es in all_oos_equity_series_list:
            if not es.empty:
                if not isinstance(es.index, pd.DatetimeIndex): es.index = pd.to_datetime(es.index)
                if es.index.tz is None: es.index = es.index.tz_localize(settings.NY_TIMEZONE_STR)
                elif es.index.tz.zone != settings.NY_TIMEZONE.zone: es.index = es.index.tz_convert(settings.NY_TIMEZONE_STR)
                processed_equity_list.append(es)
        
        if processed_equity_list:
            first_oos_timestamp = min(es.index.min() for es in processed_equity_list if not es.empty)
            base_equity_point = pd.Series([initial_capital], index=[first_oos_timestamp - pd.Timedelta(microseconds=1)])
            if base_equity_point.index.tz is None: base_equity_point.index = base_equity_point.index.tz_localize(settings.NY_TIMEZONE_STR)
            
            temp_chained_equity = pd.concat([base_equity_point] + processed_equity_list).sort_index()
            temp_chained_equity = temp_chained_equity[~temp_chained_equity.index.duplicated(keep='last')]

            min_oos_date_overall = wfo_log_df['OutOfSampleStart'].min() if not wfo_log_df.empty else None
            max_oos_date_overall = wfo_log_df['OutOfSampleEnd'].max() if not wfo_log_df.empty else None

            if min_oos_date_overall and max_oos_date_overall:
                oos_full_datetime_index = full_price_data[(full_price_data.index.date >= min_oos_date_overall) & (full_price_data.index.date <= max_oos_date_overall)].index
                if not oos_full_datetime_index.empty:
                    final_chained_oos_equity = temp_chained_equity.reindex(oos_full_datetime_index, method='ffill')
                    if not final_chained_oos_equity.empty and pd.isna(final_chained_oos_equity.iloc[0]):
                        final_chained_oos_equity.iloc[0] = initial_capital
                        final_chained_oos_equity = final_chained_oos_equity.ffill()
                else: final_chained_oos_equity = temp_chained_equity
            else: final_chained_oos_equity = temp_chained_equity
        
    if final_chained_oos_equity.empty:
         final_chained_oos_equity = pd.Series([initial_capital], index=[pd.Timestamp.now(tz=settings.NY_TIMEZONE_STR)])

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
        
        if not final_chained_oos_equity.empty and final_chained_oos_equity.notna().any():
            aggregated_oos_performance_summary['Final Capital'] = final_chained_oos_equity.iloc[-1]
            cumulative_max_oos = final_chained_oos_equity.cummax()
            drawdown_oos = (final_chained_oos_equity - cumulative_max_oos) / cumulative_max_oos
            drawdown_oos.replace([np.inf, -np.inf], np.nan, inplace=True)
            mdd_oos_val = drawdown_oos.min() * 100
            aggregated_oos_performance_summary['Max Drawdown (%)'] = mdd_oos_val if pd.notna(mdd_oos_val) and not drawdown_oos.empty else 0.0
        else:
             aggregated_oos_performance_summary['Final Capital'] = initial_capital + total_pnl_from_oos_trades

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
        elif gross_profit_oos > 0 and gross_loss_oos == 0:
            aggregated_oos_performance_summary['Profit Factor'] = np.inf
        else:
            aggregated_oos_performance_summary['Profit Factor'] = 0.0
            
    logger.info(f"WFO ({strategy_name_wfo}) complete. Processed {fold_num-1 if fold_num > 0 else 0} folds. Agg OOS P&L: {aggregated_oos_performance_summary['Total P&L']:.2f}")
    return wfo_log_df, final_oos_trades_aggregated_df, final_chained_oos_equity, aggregated_oos_performance_summary
