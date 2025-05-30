# services/optimizer/search_algorithms.py
"""
Implementations for Grid Search and Random Search optimization algorithms.
Includes parallel processing for faster execution.
"""
import pandas as pd
import numpy as np
import itertools
import random
from datetime import time as dt_time
import multiprocessing
import os # For os.cpu_count()

from config import settings
from utils.logger import get_logger
from .optimization_utils import _run_single_backtest_for_optimization

logger = get_logger(__name__)

def run_grid_search(
    price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    param_value_map: dict,
    data_interval_str: str,
    optimizer_control_params: dict,
    progress_callback=None
) -> pd.DataFrame:
    """Performs grid search optimization, potentially in parallel."""
    param_names = list(param_value_map.keys())
    param_values_for_product = []
    for name in param_names:
        values = param_value_map[name]
        if not isinstance(values, (list, tuple, np.ndarray)):
            logger.warning(f"Grid Search: Parameter '{name}' values not iterable: {values}. Wrapping.")
            param_values_for_product.append([values])
        elif len(values) == 0:
            logger.error(f"Grid Search: Parameter '{name}' has empty list of values.")
            return pd.DataFrame()
        else:
            param_values_for_product.append(values)

    value_combinations = list(itertools.product(*param_values_for_product))
    total_combinations = len(value_combinations)
    if total_combinations == 0:
        logger.warning("Grid Search: No parameter combinations. Check param_value_map.")
        return pd.DataFrame()

    strategy_name_gs = optimizer_control_params.get('strategy_name', settings.DEFAULT_STRATEGY)
    logger.info(f"Grid Search ({strategy_name_gs}): Params: {param_names}. Combinations: {total_combinations}. Interval: {data_interval_str}. Parallel Cores: {settings.OPTIMIZER_NUM_CORES}")
    
    results_list = []
    
    # Prepare arguments for each task
    tasks_args = []
    for combo_values in value_combinations:
        current_strategy_params_from_grid = dict(zip(param_names, combo_values))
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
        
        # Arguments for _run_single_backtest_for_optimization
        # IMPORTANT: price_data.copy() is used to ensure each process gets its own copy,
        # preventing potential issues if the underlying data were mutated (though it shouldn't be).
        # This can increase memory usage for very large DataFrames. If memory is a concern and
        # _run_single_backtest_for_optimization guarantees no mutation, .copy() might be omittable
        # but it's safer with it.
        task_arg_tuple = (
            params_for_bt_run, 
            price_data.copy(), # Pass a copy of the DataFrame
            initial_capital, 
            risk_per_trade_percent, 
            data_interval_str
        )
        tasks_args.append(task_arg_tuple)

    if settings.OPTIMIZER_NUM_CORES > 1 and total_combinations > 1:
        logger.info(f"Running Grid Search in parallel with {settings.OPTIMIZER_NUM_CORES} processes.")
        try:
            # Using try-finally to ensure pool is closed
            # Using a context manager for the pool is even better in Python 3.3+
            # with multiprocessing.Pool(processes=settings.OPTIMIZER_NUM_CORES) as pool:
            pool = multiprocessing.Pool(processes=settings.OPTIMIZER_NUM_CORES)
            async_results = [pool.apply_async(_run_single_backtest_for_optimization, args=task_args) for task_args in tasks_args]
            
            pool.close() # No more tasks will be submitted to the pool

            for i, async_result in enumerate(async_results):
                try:
                    perf_metrics_dict = async_result.get() # Blocks until this result is ready
                    perf_metrics_dict.pop('_trades_df', None)
                    perf_metrics_dict.pop('_equity_series', None)
                    results_list.append(perf_metrics_dict)
                except Exception as e:
                    # Log error for this specific task, using params from tasks_args if possible
                    # This requires careful indexing or storing params alongside async_results
                    failed_params = tasks_args[i][0] # Get the 'params_for_bt_run' for the failed task
                    logger.error(f"Error in parallel Grid Search task ({strategy_name_gs}) with params {failed_params}: {e}", exc_info=True)
                    err_log_params = {k: (v.strftime("%H:%M") if isinstance(v, dt_time) else v) for k,v in failed_params.items() if k not in ['strategy_name']}
                    error_entry = {'Strategy': strategy_name_gs, **err_log_params, 'Total P&L': np.nan, 'Error': str(e)}
                    results_list.append(error_entry)
                
                if progress_callback:
                    progress_callback((i + 1) / total_combinations, f"Grid Search ({strategy_name_gs})")
            
            pool.join() # Wait for all worker processes to exit
            logger.info("Parallel Grid Search tasks complete.")

        except Exception as e_pool:
            logger.error(f"Error during parallel Grid Search pool execution: {e_pool}", exc_info=True)
            # Fallback or re-raise, or append error entries for all remaining tasks
            # For simplicity, we'll just have the errors from individual tasks if they occur.
            # If the pool itself fails to start, this will catch it.
    else:
        logger.info("Running Grid Search sequentially.")
        for i, task_args_tuple in enumerate(tasks_args):
            # task_args_tuple is (params_for_bt_run, price_data_copy, initial_capital, ...)
            params_for_bt_run = task_args_tuple[0] # Extract the specific params for logging
            try:
                perf_metrics_dict = _run_single_backtest_for_optimization(*task_args_tuple)
                perf_metrics_dict.pop('_trades_df', None)
                perf_metrics_dict.pop('_equity_series', None)
                results_list.append(perf_metrics_dict)
            except Exception as e:
                logger.error(f"Error in sequential Grid Search iteration ({strategy_name_gs}) with params {params_for_bt_run}: {e}", exc_info=True)
                err_log_params = {k: (v.strftime("%H:%M") if isinstance(v, dt_time) else v) for k,v in params_for_bt_run.items() if k not in ['strategy_name']}
                error_entry = {'Strategy': strategy_name_gs, **err_log_params, 'Total P&L': np.nan, 'Error': str(e)}
                results_list.append(error_entry)
            
            if progress_callback:
                progress_callback((i + 1) / total_combinations, f"Grid Search ({strategy_name_gs})")
            
    return pd.DataFrame(results_list)


def run_random_search(
    price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    param_config_map: dict,
    data_interval_str: str,
    optimizer_control_params: dict,
    progress_callback=None
) -> pd.DataFrame:
    """Performs random search optimization, potentially in parallel."""
    num_iterations = optimizer_control_params.get('iterations', settings.DEFAULT_RANDOM_SEARCH_ITERATIONS)
    if num_iterations <= 0:
        logger.warning("Random Search: Non-positive iterations. No search performed.")
        return pd.DataFrame()

    strategy_name_rs = optimizer_control_params.get('strategy_name', settings.DEFAULT_STRATEGY)
    logger.info(f"Random Search ({strategy_name_rs}): Iterations: {num_iterations}. Interval: {data_interval_str}. Parallel Cores: {settings.OPTIMIZER_NUM_CORES}")
    results_list = []

    tasks_args = []
    for _ in range(num_iterations):
        current_random_params = {}
        for p_name, p_config_item in param_config_map.items():
            if isinstance(p_config_item, list):
                current_random_params[p_name] = random.choice(p_config_item)
            elif isinstance(p_config_item, tuple) and len(p_config_item) == 2:
                p_min, p_max = p_config_item
                if isinstance(p_min, float) or isinstance(p_max, float):
                    val = random.uniform(float(p_min), float(p_max))
                    if "points" in p_name or "rrr" in p_name: val = round(val, 2)
                elif isinstance(p_min, int) and isinstance(p_max, int):
                    val = random.randint(int(p_min), int(p_max))
                else:
                    logger.warning(f"Random Search ({strategy_name_rs}): Param '{p_name}' tuple config with unsupported types: {p_config_item}. Using default or skipping.")
                    # Decide on a fallback or skip this parameter for this iteration
                    continue 
                current_random_params[p_name] = val
            else:
                logger.warning(f"Random Search ({strategy_name_rs}): Skipping param '{p_name}' due to unexpected config: {p_config_item}")
                continue
        
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
        
        task_arg_tuple = (
            params_for_bt_run, 
            price_data.copy(), # Pass a copy
            initial_capital, 
            risk_per_trade_percent, 
            data_interval_str
        )
        tasks_args.append(task_arg_tuple)

    if settings.OPTIMIZER_NUM_CORES > 1 and num_iterations > 1:
        logger.info(f"Running Random Search in parallel with {settings.OPTIMIZER_NUM_CORES} processes.")
        try:
            pool = multiprocessing.Pool(processes=settings.OPTIMIZER_NUM_CORES)
            async_results = [pool.apply_async(_run_single_backtest_for_optimization, args=task_args) for task_args in tasks_args]
            pool.close()

            for i, async_result in enumerate(async_results):
                try:
                    perf_metrics_dict = async_result.get()
                    perf_metrics_dict.pop('_trades_df', None)
                    perf_metrics_dict.pop('_equity_series', None)
                    results_list.append(perf_metrics_dict)
                except Exception as e:
                    failed_params = tasks_args[i][0]
                    logger.error(f"Error in parallel Random Search task ({strategy_name_rs}) with params {failed_params}: {e}", exc_info=True)
                    err_log_params = {k: (v.strftime("%H:%M") if isinstance(v, dt_time) else v) for k,v in failed_params.items() if k not in ['strategy_name']}
                    error_entry = {'Strategy': strategy_name_rs, **err_log_params, 'Total P&L': np.nan, 'Error': str(e)}
                    results_list.append(error_entry)

                if progress_callback:
                    progress_callback((i + 1) / num_iterations, f"Random Search ({strategy_name_rs})")
            pool.join()
            logger.info("Parallel Random Search tasks complete.")
        except Exception as e_pool:
            logger.error(f"Error during parallel Random Search pool execution: {e_pool}", exc_info=True)

    else:
        logger.info("Running Random Search sequentially.")
        for i, task_args_tuple in enumerate(tasks_args):
            params_for_bt_run = task_args_tuple[0]
            try:
                perf_metrics_dict = _run_single_backtest_for_optimization(*task_args_tuple)
                perf_metrics_dict.pop('_trades_df', None)
                perf_metrics_dict.pop('_equity_series', None)
                results_list.append(perf_metrics_dict)
            except Exception as e:
                logger.error(f"Error in sequential Random Search iteration ({strategy_name_rs}) with params {params_for_bt_run}: {e}", exc_info=True)
                err_log_params = {k: (v.strftime("%H:%M") if isinstance(v, dt_time) else v) for k,v in params_for_bt_run.items() if k not in ['strategy_name']}
                error_entry = {'Strategy': strategy_name_rs, **err_log_params, 'Total P&L': np.nan, 'Error': str(e)}
                results_list.append(error_entry)

            if progress_callback:
                progress_callback((i + 1) / num_iterations, f"Random Search ({strategy_name_rs})")
                
    return pd.DataFrame(results_list)
