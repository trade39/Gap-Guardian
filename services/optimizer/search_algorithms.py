# services/optimizer/search_algorithms.py
"""
Implementations for Grid Search and Random Search optimization algorithms.
"""
import pandas as pd
import numpy as np
import itertools
import random
from datetime import time as dt_time

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
    """Performs grid search optimization."""
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
    logger.info(f"Grid Search ({strategy_name_gs}): Params: {param_names}. Combinations: {total_combinations}. Interval: {data_interval_str}")
    results_list = []

    for i, combo_values in enumerate(value_combinations):
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
        
        try:
            perf_metrics_dict = _run_single_backtest_for_optimization(params_for_bt_run, price_data, initial_capital, risk_per_trade_percent, data_interval_str)
            perf_metrics_dict.pop('_trades_df', None)
            perf_metrics_dict.pop('_equity_series', None)
            results_list.append(perf_metrics_dict)
        except Exception as e:
            logger.error(f"Error in Grid Search iteration ({strategy_name_gs}) with params {params_for_bt_run}: {e}", exc_info=True)
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
    """Performs random search optimization."""
    num_iterations = optimizer_control_params.get('iterations', settings.DEFAULT_RANDOM_SEARCH_ITERATIONS)
    if num_iterations <= 0:
        logger.warning("Random Search: Non-positive iterations. No search performed.")
        return pd.DataFrame()

    strategy_name_rs = optimizer_control_params.get('strategy_name', settings.DEFAULT_STRATEGY)
    logger.info(f"Random Search ({strategy_name_rs}): Iterations: {num_iterations}. Interval: {data_interval_str}")
    results_list = []

    for i in range(num_iterations):
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
                    logger.warning(f"Random Search ({strategy_name_rs}): Param '{p_name}' tuple config with unsupported types: {p_config_item}. Skipping.")
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
            
        try:
            perf_metrics_dict = _run_single_backtest_for_optimization(params_for_bt_run, price_data, initial_capital, risk_per_trade_percent, data_interval_str)
            perf_metrics_dict.pop('_trades_df', None)
            perf_metrics_dict.pop('_equity_series', None)
            results_list.append(perf_metrics_dict)
        except Exception as e:
            logger.error(f"Error in Random Search iteration ({strategy_name_rs}) with params {params_for_bt_run}: {e}", exc_info=True)
            err_log_params = {k: (v.strftime("%H:%M") if isinstance(v, dt_time) else v) for k,v in params_for_bt_run.items() if k not in ['strategy_name']}
            error_entry = {'Strategy': strategy_name_rs, **err_log_params, 'Total P&L': np.nan, 'Error': str(e)}
            results_list.append(error_entry)

        if progress_callback:
            progress_callback((i + 1) / num_iterations, f"Random Search ({strategy_name_rs})")
            
    return pd.DataFrame(results_list)
