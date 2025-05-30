# services/optimizer/optimization_utils.py
"""
Utility functions for the optimization process, primarily running a single backtest iteration.
"""
import pandas as pd
import numpy as np
from datetime import time as dt_time

from services import strategy_engine, backtester # Main services
from config import settings
from utils.logger import get_logger
from .metrics_calculator import _calculate_daily_returns, calculate_sharpe_ratio, calculate_sortino_ratio

logger = get_logger(__name__)

def _run_single_backtest_for_optimization(
    params: dict, 
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
    sl = params['SL Points']
    rrr = params['RRR']
    
    signal_gen_params_for_engine = {
        'strategy_name': strategy_name_opt,
        'stop_loss_points': float(sl),
        'rrr': float(rrr)
    }
    if strategy_name_opt == "Gap Guardian":
        est_val = params.get('EntryStartTime')
        eet_val = params.get('EntryEndTime')
        signal_gen_params_for_engine['entry_start_time'] = est_val if isinstance(est_val, dt_time) else dt_time(int(est_val.hour), int(est_val.minute)) if hasattr(est_val, 'hour') else settings.DEFAULT_ENTRY_WINDOW_START_HOUR
        signal_gen_params_for_engine['entry_end_time'] = eet_val if isinstance(eet_val, dt_time) else dt_time(int(eet_val.hour), int(eet_val.minute)) if hasattr(eet_val, 'hour') else settings.DEFAULT_ENTRY_WINDOW_END_HOUR

    signals_df = strategy_engine.generate_signals(price_data.copy(), **signal_gen_params_for_engine)
    trades_df, equity_s, perf_metrics = backtester.run_backtest(
        price_data.copy(), signals_df, initial_capital, 
        risk_per_trade_percent, float(sl), data_interval_str
    )
    
    if not isinstance(equity_s.index, pd.DatetimeIndex):
        logger.warning(f"_run_single_backtest_for_optimization: equity_s.index not DatetimeIndex. Attempting conversion.")
        try:
            equity_s.index = pd.to_datetime(equity_s.index)
        except Exception as e_conv_idx:
            logger.error(f"Failed to convert equity_s index to DatetimeIndex: {e_conv_idx}", exc_info=True)
            equity_s = pd.Series(dtype=float, index=pd.to_datetime([]))

    if equity_s.index.tz is None and not equity_s.empty:
        try:
            equity_s.index = equity_s.index.tz_localize(settings.NY_TIMEZONE_STR)
        except Exception as e_loc_idx:
            logger.error(f"Failed to localize equity_s index to NY: {e_loc_idx}. Trying UTC.", exc_info=True)
            try:
                equity_s.index = equity_s.index.tz_localize('UTC').tz_convert(settings.NY_TIMEZONE_STR)
            except Exception as e_utc_conv:
                logger.error(f"Failed to convert equity_s index from UTC to NY: {e_utc_conv}", exc_info=True)
                equity_s.index = pd.to_datetime(equity_s.index).tz_localize(None)

    daily_ret = _calculate_daily_returns(equity_s.copy())

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
        'Sharpe Ratio (Annualized)': calculate_sharpe_ratio(daily_ret.copy()),
        'Sortino Ratio (Annualized)': calculate_sortino_ratio(daily_ret.copy()),
        'Average Trade P&L': perf_metrics.get('Average Trade P&L', np.nan),
        'Average Winning Trade': perf_metrics.get('Average Winning Trade', np.nan),
        'Average Losing Trade': perf_metrics.get('Average Losing Trade', np.nan),
        '_trades_df': trades_df,
        '_equity_series': equity_s
    }
    return result
