# services/optimizer.py
"""
Performs parameter optimization for trading strategies using Grid Search, Random Search,
and Walk-Forward Optimization (WFO).
"""
import pandas as pd
import numpy as np
import itertools
import random
from tqdm import tqdm # For console progress if run standalone
from datetime import timedelta

from services import strategy_engine, backtester
from utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

def _calculate_daily_returns(equity_series: pd.Series) -> pd.Series:
    """Helper to calculate daily returns from an equity series."""
    if equity_series.empty:
        return pd.Series(dtype=float)
    daily_equity = equity_series.resample('D').last().ffill()
    daily_returns = daily_equity.pct_change().fillna(0)
    return daily_returns

def calculate_sharpe_ratio(returns_series, risk_free_rate=settings.RISK_FREE_RATE, trading_days_per_year=settings.TRADING_DAYS_PER_YEAR):
    if returns_series.empty or len(returns_series) < settings.MIN_TRADES_FOR_METRICS or returns_series.std() == 0: # MIN_TRADES_FOR_METRICS here is a proxy for sufficient data points
        return np.nan # Changed from 0.0 to np.nan for better "no result" indication
    excess_returns = returns_series - (risk_free_rate / trading_days_per_year)
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    return sharpe_ratio * np.sqrt(trading_days_per_year)

def calculate_sortino_ratio(returns_series, risk_free_rate=settings.RISK_FREE_RATE, trading_days_per_year=settings.TRADING_DAYS_PER_YEAR):
    if returns_series.empty or len(returns_series) < settings.MIN_TRADES_FOR_METRICS:
        return np.nan
    target_return = risk_free_rate / trading_days_per_year
    excess_returns = returns_series - target_return
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.empty or downside_returns.std() == 0:
        if excess_returns.mean() > 0 : return np.inf
        return np.nan # Changed from 0.0
    downside_std = downside_returns.std()
    sortino_ratio = excess_returns.mean() / downside_std
    return sortino_ratio * np.sqrt(trading_days_per_year)

def _run_single_backtest_for_optimization(
    params: dict, # Contains 'SL Points', 'RRR'
    price_data: pd.DataFrame,
    initial_capital: float,
    risk_per_trade_percent: float
) -> dict:
    """Runs a single backtest iteration and returns key performance metrics."""
    sl = params['SL Points']
    rrr = params['RRR']

    signals_df = strategy_engine.generate_signals(price_data.copy(), sl, rrr)
    trades_df, equity_series, performance_metrics = backtester.run_backtest(
        price_data.copy(), signals_df, initial_capital, risk_per_trade_percent, sl
    )
    
    daily_returns = _calculate_daily_returns(equity_series)

    return {
        'SL Points': sl,
        'RRR': rrr,
        'Total P&L': performance_metrics.get('Total P&L', 0),
        'Profit Factor': performance_metrics.get('Profit Factor', 0),
        'Win Rate': performance_metrics.get('Win Rate', 0),
        'Max Drawdown (%)': performance_metrics.get('Max Drawdown (%)', 0),
        'Total Trades': performance_metrics.get('Total Trades', 0),
        'Sharpe Ratio (Annualized)': calculate_sharpe_ratio(daily_returns),
        'Sortino Ratio (Annualized)': calculate_sortino_ratio(daily_returns),
        # Store trades and equity for WFO out-of-sample aggregation
        '_trades_df': trades_df,
        '_equity_series': equity_series
    }

def run_grid_search(
    price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    sl_points_values: list, rrr_values: list,
    progress_callback=None
) -> pd.DataFrame:
    logger.info(f"Starting Grid Search. SL points: {len(sl_points_values)}, RRR values: {len(rrr_values)}.")
    results = []
    param_combinations = list(itertools.product(sl_points_values, rrr_values))
    total_combinations = len(param_combinations)

    for i, (sl, rrr) in enumerate(param_combinations):
        try:
            current_params = {'SL Points': sl, 'RRR': rrr}
            perf = _run_single_backtest_for_optimization(
                current_params, price_data, initial_capital, risk_per_trade_percent
            )
            # Remove temporary keys not needed for the main optimization results table
            perf.pop('_trades_df', None)
            perf.pop('_equity_series', None)
            results.append(perf)
        except Exception as e:
            logger.error(f"Error during Grid Search for SL={sl}, RRR={rrr}: {e}", exc_info=True)
            results.append({'SL Points': sl, 'RRR': rrr, 'Total P&L': np.nan}) # Mark error
        if progress_callback: progress_callback((i + 1) / total_combinations, "Grid Search")
    return pd.DataFrame(results)

def run_random_search(
    price_data: pd.DataFrame, initial_capital: float, risk_per_trade_percent: float,
    sl_points_range: tuple, rrr_range: tuple, num_iterations: int,
    progress_callback=None
) -> pd.DataFrame:
    logger.info(f"Starting Random Search. Iterations: {num_iterations}.")
    results = []
    sl_min, sl_max = sl_points_range
    rrr_min, rrr_max = rrr_range

    for i in range(num_iterations):
        sl = round(random.uniform(sl_min, sl_max), 2) # Round to sensible precision
        rrr = round(random.uniform(rrr_min, rrr_max), 1)

        try:
            current_params = {'SL Points': sl, 'RRR': rrr}
            perf = _run_single_backtest_for_optimization(
                current_params, price_data, initial_capital, risk_per_trade_percent
            )
            perf.pop('_trades_df', None)
            perf.pop('_equity_series', None)
            results.append(perf)
        except Exception as e:
            logger.error(f"Error during Random Search for SL={sl}, RRR={rrr}: {e}", exc_info=True)
            results.append({'SL Points': sl, 'RRR': rrr, 'Total P&L': np.nan})
        if progress_callback: progress_callback((i + 1) / num_iterations, "Random Search")
    return pd.DataFrame(results)


def run_walk_forward_optimization(
    full_price_data: pd.DataFrame,
    initial_capital: float,
    risk_per_trade_percent: float,
    wfo_params: dict, # {in_sample_days, out_of_sample_days, step_days}
    opt_algo: str, # "Grid Search" or "Random Search"
    opt_params_config: dict, # {sl_range/values, rrr_range/values, iterations, metric_to_optimize}
    progress_callback=None # For overall WFO progress
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """
    Performs Walk-Forward Optimization.
    Returns:
        - DataFrame: Log of chosen parameters for each WFO fold.
        - DataFrame: Concatenated out-of-sample trades.
        - Series: Concatenated out-of-sample equity curve.
        - dict: Aggregated out-of-sample performance metrics.
    """
    in_sample_days = wfo_params['in_sample_days']
    out_of_sample_days = wfo_params['out_of_sample_days']
    step_days = wfo_params['step_days'] # How much to slide the window forward
    metric_to_optimize = opt_params_config['metric_to_optimize']

    logger.info(f"Starting Walk-Forward Optimization: In-sample={in_sample_days}d, OOS={out_of_sample_days}d, Step={step_days}d. Optimizing for {metric_to_optimize} using {opt_algo}.")

    all_oos_trades = []
    all_oos_equity_segments = []
    wfo_log = [] # To log chosen params and OOS period performance

    # Ensure full_price_data index is datetime
    if not isinstance(full_price_data.index, pd.DatetimeIndex):
        logger.error("WFO: Price data index is not DatetimeIndex.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float), {}

    # Determine the overall date range for WFO
    start_date_overall = full_price_data.index.min().date()
    end_date_overall = full_price_data.index.max().date()
    
    current_in_sample_start_date = start_date_overall
    fold_num = 0

    total_oos_periods_estimate = (end_date_overall - (start_date_overall + timedelta(days=in_sample_days)) ).days // step_days
    total_oos_periods_estimate = max(1, total_oos_periods_estimate) # Ensure at least 1 for progress

    while True:
        fold_num += 1
        # Define current in-sample period
        current_in_sample_end_date = current_in_sample_start_date + timedelta(days=in_sample_days -1) # -1 because it's inclusive

        # Define current out-of-sample period
        current_oos_start_date = current_in_sample_end_date + timedelta(days=1)
        current_oos_end_date = current_oos_start_date + timedelta(days=out_of_sample_days -1)

        logger.info(f"WFO Fold {fold_num}: In-Sample [{current_in_sample_start_date} - {current_in_sample_end_date}], OOS [{current_oos_start_date} - {current_oos_end_date}]")

        if current_in_sample_end_date > end_date_overall or current_oos_start_date > end_date_overall :
            logger.info("WFO: Reached end of data for in-sample or start of OOS period.")
            break
        if current_oos_end_date > end_date_overall: # Truncate last OOS period if it extends beyond data
            current_oos_end_date = end_date_overall
            logger.info(f"WFO: Truncating last OOS period to end at {current_oos_end_date}")
            if current_oos_start_date > current_oos_end_date: # No OOS period left
                 logger.info("WFO: No OOS period left after truncation.")
                 break


        # Slice data for in-sample and out-of-sample
        # Convert date objects to datetime for slicing if index is datetime
        in_sample_data = full_price_data[
            (full_price_data.index.date >= current_in_sample_start_date) &
            (full_price_data.index.date <= current_in_sample_end_date)
        ]
        out_of_sample_data = full_price_data[
            (full_price_data.index.date >= current_oos_start_date) &
            (full_price_data.index.date <= current_oos_end_date)
        ]

        if in_sample_data.empty:
            logger.warning(f"WFO Fold {fold_num}: In-sample data is empty. Skipping to next step.")
            current_in_sample_start_date += timedelta(days=step_days)
            if progress_callback: progress_callback(fold_num / total_oos_periods_estimate, f"WFO Fold {fold_num} (Skipped In-Sample)")
            continue
        if out_of_sample_data.empty and current_oos_start_date <= current_oos_end_date : # Only log if OOS period was expected
            logger.warning(f"WFO Fold {fold_num}: Out-of-sample data is empty, though period was [{current_oos_start_date} - {current_oos_end_date}]. This might happen on non-trading days.")
            # We might still want to advance, or handle this as a "no OOS test" for this fold.
            # For now, let's assume if OOS is empty, we can't test, so we just log the chosen params for this IS period if any.

        # --- Perform In-Sample Optimization ---
        optimization_results_df = pd.DataFrame()
        if opt_algo == "Grid Search":
            optimization_results_df = run_grid_search(
                in_sample_data, initial_capital, risk_per_trade_percent,
                opt_params_config['sl_values'], opt_params_config['rrr_values'],
                progress_callback=None # No detailed progress for inner loop in WFO for now
            )
        elif opt_algo == "Random Search":
            optimization_results_df = run_random_search(
                in_sample_data, initial_capital, risk_per_trade_percent,
                opt_params_config['sl_range'], opt_params_config['rrr_range'],
                opt_params_config['iterations'],
                progress_callback=None
            )

        if optimization_results_df.empty or optimization_metric not in optimization_results_df.columns:
            logger.warning(f"WFO Fold {fold_num}: In-sample optimization yielded no results or missing metric '{metric_to_optimize}'.")
            best_sl, best_rrr = settings.DEFAULT_STOP_LOSS_POINTS, settings.DEFAULT_RRR # Fallback
        else:
            # Select best parameters based on the chosen metric
            # Handle NaN values in the metric column by dropping them before finding idxmax/idxmin
            valid_optimization_results = optimization_results_df.dropna(subset=[metric_to_optimize])
            if valid_optimization_results.empty:
                logger.warning(f"WFO Fold {fold_num}: All optimization results for '{metric_to_optimize}' were NaN.")
                best_sl, best_rrr = settings.DEFAULT_STOP_LOSS_POINTS, settings.DEFAULT_RRR
            else:
                if metric_to_optimize == "Max Drawdown (%)": # Lower is better
                    best_params_row = valid_optimization_results.loc[valid_optimization_results[metric_to_optimize].idxmin()]
                else: # Higher is better
                    best_params_row = valid_optimization_results.loc[valid_optimization_results[metric_to_optimize].idxmax()]
                best_sl = best_params_row['SL Points']
                best_rrr = best_params_row['RRR']
        
        logger.info(f"WFO Fold {fold_num}: Best In-Sample Params -> SL: {best_sl:.2f}, RRR: {best_rrr:.1f}")
        
        fold_log_entry = {
            'Fold': fold_num,
            'InSampleStart': current_in_sample_start_date,
            'InSampleEnd': current_in_sample_end_date,
            'OutOfSampleStart': current_oos_start_date,
            'OutOfSampleEnd': current_oos_end_date,
            'BestSL': best_sl,
            'BestRRR': best_rrr,
            'OptimizedMetric': metric_to_optimize,
            'OptimizedMetricValue_InSample': best_params_row.get(metric_to_optimize, np.nan) if 'best_params_row' in locals() and not valid_optimization_results.empty else np.nan
        }

        # --- Test Best Parameters on Out-Of-Sample Data ---
        if not out_of_sample_data.empty:
            oos_signals = strategy_engine.generate_signals(out_of_sample_data.copy(), best_sl, best_rrr)
            oos_trades, oos_equity, oos_perf = backtester.run_backtest(
                out_of_sample_data.copy(), oos_signals, initial_capital, # OOS backtest starts with initial_capital for each fold for simplicity, or could chain capital
                risk_per_trade_percent, best_sl
            )
            if not oos_trades.empty:
                all_oos_trades.append(oos_trades)
            if not oos_equity.empty:
                all_oos_equity_segments.append(oos_equity)
            
            # Log OOS performance for this fold
            for metric_key, metric_val in oos_perf.items():
                 fold_log_entry[f'OOS_{metric_key.replace(" (%)", "Pct").replace(" ", "")}'] = metric_val
            daily_oos_returns = _calculate_daily_returns(oos_equity)
            fold_log_entry['OOS_Sharpe'] = calculate_sharpe_ratio(daily_oos_returns)
            fold_log_entry['OOS_Sortino'] = calculate_sortino_ratio(daily_oos_returns)

        wfo_log.append(fold_log_entry)

        # Slide window for next iteration
        current_in_sample_start_date += timedelta(days=step_days)
        if progress_callback: progress_callback(fold_num / total_oos_periods_estimate, f"WFO Fold {fold_num}")


    wfo_log_df = pd.DataFrame(wfo_log)
    
    # Concatenate all OOS trades and equity
    final_oos_trades_df = pd.concat(all_oos_trades, ignore_index=True) if all_oos_trades else pd.DataFrame()
    
    # For equity curve, we need to be careful. If each OOS backtest resets capital,
    # we can't just concat. We need to chain P&Ls or returns.
    # For simplicity now, let's assume each OOS period is independent for metric calculation,
    # and we'll plot segments. A true chained equity curve is more complex.
    # Let's build a chained equity curve.
    chained_oos_equity = pd.Series(dtype=float)
    current_chained_capital = initial_capital
    
    if all_oos_equity_segments:
        # Sort segments by start time to ensure correct chaining order
        all_oos_equity_segments.sort(key=lambda s: s.index.min() if not s.empty else pd.Timestamp.max)

        temp_equity_points = []
        last_equity_val = initial_capital

        for segment in all_oos_equity_segments:
            if segment.empty:
                continue
            
            # Calculate P&L of this segment
            segment_pnl = segment.iloc[-1] - segment.iloc[0] # Assumes segment equity starts at initial_capital for that segment
                                                            # This needs refinement if backtester.run_backtest is modified to chain capital.
                                                            # For now, let's assume backtester.run_backtest returns an equity series
                                                            # that starts from initial_capital for its period.
                                                            # A better way: use the P&L from oos_trades for the segment.

            # For a simple chained view, let's adjust the segment to start from the previous segment's end capital
            # This is an approximation if each segment was run with reset capital.
            # A more accurate way is to run the backtester in a chained mode.
            # For now, let's just concatenate and then think about true chaining.
            # This simplified approach will show jumps if capital was reset.
            
            # Let's try to chain based on P&L of each segment.
            # This assumes the `oos_perf['Total P&L']` is accurate for the segment.
            # This is still tricky without modifying the core backtester to support chained capital.

            # Simplest for now: just concatenate the raw equity series. The plot will show resets.
            # Or, plot P&Ls.
            # For now, let's just return the segments, and plotting will handle them.
            pass # The concatenation of raw segments will be handled by plotting.

        # A better way to create chained equity:
        # Iterate through sorted oos_trades_df, apply P&Ls sequentially
        if not final_oos_trades_df.empty:
            # Ensure trades are sorted by entry time for proper equity calculation
            final_oos_trades_df_sorted = final_oos_trades_df.sort_values(by='EntryTime').reset_index(drop=True)
            
            temp_equity_values = [initial_capital]
            temp_equity_dates = [full_price_data.index.min()] # Start with the earliest date in the dataset

            current_capital_for_chain = initial_capital
            for _, trade in final_oos_trades_df_sorted.iterrows():
                current_capital_for_chain += trade['P&L']
                temp_equity_values.append(current_capital_for_chain)
                temp_equity_dates.append(trade['ExitTime']) # Equity changes at exit time
            
            if len(temp_equity_dates) > 0:
                chained_oos_equity = pd.Series(temp_equity_values, index=pd.to_datetime(temp_equity_dates))
                chained_oos_equity = chained_oos_equity.sort_index()
                # Fill forward to cover all dates in the OOS periods
                # This needs a full date range of all OOS periods.
                if not chained_oos_equity.empty:
                    oos_date_range = pd.date_range(start=chained_oos_equity.index.min(), end=chained_oos_equity.index.max(), freq='B') # Business day frequency
                    if oos_date_range.empty and not chained_oos_equity.empty: # If only one trade
                         chained_oos_equity_resampled = chained_oos_equity
                    elif not oos_date_range.empty:
                         chained_oos_equity_resampled = chained_oos_equity.reindex(oos_date_range).ffill()
                         # Ensure the very first point is the initial capital if not covered by reindex
                         if chained_oos_equity_resampled.index.min() > chained_oos_equity.index.min() or pd.isna(chained_oos_equity_resampled.iloc[0]):
                            first_val_series = pd.Series([initial_capital], index=[chained_oos_equity.index.min()])
                            chained_oos_equity_resampled = pd.concat([first_val_series, chained_oos_equity_resampled.dropna()])
                            chained_oos_equity_resampled = chained_oos_equity_resampled.sort_index().ffill()
                    else: # chained_oos_equity is empty
                        chained_oos_equity_resampled = pd.Series(dtype=float)

                    chained_oos_equity = chained_oos_equity_resampled.dropna() # Drop any leading NaNs if any issue with date range
            else: # No trades
                chained_oos_equity = pd.Series([initial_capital], index=[full_price_data.index.min()])


    # Calculate aggregated OOS performance from the chained equity or trades
    aggregated_oos_performance = {}
    if not final_oos_trades_df.empty:
        aggregated_oos_performance['Total Trades'] = len(final_oos_trades_df)
        aggregated_oos_performance['Total P&L'] = final_oos_trades_df['P&L'].sum()
        # ... other metrics can be recalculated from final_oos_trades_df ...
        # For now, let's use the chained equity for overall metrics
        if not chained_oos_equity.empty:
            aggregated_oos_performance['Final Capital'] = chained_oos_equity.iloc[-1]
            daily_oos_returns_chained = _calculate_daily_returns(chained_oos_equity)
            aggregated_oos_performance['Sharpe Ratio (Annualized)'] = calculate_sharpe_ratio(daily_oos_returns_chained)
            aggregated_oos_performance['Sortino Ratio (Annualized)'] = calculate_sortino_ratio(daily_oos_returns_chained)
            
            cumulative_max = chained_oos_equity.cummax()
            drawdown = (chained_oos_equity - cumulative_max) / cumulative_max
            aggregated_oos_performance['Max Drawdown (%)'] = drawdown.min() * 100 if not drawdown.empty else 0
            
            num_winning = len(final_oos_trades_df[final_oos_trades_df['P&L'] > 0])
            aggregated_oos_performance['Win Rate'] = (num_winning / len(final_oos_trades_df) * 100) if len(final_oos_trades_df) > 0 else 0
            
            gross_profit = final_oos_trades_df[final_oos_trades_df['P&L'] > 0]['P&L'].sum()
            gross_loss = final_oos_trades_df[final_oos_trades_df['P&L'] < 0]['P&L'].sum()
            aggregated_oos_performance['Profit Factor'] = abs(gross_profit / gross_loss) if gross_loss != 0 else np.inf if gross_profit > 0 else 0


    logger.info(f"Walk-Forward Optimization complete. Processed {fold_num-1} folds.")
    return wfo_log_df, final_oos_trades_df, chained_oos_equity, aggregated_oos_performance

