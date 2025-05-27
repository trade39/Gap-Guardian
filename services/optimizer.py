# services/optimizer.py
"""
Performs parameter optimization for trading strategies using grid search.
"""
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm # For progress bar in console if run standalone

from services import strategy_engine, backtester
from utils.logger import get_logger
from config import settings # For RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

logger = get_logger(__name__)

def calculate_sharpe_ratio(returns_series, risk_free_rate=settings.RISK_FREE_RATE, trading_days_per_year=settings.TRADING_DAYS_PER_YEAR):
    """Calculates the annualized Sharpe ratio."""
    if returns_series.empty or returns_series.std() == 0:
        return 0.0
    excess_returns = returns_series - (risk_free_rate / trading_days_per_year) # Daily risk-free rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    return sharpe_ratio * np.sqrt(trading_days_per_year) # Annualize

def calculate_sortino_ratio(returns_series, risk_free_rate=settings.RISK_FREE_RATE, trading_days_per_year=settings.TRADING_DAYS_PER_YEAR):
    """Calculates the annualized Sortino ratio."""
    if returns_series.empty:
        return 0.0
    target_return = risk_free_rate / trading_days_per_year # Daily target (risk-free rate)
    excess_returns = returns_series - target_return
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.empty or downside_returns.std() == 0: # No downside returns or no volatility in them
        if excess_returns.mean() > 0 : return np.inf # Positive mean return with no downside risk
        return 0.0 # No excess return and no downside risk
    
    downside_std = downside_returns.std()
    sortino_ratio = excess_returns.mean() / downside_std
    return sortino_ratio * np.sqrt(trading_days_per_year) # Annualize


def run_grid_search(
    price_data: pd.DataFrame,
    initial_capital: float,
    risk_per_trade_percent: float,
    sl_points_values: list, # List of SL points to test
    rrr_values: list,       # List of RRR values to test
    optimization_metric_name: str,
    progress_callback=None # Optional callback for Streamlit progress bar
) -> pd.DataFrame:
    """
    Performs a grid search over stop_loss_points and rrr.

    Args:
        price_data (pd.DataFrame): Historical OHLCV data.
        initial_capital (float): Starting account balance.
        risk_per_trade_percent (float): Percentage of capital to risk per trade.
        sl_points_values (list): List of stop loss point values to iterate over.
        rrr_values (list): List of RRR values to iterate over.
        optimization_metric_name (str): The name of the performance metric to optimize for.
        progress_callback (function, optional): Callback function to update progress (e.g., st.progress).

    Returns:
        pd.DataFrame: DataFrame containing parameters and their corresponding performance metrics.
                      Columns: ['SL Points', 'RRR', <optimization_metric_name>, 'Total P&L', 'Win Rate', ...]
    """
    logger.info(f"Starting grid search. SL points: {len(sl_points_values)}, RRR values: {len(rrr_values)}. Optimizing for: {optimization_metric_name}")

    results = []
    param_combinations = list(itertools.product(sl_points_values, rrr_values))
    total_combinations = len(param_combinations)
    
    # Use tqdm for console progress if not using Streamlit's progress bar
    # iterator = tqdm(param_combinations, desc="Grid Search Progress") if progress_callback is None else param_combinations
    iterator = param_combinations

    for i, (sl, rrr) in enumerate(iterator):
        logger.debug(f"Testing SL: {sl}, RRR: {rrr}")
        try:
            # 1. Generate Signals with current parameters
            signals_df = strategy_engine.generate_signals(
                price_data.copy(), # Pass a copy to avoid modifications if any
                stop_loss_points=sl,
                rrr=rrr
            )

            # 2. Run Backtest
            # Note: backtester.run_backtest already calculates many metrics.
            # We might need to add Sharpe/Sortino if not already there.
            trades_df, equity_series, performance_metrics = backtester.run_backtest(
                price_data.copy(),
                signals_df,
                initial_capital,
                risk_per_trade_percent,
                stop_loss_points_config=sl # This is the SL distance used for position sizing
            )
            
            # Calculate daily returns for Sharpe/Sortino if equity_series is available
            daily_returns = pd.Series(dtype=float)
            if not equity_series.empty:
                # Resample to daily, then calculate percentage change.
                # Fill NaNs that might occur on non-trading days or before first trade.
                daily_equity = equity_series.resample('D').last().ffill()
                daily_returns = daily_equity.pct_change().fillna(0)


            # Add advanced metrics if requested for optimization
            current_result_row = {
                'SL Points': sl,
                'RRR': rrr,
                'Total P&L': performance_metrics.get('Total P&L', 0),
                'Profit Factor': performance_metrics.get('Profit Factor', 0),
                'Win Rate': performance_metrics.get('Win Rate', 0),
                'Max Drawdown (%)': performance_metrics.get('Max Drawdown (%)', 0),
                'Total Trades': performance_metrics.get('Total Trades', 0),
                'Sharpe Ratio (Annualized)': calculate_sharpe_ratio(daily_returns),
                'Sortino Ratio (Annualized)': calculate_sortino_ratio(daily_returns)
            }
            results.append(current_result_row)

        except Exception as e:
            logger.error(f"Error during optimization for SL={sl}, RRR={rrr}: {e}", exc_info=True)
            results.append({
                'SL Points': sl,
                'RRR': rrr,
                optimization_metric_name: np.nan # Mark as NaN or some error indicator
                # Add other metrics as NaN as well
            })
        
        if progress_callback:
            progress_callback( (i + 1) / total_combinations )

    results_df = pd.DataFrame(results)
    logger.info(f"Grid search complete. Generated {len(results_df)} results.")
    return results_df

if __name__ == '__main__':
    # Example Usage (requires other modules and running from project root)
    from services.data_loader import fetch_historical_data
    from datetime import date, timedelta

    sample_ticker = "GC=F"
    start_dt = date.today() - timedelta(days=59) # Ensure valid range for 15m
    end_dt = date.today()
    
    price_data_opt = fetch_historical_data(sample_ticker, start_dt, end_dt, "15m")

    if not price_data_opt.empty:
        print(f"\nPrice data for optimization ({sample_ticker}): {len(price_data_opt)} rows")
        
        sl_values = np.linspace(10, 20, 3) # 10, 15, 20
        rrr_values = np.linspace(1.5, 2.5, 3) # 1.5, 2.0, 2.5
        
        optimization_results = run_grid_search(
            price_data_opt,
            settings.DEFAULT_INITIAL_CAPITAL,
            settings.DEFAULT_RISK_PER_TRADE_PERCENT,
            sl_values,
            rrr_values,
            "Total P&L"
        )

        print("\nOptimization Results:")
        print(optimization_results)

        if not optimization_results.empty:
            best_params = optimization_results.loc[optimization_results["Total P&L"].idxmax()]
            print("\nBest Parameters (based on Total P&L):")
            print(best_params)
    else:
        print(f"\nCould not fetch price data for {sample_ticker} to run optimization.")
