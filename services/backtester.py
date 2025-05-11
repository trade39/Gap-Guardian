# services/backtester.py
"""
Executes the backtest by simulating trades based on generated signals
and calculating profit/loss and equity curve.
"""
import pandas as pd
import numpy as np
from utils.logger import get_logger
from config.settings import STRATEGY_TIME_FRAME # To know the bar duration

logger = get_logger(__name__)

def run_backtest(
    price_data: pd.DataFrame, 
    signals: pd.DataFrame, 
    initial_capital: float, 
    risk_per_trade_percent: float,
    stop_loss_points_config: float # This is the SL *distance* from entry defined in config
    ) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Runs the backtest simulation.

    Args:
        price_data (pd.DataFrame): Historical OHLCV data (NY timezone).
        signals (pd.DataFrame): DataFrame of trade signals with 'SignalTime', 'SignalType', 
                                'EntryPrice', 'SL', 'TP'.
        initial_capital (float): Starting account balance.
        risk_per_trade_percent (float): Percentage of capital to risk per trade.
        stop_loss_points_config (float): The configured SL distance in points. This is used to calculate position size.


    Returns:
        tuple:
            - pd.DataFrame: Detailed log of executed trades.
            - pd.Series: Equity curve over time.
            - dict: Summary of performance metrics.
    """
    if price_data.empty:
        logger.warning("Price data is empty. Cannot run backtest.")
        return pd.DataFrame(), pd.Series(dtype=float), {}
    if signals.empty:
        logger.info("No signals to backtest.")
        # Return initial capital as flat equity curve
        equity = pd.Series([initial_capital] * len(price_data), index=price_data.index, name="Equity")
        return pd.DataFrame(), equity, {"Total P&L": 0, "Win Rate": 0, "Total Trades": 0, "Profit Factor": 0}

    trades_log = []
    current_capital = initial_capital
    equity = [] # List to store equity values at each bar
    equity_timestamps = [] # List to store timestamps for equity curve

    # Ensure price_data index is sorted
    price_data = price_data.sort_index()
    
    # Get bar duration for exit checks (e.g., 15 minutes)
    # This is a bit simplistic; robust way is to use freq from DatetimeIndex if available
    # or calculate diff between consecutive timestamps.
    try:
        # Attempt to infer frequency if possible, otherwise default
        if price_data.index.freqstr:
            bar_duration = pd.Timedelta(price_data.index.freqstr)
        else:
            # Calculate from first two timestamps if more than one row
            if len(price_data.index) > 1:
                 bar_duration = price_data.index[1] - price_data.index[0]
            else: # Default if only one row or freq cannot be inferred
                 bar_duration = pd.Timedelta(minutes=int(STRATEGY_TIME_FRAME.replace('m','')))
        logger.info(f"Inferred bar duration for trade processing: {bar_duration}")
    except Exception as e:
        logger.warning(f"Could not reliably infer bar duration, defaulting to {STRATEGY_TIME_FRAME}. Error: {e}")
        bar_duration = pd.Timedelta(minutes=int(STRATEGY_TIME_FRAME.replace('m','')))


    # Iterate through each bar in the price data to track equity and check for exits
    # This approach is more aligned with how equity changes over time, even between trades.
    
    active_trade = None
    last_equity_update_idx = -1

    for i, (timestamp, current_bar) in enumerate(price_data.iterrows()):
        # Process exits first
        if active_trade:
            exit_reason = None
            exit_price = None
            exit_time = None

            if active_trade['Type'] == 'Long':
                # Check SL
                if current_bar['Low'] <= active_trade['SL']:
                    exit_price = active_trade['SL']
                    exit_reason = 'Stop Loss'
                # Check TP
                elif current_bar['High'] >= active_trade['TP']:
                    exit_price = active_trade['TP']
                    exit_reason = 'Take Profit'
            
            elif active_trade['Type'] == 'Short':
                # Check SL
                if current_bar['High'] >= active_trade['SL']:
                    exit_price = active_trade['SL']
                    exit_reason = 'Stop Loss'
                # Check TP
                elif current_bar['Low'] <= active_trade['TP']:
                    exit_price = active_trade['TP']
                    exit_reason = 'Take Profit'
            
            if exit_reason:
                exit_time = timestamp # Exit occurs within this bar
                pnl = (exit_price - active_trade['EntryPrice']) * active_trade['PositionSize'] if active_trade['Type'] == 'Long' else \
                      (active_trade['EntryPrice'] - exit_price) * active_trade['PositionSize']
                
                current_capital += pnl
                active_trade['ExitPrice'] = exit_price
                active_trade['ExitTime'] = exit_time
                active_trade['P&L'] = pnl
                active_trade['ExitReason'] = exit_reason
                trades_log.append(active_trade.copy())
                logger.info(f"Trade closed: {active_trade['Type']} at {exit_price:.2f} ({exit_reason}). P&L: {pnl:.2f}. Capital: {current_capital:.2f}")
                active_trade = None


        # Check for new entries if no active trade
        # A signal's SignalTime is the close of the bar that generated it.
        # Entry is assumed at that close price, effectively at the start of the *next* bar.
        # Or, if SignalTime is start of bar, entry is at that bar's 'EntryPrice' (assumed close).
        # The current signal generation gives SignalTime = bar index (start of bar). EntryPrice = bar.Close.
        # So, a trade can be initiated if current_bar's timestamp matches a signal's timestamp.
        
        if not active_trade and timestamp in signals.index:
            signal = signals.loc[timestamp]
            # Ensure it's not a Series of Series if multiple signals at same time (should not happen with 1 trade/day)
            if isinstance(signal, pd.DataFrame): signal = signal.iloc[0]


            # Check if this is the first signal of the day (already handled by generate_signals)
            # For backtester, we just process signals as they come.

            risk_amount_per_trade = current_capital * (risk_per_trade_percent / 100.0)
            
            # stop_loss_points_config is the distance in points (e.g., 15 points for SPX)
            # Position size = Risk Amount / (Stop Loss Distance in $ per unit)
            # For yfinance data, we assume 1 unit of index/commodity. So SL distance in $ is just stop_loss_points_config.
            # This means P&L per point is effectively the position size.
            if stop_loss_points_config <= 0:
                logger.error(f"Stop loss points ({stop_loss_points_config}) must be positive. Skipping trade.")
            else:
                position_size = risk_amount_per_trade / stop_loss_points_config # This is the "multiplier" or "value per point"
            
                active_trade = {
                    'EntryTime': timestamp, # Entry at the close of this bar
                    'EntryPrice': signal['EntryPrice'],
                    'Type': signal['SignalType'],
                    'SL': signal['SL'],
                    'TP': signal['TP'],
                    'PositionSize': position_size, # This is effectively value per point
                    'InitialRiskAmount': risk_amount_per_trade,
                    'StopLossPoints': stop_loss_points_config # Store the original SL points for reference
                }
                logger.info(f"Trade opened: {active_trade['Type']} at {active_trade['EntryPrice']:.2f} on {timestamp}. "
                            f"PosSize (Value/Pt): {position_size:.2f}. Risked: ${risk_amount_per_trade:.2f}")
        
        # Update equity at each bar
        # If there's an active trade, P&L fluctuates with current price until exit.
        # For simplicity here, equity is updated when trades close or at end of data.
        # A more granular equity curve would update based on current bar's close if trade is open.
        # For now, let's update equity based on closed trades.
        # The equity curve will show jumps when trades close.
        
        # Store equity at the current timestamp
        # If a trade is active, current P&L could be mark-to-market.
        # For this version, equity only changes upon trade closure.
        equity.append(current_capital)
        equity_timestamps.append(timestamp)
        last_equity_update_idx = i


    # If a trade is still open at the end of data, close it at the last price
    if active_trade:
        last_bar = price_data.iloc[-1]
        exit_price = last_bar['Close']
        exit_time = price_data.index[-1]
        
        pnl = (exit_price - active_trade['EntryPrice']) * active_trade['PositionSize'] if active_trade['Type'] == 'Long' else \
              (active_trade['EntryPrice'] - exit_price) * active_trade['PositionSize']
        
        current_capital += pnl
        active_trade['ExitPrice'] = exit_price
        active_trade['ExitTime'] = exit_time
        active_trade['P&L'] = pnl
        active_trade['ExitReason'] = 'End of Data'
        trades_log.append(active_trade.copy())
        logger.info(f"Trade closed at end of data: {active_trade['Type']} at {exit_price:.2f}. P&L: {pnl:.2f}. Capital: {current_capital:.2f}")
        active_trade = None
        # Update the last equity point
        if equity: equity[-1] = current_capital


    # Create final DataFrames and Series
    trades_df = pd.DataFrame(trades_log)
    if not trades_df.empty:
        trades_df['EntryTime'] = pd.to_datetime(trades_df['EntryTime'])
        trades_df['ExitTime'] = pd.to_datetime(trades_df['ExitTime'])

    # Ensure equity series has the same length as price_data if no trades occurred or for full timeline
    # If equity list is shorter (e.g. only updated on trades), need to fill it.
    # The current loop updates equity at every bar.
    equity_series = pd.Series(equity, index=pd.to_datetime(equity_timestamps), name="Equity")
    if equity_series.empty and not price_data.empty: # No trades, flat equity
         equity_series = pd.Series([initial_capital] * len(price_data), index=price_data.index, name="Equity")
    elif equity_series.empty and price_data.empty: # No data, no equity
         equity_series = pd.Series(dtype=float, name="Equity")


    # Calculate Performance Metrics
    performance = {}
    if not trades_df.empty:
        performance['Total Trades'] = len(trades_df)
        performance['Total P&L'] = trades_df['P&L'].sum()
        performance['Gross Profit'] = trades_df[trades_df['P&L'] > 0]['P&L'].sum()
        performance['Gross Loss'] = trades_df[trades_df['P&L'] < 0]['P&L'].sum() # Will be negative
        
        num_winning_trades = len(trades_df[trades_df['P&L'] > 0])
        num_losing_trades = len(trades_df[trades_df['P&L'] < 0])
        
        performance['Winning Trades'] = num_winning_trades
        performance['Losing Trades'] = num_losing_trades
        performance['Win Rate'] = (num_winning_trades / performance['Total Trades'] * 100) if performance['Total Trades'] > 0 else 0
        
        if performance['Gross Loss'] == 0: # Avoid division by zero if all trades are winners or no losses
            performance['Profit Factor'] = np.inf if performance['Gross Profit'] > 0 else 0
        else:
            performance['Profit Factor'] = abs(performance['Gross Profit'] / performance['Gross Loss'])
        
        performance['Average Trade P&L'] = trades_df['P&L'].mean()
        performance['Average Winning Trade'] = trades_df[trades_df['P&L'] > 0]['P&L'].mean() if num_winning_trades > 0 else 0
        performance['Average Losing Trade'] = trades_df[trades_df['P&L'] < 0]['P&L'].mean() if num_losing_trades > 0 else 0
        
        # Max Drawdown (simple version based on equity series)
        if not equity_series.empty:
            cumulative_max = equity_series.cummax()
            drawdown = (equity_series - cumulative_max) / cumulative_max
            performance['Max Drawdown (%)'] = drawdown.min() * 100 if not drawdown.empty else 0
        else:
            performance['Max Drawdown (%)'] = 0

    else: # No trades
        performance = {
            'Total Trades': 0, 'Total P&L': 0, 'Gross Profit': 0, 'Gross Loss': 0,
            'Winning Trades': 0, 'Losing Trades': 0, 'Win Rate': 0, 'Profit Factor': 0,
            'Average Trade P&L': 0, 'Average Winning Trade': 0, 'Average Losing Trade': 0,
            'Max Drawdown (%)': 0
        }
    
    performance['Final Capital'] = current_capital
    
    logger.info(f"Backtest complete. Final Capital: {current_capital:.2f}. Total P&L: {performance.get('Total P&L', 0):.2f}")
    return trades_df, equity_series, performance

if __name__ == '__main__':
    # Example Usage (requires other modules and running from project root)
    from services.data_loader import fetch_historical_data
    from services.strategy_engine import generate_signals
    from config.settings import DEFAULT_INITIAL_CAPITAL, DEFAULT_RISK_PER_TRADE_PERCENT, DEFAULT_STOP_LOSS_POINTS, DEFAULT_RRR
    from datetime import date

    sample_ticker = "GC=F" # Gold
    start_dt = date.today() - pd.Timedelta(days=59)
    end_dt = date.today()
    
    price_data_bt = fetch_historical_data(sample_ticker, start_dt, end_dt, "15m")

    if not price_data_bt.empty:
        print(f"\nPrice data for backtest ({sample_ticker}): {len(price_data_bt)} rows")
        
        signals_bt = generate_signals(price_data_bt, DEFAULT_STOP_LOSS_POINTS, DEFAULT_RRR)
        
        if not signals_bt.empty:
            print(f"\nGenerated {len(signals_bt)} signals for backtest.")
            print(signals_bt.head())

            trades, equity_curve, performance_metrics = run_backtest(
                price_data_bt,
                signals_bt,
                DEFAULT_INITIAL_CAPITAL,
                DEFAULT_RISK_PER_TRADE_PERCENT,
                DEFAULT_STOP_LOSS_POINTS # This is the configured SL distance
            )

            print("\nBacktest Performance Metrics:")
            for k, v in performance_metrics.items():
                print(f"{k}: {v:.2f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}")

            if not trades.empty:
                print("\nTrades Log (first 5):")
                print(trades.head())
            else:
                print("\nNo trades were executed in the backtest.")

            if not equity_curve.empty:
                print("\nEquity Curve (last 5 points):")
                print(equity_curve.tail())
            else:
                print("\nEquity curve is empty.")

        else:
            print(f"\nNo signals generated for {sample_ticker}, cannot run backtest.")
    else:
        print(f"\nCould not fetch price data for {sample_ticker} to run backtest.")
