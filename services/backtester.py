# services/backtester.py
"""
Executes the backtest by simulating trades based on generated signals
and calculating profit/loss and equity curve, including transaction costs.
"""
import pandas as pd
import numpy as np
from utils.logger import get_logger
# Assuming settings are imported for default values if needed, though not directly used here
# from config import settings

logger = get_logger(__name__)

def _parse_interval_to_timedelta(interval_str: str) -> pd.Timedelta | None:
    """
    Parses a yfinance-style interval string (e.g., "15m", "1h", "1d")
    into a pandas Timedelta object. Returns None if parsing fails.
    """
    try:
        if 'm' in interval_str and 'mo' not in interval_str:
            return pd.Timedelta(minutes=int(interval_str.replace('m', '')))
        elif 'h' in interval_str:
            return pd.Timedelta(hours=int(interval_str.replace('h', '')))
        elif 'd' in interval_str:
            return pd.Timedelta(days=int(interval_str.replace('d', '')))
        elif 'wk' in interval_str:
            return pd.Timedelta(weeks=int(interval_str.replace('wk', '')))
        logger.warning(f"Interval string '{interval_str}' not supported for Timedelta parsing for bar duration.")
        return None
    except ValueError:
        logger.error(f"Could not parse interval string '{interval_str}' to Timedelta.")
        return None

def run_backtest(
    price_data: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float,
    risk_per_trade_percent: float,
    stop_loss_points_config: float, # This is the risk distance for position sizing
    data_interval_str: str,
    # New parameters for transaction costs
    commission_type: str = "None", # "None", "Fixed per Trade", "Percentage of Trade Value"
    commission_rate: float = 0.0,  # Actual rate (e.g., 1.0 for fixed, 0.001 for 0.1%)
    slippage_points: float = 0.0   # Points per side
    ) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Runs the backtest simulation including commission and slippage.
    Args:
        price_data (pd.DataFrame): OHLCV data.
        signals (pd.DataFrame): Trade signals with 'EntryPrice', 'SL', 'TP', 'SignalType'.
        initial_capital (float): Starting capital.
        risk_per_trade_percent (float): Risk per trade as a percentage of current capital.
        stop_loss_points_config (float): The distance in points defining the risk for position sizing.
        data_interval_str (str): Interval of the price data (e.g., "15m").
        commission_type (str): Type of commission.
        commission_rate (float): Rate for commission (actual value, not percentage for "Percentage" type).
        slippage_points (float): Slippage in points applied per side of the trade.
    """
    if price_data.empty:
        logger.warning("Price data is empty. Cannot run backtest.")
        return pd.DataFrame(), pd.Series(dtype=float), {}

    equity_points = {} 
    start_point_equity_time = price_data.index.min() - pd.Timedelta(microseconds=1) if not price_data.empty else pd.Timestamp('1970-01-01', tz='UTC')
    equity_points[start_point_equity_time] = initial_capital
    current_capital = initial_capital
    trades_log = []
    price_data = price_data.sort_index()

    logger.info(f"Backtesting with Initial Capital: ${initial_capital:,.2f}, Risk/Trade: {risk_per_trade_percent}%, SL (for sizing): {stop_loss_points_config} pts")
    logger.info(f"Transaction Costs: Commission Type='{commission_type}', Rate={commission_rate}, Slippage={slippage_points} pts/side")

    active_trade = None
    for i, (timestamp, current_bar) in enumerate(price_data.iterrows()):
        # --- Check for exits before new entries ---
        if active_trade:
            exit_reason, exit_price_raw = None, None
            
            # Determine raw exit price based on SL/TP hit by current_bar's Low/High
            if active_trade['Type'] == 'Long':
                if current_bar['Low'] <= active_trade['SL']: 
                    exit_price_raw, exit_reason = active_trade['SL'], 'Stop Loss'
                elif current_bar['High'] >= active_trade['TP']: 
                    exit_price_raw, exit_reason = active_trade['TP'], 'Take Profit'
            elif active_trade['Type'] == 'Short':
                if current_bar['High'] >= active_trade['SL']: 
                    exit_price_raw, exit_reason = active_trade['SL'], 'Stop Loss'
                elif current_bar['Low'] <= active_trade['TP']: 
                    exit_price_raw, exit_reason = active_trade['TP'], 'Take Profit'
            
            if exit_reason:
                # Apply slippage to exit
                actual_exit_price = exit_price_raw
                slippage_on_exit = 0
                if active_trade['Type'] == 'Long': # Selling to exit
                    actual_exit_price -= slippage_points
                    slippage_on_exit = -slippage_points
                elif active_trade['Type'] == 'Short': # Buying to cover
                    actual_exit_price += slippage_points
                    slippage_on_exit = slippage_points

                # Calculate P&L before commission
                pnl_before_commission = (actual_exit_price - active_trade['ActualEntryPrice']) * active_trade['PositionSize'] \
                                        if active_trade['Type'] == 'Long' else \
                                        (active_trade['ActualEntryPrice'] - actual_exit_price) * active_trade['PositionSize']

                # Calculate commission (applied per side, so entry commission already accounted for if any)
                # Here we calculate exit commission. If commission_rate is for round-trip, adjust logic.
                # Assuming commission_rate is per side for Fixed and Percentage types.
                commission_on_exit = 0
                if commission_type == "Fixed per Trade":
                    commission_on_exit = commission_rate 
                elif commission_type == "Percentage of Trade Value":
                    exit_trade_value = actual_exit_price * active_trade['PositionSize']
                    commission_on_exit = abs(exit_trade_value * commission_rate) # commission_rate is already decimal

                total_commission_for_trade = active_trade.get('CommissionOnEntry', 0) + commission_on_exit
                pnl_after_commission = pnl_before_commission - commission_on_exit # Only subtract exit commission here

                current_capital += pnl_after_commission # Capital was already adjusted for entry commission
                
                active_trade.update({
                    'ExitPrice': actual_exit_price, 
                    'ExitTime': timestamp, 
                    'P&L': pnl_after_commission + active_trade.get('PnlAdjustEntryComm',0), # Total P&L for the trade
                    'RawP&L': pnl_before_commission + active_trade.get('RawPnlAdjustEntryComm',0), # Total Raw P&L
                    'ExitReason': exit_reason,
                    'SlippageOnExit': slippage_on_exit,
                    'CommissionOnExit': commission_on_exit,
                    'TotalCommission': total_commission_for_trade
                })
                trades_log.append(active_trade.copy())
                equity_points[timestamp] = current_capital
                logger.debug(f"Trade closed: {active_trade['Type']} at {actual_exit_price:.2f} ({exit_reason}). Net P&L: {active_trade['P&L']:.2f}. Capital: {current_capital:.2f}")
                active_trade = None

        # --- Check for new entries ---
        if not active_trade and timestamp in signals.index:
            signal_data = signals.loc[signals.index == timestamp] # Use boolean indexing for robustness
            if isinstance(signal_data, pd.DataFrame) and not signal_data.empty: 
                signal = signal_data.iloc[0] # Take the first signal if multiple at same timestamp
            elif isinstance(signal_data, pd.Series):
                signal = signal_data
            else:
                continue # No valid signal

            risk_amount_trade = current_capital * (risk_per_trade_percent / 100.0)
            if stop_loss_points_config <= 0:
                logger.error(f"Stop loss points for sizing ({stop_loss_points_config}) is non-positive. Skipping trade at {timestamp}.")
                continue
            
            position_size = risk_amount_trade / stop_loss_points_config
            if position_size <= 0:
                logger.warning(f"Calculated position size is non-positive ({position_size:.4f}). Skipping trade at {timestamp}.")
                continue

            # Apply slippage to entry
            entry_price_signal = signal['EntryPrice']
            actual_entry_price = entry_price_signal
            slippage_on_entry = 0
            if signal['SignalType'] == 'Long': # Buying to enter
                actual_entry_price += slippage_points
                slippage_on_entry = slippage_points
            elif signal['SignalType'] == 'Short': # Selling to enter
                actual_entry_price -= slippage_points
                slippage_on_entry = -slippage_points # Negative for short entry slippage

            # Calculate commission on entry
            commission_on_entry = 0
            if commission_type == "Fixed per Trade":
                commission_on_entry = commission_rate
            elif commission_type == "Percentage of Trade Value":
                entry_trade_value = actual_entry_price * position_size
                commission_on_entry = abs(entry_trade_value * commission_rate) # commission_rate is decimal

            # Adjust capital immediately for entry commission
            current_capital -= commission_on_entry
            equity_points[timestamp] = current_capital # Log equity change due to entry commission

            active_trade = {
                'EntryTime': timestamp, 
                'SignalEntryPrice': entry_price_signal,
                'ActualEntryPrice': actual_entry_price, 
                'Type': signal['SignalType'],
                'SL': signal['SL'], # SL/TP are based on signal price, not actual fill
                'TP': signal['TP'], 
                'PositionSize': position_size,
                'InitialRiskAmount': risk_amount_trade, 
                'StopLossPointsSizing': stop_loss_points_config,
                'SlippageOnEntry': slippage_on_entry,
                'CommissionOnEntry': commission_on_entry,
                'PnlAdjustEntryComm': -commission_on_entry, # P&L impact of entry commission
                'RawPnlAdjustEntryComm': 0 # Raw P&L is not affected by entry commission directly, but net P&L is
            }
            logger.debug(f"Trade opened: {active_trade['Type']} at signal {entry_price_signal:.2f} (actual {actual_entry_price:.2f}) on {timestamp}. PosSize: {position_size:.4f}. Entry Comm: {commission_on_entry:.2f}. Capital after Entry Comm: {current_capital:.2f}")
    
    # --- Handle trade open at end of data ---
    if active_trade:
        last_bar_close = price_data['Close'].iloc[-1]
        # Apply slippage to this forced exit
        actual_exit_price_eod = last_bar_close
        slippage_on_eod_exit = 0
        if active_trade['Type'] == 'Long':
            actual_exit_price_eod -= slippage_points
            slippage_on_eod_exit = -slippage_points
        elif active_trade['Type'] == 'Short':
            actual_exit_price_eod += slippage_points
            slippage_on_eod_exit = slippage_points
        
        pnl_before_commission_eod = (actual_exit_price_eod - active_trade['ActualEntryPrice']) * active_trade['PositionSize'] \
                                    if active_trade['Type'] == 'Long' else \
                                    (active_trade['ActualEntryPrice'] - actual_exit_price_eod) * active_trade['PositionSize']

        commission_on_eod_exit = 0
        if commission_type == "Fixed per Trade":
            commission_on_eod_exit = commission_rate
        elif commission_type == "Percentage of Trade Value":
            exit_trade_value_eod = actual_exit_price_eod * active_trade['PositionSize']
            commission_on_eod_exit = abs(exit_trade_value_eod * commission_rate)

        total_commission_for_trade_eod = active_trade.get('CommissionOnEntry', 0) + commission_on_eod_exit
        pnl_after_commission_eod = pnl_before_commission_eod - commission_on_eod_exit

        current_capital += pnl_after_commission_eod # Capital was adjusted for entry commission

        active_trade.update({
            'ExitPrice': actual_exit_price_eod, 
            'ExitTime': price_data.index[-1], 
            'P&L': pnl_after_commission_eod + active_trade.get('PnlAdjustEntryComm',0),
            'RawP&L': pnl_before_commission_eod + active_trade.get('RawPnlAdjustEntryComm',0),
            'ExitReason': 'End of Data',
            'SlippageOnExit': slippage_on_eod_exit,
            'CommissionOnExit': commission_on_eod_exit,
            'TotalCommission': total_commission_for_trade_eod
        })
        trades_log.append(active_trade.copy())
        equity_points[price_data.index[-1]] = current_capital
        logger.debug(f"Trade closed at EOD: {active_trade['Type']} at {actual_exit_price_eod:.2f}. Net P&L: {active_trade['P&L']:.2f}. Capital: {current_capital:.2f}")

    trades_df = pd.DataFrame(trades_log)
    if not trades_df.empty:
        for col in ['EntryTime', 'ExitTime']:
            if col in trades_df.columns:
                 trades_df[col] = pd.to_datetime(trades_df[col])

    # Construct final equity series
    if not price_data.empty:
        if price_data.index[-1] not in equity_points: # Ensure last capital point is there
            equity_points[price_data.index[-1]] = current_capital
        
        equity_series_raw = pd.Series(equity_points).sort_index()
        equity_series = equity_series_raw.reindex(price_data.index, method='ffill')
        
        if equity_series.empty or pd.isna(equity_series.iloc[0]):
             temp_equity = pd.Series(index=price_data.index, dtype=float)
             temp_equity.iloc[0] = initial_capital # Start with initial capital
             # If equity_points has data (e.g. commissions changed capital before first trade bar)
             # we need to merge carefully or ensure the first point reflects this.
             # For now, this covers the case of no trades or trades starting later.
             equity_series = temp_equity.ffill() 
             if not equity_series_raw.empty:
                 equity_series.update(equity_series_raw) 
                 equity_series = equity_series.ffill()   

        if not equity_series.empty and equity_series.index.min() == price_data.index.min() and pd.isna(equity_series.iloc[0]):
            # If the very first point is still NaN after reindex/ffill, it means no trades/commissions affected it.
            # It should be initial_capital.
            # However, if commissions were applied at t0, equity_points[t0] would exist.
            # This handles the case where price_data starts, and no trades/commissions happen on the first bar.
            first_price_data_timestamp = price_data.index.min()
            if first_price_data_timestamp not in equity_points: # If no explicit equity point at t0
                 equity_series.loc[first_price_data_timestamp] = initial_capital
                 equity_series = equity_series.sort_index().ffill()


    else: 
        equity_series = pd.Series([initial_capital], index=[pd.Timestamp('1970-01-01', tz='UTC')], dtype=float)
    
    equity_series.name = "Equity"

    # Calculate performance metrics
    performance = {
        'Final Capital': current_capital, 'Total Trades': 0, 'Total P&L': 0.0, 
        'Win Rate': 0.0, 'Profit Factor': 0.0, 'Max Drawdown (%)': 0.0,
        'Average Trade P&L': np.nan, 'Average Winning Trade': np.nan, 'Average Losing Trade': np.nan,
        'Total Commissions Paid': 0.0, 'Total Slippage Impact': 0.0
    }
    if not trades_df.empty:
        performance['Total Trades'] = len(trades_df)
        performance['Total P&L'] = trades_df['P&L'].sum() # Net P&L
        
        # Gross Profit and Loss based on Raw P&L (before exit commission, but after slippage)
        # To calculate Profit Factor more traditionally (gross profit / gross loss before any costs)
        # one might use a P&L column that excludes all commissions.
        # For now, using the P&L column that includes entry commission impact but not exit.
        # Let's use 'RawP&L' which is P&L after slippage but before any commissions.
        if 'RawP&L' in trades_df.columns:
            gross_profit_raw = trades_df[trades_df['RawP&L'] > 0]['RawP&L'].sum()
            gross_loss_raw = abs(trades_df[trades_df['RawP&L'] < 0]['RawP&L'].sum())
        else: # Fallback if RawP&L is not there (should be)
            gross_profit_raw = trades_df[trades_df['P&L'] > 0]['P&L'].sum() # This would be net if RawP&L missing
            gross_loss_raw = abs(trades_df[trades_df['P&L'] < 0]['P&L'].sum())


        performance['Winning Trades'] = len(trades_df[trades_df['P&L'] > 0])
        performance['Losing Trades'] = len(trades_df[trades_df['P&L'] < 0])
        performance['Win Rate'] = (performance['Winning Trades'] / performance['Total Trades'] * 100) if performance['Total Trades'] > 0 else 0.0
        
        performance['Profit Factor'] = (gross_profit_raw / gross_loss_raw) if gross_loss_raw > 0 else np.inf if gross_profit_raw > 0 else 0.0
        
        performance['Average Trade P&L'] = trades_df['P&L'].mean() if performance['Total Trades'] > 0 else np.nan
        performance['Average Winning Trade'] = trades_df[trades_df['P&L'] > 0]['P&L'].mean() if performance['Winning Trades'] > 0 else np.nan
        performance['Average Losing Trade'] = trades_df[trades_df['P&L'] < 0]['P&L'].mean() if performance['Losing Trades'] > 0 else np.nan
        
        performance['Total Commissions Paid'] = trades_df['TotalCommission'].sum() if 'TotalCommission' in trades_df.columns else 0.0
        total_slippage_impact_val = 0
        if 'SlippageOnEntry' in trades_df.columns:
            total_slippage_impact_val += (trades_df['SlippageOnEntry'] * trades_df['PositionSize'] * np.where(trades_df['Type'] == 'Long', -1, 1)).sum()
        if 'SlippageOnExit' in trades_df.columns:
            total_slippage_impact_val += (trades_df['SlippageOnExit'] * trades_df['PositionSize'] * np.where(trades_df['Type'] == 'Long', -1, 1)).sum()
        performance['Total Slippage Impact'] = total_slippage_impact_val


        if not equity_series.empty and equity_series.notna().any():
            cumulative_max = equity_series.cummax()
            drawdown = (equity_series - cumulative_max) / cumulative_max
            drawdown.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero if equity hits 0
            mdd_val = drawdown.min() * 100
            performance['Max Drawdown (%)'] = mdd_val if pd.notna(mdd_val) and not drawdown.empty else 0.0
        else: 
            performance['Max Drawdown (%)'] = 0.0
    
    logger.info(f"Backtest complete. Final Capital: {current_capital:,.2f}. Net P&L: {performance.get('Total P&L', 0):,.2f}. Trades: {performance['Total Trades']}.")
    return trades_df, equity_series, performance
