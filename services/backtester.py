# services/backtester.py
"""
Executes the backtest by simulating trades based on generated signals
and calculating profit/loss, equity curve, and advanced performance metrics.
"""
import pandas as pd
import numpy as np
from utils.logger import get_logger
from config import settings # For RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
# Assuming metrics_calculator is accessible for Sharpe Ratio
from services.optimizer.metrics_calculator import _calculate_daily_returns, calculate_sharpe_ratio 

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
    stop_loss_points_config: float, 
    data_interval_str: str,
    commission_type: str = "None", 
    commission_rate: float = 0.0,  
    slippage_points: float = 0.0   
    ) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Runs the backtest simulation including commission, slippage, and advanced metrics.
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
        if active_trade:
            exit_reason, exit_price_raw = None, None
            
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
                actual_exit_price = exit_price_raw
                slippage_on_exit = 0
                if active_trade['Type'] == 'Long': 
                    actual_exit_price -= slippage_points
                    slippage_on_exit = -slippage_points
                elif active_trade['Type'] == 'Short': 
                    actual_exit_price += slippage_points
                    slippage_on_exit = slippage_points

                pnl_before_commission = (actual_exit_price - active_trade['ActualEntryPrice']) * active_trade['PositionSize'] \
                                        if active_trade['Type'] == 'Long' else \
                                        (active_trade['ActualEntryPrice'] - actual_exit_price) * active_trade['PositionSize']

                commission_on_exit = 0
                if commission_type == "Fixed per Trade":
                    commission_on_exit = commission_rate 
                elif commission_type == "Percentage of Trade Value":
                    exit_trade_value = actual_exit_price * active_trade['PositionSize']
                    commission_on_exit = abs(exit_trade_value * commission_rate) 

                total_commission_for_trade = active_trade.get('CommissionOnEntry', 0) + commission_on_exit
                # P&L for this exit event (already accounts for entry commission via current_capital)
                pnl_this_event = pnl_before_commission - commission_on_exit 
                current_capital += pnl_this_event
                
                active_trade.update({
                    'ExitPrice': actual_exit_price, 
                    'ExitTime': timestamp, 
                    'P&L': pnl_before_commission - total_commission_for_trade, # Net P&L for the trade
                    'RawP&L': pnl_before_commission, # P&L after slippage, before this exit's commission
                    'ExitReason': exit_reason,
                    'SlippageOnExit': slippage_on_exit,
                    'CommissionOnExit': commission_on_exit,
                    'TotalCommission': total_commission_for_trade
                })
                trades_log.append(active_trade.copy())
                equity_points[timestamp] = current_capital
                logger.debug(f"Trade closed: {active_trade['Type']} at {actual_exit_price:.2f} ({exit_reason}). Net P&L: {active_trade['P&L']:.2f}. Capital: {current_capital:.2f}")
                active_trade = None

        if not active_trade and timestamp in signals.index:
            signal_data = signals.loc[signals.index == timestamp] 
            if isinstance(signal_data, pd.DataFrame) and not signal_data.empty: 
                signal = signal_data.iloc[0]
            elif isinstance(signal_data, pd.Series):
                signal = signal_data
            else:
                continue

            risk_amount_trade = current_capital * (risk_per_trade_percent / 100.0)
            if stop_loss_points_config <= 0:
                logger.error(f"Stop loss points for sizing ({stop_loss_points_config}) is non-positive. Skipping trade at {timestamp}.")
                continue
            
            position_size = risk_amount_trade / stop_loss_points_config
            if position_size <= 0:
                logger.warning(f"Calculated position size is non-positive ({position_size:.4f}). Skipping trade at {timestamp}.")
                continue

            entry_price_signal = signal['EntryPrice']
            actual_entry_price = entry_price_signal
            slippage_on_entry = 0
            if signal['SignalType'] == 'Long': 
                actual_entry_price += slippage_points
                slippage_on_entry = slippage_points
            elif signal['SignalType'] == 'Short': 
                actual_entry_price -= slippage_points
                slippage_on_entry = -slippage_points 

            commission_on_entry = 0
            if commission_type == "Fixed per Trade":
                commission_on_entry = commission_rate
            elif commission_type == "Percentage of Trade Value":
                entry_trade_value = actual_entry_price * position_size
                commission_on_entry = abs(entry_trade_value * commission_rate)

            current_capital -= commission_on_entry # Deduct entry commission from capital
            equity_points[timestamp] = current_capital 

            active_trade = {
                'EntryTime': timestamp, 
                'SignalEntryPrice': entry_price_signal,
                'ActualEntryPrice': actual_entry_price, 
                'Type': signal['SignalType'],
                'SL': signal['SL'], 
                'TP': signal['TP'], 
                'PositionSize': position_size,
                'InitialRiskAmount': risk_amount_trade, 
                'StopLossPointsSizing': stop_loss_points_config,
                'SlippageOnEntry': slippage_on_entry,
                'CommissionOnEntry': commission_on_entry,
                # 'PnlAdjustEntryComm' and 'RawPnlAdjustEntryComm' are removed as P&L is calculated at exit
            }
            logger.debug(f"Trade opened: {active_trade['Type']} at signal {entry_price_signal:.2f} (actual {actual_entry_price:.2f}) on {timestamp}. PosSize: {position_size:.4f}. Entry Comm: {commission_on_entry:.2f}. Capital after Entry Comm: {current_capital:.2f}")
    
    if active_trade:
        last_bar_close = price_data['Close'].iloc[-1]
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
        pnl_this_event_eod = pnl_before_commission_eod - commission_on_eod_exit
        current_capital += pnl_this_event_eod

        active_trade.update({
            'ExitPrice': actual_exit_price_eod, 
            'ExitTime': price_data.index[-1], 
            'P&L': pnl_before_commission_eod - total_commission_for_trade_eod, # Net P&L for the trade
            'RawP&L': pnl_before_commission_eod, # P&L after slippage, before this exit's commission
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

    if not price_data.empty:
        if not equity_points or price_data.index[-1] not in equity_points:
            equity_points[price_data.index[-1]] = current_capital
        
        equity_series_raw = pd.Series(equity_points).sort_index()
        equity_series = equity_series_raw.reindex(price_data.index, method='ffill')
        
        if equity_series.empty or pd.isna(equity_series.iloc[0]):
             first_timestamp = price_data.index.min()
             equity_series.loc[first_timestamp] = initial_capital # Ensure it starts with initial capital
             equity_series = equity_series.ffill() 
             if not equity_series_raw.empty: # Re-apply actual points if they existed
                 equity_series.update(equity_series_raw) 
                 equity_series = equity_series.ffill()
    else: 
        equity_series = pd.Series([initial_capital], index=[pd.Timestamp('1970-01-01', tz='UTC')], dtype=float)
    
    equity_series.name = "Equity"

    # --- Performance Metrics Calculation ---
    performance = {
        'Final Capital': current_capital, 'Total Trades': 0, 'Total P&L': 0.0, 
        'Win Rate': 0.0, 'Profit Factor': 0.0, 'Max Drawdown (%)': 0.0,
        'Average Trade P&L': np.nan, 'Average Winning Trade': np.nan, 'Average Losing Trade': np.nan,
        'Total Commissions Paid': 0.0, 'Total Slippage Impact (approx)': 0.0, # Renamed
        'Sharpe Ratio (Annualized)': np.nan, 'Expected Value': np.nan, 'Recovery Factor': np.nan
    }

    if not trades_df.empty:
        performance['Total Trades'] = len(trades_df)
        performance['Total P&L'] = trades_df['P&L'].sum() # Net P&L after all costs
        
        # Use RawP&L (after slippage, before exit commission) for Profit Factor calculation if available
        # This gives a sense of edge before per-trade fixed costs.
        raw_pnl_col = 'RawP&L' if 'RawP&L' in trades_df.columns else 'P&L' # Fallback to P&L
        gross_profit_raw = trades_df[trades_df[raw_pnl_col] > 0][raw_pnl_col].sum()
        gross_loss_raw = abs(trades_df[trades_df[raw_pnl_col] < 0][raw_pnl_col].sum())

        performance['Winning Trades'] = len(trades_df[trades_df['P&L'] > 0])
        performance['Losing Trades'] = len(trades_df[trades_df['P&L'] < 0])
        performance['Win Rate'] = (performance['Winning Trades'] / performance['Total Trades'] * 100) if performance['Total Trades'] > 0 else 0.0
        
        performance['Profit Factor'] = (gross_profit_raw / gross_loss_raw) if gross_loss_raw > 0 else np.inf if gross_profit_raw > 0 else 0.0
        
        avg_win_amount = trades_df[trades_df['P&L'] > 0]['P&L'].mean() if performance['Winning Trades'] > 0 else 0.0
        avg_loss_amount = abs(trades_df[trades_df['P&L'] < 0]['P&L'].mean()) if performance['Losing Trades'] > 0 else 0.0
        
        performance['Average Trade P&L'] = trades_df['P&L'].mean() if performance['Total Trades'] > 0 else np.nan
        performance['Average Winning Trade'] = avg_win_amount
        performance['Average Losing Trade'] = -avg_loss_amount # Typically displayed as negative

        if performance['Total Trades'] > 0:
            win_rate_decimal = performance['Win Rate'] / 100.0
            loss_rate_decimal = 1.0 - win_rate_decimal if performance['Total Trades'] > performance['Winning Trades'] + performance['Losing Trades'] else (performance['Losing Trades'] / performance['Total Trades'])
            performance['Expected Value'] = (win_rate_decimal * avg_win_amount) - (loss_rate_decimal * avg_loss_amount)
        
        performance['Total Commissions Paid'] = trades_df['TotalCommission'].sum() if 'TotalCommission' in trades_df.columns else 0.0
        
        # Approximate total slippage cost
        total_slippage_cost = 0
        if 'SlippageOnEntry' in trades_df.columns and 'PositionSize' in trades_df.columns:
            # Slippage cost is value of slippage_points * position_size
            # For long entry, positive slippage_points means higher entry price (cost)
            # For short entry, negative slippage_points means lower entry price (cost)
            total_slippage_cost -= (trades_df['SlippageOnEntry'] * trades_df['PositionSize']).sum()
        if 'SlippageOnExit' in trades_df.columns and 'PositionSize' in trades_df.columns:
            # For long exit, negative slippage_points means lower exit price (cost)
            # For short exit, positive slippage_points means higher exit price (cost)
            total_slippage_cost -= (trades_df['SlippageOnExit'] * trades_df['PositionSize']).sum()
        performance['Total Slippage Impact (approx)'] = total_slippage_cost


        # Max Drawdown Calculation (from equity curve)
        abs_max_drawdown_value = 0.0
        if not equity_series.empty and equity_series.notna().any():
            cumulative_max_equity = equity_series.cummax()
            drawdown_series = cumulative_max_equity - equity_series # Absolute drawdown values
            abs_max_drawdown_value = drawdown_series.max() if drawdown_series.notna().any() and not drawdown_series.empty else 0.0
            
            drawdown_percent_series = (equity_series - cumulative_max_equity) / cumulative_max_equity
            drawdown_percent_series.replace([np.inf, -np.inf], np.nan, inplace=True)
            mdd_percent_val = drawdown_percent_series.min() * 100
            performance['Max Drawdown (%)'] = mdd_percent_val if pd.notna(mdd_percent_val) and not drawdown_percent_series.empty else 0.0
        else: 
            performance['Max Drawdown (%)'] = 0.0
            abs_max_drawdown_value = 0.0 # Should be initial_capital if P&L is negative

        if abs_max_drawdown_value > 0 and performance['Total P&L'] > 0 : # Avoid division by zero or meaningless ratio
            performance['Recovery Factor'] = performance['Total P&L'] / abs_max_drawdown_value
        elif performance['Total P&L'] > 0 and abs_max_drawdown_value == 0: # Positive P&L with no drawdown
             performance['Recovery Factor'] = np.inf
        else:
            performance['Recovery Factor'] = 0.0


        # Sharpe Ratio Calculation
        daily_returns = _calculate_daily_returns(equity_series.copy()) # ensure copy
        performance['Sharpe Ratio (Annualized)'] = calculate_sharpe_ratio(
            daily_returns, 
            risk_free_rate=settings.RISK_FREE_RATE, 
            trading_days_per_year=settings.TRADING_DAYS_PER_YEAR
        )
    
    logger.info(f"Backtest complete. Final Capital: {current_capital:,.2f}. Net P&L: {performance.get('Total P&L', 0):,.2f}. Trades: {performance['Total Trades']}.")
    logger.info(f"New Metrics: Sharpe={performance.get('Sharpe Ratio (Annualized)', np.nan):.2f}, Expectancy=${performance.get('Expected Value', np.nan):.2f}, RecoveryFactor={performance.get('Recovery Factor', np.nan):.2f}")
    return trades_df, equity_series, performance
