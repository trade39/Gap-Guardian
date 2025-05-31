# services/backtester.py
"""
Executes the backtest by simulating trades based on generated signals
and calculating profit/loss, equity curve, and advanced performance metrics for reporting.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from utils.logger import get_logger
from config import settings 
from services.optimizer.metrics_calculator import _calculate_daily_returns, calculate_sharpe_ratio 

logger = get_logger(__name__)

def _parse_interval_to_timedelta(interval_str: str) -> pd.Timedelta | None:
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

def _calculate_consecutive_stats(trades_df: pd.DataFrame):
    if trades_df.empty:
        return 0, 0.0, 0, 0.0, 0, 0 # max_consecutive_wins, max_consecutive_profit, max_consecutive_losses, max_consecutive_loss_sum, avg_consecutive_wins, avg_consecutive_losses

    wins_streaks = []
    losses_streaks = []
    current_win_streak = 0
    current_loss_streak = 0
    current_win_streak_pnl = 0.0
    current_loss_streak_pnl = 0.0
    
    all_win_streaks_counts = []
    all_loss_streaks_counts = []

    for _, trade in trades_df.iterrows():
        if trade['P&L'] > 0:
            current_win_streak += 1
            current_win_streak_pnl += trade['P&L']
            if current_loss_streak > 0:
                losses_streaks.append({'count': current_loss_streak, 'pnl': current_loss_streak_pnl})
                all_loss_streaks_counts.append(current_loss_streak)
            current_loss_streak = 0
            current_loss_streak_pnl = 0.0
        elif trade['P&L'] < 0:
            current_loss_streak += 1
            current_loss_streak_pnl += trade['P&L'] # P&L is negative
            if current_win_streak > 0:
                wins_streaks.append({'count': current_win_streak, 'pnl': current_win_streak_pnl})
                all_win_streaks_counts.append(current_win_streak)
            current_win_streak = 0
            current_win_streak_pnl = 0.0
        # else: P&L is 0, streak continues as is or resets both, depending on definition.
        # For simplicity, break streaks on zero P&L too.
        else: 
            if current_win_streak > 0:
                wins_streaks.append({'count': current_win_streak, 'pnl': current_win_streak_pnl})
                all_win_streaks_counts.append(current_win_streak)
            if current_loss_streak > 0:
                losses_streaks.append({'count': current_loss_streak, 'pnl': current_loss_streak_pnl})
                all_loss_streaks_counts.append(current_loss_streak)
            current_win_streak = 0
            current_win_streak_pnl = 0.0
            current_loss_streak = 0
            current_loss_streak_pnl = 0.0


    # Append final streak
    if current_win_streak > 0:
        wins_streaks.append({'count': current_win_streak, 'pnl': current_win_streak_pnl})
        all_win_streaks_counts.append(current_win_streak)
    if current_loss_streak > 0:
        losses_streaks.append({'count': current_loss_streak, 'pnl': current_loss_streak_pnl})
        all_loss_streaks_counts.append(current_loss_streak)

    max_consecutive_wins = max(s['count'] for s in wins_streaks) if wins_streaks else 0
    max_consecutive_profit = max(s['pnl'] for s in wins_streaks) if wins_streaks else 0.0
    max_consecutive_losses = max(s['count'] for s in losses_streaks) if losses_streaks else 0
    max_consecutive_loss_sum = min(s['pnl'] for s in losses_streaks) if losses_streaks else 0.0 # Sum of losses, so it's negative

    avg_consecutive_wins = np.mean(all_win_streaks_counts) if all_win_streaks_counts else 0
    avg_consecutive_losses = np.mean(all_loss_streaks_counts) if all_loss_streaks_counts else 0
    
    return max_consecutive_wins, max_consecutive_profit, max_consecutive_losses, max_consecutive_loss_sum, int(round(avg_consecutive_wins)), int(round(avg_consecutive_losses))


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
    Runs the backtest simulation including commission, slippage, and advanced performance metrics.
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
                pnl_this_event = pnl_before_commission - commission_on_exit 
                current_capital += pnl_this_event
                
                active_trade.update({
                    'ExitPrice': actual_exit_price, 
                    'ExitTime': timestamp, 
                    'P&L': pnl_before_commission - total_commission_for_trade, 
                    'RawP&L': pnl_before_commission, 
                    'ExitReason': exit_reason,
                    'SlippageOnExit': slippage_on_exit,
                    'CommissionOnExit': commission_on_exit,
                    'TotalCommission': total_commission_for_trade
                })
                trades_log.append(active_trade.copy())
                equity_points[timestamp] = current_capital
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
                continue
            
            position_size = risk_amount_trade / stop_loss_points_config
            if position_size <= 0:
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

            current_capital -= commission_on_entry
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
            }
    
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
            'P&L': pnl_before_commission_eod - total_commission_for_trade_eod, 
            'RawP&L': pnl_before_commission_eod, 
            'ExitReason': 'End of Data',
            'SlippageOnExit': slippage_on_eod_exit,
            'CommissionOnExit': commission_on_eod_exit,
            'TotalCommission': total_commission_for_trade_eod
        })
        trades_log.append(active_trade.copy())
        equity_points[price_data.index[-1]] = current_capital
        
    trades_df = pd.DataFrame(trades_log)
    if not trades_df.empty:
        for col in ['EntryTime', 'ExitTime']:
            if col in trades_df.columns:
                 trades_df[col] = pd.to_datetime(trades_df[col])
        if 'EntryTime' in trades_df.columns and 'ExitTime' in trades_df.columns:
             trades_df['HoldingPeriod'] = trades_df['ExitTime'] - trades_df['EntryTime']
        else:
             trades_df['HoldingPeriod'] = pd.NaT


    if not price_data.empty:
        if not equity_points or price_data.index[-1] not in equity_points:
            equity_points[price_data.index[-1]] = current_capital
        equity_series_raw = pd.Series(equity_points).sort_index()
        equity_series = equity_series_raw.reindex(price_data.index, method='ffill')
        if equity_series.empty or pd.isna(equity_series.iloc[0]):
             first_timestamp = price_data.index.min()
             equity_series.loc[first_timestamp] = initial_capital
             equity_series = equity_series.ffill() 
             if not equity_series_raw.empty:
                 equity_series.update(equity_series_raw) 
                 equity_series = equity_series.ffill()   
    else: 
        equity_series = pd.Series([initial_capital], index=[pd.Timestamp('1970-01-01', tz='UTC')], dtype=float)
    equity_series.name = "Equity"

    performance = {
        'Initial Capital': initial_capital, 'Final Capital': current_capital, 
        'Total Trades': 0, 'Total Net P&L': 0.0, 'Gross Profit': 0.0, 'Gross Loss': 0.0,
        'Win Rate': 0.0, 'Profit Factor': 0.0, 'Max Drawdown (%)': 0.0, 'Max Drawdown ($)': 0.0,
        'Average Trade P&L': np.nan, 'Average Winning Trade': np.nan, 'Average Losing Trade': np.nan,
        'Total Commissions Paid': 0.0, 'Total Slippage Impact (approx)': 0.0,
        'Sharpe Ratio (Annualized)': np.nan, 'Expected Value': np.nan, 'Recovery Factor': np.nan,
        'Short Trades': 0, 'Short Trades Won': 0, 'Long Trades': 0, 'Long Trades Won': 0,
        'Largest Profit Trade': np.nan, 'Largest Loss Trade': np.nan,
        'Max Consecutive Wins': 0, 'Max Consecutive Wins ($)': 0.0,
        'Max Consecutive Losses': 0, 'Max Consecutive Losses ($)': 0.0,
        'Avg Consecutive Wins': 0, 'Avg Consecutive Losses': 0,
        'Min Position Holding Time': pd.NaT, 'Max Position Holding Time': pd.NaT, 'Avg Position Holding Time': pd.NaT
    }

    if not trades_df.empty:
        performance['Total Trades'] = len(trades_df)
        performance['Total Net P&L'] = trades_df['P&L'].sum()
        
        # Gross Profit/Loss using RawP&L (after slippage, before individual trade commissions)
        # This is closer to how MT4 might define it before explicit commission costs.
        raw_pnl_for_gross_col = 'RawP&L' if 'RawP&L' in trades_df.columns else 'P&L'
        gross_profit = trades_df[trades_df[raw_pnl_for_gross_col] > 0][raw_pnl_for_gross_col].sum()
        gross_loss = abs(trades_df[trades_df[raw_pnl_for_gross_col] < 0][raw_pnl_for_gross_col].sum()) # Absolute value for gross loss sum
        performance['Gross Profit'] = gross_profit
        performance['Gross Loss'] = gross_loss # Store as positive value

        winning_trades_df = trades_df[trades_df['P&L'] > 0]
        losing_trades_df = trades_df[trades_df['P&L'] < 0]
        performance['Winning Trades'] = len(winning_trades_df)
        performance['Losing Trades'] = len(losing_trades_df)
        performance['Win Rate'] = (performance['Winning Trades'] / performance['Total Trades'] * 100) if performance['Total Trades'] > 0 else 0.0
        performance['Profit Factor'] = (gross_profit / gross_loss) if gross_loss > 0 else np.inf if gross_profit > 0 else 0.0
        
        avg_win_amount = winning_trades_df['P&L'].mean() if not winning_trades_df.empty else 0.0
        avg_loss_amount = abs(losing_trades_df['P&L'].mean()) if not losing_trades_df.empty else 0.0 # Absolute for calculation
        
        performance['Average Trade P&L'] = trades_df['P&L'].mean()
        performance['Average Winning Trade'] = avg_win_amount
        performance['Average Losing Trade'] = -avg_loss_amount # Display as negative

        if performance['Total Trades'] > 0:
            win_rate_decimal = performance['Winning Trades'] / performance['Total Trades']
            loss_rate_decimal = performance['Losing Trades'] / performance['Total Trades']
            performance['Expected Value'] = (win_rate_decimal * avg_win_amount) - (loss_rate_decimal * avg_loss_amount)
        
        performance['Total Commissions Paid'] = trades_df['TotalCommission'].sum() if 'TotalCommission' in trades_df.columns else 0.0
        total_slippage_cost = 0
        if 'SlippageOnEntry' in trades_df.columns and 'PositionSize' in trades_df.columns:
            total_slippage_cost -= (trades_df['SlippageOnEntry'] * trades_df['PositionSize']).sum()
        if 'SlippageOnExit' in trades_df.columns and 'PositionSize' in trades_df.columns:
            total_slippage_cost -= (trades_df['SlippageOnExit'] * trades_df['PositionSize']).sum()
        performance['Total Slippage Impact (approx)'] = total_slippage_cost

        abs_max_drawdown_value = 0.0
        if not equity_series.empty and equity_series.notna().any():
            cumulative_max_equity = equity_series.cummax()
            drawdown_series_val = cumulative_max_equity - equity_series
            abs_max_drawdown_value = drawdown_series_val.max() if drawdown_series_val.notna().any() and not drawdown_series_val.empty else 0.0
            performance['Max Drawdown ($)'] = abs_max_drawdown_value
            
            drawdown_percent_series = (equity_series - cumulative_max_equity) / cumulative_max_equity
            drawdown_percent_series.replace([np.inf, -np.inf], np.nan, inplace=True)
            mdd_percent_val = drawdown_percent_series.min() * 100
            performance['Max Drawdown (%)'] = mdd_percent_val if pd.notna(mdd_percent_val) and not drawdown_percent_series.empty else 0.0
        
        if abs_max_drawdown_value > 0 and performance['Total Net P&L'] > 0 : 
            performance['Recovery Factor'] = performance['Total Net P&L'] / abs_max_drawdown_value
        elif performance['Total Net P&L'] > 0 and abs_max_drawdown_value == 0:
             performance['Recovery Factor'] = np.inf
        else:
            performance['Recovery Factor'] = 0.0

        daily_returns = _calculate_daily_returns(equity_series.copy())
        performance['Sharpe Ratio (Annualized)'] = calculate_sharpe_ratio(
            daily_returns, risk_free_rate=settings.RISK_FREE_RATE, trading_days_per_year=settings.TRADING_DAYS_PER_YEAR
        )

        performance['Short Trades'] = len(trades_df[trades_df['Type'] == 'Short'])
        performance['Short Trades Won'] = len(trades_df[(trades_df['Type'] == 'Short') & (trades_df['P&L'] > 0)])
        performance['Long Trades'] = len(trades_df[trades_df['Type'] == 'Long'])
        performance['Long Trades Won'] = len(trades_df[(trades_df['Type'] == 'Long') & (trades_df['P&L'] > 0)])
        performance['Largest Profit Trade'] = trades_df['P&L'].max() if performance['Winning Trades'] > 0 else 0.0
        performance['Largest Loss Trade'] = trades_df['P&L'].min() if performance['Losing Trades'] > 0 else 0.0
        
        (mcw, mcwp, mcl, mclp, acw, acl) = _calculate_consecutive_stats(trades_df)
        performance['Max Consecutive Wins'] = mcw
        performance['Max Consecutive Wins ($)'] = mcwp
        performance['Max Consecutive Losses'] = mcl
        performance['Max Consecutive Losses ($)'] = mclp # This will be negative
        performance['Avg Consecutive Wins'] = acw
        performance['Avg Consecutive Losses'] = acl

        if 'HoldingPeriod' in trades_df.columns and trades_df['HoldingPeriod'].notna().any():
            performance['Min Position Holding Time'] = trades_df['HoldingPeriod'].min()
            performance['Max Position Holding Time'] = trades_df['HoldingPeriod'].max()
            performance['Avg Position Holding Time'] = trades_df['HoldingPeriod'].mean()
    
    logger.info(f"Backtest complete. Final Capital: {current_capital:,.2f}. Net P&L: {performance.get('Total Net P&L', 0):,.2f}. Trades: {performance['Total Trades']}.")
    logger.info(f"Extended Metrics: Sharpe={performance.get('Sharpe Ratio (Annualized)', np.nan):.2f}, Expectancy=${performance.get('Expected Value', np.nan):.2f}, RecoveryFactor={performance.get('Recovery Factor', np.nan):.2f}")
    return trades_df, equity_series, performance
