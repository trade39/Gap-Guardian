# services/strategy_engine.py
"""
Implements the core logic for the Gap Guardian trading strategy.
Identifies entry signals based on price action within a specified time window,
adapting to the data's timeframe.
"""
import pandas as pd
import numpy as np
from datetime import time as dt_time # Explicitly import time as dt_time
from config.settings import NY_TIMEZONE # Removed unused specific hour/minute imports
from utils.logger import get_logger

logger = get_logger(__name__)

def generate_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float,
    entry_start_time: dt_time, # Pass datetime.time object
    entry_end_time: dt_time    # Pass datetime.time object
) -> pd.DataFrame:
    """
    Generates trading signals based on the Gap Guardian strategy.

    Args:
        data (pd.DataFrame): Price data (OHLC) indexed by Datetime (must be NY timezone).
        stop_loss_points (float): Stop loss distance in price points.
        rrr (float): Risk-Reward Ratio.
        entry_start_time (datetime.time): The start of the entry window (NY time).
        entry_end_time (datetime.time): The end of the entry window (NY time).
                                         Signals can form on bars starting *before* this time.
    Returns:
        pd.DataFrame: DataFrame containing trade signals.
    """
    if data.empty:
        logger.warning("Input data for signal generation is empty.")
        return pd.DataFrame()

    if data.index.tz != NY_TIMEZONE:
        logger.error(f"Data must be in New York timezone ({NY_TIMEZONE}). Current: {data.index.tz}")
        return pd.DataFrame()

    logger.info(f"Generating signals with SL points: {stop_loss_points}, RRR: {rrr}, "
                f"Entry Window: {entry_start_time.strftime('%H:%M')}-{entry_end_time.strftime('%H:%M')} NYT.")

    signals_list = []

    # Group data by date to process one day at a time
    for date_val, day_data in data.groupby(data.index.date):
        if day_data.empty:
            continue
        
        # Find the "opening bar" for the strategy:
        # This is the first bar of the day_data whose start time is >= entry_start_time.
        opening_bar_candidates = day_data[day_data.index.time >= entry_start_time]
        if opening_bar_candidates.empty:
            # logger.debug(f"No data at or after entry_start_time {entry_start_time.strftime('%H:%M')} on {date_val}. Skipping day.")
            continue
        
        opening_bar_data = opening_bar_candidates.iloc[0:1] # Get the first such bar
        if opening_bar_data.empty: # Should not happen if candidates were not empty
            logger.warning(f"Could not determine opening bar for {date_val} at or after {entry_start_time.strftime('%H:%M')}.")
            continue
            
        opening_bar_timestamp = opening_bar_data.index[0]
        opening_range_high = opening_bar_data['High'].iloc[0]
        opening_range_low = opening_bar_data['Low'].iloc[0]
        
        # logger.debug(f"Day {date_val}: Opening bar at {opening_bar_timestamp.strftime('%H:%M')}, H: {opening_range_high}, L: {opening_range_low}")

        # Iterate through bars within the entry window (after the opening bar)
        # Signal scan window: bars starting *after* the opening bar's start time,
        # AND starting *before* the entry_end_time.
        signal_scan_window_data = day_data[
            (day_data.index > opening_bar_timestamp) &          # Strictly after the opening bar
            (day_data.index.time < entry_end_time)              # Bar starts before the window closes
        ]

        if signal_scan_window_data.empty:
            # logger.debug(f"No bars in signal scan window for {date_val} after {opening_bar_timestamp.time()} and before {entry_end_time}.")
            continue

        for idx, bar in signal_scan_window_data.iterrows():
            # idx is the start time of the current bar. Signal is confirmed at its close.
            signal_time = idx # Entry is at the close of this bar (current logic)
            
            # Check for Long Signal (False Breakdown of opening_range_low)
            if bar['Low'] < opening_range_low and bar['Close'] > opening_range_low:
                entry_price = bar['Close']
                sl = entry_price - stop_loss_points
                tp = entry_price + (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': signal_time, 'SignalType': 'Long', 'EntryPrice': entry_price,
                    'SL': sl, 'TP': tp, 'Reason': f"False breakdown of {opening_range_low:.2f}"
                })
                logger.info(f"Long signal on {date_val} at {signal_time.strftime('%H:%M')}: Entry={entry_price:.2f}, SL={sl:.2f}, TP={tp:.2f}")
                break # Max one trade per day

            # Check for Short Signal (False Breakout of opening_range_high)
            elif bar['High'] > opening_range_high and bar['Close'] < opening_range_high:
                entry_price = bar['Close']
                sl = entry_price + stop_loss_points
                tp = entry_price - (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': signal_time, 'SignalType': 'Short', 'EntryPrice': entry_price,
                    'SL': sl, 'TP': tp, 'Reason': f"False breakout of {opening_range_high:.2f}"
                })
                logger.info(f"Short signal on {date_val} at {signal_time.strftime('%H:%M')}: Entry={entry_price:.2f}, SL={sl:.2f}, TP={tp:.2f}")
                break # Max one trade per day
        
    if not signals_list:
        # logger.info("No signals generated overall.") # This can be too verbose if called many times
        return pd.DataFrame()
        
    signals_df = pd.DataFrame(signals_list)
    signals_df['SignalTime'] = pd.to_datetime(signals_df['SignalTime'])
    signals_df.set_index('SignalTime', inplace=True, drop=False)
    signals_df.sort_index(inplace=True)
    
    logger.info(f"Generated {len(signals_df)} signals in total.")
    return signals_df

if __name__ == '__main__':
    from services.data_loader import fetch_historical_data
    from config.settings import DEFAULT_STOP_LOSS_POINTS, DEFAULT_RRR, DEFAULT_ENTRY_WINDOW_START_HOUR, DEFAULT_ENTRY_WINDOW_START_MINUTE, DEFAULT_ENTRY_WINDOW_END_HOUR, DEFAULT_ENTRY_WINDOW_END_MINUTE
    from datetime import date as dt_date # Alias to avoid conflict

    sample_ticker = "^GSPC"
    start_d = dt_date.today() - pd.Timedelta(days=59)
    end_d = dt_date.today()
    
    # Test with default entry window times
    test_entry_start_time = dt_time(DEFAULT_ENTRY_WINDOW_START_HOUR, DEFAULT_ENTRY_WINDOW_START_MINUTE)
    test_entry_end_time = dt_time(DEFAULT_ENTRY_WINDOW_END_HOUR, DEFAULT_ENTRY_WINDOW_END_MINUTE)

    for tf_display, tf_value in {"15 Minutes": "15m", "1 Hour": "1h", "Daily": "1d"}.items():
        print(f"\n--- Testing {sample_ticker} on {tf_display} ({tf_value}) ---")
        price_data_test = fetch_historical_data(sample_ticker, start_d, end_d, tf_value)
        if not price_data_test.empty:
            print(f"Price data ({len(price_data_test)} rows) from {price_data_test.index.min()} to {price_data_test.index.max()}")
            signals = generate_signals(
                price_data_test,
                DEFAULT_STOP_LOSS_POINTS,
                DEFAULT_RRR,
                test_entry_start_time,
                test_entry_end_time
            )
            if not signals.empty: print(f"Generated Signals:\n{signals.head()}")
            else: print("No signals generated for this timeframe.")
        else: print(f"Could not fetch price data for {tf_value}.")
