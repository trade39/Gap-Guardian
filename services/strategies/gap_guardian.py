# services/strategies/gap_guardian.py
"""
Signal generation logic for the Gap Guardian strategy.
Relies on standard Python import mechanisms, assuming project root is in sys.path.
"""
import pandas as pd
from datetime import time as dt_time

# Project-specific imports - these should work if app.py is run from project root
from utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

def generate_gap_guardian_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float,
    entry_start_time: dt_time,
    entry_end_time: dt_time
) -> pd.DataFrame:
    """
    Generates trading signals based on the Gap Guardian strategy.

    Args:
        data (pd.DataFrame): Price data (OHLC) indexed by Datetime (must be NY timezone).
        stop_loss_points (float): Stop loss distance in price points.
        rrr (float): Risk-Reward Ratio.
        entry_start_time (dt_time): The start of the entry window (NY time).
        entry_end_time (dt_time): The end of the entry window (NY time).

    Returns:
        pd.DataFrame: DataFrame containing trade signals. Columns:
                      'SignalTime', 'SignalType', 'EntryPrice', 'SL', 'TP', 'Reason'.
    """
    signals_list = []
    # Group data by date to process one day at a time
    for date_val, day_data in data.groupby(data.index.date):
        if day_data.empty:
            continue
        
        # Find the "opening bar" for the strategy:
        # This is the first bar that occurs at or after the entry_start_time on that day.
        opening_bar_candidates = day_data[day_data.index.time >= entry_start_time]
        if opening_bar_candidates.empty:
            # No bars on this day at or after the entry start time
            continue
        
        opening_bar_data = opening_bar_candidates.iloc[0:1] # Get the first such bar
        if opening_bar_data.empty: # Should not happen if opening_bar_candidates was not empty
            continue
            
        opening_bar_timestamp = opening_bar_data.index[0]
        opening_range_high = opening_bar_data['High'].iloc[0]
        opening_range_low = opening_bar_data['Low'].iloc[0]
        
        # Iterate through bars within the entry window (after the opening bar)
        # to look for the false break setup.
        signal_scan_window_data = day_data[
            (day_data.index > opening_bar_timestamp) & # Must be after the opening bar
            (day_data.index.time < entry_end_time)     # And before the entry window closes
        ]

        if signal_scan_window_data.empty:
            continue

        for idx, bar in signal_scan_window_data.iterrows():
            signal_time = idx # Timestamp of the current bar being evaluated
            
            # Check for Long Signal:
            # 1. Price breaks below the opening range low.
            # 2. Price then closes back above the opening range low.
            # All within the specified entry window.
            if bar['Low'] < opening_range_low and bar['Close'] > opening_range_low:
                entry_price = bar['Close'] # Enter on the close of the signal bar
                sl = entry_price - stop_loss_points
                tp = entry_price + (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': signal_time,
                    'SignalType': 'Long',
                    'EntryPrice': entry_price,
                    'SL': sl,
                    'TP': tp,
                    'Reason': f"GG: False breakdown of ORL {opening_range_low:.2f}"
                })
                logger.debug(f"GG Long signal on {date_val} at {signal_time.strftime('%H:%M')}: Entry={entry_price:.2f}, ORL={opening_range_low:.2f}, Bar Low={bar['Low']:.2f}, Bar Close={bar['Close']:.2f}")
                break # Max one trade per day for this strategy

            # Check for Short Signal:
            # 1. Price breaks above the opening range high.
            # 2. Price then closes back below the opening range high.
            # All within the specified entry window.
            elif bar['High'] > opening_range_high and bar['Close'] < opening_range_high:
                entry_price = bar['Close'] # Enter on the close of the signal bar
                sl = entry_price + stop_loss_points
                tp = entry_price - (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': signal_time,
                    'SignalType': 'Short',
                    'EntryPrice': entry_price,
                    'SL': sl,
                    'TP': tp,
                    'Reason': f"GG: False breakout of ORH {opening_range_high:.2f}"
                })
                logger.debug(f"GG Short signal on {date_val} at {signal_time.strftime('%H:%M')}: Entry={entry_price:.2f}, ORH={opening_range_high:.2f}, Bar High={bar['High']:.2f}, Bar Close={bar['Close']:.2f}")
                break # Max one trade per day
        
    return pd.DataFrame(signals_list)
