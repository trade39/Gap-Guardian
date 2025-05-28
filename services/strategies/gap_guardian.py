# services/strategies/gap_guardian.py
"""
Signal generation logic for the Gap Guardian strategy.
Relies on standard Python import mechanisms, assuming project root is in sys.path.
"""
import sys
import os
import pandas as pd
from datetime import time as dt_time

# --- sys.path diagnostics for gap_guardian.py ---
GAP_GUARDIAN_FILE_PATH = os.path.abspath(__file__)
print(f"--- [DEBUG gap_guardian.py @ {os.path.basename(GAP_GUARDIAN_FILE_PATH)}] ---")
print(f"    GAP_GUARDIAN_FILE_PATH: {GAP_GUARDIAN_FILE_PATH}")
print(f"    sys.path as seen by gap_guardian.py: {sys.path}")
# We expect the project root (e.g., '/mount/src/gap-guardian') to be in sys.path, added by app.py.
project_root_candidate_gg = None
for path_entry in sys.path:
    if os.path.isdir(os.path.join(path_entry, 'utils')) and \
       os.path.isdir(os.path.join(path_entry, 'config')) and \
       os.path.isdir(os.path.join(path_entry, 'services')):
        project_root_candidate_gg = path_entry
        break
print(f"    Attempted to find project root in sys.path: {project_root_candidate_gg if project_root_candidate_gg else 'Not found based on subdirs.'}")
print(f"--- [END DEBUG gap_guardian.py @ {os.path.basename(GAP_GUARDIAN_FILE_PATH)}] ---")
# --- end of sys.path diagnostics ---

print(f"--- [DEBUG gap_guardian.py @ {os.path.basename(GAP_GUARDIAN_FILE_PATH)}] Attempting to import utils.logger and config.settings ---")
try:
    # Project-specific imports - these should work if app.py set sys.path correctly
    from utils.logger import get_logger
    from config import settings
    print(f"--- [DEBUG gap_guardian.py @ {os.path.basename(GAP_GUARDIAN_FILE_PATH)}] Successfully imported utils.logger and config.settings ---")
except ImportError as e:
    print(f"--- [CRITICAL ERROR gap_guardian.py @ {os.path.basename(GAP_GUARDIAN_FILE_PATH)}] Failed to import utils or config. Error: {e} ---")
    print(f"    Current sys.path during this error: {sys.path}")
    raise # Re-raise to make the failure clear


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
        
        opening_bar_candidates = day_data[day_data.index.time >= entry_start_time]
        if opening_bar_candidates.empty:
            continue
        
        opening_bar_data = opening_bar_candidates.iloc[0:1] 
        if opening_bar_data.empty: 
            continue
            
        opening_bar_timestamp = opening_bar_data.index[0]
        opening_range_high = opening_bar_data['High'].iloc[0]
        opening_range_low = opening_bar_data['Low'].iloc[0]
        
        signal_scan_window_data = day_data[
            (day_data.index > opening_bar_timestamp) & 
            (day_data.index.time < entry_end_time)    
        ]

        if signal_scan_window_data.empty:
            continue

        for idx, bar in signal_scan_window_data.iterrows():
            signal_time = idx 
            
            if bar['Low'] < opening_range_low and bar['Close'] > opening_range_low:
                entry_price = bar['Close'] 
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
                break 

            elif bar['High'] > opening_range_high and bar['Close'] < opening_range_high:
                entry_price = bar['Close'] 
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
                break 
        
    return pd.DataFrame(signals_list)
