# services/strategies/gap_guardian.py
"""
Signal generation logic for the Gap Guardian strategy.
Includes sys.path adjustment for robust imports.
"""
import sys
import os
import pandas as pd # Keep this here as it's a standard library
from datetime import time as dt_time # Standard library

# --- Robust sys.path adjustment ---
# This block MUST be at the top before any project-specific imports.
_project_root_found = False
try:
    # Try to determine the project root assuming a fixed structure:
    # Gap-Guardian-Strategy-Backtester-main/services/strategies/this_file.py
    current_file_path = os.path.abspath(__file__)
    # Navigate three levels up to reach the project root from services/strategies/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    if os.path.isdir(os.path.join(project_root, 'utils')) and \
       os.path.isdir(os.path.join(project_root, 'config')):
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            # print(f"DEBUG (gap_guardian.py): Added to sys.path: {project_root}") # Optional for debugging
        _project_root_found = True
    else:
        print(f"WARNING (gap_guardian.py): Calculated project root '{project_root}' does not contain 'utils' or 'config' folders. Imports might fail.")

except Exception as e:
    print(f"ERROR (gap_guardian.py): Could not adjust sys.path: {e}")
# --- End of sys.path adjustment ---

# Now attempt project-specific imports
if _project_root_found:
    try:
        from utils.logger import get_logger
        from config import settings
    except ImportError as e:
        print(f"CRITICAL ERROR (gap_guardian.py): Failed to import 'utils.logger' or 'config.settings' after sys.path adjustment. Sys.path: {sys.path}. Error: {e}")
        # If these imports fail, the module cannot function.
        # Raising the error will make it clear during Streamlit startup.
        raise ImportError(f"gap_guardian.py: Could not import critical modules: {e}") from e
else:
    # Fallback if project root wasn't confidently found, try direct absolute imports
    # This might work if sys.path is already correctly configured by the environment
    try:
        from utils.logger import get_logger
        from config import settings
        print("INFO (gap_guardian.py): Using fallback absolute imports (sys.path might have been pre-configured).")
    except ImportError as e:
        print(f"CRITICAL ERROR (gap_guardian.py): Fallback absolute imports also failed. Sys.path: {sys.path}. Error: {e}")
        raise ImportError(f"gap_guardian.py: Critical module import failed: {e}") from e


logger = get_logger(__name__)

def generate_gap_guardian_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float,
    entry_start_time: dt_time,
    entry_end_time: dt_time
) -> pd.DataFrame:
    signals_list = []
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
                    'SignalTime': signal_time, 'SignalType': 'Long', 'EntryPrice': entry_price,
                    'SL': sl, 'TP': tp, 'Reason': f"GG: False breakdown of {opening_range_low:.2f}"
                })
                logger.debug(f"GG Long signal on {date_val} at {signal_time.strftime('%H:%M')}: Entry={entry_price:.2f}")
                break 
            elif bar['High'] > opening_range_high and bar['Close'] < opening_range_high:
                entry_price = bar['Close']
                sl = entry_price + stop_loss_points
                tp = entry_price - (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': signal_time, 'SignalType': 'Short', 'EntryPrice': entry_price,
                    'SL': sl, 'TP': tp, 'Reason': f"GG: False breakout of {opening_range_high:.2f}"
                })
                logger.debug(f"GG Short signal on {date_val} at {signal_time.strftime('%H:%M')}: Entry={entry_price:.2f}")
                break
    return pd.DataFrame(signals_list)
```python
