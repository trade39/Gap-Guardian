# services/strategies/unicorn.py
"""
Signal generation logic for the Unicorn strategy (Breaker Block + FVG).
Includes sys.path adjustment for robust imports.
"""
import sys
import os
import pandas as pd
import numpy as np

# --- Robust sys.path adjustment ---
_project_root_found = False
try:
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    if os.path.isdir(os.path.join(project_root, 'utils')) and \
       os.path.isdir(os.path.join(project_root, 'config')):
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        _project_root_found = True
    else:
        print(f"WARNING (unicorn.py): Calculated project root '{project_root}' does not contain 'utils' or 'config' folders.")
except Exception as e:
    print(f"ERROR (unicorn.py): Could not adjust sys.path: {e}")

if _project_root_found:
    try:
        from config import settings
        from utils.logger import get_logger
    except ImportError as e:
        print(f"CRITICAL ERROR (unicorn.py): Failed to import 'utils.logger' or 'config.settings'. Error: {e}")
        raise ImportError(f"unicorn.py: Could not import critical modules: {e}") from e
else:
    try:
        from config import settings
        from utils.logger import get_logger
        print("INFO (unicorn.py): Using fallback absolute imports.")
    except ImportError as e:
        print(f"CRITICAL ERROR (unicorn.py): Fallback absolute imports also failed. Error: {e}")
        raise ImportError(f"unicorn.py: Critical module import failed: {e}") from e

logger = get_logger(__name__)

def _find_fvg(data: pd.DataFrame, bar_index: int, direction: str = "bullish") -> tuple[float, float] | None:
    if bar_index + 3 >= len(data):
        return None
    bar1, bar2, bar3 = data.iloc[bar_index + 1], data.iloc[bar_index + 2], data.iloc[bar_index + 3]
    if direction == "bullish" and bar1['High'] < bar3['Low'] and bar2['Close'] > bar2['Open']:
        return bar1['High'], bar3['Low']
    elif direction == "bearish" and bar1['Low'] > bar3['High'] and bar2['Close'] < bar2['Open']:
        return bar3['High'], bar1['Low']
    return None

def _find_swing_points(data: pd.DataFrame, n: int = settings.UNICORN_SWING_LOOKBACK) -> pd.DataFrame:
    data_copy = data.copy()
    data_copy['SwingHigh'] = np.nan
    data_copy['SwingLow'] = np.nan
    for i in range(n, len(data_copy) - n):
        is_swing_high = True
        for j in range(1, n + 1):
            if data_copy['High'].iloc[i] < data_copy['High'].iloc[i-j] or \
               data_copy['High'].iloc[i] < data_copy['High'].iloc[i+j]:
                is_swing_high = False; break
        if is_swing_high and data_copy['High'].iloc[i] > data_copy['High'].iloc[i-1] and data_copy['High'].iloc[i] > data_copy['High'].iloc[i+1]:
            data_copy.loc[data_copy.index[i], 'SwingHigh'] = data_copy['High'].iloc[i]
        is_swing_low = True
        for j in range(1, n + 1):
            if data_copy['Low'].iloc[i] > data_copy['Low'].iloc[i-j] or \
               data_copy['Low'].iloc[i] > data_copy['Low'].iloc[i+j]:
                is_swing_low = False; break
        if is_swing_low and data_copy['Low'].iloc[i] < data_copy['Low'].iloc[i-1] and data_copy['Low'].iloc[i] < data_copy['Low'].iloc[i+1]:
            data_copy.loc[data_copy.index[i], 'SwingLow'] = data_copy['Low'].iloc[i]
    return data_copy

def generate_unicorn_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float
) -> pd.DataFrame:
    signals_list = []
    if len(data) < (settings.UNICORN_SWING_LOOKBACK * 2 + 5):
        logger.warning("Unicorn: Not enough data for signal generation.")
        return pd.DataFrame()
    for i in range(len(data) - 3):
        current_bar = data.iloc[i]; current_bar_time = data.index[i]
        if i >= 3:
            bullish_fvg = _find_fvg(data, i - 3, "bullish")
            if bullish_fvg:
                fvg_b_actual_low, fvg_b_actual_high = bullish_fvg[0], bullish_fvg[1]
                if current_bar['Low'] <= fvg_b_actual_high and current_bar['Close'] > fvg_b_actual_low and current_bar['Close'] > current_bar['Open']:
                    entry_price = current_bar['Close']; sl_price = entry_price - stop_loss_points; tp_price = entry_price + (stop_loss_points * rrr)
                    signals_list.append({'SignalTime': current_bar_time, 'SignalType': 'Long', 'EntryPrice': entry_price, 'SL': sl_price, 'TP': tp_price, 'Reason': f"Unicorn (Bullish FVG Entry): {fvg_b_actual_low:.2f}-{fvg_b_actual_high:.2f}"})
                    logger.debug(f"Unicorn Long @ {current_bar_time}, FVG: {fvg_b_actual_low:.2f}-{fvg_b_actual_high:.2f}, Entry: {entry_price:.2f}")
            bearish_fvg = _find_fvg(data, i - 3, "bearish")
            if bearish_fvg:
                fvg_s_actual_low, fvg_s_actual_high = bearish_fvg[0], bearish_fvg[1]
                if current_bar['High'] >= fvg_s_actual_low and current_bar['Close'] < fvg_s_actual_high and current_bar['Close'] < current_bar['Open']:
                    entry_price = current_bar['Close']; sl_price = entry_price + stop_loss_points; tp_price = entry_price - (stop_loss_points * rrr)
                    signals_list.append({'SignalTime': current_bar_time, 'SignalType': 'Short', 'EntryPrice': entry_price, 'SL': sl_price, 'TP': tp_price, 'Reason': f"Unicorn (Bearish FVG Entry): {fvg_s_actual_low:.2f}-{fvg_s_actual_high:.2f}"})
                    logger.debug(f"Unicorn Short @ {current_bar_time}, FVG: {fvg_s_actual_low:.2f}-{fvg_s_actual_high:.2f}, Entry: {entry_price:.2f}")
    if not signals_list: logger.debug("Unicorn: No signals generated (simplified FVG logic).")
    else: logger.info(f"Unicorn: Generated {len(signals_list)} signals (simplified FVG logic).")
    logger.warning("Unicorn strategy currently uses a simplified FVG entry logic.")
    return pd.DataFrame(signals_list)
```python
