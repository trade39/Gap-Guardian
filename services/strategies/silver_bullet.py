# services/strategies/silver_bullet.py
"""
Signal generation logic for the Silver Bullet strategy.
Includes sys.path adjustment for robust imports.
"""
import sys
import os
import pandas as pd

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
        print(f"WARNING (silver_bullet.py): Calculated project root '{project_root}' does not contain 'utils' or 'config' folders.")
except Exception as e:
    print(f"ERROR (silver_bullet.py): Could not adjust sys.path: {e}")

if _project_root_found:
    try:
        from config import settings
        from utils.logger import get_logger
    except ImportError as e:
        print(f"CRITICAL ERROR (silver_bullet.py): Failed to import 'utils.logger' or 'config.settings'. Error: {e}")
        raise ImportError(f"silver_bullet.py: Could not import critical modules: {e}") from e
else:
    try:
        from config import settings
        from utils.logger import get_logger
        print("INFO (silver_bullet.py): Using fallback absolute imports.")
    except ImportError as e:
        print(f"CRITICAL ERROR (silver_bullet.py): Fallback absolute imports also failed. Error: {e}")
        raise ImportError(f"silver_bullet.py: Critical module import failed: {e}") from e

logger = get_logger(__name__)

def _find_fvg_for_silver_bullet(data: pd.DataFrame, bar_index: int, direction: str = "bullish") -> tuple[float, float] | None:
    if bar_index + 3 >= len(data):
        return None
    bar1, bar2, bar3 = data.iloc[bar_index + 1], data.iloc[bar_index + 2], data.iloc[bar_index + 3]
    if direction == "bullish" and bar1['High'] < bar3['Low'] and bar2['Close'] > bar2['Open']:
        return bar1['High'], bar3['Low']
    elif direction == "bearish" and bar1['Low'] > bar3['High'] and bar2['Close'] < bar2['Open']:
        return bar3['High'], bar1['Low']
    return None

def generate_silver_bullet_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float
) -> pd.DataFrame:
    signals_list = []
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None or data.index.tz.zone != settings.NY_TIMEZONE.zone:
        logger.error(f"Silver Bullet: Data must have NY-localized DatetimeIndex. Current tz: {data.index.tz}")
        return pd.DataFrame()
    for i in range(len(data) - 3):
        current_bar = data.iloc[i]; current_bar_time_dt = data.index[i]; current_bar_ny_time_obj = current_bar_time_dt.time()
        in_sb_window = any(start_t <= current_bar_ny_time_obj < end_t for start_t, end_t in settings.SILVER_BULLET_WINDOWS_NY)
        if not in_sb_window: continue
        if i >= 3:
            bullish_fvg = _find_fvg_for_silver_bullet(data, i - 3, "bullish")
            if bullish_fvg:
                fvg_b_actual_low, fvg_b_actual_high = bullish_fvg[0], bullish_fvg[1]
                if current_bar['Low'] <= fvg_b_actual_high and current_bar['Close'] > fvg_b_actual_low and current_bar['Close'] > current_bar['Open']:
                    entry_price = current_bar['Close']; sl = entry_price - stop_loss_points; tp = entry_price + (stop_loss_points * rrr)
                    signals_list.append({'SignalTime': current_bar_time_dt, 'SignalType': 'Long', 'EntryPrice': entry_price, 'SL': sl, 'TP': tp, 'Reason': f"SB Long: FVG {fvg_b_actual_low:.2f}-{fvg_b_actual_high:.2f} in window"})
                    logger.debug(f"SB Long @ {current_bar_time_dt}, FVG: {fvg_b_actual_low:.2f}-{fvg_b_actual_high:.2f}, Entry: {entry_price:.2f}")
                    continue 
            bearish_fvg = _find_fvg_for_silver_bullet(data, i - 3, "bearish")
            if bearish_fvg:
                fvg_s_actual_low, fvg_s_actual_high = bearish_fvg[0], bearish_fvg[1]
                if current_bar['High'] >= fvg_s_actual_low and current_bar['Close'] < fvg_s_actual_high and current_bar['Close'] < current_bar['Open']:
                    entry_price = current_bar['Close']; sl = entry_price + stop_loss_points; tp = entry_price - (stop_loss_points * rrr)
                    signals_list.append({'SignalTime': current_bar_time_dt, 'SignalType': 'Short', 'EntryPrice': entry_price, 'SL': sl, 'TP': tp, 'Reason': f"SB Short: FVG {fvg_s_actual_low:.2f}-{fvg_s_actual_high:.2f} in window"})
                    logger.debug(f"SB Short @ {current_bar_time_dt}, FVG: {fvg_s_actual_low:.2f}-{fvg_s_actual_high:.2f}, Entry: {entry_price:.2f}")
                    continue
    if not signals_list: logger.debug("Silver Bullet: No signals generated.")
    else: logger.info(f"Silver Bullet: Generated {len(signals_list)} signals.")
    return pd.DataFrame(signals_list)
