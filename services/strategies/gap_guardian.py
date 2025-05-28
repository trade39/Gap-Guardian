# services/strategies/gap_guardian.py
"""
Signal generation logic for the Gap Guardian strategy.
Includes sys.path adjustment for robust imports.
"""
import sys
import os
import pandas as pd
from datetime import time as dt_time

# Ensure the project root is in sys.path for absolute imports
# Assumes this file is in Gap-Guardian-Strategy-Backtester-main/services/strategies/
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
        # print(f"DEBUG: Added to sys.path from gap_guardian.py: {PROJECT_ROOT}") # Optional debug print
    from utils.logger import get_logger
    from config import settings
except ImportError as e:
    print(f"CRITICAL ERROR in services/strategies/gap_guardian.py: Could not set up imports: {e}")
    # If these base imports fail, the module cannot function.
    # Re-raise or define a placeholder to make the problem obvious.
    raise

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
# services/strategies/unicorn.py
"""
Signal generation logic for the Unicorn strategy (Breaker Block + FVG).
Includes sys.path adjustment for robust imports.
"""
import sys
import os
import pandas as pd
import numpy as np

# Ensure the project root is in sys.path for absolute imports
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from config import settings
    from utils.logger import get_logger
except ImportError as e:
    print(f"CRITICAL ERROR in services/strategies/unicorn.py: Could not set up imports: {e}")
    raise

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
# services/strategies/silver_bullet.py
"""
Signal generation logic for the Silver Bullet strategy.
Includes sys.path adjustment for robust imports.
"""
import sys
import os
import pandas as pd

# Ensure the project root is in sys.path for absolute imports
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from config import settings
    from utils.logger import get_logger
except ImportError as e:
    print(f"CRITICAL ERROR in services/strategies/silver_bullet.py: Could not set up imports: {e}")
    raise

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
