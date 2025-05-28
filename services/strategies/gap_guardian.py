# services/strategies/gap_guardian.py
"""
Signal generation logic for the Gap Guardian strategy.
"""
import pandas as pd
from datetime import time as dt_time
from utils.logger import get_logger # Corrected: Absolute import
from config import settings # Corrected: Absolute import

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
        pd.DataFrame: DataFrame containing trade signals.
    """
    signals_list = []
    # Group data by date to process one day at a time
    for date_val, day_data in data.groupby(data.index.date):
        if day_data.empty:
            continue
        
        # Find the "opening bar" for the strategy:
        opening_bar_candidates = day_data[day_data.index.time >= entry_start_time]
        if opening_bar_candidates.empty:
            continue
        
        opening_bar_data = opening_bar_candidates.iloc[0:1] # Get the first such bar
        if opening_bar_data.empty:
            continue
            
        opening_bar_timestamp = opening_bar_data.index[0]
        opening_range_high = opening_bar_data['High'].iloc[0]
        opening_range_low = opening_bar_data['Low'].iloc[0]
        
        # Iterate through bars within the entry window (after the opening bar)
        signal_scan_window_data = day_data[
            (day_data.index > opening_bar_timestamp) &
            (day_data.index.time < entry_end_time)
        ]

        if signal_scan_window_data.empty:
            continue

        for idx, bar in signal_scan_window_data.iterrows():
            signal_time = idx
            
            # Check for Long Signal
            if bar['Low'] < opening_range_low and bar['Close'] > opening_range_low:
                entry_price = bar['Close']
                sl = entry_price - stop_loss_points
                tp = entry_price + (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': signal_time, 'SignalType': 'Long', 'EntryPrice': entry_price,
                    'SL': sl, 'TP': tp, 'Reason': f"GG: False breakdown of {opening_range_low:.2f}"
                })
                logger.debug(f"GG Long signal on {date_val} at {signal_time.strftime('%H:%M')}: Entry={entry_price:.2f}")
                break # Max one trade per day

            # Check for Short Signal
            elif bar['High'] > opening_range_high and bar['Close'] < opening_range_high:
                entry_price = bar['Close']
                sl = entry_price + stop_loss_points
                tp = entry_price - (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': signal_time, 'SignalType': 'Short', 'EntryPrice': entry_price,
                    'SL': sl, 'TP': tp, 'Reason': f"GG: False breakout of {opening_range_high:.2f}"
                })
                logger.debug(f"GG Short signal on {date_val} at {signal_time.strftime('%H:%M')}: Entry={entry_price:.2f}")
                break # Max one trade per day
        
    return pd.DataFrame(signals_list)

```python
# services/strategies/unicorn.py
"""
Signal generation logic for the Unicorn strategy (Breaker Block + FVG).
"""
import pandas as pd
import numpy as np
from config import settings # Corrected: Absolute import
from utils.logger import get_logger # Corrected: Absolute import

logger = get_logger(__name__)

# --- Helper Functions for Unicorn Strategy ---
def _find_fvg(data: pd.DataFrame, bar_index: int, direction: str = "bullish") -> tuple[float, float] | None:
    """
    Identifies a Fair Value Gap (FVG) based on a 3-bar pattern.
    FVG is identified based on bars [bar_index+1, bar_index+2, bar_index+3].
    The FVG itself is the price range on bar_index+2.
    """
    if bar_index + 3 >= len(data):
        return None
    bar1, bar2, bar3 = data.iloc[bar_index + 1], data.iloc[bar_index + 2], data.iloc[bar_index + 3]
    if direction == "bullish" and bar1['High'] < bar3['Low'] and bar2['Close'] > bar2['Open']:
        return bar1['High'], bar3['Low'] # FVG is between bar1.High and bar3.Low
    elif direction == "bearish" and bar1['Low'] > bar3['High'] and bar2['Close'] < bar2['Open']:
        return bar3['High'], bar1['Low'] # FVG is between bar3.High and bar1.Low
    return None

def _find_swing_points(data: pd.DataFrame, n: int = settings.UNICORN_SWING_LOOKBACK) -> pd.DataFrame:
    """Identifies swing highs and lows."""
    data_copy = data.copy()
    data_copy['SwingHigh'] = np.nan
    data_copy['SwingLow'] = np.nan
    # Corrected swing point logic for clarity and common definition
    for i in range(n, len(data_copy) - n):
        # Check for Swing High: High is greater than n bars on both sides
        is_swing_high = True
        for j in range(1, n + 1):
            if data_copy['High'].iloc[i] < data_copy['High'].iloc[i-j] or \
               data_copy['High'].iloc[i] < data_copy['High'].iloc[i+j]:
                is_swing_high = False
                break
        if is_swing_high:
            # Check if it's strictly higher than immediate neighbors for a sharper peak
            if data_copy['High'].iloc[i] > data_copy['High'].iloc[i-1] and \
               data_copy['High'].iloc[i] > data_copy['High'].iloc[i+1]:
                data_copy.loc[data_copy.index[i], 'SwingHigh'] = data_copy['High'].iloc[i]

        # Check for Swing Low: Low is smaller than n bars on both sides
        is_swing_low = True
        for j in range(1, n + 1):
            if data_copy['Low'].iloc[i] > data_copy['Low'].iloc[i-j] or \
               data_copy['Low'].iloc[i] > data_copy['Low'].iloc[i+j]:
                is_swing_low = False
                break
        if is_swing_low:
             # Check if it's strictly lower than immediate neighbors for a sharper trough
            if data_copy['Low'].iloc[i] < data_copy['Low'].iloc[i-1] and \
               data_copy['Low'].iloc[i] < data_copy['Low'].iloc[i+1]:
                data_copy.loc[data_copy.index[i], 'SwingLow'] = data_copy['Low'].iloc[i]
    return data_copy

def generate_unicorn_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float
) -> pd.DataFrame:
    """
    Generates signals for the Unicorn strategy.
    Current implementation uses a simplified FVG entry.
    Full Breaker + FVG overlap requires more complex pattern recognition.
    """
    signals_list = []
    if len(data) < (settings.UNICORN_SWING_LOOKBACK * 2 + 5): # Min data for swings and FVG patterns
        logger.warning("Unicorn: Not enough data for signal generation.")
        return pd.DataFrame()

    # data_with_swings = _find_swing_points(data, n=settings.UNICORN_SWING_LOOKBACK) # For full Unicorn

    for i in range(len(data) - 3): # Iterate to allow FVG check using i, i+1, i+2 relative to current processing point
        current_bar = data.iloc[i]
        current_bar_time = data.index[i]

        # Simplified FVG Entry Logic:
        # Look for an FVG that formed recently (e.g., completed on bar i-1, formed by i-3, i-2, i-1)
        # and current bar 'i' is the entry bar if it retraces into that FVG.
        if i >= 3: 
            # Bullish FVG check: FVG formed by (i-3), (i-2), (i-1). Bar (i-2) is the one with the gap.
            # FVG range is [(i-3).High, (i-1).Low]
            bullish_fvg = _find_fvg(data, i - 3, "bullish") # Pass i-3 to check pattern starting at i-2
            if bullish_fvg:
                # For bullish FVG: fvg_low is bar(i-3).High, fvg_high is bar(i-1).Low
                fvg_b_actual_low, fvg_b_actual_high = bullish_fvg[0], bullish_fvg[1]
                
                # Entry condition: current_bar (i) dips into the FVG and shows bullish reaction
                if current_bar['Low'] <= fvg_b_actual_high and current_bar['Close'] > fvg_b_actual_low and current_bar['Close'] > current_bar['Open']:
                    entry_price = current_bar['Close']
                    sl_price = entry_price - stop_loss_points 
                    tp_price = entry_price + (stop_loss_points * rrr)
                    signals_list.append({
                        'SignalTime': current_bar_time, 'SignalType': 'Long',
                        'EntryPrice': entry_price, 'SL': sl_price, 'TP': tp_price,
                        'Reason': f"Unicorn (Bullish FVG Entry): {fvg_b_actual_low:.2f}-{fvg_b_actual_high:.2f}"
                    })
                    logger.debug(f"Unicorn Long @ {current_bar_time}, FVG: {fvg_b_actual_low:.2f}-{fvg_b_actual_high:.2f}, Entry: {entry_price:.2f}")
                    # Consider adding logic to only take one signal per FVG or per day.

            # Bearish FVG check: FVG formed by (i-3), (i-2), (i-1). Bar (i-2) is the one with the gap.
            # FVG range is [(i-1).High, (i-3).Low]
            bearish_fvg = _find_fvg(data, i - 3, "bearish") # Pass i-3 to check pattern starting at i-2
            if bearish_fvg:
                # For bearish FVG: fvg_low is bar(i-1).High, fvg_high is bar(i-3).Low
                fvg_s_actual_low, fvg_s_actual_high = bearish_fvg[0], bearish_fvg[1]

                if current_bar['High'] >= fvg_s_actual_low and current_bar['Close'] < fvg_s_actual_high and current_bar['Close'] < current_bar['Open']:
                    entry_price = current_bar['Close']
                    sl_price = entry_price + stop_loss_points 
                    tp_price = entry_price - (stop_loss_points * rrr)
                    signals_list.append({
                        'SignalTime': current_bar_time, 'SignalType': 'Short',
                        'EntryPrice': entry_price, 'SL': sl_price, 'TP': tp_price,
                        'Reason': f"Unicorn (Bearish FVG Entry): {fvg_s_actual_low:.2f}-{fvg_s_actual_high:.2f}"
                    })
                    logger.debug(f"Unicorn Short @ {current_bar_time}, FVG: {fvg_s_actual_low:.2f}-{fvg_s_actual_high:.2f}, Entry: {entry_price:.2f}")

    if not signals_list:
        logger.debug("Unicorn: No signals generated (using simplified FVG logic).")
    else:
        logger.info(f"Unicorn: Generated {len(signals_list)} signals (simplified FVG logic).")
    
    logger.warning("Unicorn strategy currently uses a simplified FVG entry logic. Full Breaker+FVG overlap is complex and not fully implemented.")
    return pd.DataFrame(signals_list)
```python
# services/strategies/silver_bullet.py
"""
Signal generation logic for the Silver Bullet strategy.
"""
import pandas as pd
from config import settings # Corrected: Absolute import
from utils.logger import get_logger # Corrected: Absolute import

logger = get_logger(__name__)

def _find_fvg_for_silver_bullet(data: pd.DataFrame, bar_index: int, direction: str = "bullish") -> tuple[float, float] | None:
    """
    Identifies a Fair Value Gap (FVG) based on a 3-bar pattern.
    FVG is identified based on bars [bar_index+1, bar_index+2, bar_index+3].
    The FVG itself is the price range on bar_index+2.
    (This is identical to the one in unicorn.py, can be refactored to a common place)
    """
    if bar_index + 3 >= len(data): # Need bars data[bar_index+1], data[bar_index+2], data[bar_index+3]
        return None
    bar1, bar2, bar3 = data.iloc[bar_index + 1], data.iloc[bar_index + 2], data.iloc[bar_index + 3]
    if direction == "bullish" and bar1['High'] < bar3['Low'] and bar2['Close'] > bar2['Open']:
        return bar1['High'], bar3['Low'] # FVG is between bar1.High and bar3.Low
    elif direction == "bearish" and bar1['Low'] > bar3['High'] and bar2['Close'] < bar2['Open']:
        return bar3['High'], bar1['Low'] # FVG is between bar3.High and bar1.Low
    return None


def generate_silver_bullet_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float
) -> pd.DataFrame:
    """
    Generates signals for the Silver Bullet strategy.
    Logic: FVG entry within specific 1-hour NY time windows.
    """
    signals_list = []
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None or data.index.tz.zone != settings.NY_TIMEZONE.zone:
        logger.error(f"Silver Bullet: Data must have NY-localized DatetimeIndex. Current tz: {data.index.tz}")
        return pd.DataFrame()

    for i in range(len(data) - 3): # Iterate to allow FVG check, entry on bar 'i' into FVG from i-3,i-2,i-1
        current_bar = data.iloc[i]
        current_bar_time_dt = data.index[i] 
        current_bar_ny_time_obj = current_bar_time_dt.time()

        in_sb_window = False
        for start_t, end_t in settings.SILVER_BULLET_WINDOWS_NY:
            if start_t <= current_bar_ny_time_obj < end_t:
                in_sb_window = True
                break
        
        if not in_sb_window:
            continue

        # Entry on current_bar 'i' into an FVG that *just completed* on bar 'i-1' (formed by i-3, i-2, i-1)
        if i >= 3:
            # Bullish FVG check: FVG formed by (i-3), (i-2), (i-1). Bar (i-2) is the one with the gap.
            # FVG range is [(i-3).High, (i-1).Low]
            bullish_fvg = _find_fvg_for_silver_bullet(data, i - 3, "bullish") # Pass i-3 to check pattern starting at i-2
            if bullish_fvg:
                fvg_b_actual_low, fvg_b_actual_high = bullish_fvg[0], bullish_fvg[1]
                
                if current_bar['Low'] <= fvg_b_actual_high and current_bar['Close'] > fvg_b_actual_low and current_bar['Close'] > current_bar['Open']:
                    entry_price = current_bar['Close']
                    sl = entry_price - stop_loss_points
                    tp = entry_price + (stop_loss_points * rrr)
                    signals_list.append({
                        'SignalTime': current_bar_time_dt, 'SignalType': 'Long', 'EntryPrice': entry_price,
                        'SL': sl, 'TP': tp, 'Reason': f"SB Long: FVG {fvg_b_actual_low:.2f}-{fvg_b_actual_high:.2f} in window"
                    })
                    logger.debug(f"SB Long @ {current_bar_time_dt}, FVG: {fvg_b_actual_low:.2f}-{fvg_b_actual_high:.2f}, Entry: {entry_price:.2f}")
                    continue 

            # Bearish FVG check: FVG formed by (i-3), (i-2), (i-1). Bar (i-2) is the one with the gap.
            # FVG range is [(i-1).High, (i-3).Low]
            bearish_fvg = _find_fvg_for_silver_bullet(data, i - 3, "bearish") # Pass i-3 to check pattern starting at i-2
            if bearish_fvg:
                fvg_s_actual_low, fvg_s_actual_high = bearish_fvg[0], bearish_fvg[1]

                if current_bar['High'] >= fvg_s_actual_low and current_bar['Close'] < fvg_s_actual_high and current_bar['Close'] < current_bar['Open']:
                    entry_price = current_bar['Close']
                    sl = entry_price + stop_loss_points
                    tp = entry_price - (stop_loss_points * rrr)
                    signals_list.append({
                        'SignalTime': current_bar_time_dt, 'SignalType': 'Short', 'EntryPrice': entry_price,
                        'SL': sl, 'TP': tp, 'Reason': f"SB Short: FVG {fvg_s_actual_low:.2f}-{fvg_s_actual_high:.2f} in window"
                    })
                    logger.debug(f"SB Short @ {current_bar_time_dt}, FVG: {fvg_s_actual_low:.2f}-{fvg_s_actual_high:.2f}, Entry: {entry_price:.2f}")
                    continue
                    
    if not signals_list:
        logger.debug("Silver Bullet: No signals generated.")
    else:
        logger.info(f"Silver Bullet: Generated {len(signals_list)} signals.")
    return pd.DataFrame(signals_list)
