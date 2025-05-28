# services/strategies/__init__.py
# This file makes the 'strategies' directory a Python package.
# It can also be used to selectively import strategy functions for easier access.

from .gap_guardian import generate_gap_guardian_signals
from .unicorn import generate_unicorn_signals
from .silver_bullet import generate_silver_bullet_signals

__all__ = [
    "generate_gap_guardian_signals",
    "generate_unicorn_signals",
    "generate_silver_bullet_signals"
]
```python
# services/strategies/gap_guardian.py
"""
Signal generation logic for the Gap Guardian strategy.
"""
import pandas as pd
from datetime import time as dt_time
from utils.logger import get_logger # Assuming logger is in ../../utils/

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
from config import settings # Assuming settings is in ../../config/
from utils.logger import get_logger # Assuming logger is in ../../utils/

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
        return bar1['High'], bar3['Low']
    elif direction == "bearish" and bar1['Low'] > bar3['High'] and bar2['Close'] < bar2['Open']:
        return bar3['High'], bar1['Low']
    return None

def _find_swing_points(data: pd.DataFrame, n: int = settings.UNICORN_SWING_LOOKBACK) -> pd.DataFrame:
    """Identifies swing highs and lows."""
    data_copy = data.copy()
    data_copy['SwingHigh'] = np.nan
    data_copy['SwingLow'] = np.nan
    for i in range(n, len(data_copy) - n):
        is_sh = all(data_copy['High'].iloc[i] >= data_copy['High'].iloc[i-j] for j in range(1, n+1)) and \
                all(data_copy['High'].iloc[i] >= data_copy['High'].iloc[i+j] for j in range(1, n+1))
        if is_sh: data_copy.loc[data_copy.index[i], 'SwingHigh'] = data_copy['High'].iloc[i]
        
        is_sl = all(data_copy['Low'].iloc[i] <= data_copy['Low'].iloc[i-j] for j in range(1, n+1)) and \
                all(data_copy['Low'].iloc[i] <= data_copy['Low'].iloc[i+j] for j in range(1, n+1))
        if is_sl: data_copy.loc[data_copy.index[i], 'SwingLow'] = data_copy['Low'].iloc[i]
    return data_copy

def generate_unicorn_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float
) -> pd.DataFrame:
    """
    Generates signals for the Unicorn strategy.
    Current implementation is a simplified FVG entry.
    Full Breaker + FVG overlap requires more complex pattern recognition.
    """
    signals_list = []
    if len(data) < (settings.UNICORN_SWING_LOOKBACK * 2 + 5):
        logger.warning("Unicorn: Not enough data for signal generation.")
        return pd.DataFrame()

    # data_with_swings = _find_swing_points(data, n=settings.UNICORN_SWING_LOOKBACK)
    # Full Unicorn logic would use data_with_swings to identify breakers.
    # For this simplified version, we'll focus on FVG entries.

    for i in range(len(data) - 3): # Iterate to allow FVG check using i, i+1, i+2
        current_bar = data.iloc[i]
        current_bar_time = data.index[i]

        # Bullish FVG Entry (Simplified Unicorn)
        # FVG formed by bars i, i+1, i+2. Entry on bar i+3 if it retraces.
        # Or, for entry on current_bar 'i' into an FVG that *just completed* on bar 'i-1' (formed by i-3, i-2, i-1)
        if i >= 3: # Ensure we have bars i-3, i-2, i-1 for FVG, and 'i' for potential entry
            bullish_fvg = _find_fvg(data, i - 3, "bullish") # FVG is between (i-2).High and (i).Low
            if bullish_fvg:
                fvg_b_low, fvg_b_high = bullish_fvg # Note: for bullish FVG, this is (bar(i-2).High, bar(i).Low)
                # Entry condition: current_bar (i) dips into the FVG and shows bullish reaction
                if current_bar['Low'] <= fvg_b_high and current_bar['Close'] > fvg_b_low and current_bar['Close'] > current_bar['Open']:
                    entry_price = current_bar['Close']
                    sl_price = entry_price - stop_loss_points # Or below FVG low / structure
                    tp_price = entry_price + (stop_loss_points * rrr)
                    signals_list.append({
                        'SignalTime': current_bar_time, 'SignalType': 'Long',
                        'EntryPrice': entry_price, 'SL': sl_price, 'TP': tp_price,
                        'Reason': f"Unicorn (Bullish FVG Entry): {fvg_b_low:.2f}-{fvg_b_high:.2f}"
                    })
                    logger.debug(f"Unicorn Long @ {current_bar_time}, FVG: {fvg_b_low:.2f}-{fvg_b_high:.2f}, Entry: {entry_price:.2f}")
                    # Potentially break or add logic to avoid multiple signals for the same setup

        # Bearish FVG Entry (Simplified Unicorn)
        if i >= 3:
            bearish_fvg = _find_fvg(data, i - 3, "bearish") # FVG is between (i).High and (i-2).Low
            if bearish_fvg:
                fvg_s_low, fvg_s_high = bearish_fvg # Note: for bearish FVG, this is (bar(i).High, bar(i-2).Low)
                if current_bar['High'] >= fvg_s_low and current_bar['Close'] < fvg_s_high and current_bar['Close'] < current_bar['Open']:
                    entry_price = current_bar['Close']
                    sl_price = entry_price + stop_loss_points # Or above FVG high / structure
                    tp_price = entry_price - (stop_loss_points * rrr)
                    signals_list.append({
                        'SignalTime': current_bar_time, 'SignalType': 'Short',
                        'EntryPrice': entry_price, 'SL': sl_price, 'TP': tp_price,
                        'Reason': f"Unicorn (Bearish FVG Entry): {fvg_s_low:.2f}-{fvg_s_high:.2f}"
                    })
                    logger.debug(f"Unicorn Short @ {current_bar_time}, FVG: {fvg_s_low:.2f}-{fvg_s_high:.2f}, Entry: {entry_price:.2f}")

    if not signals_list:
        logger.debug("Unicorn: No signals generated (using simplified FVG logic).")
    else:
        logger.info(f"Unicorn: Generated {len(signals_list)} signals (simplified FVG logic).")
    
    # A more robust Unicorn would involve:
    # 1. Identifying swing points using _find_swing_points.
    # 2. Detecting the Bullish/Bearish Breaker structure (sweep of prior swing, then displacement breaking subsequent swing).
    # 3. Identifying the breaker candle/zone.
    # 4. Finding an FVG formed *during the displacement*.
    # 5. Checking for overlap between the breaker zone and the FVG.
    # 6. Entering on retracement into the overlapping zone.
    # This is a significantly more complex piece of pattern recognition.
    logger.warning("Unicorn strategy currently uses a simplified FVG entry logic. Full Breaker+FVG overlap is complex and not fully implemented.")
    return pd.DataFrame(signals_list)
```python
# services/strategies/silver_bullet.py
"""
Signal generation logic for the Silver Bullet strategy.
"""
import pandas as pd
from config import settings # Assuming settings is in ../../config/
from utils.logger import get_logger # Assuming logger is in ../../utils/
# Import _find_fvg if it's made common, or define locally if specific variant needed.
# For now, let's assume it might be common or we can copy it if strictly needed.
# Re-defining _find_fvg here for clarity if it's slightly different for SB or until it's common.
# from .unicorn import _find_fvg # Or define/copy if needed

logger = get_logger(__name__)

def _find_fvg_for_silver_bullet(data: pd.DataFrame, bar_index: int, direction: str = "bullish") -> tuple[float, float] | None:
    """
    Identifies a Fair Value Gap (FVG) based on a 3-bar pattern.
    FVG is identified based on bars [bar_index+1, bar_index+2, bar_index+3].
    The FVG itself is the price range on bar_index+2.
    (This is identical to the one in unicorn.py, can be refactored to a common place)
    """
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
    """
    Generates signals for the Silver Bullet strategy.
    Logic: FVG entry within specific 1-hour NY time windows.
    """
    signals_list = []
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz != settings.NY_TIMEZONE:
        logger.error("Silver Bullet: Data must have NY-localized DatetimeIndex.")
        return pd.DataFrame()

    for i in range(len(data) - 3): # Iterate to allow FVG check, entry on bar 'i' into FVG from i-3,i-2,i-1
        current_bar = data.iloc[i]
        current_bar_time_dt = data.index[i] # This is a datetime object
        current_bar_ny_time_obj = current_bar_time_dt.time() # This is a datetime.time object

        in_sb_window = False
        for start_t, end_t in settings.SILVER_BULLET_WINDOWS_NY:
            if start_t <= current_bar_ny_time_obj < end_t:
                in_sb_window = True
                break
        
        if not in_sb_window:
            continue

        # Entry on current_bar 'i' into an FVG that *just completed* on bar 'i-1' (formed by i-3, i-2, i-1)
        if i >= 3:
            # Bullish FVG check
            # FVG formed by bars (i-3), (i-2), (i-1). Bar (i-2) is the one with the gap.
            # FVG range is [(i-3).High, (i-1).Low]
            bullish_fvg = _find_fvg_for_silver_bullet(data, i - 3, "bullish")
            if bullish_fvg:
                fvg_low, fvg_high = bullish_fvg # This is bar(i-3).High, bar(i-1).Low
                # Entry if current_bar.Low dips into FVG and Close is bullish
                if current_bar['Low'] <= fvg_high and current_bar['Close'] > fvg_low and current_bar['Close'] > current_bar['Open']:
                    entry_price = current_bar['Close']
                    sl = entry_price - stop_loss_points
                    tp = entry_price + (stop_loss_points * rrr)
                    signals_list.append({
                        'SignalTime': current_bar_time_dt, 'SignalType': 'Long', 'EntryPrice': entry_price,
                        'SL': sl, 'TP': tp, 'Reason': f"SB Long: FVG {fvg_low:.2f}-{fvg_high:.2f} in window"
                    })
                    logger.debug(f"SB Long @ {current_bar_time_dt}, FVG: {fvg_low:.2f}-{fvg_high:.2f}, Entry: {entry_price:.2f}")
                    continue # Avoid multiple signals on same FVG with subsequent bars

            # Bearish FVG check
            # FVG formed by bars (i-3), (i-2), (i-1). Bar (i-2) is the one with the gap.
            # FVG range is [(i-1).High, (i-3).Low]
            bearish_fvg = _find_fvg_for_silver_bullet(data, i - 3, "bearish")
            if bearish_fvg:
                fvg_low, fvg_high = bearish_fvg # This is bar(i-1).High, bar(i-3).Low
                if current_bar['High'] >= fvg_low and current_bar['Close'] < fvg_high and current_bar['Close'] < current_bar['Open']:
                    entry_price = current_bar['Close']
                    sl = entry_price + stop_loss_points
                    tp = entry_price - (stop_loss_points * rrr)
                    signals_list.append({
                        'SignalTime': current_bar_time_dt, 'SignalType': 'Short', 'EntryPrice': entry_price,
                        'SL': sl, 'TP': tp, 'Reason': f"SB Short: FVG {fvg_low:.2f}-{fvg_high:.2f} in window"
                    })
                    logger.debug(f"SB Short @ {current_bar_time_dt}, FVG: {fvg_low:.2f}-{fvg_high:.2f}, Entry: {entry_price:.2f}")
                    continue
                    
    if not signals_list:
        logger.debug("Silver Bullet: No signals generated.")
    else:
        logger.info(f"Silver Bullet: Generated {len(signals_list)} signals.")
    return pd.DataFrame(signals_list)
