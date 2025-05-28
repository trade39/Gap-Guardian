# services/strategies/silver_bullet.py
"""
Signal generation logic for the Silver Bullet strategy.
Relies on standard Python import mechanisms, assuming project root is in sys.path.
Logic: FVG entry within specific 1-hour NY time windows.
"""
import pandas as pd

# Project-specific imports
from config import settings
from utils.logger import get_logger
# Assuming _find_fvg is defined in unicorn and we want to use that exact one.
# If it were in a common utils file: from utils.technical_analysis import _find_fvg
# For now, let's use the one from unicorn.py, which means it needs to be importable.
# This creates a dependency between strategies if not careful.
# A better approach would be a common TA utility file.
# For this fix, let's assume unicorn._find_fvg is accessible or redefine locally.
# To avoid inter-strategy dependency for now, let's copy/paste _find_fvg if needed.
# The _find_fvg in unicorn.py was:
# def _find_fvg(data: pd.DataFrame, bar_index: int, direction: str = "bullish") -> tuple[float, float] | None:
#     if bar_index + 3 >= len(data): return None
#     bar1, bar2, bar3 = data.iloc[bar_index + 1], data.iloc[bar_index + 2], data.iloc[bar_index + 3]
#     if direction == "bullish" and bar1['High'] < bar3['Low'] and bar2['Close'] > bar2['Open']:
#         return bar1['High'], bar3['Low']
#     elif direction == "bearish" and bar1['Low'] > bar3['High'] and bar2['Close'] < bar2['Open']:
#         return bar3['High'], bar1['Low']
#     return None
# This is the same definition as used in the original silver_bullet.py. Let's keep it local for now.

logger = get_logger(__name__)

def _find_fvg_for_silver_bullet(data: pd.DataFrame, bar_index: int, direction: str = "bullish") -> tuple[float, float] | None:
    """
    Identifies a Fair Value Gap (FVG) based on a 3-bar pattern.
    The FVG is identified based on the bars at indices: bar_index+1, bar_index+2, bar_index+3.
    The FVG itself is the price range on bar_index+2, defined by the wicks of bar_index+1 and bar_index+3.
    """
    if bar_index + 3 >= len(data): # Need at least bar_index+1, bar_index+2, bar_index+3
        return None
    
    bar1 = data.iloc[bar_index + 1] # First bar of the 3-bar pattern
    bar2 = data.iloc[bar_index + 2] # Middle bar (where the gap is)
    bar3 = data.iloc[bar_index + 3] # Third bar of the 3-bar pattern

    if direction == "bullish":
        # Bullish FVG: bar1.High < bar3.Low, and bar2 is a bullish candle (strong move)
        if bar1['High'] < bar3['Low'] and bar2['Close'] > bar2['Open']:
            return bar1['High'], bar3['Low'] # FVG is the space between bar1's high and bar3's low
    elif direction == "bearish":
        # Bearish FVG: bar1.Low > bar3.High, and bar2 is a bearish candle (strong move)
        if bar1['Low'] > bar3['High'] and bar2['Close'] < bar2['Open']:
            return bar3['High'], bar1['Low'] # FVG is the space between bar3's high and bar1's low
    return None


def generate_silver_bullet_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float
) -> pd.DataFrame:
    """
    Generates signals for the Silver Bullet strategy.
    Logic: FVG entry within specific 1-hour NY time windows.
    Data must have a NY-localized DatetimeIndex.
    """
    signals_list = []
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz is None or data.index.tz.zone != settings.NY_TIMEZONE.zone:
        logger.error(f"Silver Bullet: Data must have NY-localized DatetimeIndex. Current tz: {data.index.tz}")
        return pd.DataFrame()

    # Iterate through the data. Entry on current_bar 'i' into an FVG that completed on bar 'i-1'
    # (formed by i-3, i-2, i-1).
    for i in range(len(data)):
        if i < 3: # Need at least 3 previous bars to check for an FVG ending at i-1
            continue

        current_bar = data.iloc[i]
        current_bar_time_dt = data.index[i]       # This is a datetime object
        current_bar_ny_time_obj = current_bar_time_dt.time() # This is a datetime.time object

        # Check if the current bar's time is within any of the Silver Bullet windows
        in_sb_window = False
        for start_t, end_t in settings.SILVER_BULLET_WINDOWS_NY:
            if start_t <= current_bar_ny_time_obj < end_t:
                in_sb_window = True
                break
        
        if not in_sb_window:
            continue

        # Check for FVG formed by bars (i-3), (i-2), (i-1).
        # `bar_index` for `_find_fvg_for_silver_bullet` should be `i-4`
        # so that bar1 = data[i-3], bar2 = data[i-2], bar3 = data[i-1]
        fvg_check_idx = i - 4
        if fvg_check_idx < 0: continue


        # Bullish FVG check
        bullish_fvg_details = _find_fvg_for_silver_bullet(data, fvg_check_idx, "bullish")
        if bullish_fvg_details:
            fvg_b_low_boundary, fvg_b_high_boundary = bullish_fvg_details # (bar1.High, bar3.Low)
            # Entry if current_bar.Low dips into FVG (i.e., below fvg_b_high_boundary)
            # and Close is bullish and above the FVG's low (fvg_b_low_boundary)
            if current_bar['Low'] <= fvg_b_high_boundary and \
               current_bar['Close'] > fvg_b_low_boundary and \
               current_bar['Close'] > current_bar['Open']: # Bullish close
                entry_price = current_bar['Close']
                sl = entry_price - stop_loss_points
                tp = entry_price + (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': current_bar_time_dt, 'SignalType': 'Long', 'EntryPrice': entry_price,
                    'SL': sl, 'TP': tp, 'Reason': f"SB Long: FVG {fvg_b_low_boundary:.2f}-{fvg_b_high_boundary:.2f} in window"
                })
                logger.debug(f"SB Long @ {current_bar_time_dt}, FVG: {fvg_b_low_boundary:.2f}-{fvg_b_high_boundary:.2f}, Entry: {entry_price:.2f}")
                continue # Avoid multiple signals on same FVG with subsequent bars if logic allows

        # Bearish FVG check
        bearish_fvg_details = _find_fvg_for_silver_bullet(data, fvg_check_idx, "bearish")
        if bearish_fvg_details:
            fvg_s_low_boundary, fvg_s_high_boundary = bearish_fvg_details # (bar3.High, bar1.Low)
            # Entry if current_bar.High rallies into FVG (i.e., above fvg_s_low_boundary)
            # and Close is bearish and below the FVG's high (fvg_s_high_boundary)
            if current_bar['High'] >= fvg_s_low_boundary and \
               current_bar['Close'] < fvg_s_high_boundary and \
               current_bar['Close'] < current_bar['Open']: # Bearish close
                entry_price = current_bar['Close']
                sl = entry_price + stop_loss_points
                tp = entry_price - (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': current_bar_time_dt, 'SignalType': 'Short', 'EntryPrice': entry_price,
                    'SL': sl, 'TP': tp, 'Reason': f"SB Short: FVG {fvg_s_low_boundary:.2f}-{fvg_s_high_boundary:.2f} in window"
                })
                logger.debug(f"SB Short @ {current_bar_time_dt}, FVG: {fvg_s_low_boundary:.2f}-{fvg_s_high_boundary:.2f}, Entry: {entry_price:.2f}")
                continue
                    
    if not signals_list:
        logger.debug("Silver Bullet: No signals generated.")
    else:
        logger.info(f"Silver Bullet: Generated {len(signals_list)} signals.")
    return pd.DataFrame(signals_list)
