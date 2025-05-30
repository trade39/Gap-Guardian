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
from utils.technical_analysis import find_fvg # Centralized TA function

logger = get_logger(__name__)

# Local _find_fvg_for_silver_bullet is now removed, using centralized find_fvg.

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
    # `find_fvg` expects `bar_index` to be the bar *before* the 3-bar FVG pattern.
    # So, to check an FVG formed by bars (i-3, i-2, i-1), `bar_index` for `find_fvg` should be `i-4`.
    for i in range(len(data)):
        if i < 3: # Need at least 3 previous bars (i-3, i-2, i-1) to check for an FVG, plus current bar (i) for entry.
                  # So, i-4 must be >= 0. This means i must be >= 4.
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
        
        fvg_check_idx = i - 4 # Index of bar *before* the potential 3-bar FVG pattern (i-3, i-2, i-1)
        if fvg_check_idx < 0: # Should be caught by i < 3 check.
            continue

        # Bullish FVG check
        bullish_fvg_details = find_fvg(data, fvg_check_idx, "bullish")
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
        bearish_fvg_details = find_fvg(data, fvg_check_idx, "bearish")
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
