# services/strategies/unicorn.py
"""
Signal generation logic for the Unicorn strategy (Breaker Block + FVG).
Relies on standard Python import mechanisms, assuming project root is in sys.path.
Current implementation is a simplified FVG entry.
"""
import pandas as pd
import numpy as np

# Project-specific imports
from config import settings
from utils.logger import get_logger
from utils.technical_analysis import find_fvg, find_swing_points # Centralized TA functions

logger = get_logger(__name__)

# --- Helper functions _find_fvg and _find_swing_points are now imported from utils.technical_analysis ---

def generate_unicorn_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float
) -> pd.DataFrame:
    """
    Generates signals for the Unicorn strategy.
    Current implementation is a simplified FVG entry: looks for an FVG that just formed
    and enters if the current bar retraces into it and shows a reaction.
    Full Breaker + FVG overlap requires more complex pattern recognition.
    """
    signals_list = []
    # Ensure enough data for swing point calculation (n bars on each side) and FVG (3 bars)
    # Minimum data: (n*2 for swings) + (3 for FVG pattern) + (1 for current bar)
    # If n = settings.UNICORN_SWING_LOOKBACK (e.g., 5), then 5*2 + 3 + 1 = 14 bars minimum.
    # Let's use a slightly more conservative check.
    if len(data) < (settings.UNICORN_SWING_LOOKBACK * 2 + 5):
        logger.warning(f"Unicorn: Not enough data for signal generation. Need at least {settings.UNICORN_SWING_LOOKBACK * 2 + 5} bars, got {len(data)}.")
        return pd.DataFrame()

    # Note: Full Unicorn logic would use `find_swing_points` to identify breaker blocks.
    # The current simplified version focuses on FVG entries.
    # data_with_swings = find_swing_points(data.copy(), n=settings.UNICORN_SWING_LOOKBACK)

    # Iterate through the data, allowing for FVG lookback.
    # If FVG is formed by bars (k-2, k-1, k), we check entry on bar k+1.
    # So, if current_bar is at index `i`, the FVG would have formed on bars (i-3, i-2, i-1).
    # `find_fvg` expects `bar_index` to be the bar *before* the 3-bar FVG pattern.
    # So, to check an FVG formed by bars (i-3, i-2, i-1), `bar_index` for `find_fvg` should be `i-4`.
    for i in range(len(data)):
        if i < 3: # Need at least 3 previous bars (i-3, i-2, i-1) to check for an FVG, plus current bar (i) for entry.
                  # So, i-4 must be >= 0. This means i must be >= 4.
            continue

        current_bar = data.iloc[i]
        current_bar_time = data.index[i]
        
        fvg_check_idx = i - 4 # Index of bar *before* the potential 3-bar FVG pattern (i-3, i-2, i-1)
        if fvg_check_idx < 0: # Should be caught by the i < 3 check, but good for safety.
            continue

        # Bullish FVG Entry (Simplified Unicorn)
        bullish_fvg_details = find_fvg(data, fvg_check_idx, "bullish")
        if bullish_fvg_details:
            fvg_b_low_boundary, fvg_b_high_boundary = bullish_fvg_details # (bar1.High, bar3.Low)
            # Entry condition: current_bar (i) dips into the FVG and shows bullish reaction
            if current_bar['Low'] <= fvg_b_high_boundary and \
               current_bar['Close'] > fvg_b_low_boundary and \
               current_bar['Close'] > current_bar['Open']: # Bullish close
                entry_price = current_bar['Close']
                sl_price = entry_price - stop_loss_points
                tp_price = entry_price + (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': current_bar_time, 'SignalType': 'Long',
                    'EntryPrice': entry_price, 'SL': sl_price, 'TP': tp_price,
                    'Reason': f"Unicorn (Bullish FVG Entry): {fvg_b_low_boundary:.2f}-{fvg_b_high_boundary:.2f}"
                })
                logger.debug(f"Unicorn Long @ {current_bar_time}, FVG: {fvg_b_low_boundary:.2f}-{fvg_b_high_boundary:.2f}, Entry: {entry_price:.2f}")
                # Potentially break or add logic to avoid multiple signals for the same setup if desired

        # Bearish FVG Entry (Simplified Unicorn)
        bearish_fvg_details = find_fvg(data, fvg_check_idx, "bearish")
        if bearish_fvg_details:
            fvg_s_low_boundary, fvg_s_high_boundary = bearish_fvg_details # (bar3.High, bar1.Low)
            if current_bar['High'] >= fvg_s_low_boundary and \
               current_bar['Close'] < fvg_s_high_boundary and \
               current_bar['Close'] < current_bar['Open']: # Bearish close
                entry_price = current_bar['Close']
                sl_price = entry_price + stop_loss_points
                tp_price = entry_price - (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': current_bar_time, 'SignalType': 'Short',
                    'EntryPrice': entry_price, 'SL': sl_price, 'TP': tp_price,
                    'Reason': f"Unicorn (Bearish FVG Entry): {fvg_s_low_boundary:.2f}-{fvg_s_high_boundary:.2f}"
                })
                logger.debug(f"Unicorn Short @ {current_bar_time}, FVG: {fvg_s_low_boundary:.2f}-{fvg_s_high_boundary:.2f}, Entry: {entry_price:.2f}")

    if not signals_list:
        logger.debug("Unicorn: No signals generated (using simplified FVG logic).")
    else:
        logger.info(f"Unicorn: Generated {len(signals_list)} signals (simplified FVG logic).")
    
    logger.warning("Unicorn strategy currently uses a simplified FVG entry logic. Full Breaker+FVG overlap is complex and not fully implemented.")
    return pd.DataFrame(signals_list)
