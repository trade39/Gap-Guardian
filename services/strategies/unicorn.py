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

logger = get_logger(__name__)

# --- Helper Functions for Unicorn Strategy ---
def _find_fvg(data: pd.DataFrame, bar_index: int, direction: str = "bullish") -> tuple[float, float] | None:
    """
    Identifies a Fair Value Gap (FVG) based on a 3-bar pattern.
    The FVG is identified based on the bars at indices: bar_index+1, bar_index+2, bar_index+3.
    The FVG itself is the price range on bar_index+2, defined by the wicks of bar_index+1 and bar_index+3.

    Args:
        data (pd.DataFrame): OHLC price data.
        bar_index (int): Index of the bar *before* the 3-bar pattern starts.
                         So, pattern is data[bar_index+1], data[bar_index+2], data[bar_index+3].
        direction (str): "bullish" or "bearish".

    Returns:
        tuple[float, float] | None: (FVG_low, FVG_high) or None if no FVG.
                                     For bullish FVG: (bar1.High, bar3.Low)
                                     For bearish FVG: (bar3.High, bar1.Low)
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

def _find_swing_points(data: pd.DataFrame, n: int = settings.UNICORN_SWING_LOOKBACK) -> pd.DataFrame:
    """
    Identifies swing highs and lows based on 'n' bars on each side.
    A swing high is higher than 'n' bars to its left and 'n' bars to its right.
    A swing low is lower than 'n' bars to its left and 'n' bars to its right.
    This is a simple definition; more complex ones exist (e.g., fractal based).
    """
    data_copy = data.copy()
    data_copy['SwingHigh'] = np.nan
    data_copy['SwingLow'] = np.nan

    # Iterate from n to len-n to allow lookback and lookforward
    for i in range(n, len(data_copy) - n):
        # Check for Swing High
        is_swing_high = True
        for j in range(1, n + 1):
            if data_copy['High'].iloc[i] < data_copy['High'].iloc[i-j] or \
               data_copy['High'].iloc[i] < data_copy['High'].iloc[i+j]:
                is_swing_high = False
                break
        if is_swing_high:
             # Additional check: current high must be strictly greater than immediate neighbors
             # to avoid flat tops being marked multiple times if n=1.
            if data_copy['High'].iloc[i] > data_copy['High'].iloc[i-1] and \
               data_copy['High'].iloc[i] > data_copy['High'].iloc[i+1]:
                data_copy.loc[data_copy.index[i], 'SwingHigh'] = data_copy['High'].iloc[i]

        # Check for Swing Low
        is_swing_low = True
        for j in range(1, n + 1):
            if data_copy['Low'].iloc[i] > data_copy['Low'].iloc[i-j] or \
               data_copy['Low'].iloc[i] > data_copy['Low'].iloc[i+j]:
                is_swing_low = False
                break
        if is_swing_low:
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
    Current implementation is a simplified FVG entry: looks for an FVG that just formed
    and enters if the current bar retraces into it and shows a reaction.
    Full Breaker + FVG overlap requires more complex pattern recognition.
    """
    signals_list = []
    if len(data) < (settings.UNICORN_SWING_LOOKBACK * 2 + 5): # Need enough data for swings and FVG checks
        logger.warning("Unicorn: Not enough data for signal generation.")
        return pd.DataFrame()

    # Note: Full Unicorn logic would use `_find_swing_points` to identify breaker blocks.
    # The current simplified version focuses on FVG entries.
    # data_with_swings = _find_swing_points(data.copy(), n=settings.UNICORN_SWING_LOOKBACK)

    # Iterate through the data, allowing for FVG lookback.
    # If FVG is formed by bars (k-2, k-1, k), we check entry on bar k+1.
    # So, if current_bar is at index `i`, the FVG would have formed on bars (i-3, i-2, i-1).
    for i in range(len(data)): # Iterate up to the second to last bar to allow FVG check on i-3, i-2, i-1
        if i < 3: # Need at least 3 previous bars to check for an FVG ending at i-1
            continue

        current_bar = data.iloc[i]
        current_bar_time = data.index[i]

        # Bullish FVG Entry (Simplified Unicorn)
        # FVG formed by bars (i-3), (i-2), (i-1). Bar (i-2) is the one with the gap.
        # FVG range is [(i-3).High, (i-1).Low]. We look for entry on current_bar (i).
        bullish_fvg = _find_fvg(data, i - 3, "bullish") # Pass index of bar *before* 3-bar pattern
        if bullish_fvg:
            # For bullish FVG, _find_fvg returns (bar(i-2).High, bar(i).Low) if pattern is (i-2,i-1,i)
            # If pattern is (i-3,i-2,i-1), then it's (bar(i-2).High, bar(i).Low)
            # Corrected: if pattern is data[idx+1], data[idx+2], data[idx+3]
            # For bullish: data[idx+1].High, data[idx+3].Low
            # So, if _find_fvg(data, i-3, ...), pattern is data[i-2], data[i-1], data[i]
            # and FVG is (data[i-2].High, data[i].Low) -- this seems off.
            # Let's re-verify _find_fvg:
            # bar1=data[idx+1], bar2=data[idx+2], bar3=data[idx+3]
            # Bullish: bar1.High < bar3.Low. Returns (bar1.High, bar3.Low)
            # So, if we call _find_fvg(data, i-3, ...):
            # bar1=data[i-2], bar2=data[i-1], bar3=data[i]
            # FVG is (data[i-2].High, data[i].Low)
            # This means the FVG is formed by the candle at i-1, using wicks of i-2 and i.
            # This is not standard. Standard FVG: 3 candles. candle1, candle2, candle3.
            # Bullish FVG: candle1.high < candle3.low. FVG is the space.
            # Let's assume _find_fvg is correct and it means FVG is between bar1.High and bar3.Low.
            # If we want FVG formed by (i-3, i-2, i-1), then call _find_fvg(data, i-4, ...)
            # Let's use the provided _find_fvg and assume its indexing logic.
            # If we want to check an FVG that completed with bar `i-1`, formed by `i-3, i-2, i-1`.
            # Then `bar_index` for `_find_fvg` should be `i-4`.
            # `bar1 = data[i-3]`, `bar2 = data[i-2]`, `bar3 = data[i-1]`
            # FVG is `(data[i-3].High, data[i-1].Low)`

            fvg_check_idx = i - 4 # To check FVG formed by bars i-3, i-2, i-1
            if fvg_check_idx < 0: continue
            
            bullish_fvg_details = _find_fvg(data, fvg_check_idx, "bullish")
            if bullish_fvg_details:
                fvg_b_low_boundary, fvg_b_high_boundary = bullish_fvg_details # (bar1.High, bar3.Low)
                # Entry condition: current_bar (i) dips into the FVG and shows bullish reaction
                if current_bar['Low'] <= fvg_b_high_boundary and \
                   current_bar['Close'] > fvg_b_low_boundary and \
                   current_bar['Close'] > current_bar['Open']: # Bullish close
                    entry_price = current_bar['Close']
                    sl_price = entry_price - stop_loss_points # Or below FVG low / structure
                    tp_price = entry_price + (stop_loss_points * rrr)
                    signals_list.append({
                        'SignalTime': current_bar_time, 'SignalType': 'Long',
                        'EntryPrice': entry_price, 'SL': sl_price, 'TP': tp_price,
                        'Reason': f"Unicorn (Bullish FVG Entry): {fvg_b_low_boundary:.2f}-{fvg_b_high_boundary:.2f}"
                    })
                    logger.debug(f"Unicorn Long @ {current_bar_time}, FVG: {fvg_b_low_boundary:.2f}-{fvg_b_high_boundary:.2f}, Entry: {entry_price:.2f}")
                    # Potentially break or add logic to avoid multiple signals for the same setup

        # Bearish FVG Entry (Simplified Unicorn)
            bearish_fvg_details = _find_fvg(data, fvg_check_idx, "bearish")
            if bearish_fvg_details:
                fvg_s_low_boundary, fvg_s_high_boundary = bearish_fvg_details # (bar3.High, bar1.Low)
                if current_bar['High'] >= fvg_s_low_boundary and \
                   current_bar['Close'] < fvg_s_high_boundary and \
                   current_bar['Close'] < current_bar['Open']: # Bearish close
                    entry_price = current_bar['Close']
                    sl_price = entry_price + stop_loss_points # Or above FVG high / structure
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
