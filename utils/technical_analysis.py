# utils/technical_analysis.py
"""
Common technical analysis helper functions.
"""
import pandas as pd
import numpy as np
from config import settings # For UNICORN_SWING_LOOKBACK

def find_fvg(data: pd.DataFrame, bar_index: int, direction: str = "bullish") -> tuple[float, float] | None:
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
                                     For bullish FVG: (bar1.High, bar3.Low) - FVG is the space.
                                     For bearish FVG: (bar3.High, bar1.Low) - FVG is the space.
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

def find_swing_points(data: pd.DataFrame, n: int = settings.UNICORN_SWING_LOOKBACK) -> pd.DataFrame:
    """
    Identifies swing highs and lows based on 'n' bars on each side.
    A swing high is higher than 'n' bars to its left and 'n' bars to its right.
    A swing low is lower than 'n' bars to its left and 'n' bars to its right.
    
    Args:
        data (pd.DataFrame): OHLC price data.
        n (int): Number of bars on each side to define a swing point. 
                 Defaults to settings.UNICORN_SWING_LOOKBACK.

    Returns:
        pd.DataFrame: A copy of the input data with 'SwingHigh' and 'SwingLow' columns added.
                      SwingHigh contains the high price at swing high points, NaN otherwise.
                      SwingLow contains the low price at swing low points, NaN otherwise.
    """
    data_copy = data.copy()
    data_copy['SwingHigh'] = np.nan
    data_copy['SwingLow'] = np.nan

    # Iterate from n to len-n to allow lookback and lookforward
    for i in range(n, len(data_copy) - n):
        # Check for Swing High
        is_swing_high = True
        for j in range(1, n + 1):
            if data_copy['High'].iloc[i] <= data_copy['High'].iloc[i-j] or \
               data_copy['High'].iloc[i] <= data_copy['High'].iloc[i+j]:
                is_swing_high = False
                break
        if is_swing_high:
            # Ensure it's strictly greater than immediate neighbors to avoid flat tops for n=1
            if data_copy['High'].iloc[i] > data_copy['High'].iloc[i-1] and \
               data_copy['High'].iloc[i] > data_copy['High'].iloc[i+1]:
                data_copy.loc[data_copy.index[i], 'SwingHigh'] = data_copy['High'].iloc[i]

        # Check for Swing Low
        is_swing_low = True
        for j in range(1, n + 1):
            if data_copy['Low'].iloc[i] >= data_copy['Low'].iloc[i-j] or \
               data_copy['Low'].iloc[i] >= data_copy['Low'].iloc[i+j]:
                is_swing_low = False
                break
        if is_swing_low:
            # Ensure it's strictly lower than immediate neighbors
            if data_copy['Low'].iloc[i] < data_copy['Low'].iloc[i-1] and \
               data_copy['Low'].iloc[i] < data_copy['Low'].iloc[i+1]:
                data_copy.loc[data_copy.index[i], 'SwingLow'] = data_copy['Low'].iloc[i]
                
    return data_copy
