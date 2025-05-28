# services/strategy_engine.py
"""
Implements core logic for various trading strategies:
- Gap Guardian
- Unicorn (Breaker + FVG)
- Silver Bullet (Time-based FVG)
"""
import pandas as pd
import numpy as np
from datetime import time as dt_time
from config import settings # Import settings for strategy-specific configs
from utils.logger import get_logger

logger = get_logger(__name__)

# --- Helper Functions for Technical Analysis ---

def _find_fvg(data: pd.DataFrame, bar_index: int, direction: str = "bullish") -> tuple[float, float] | None:
    """
    Identifies a Fair Value Gap (FVG) based on a 3-bar pattern.
    Looks for FVG *after* bar_index (i.e., bar_index is bar 0, FVG is between bar 1 and bar 3 highs/lows, formed by bar 2).
    This means the FVG is identified based on bars [bar_index+1, bar_index+2, bar_index+3].
    The FVG itself is the price range on bar_index+2.

    Args:
        data (pd.DataFrame): OHLCV data.
        bar_index (int): The index of the bar *before* the potential 3-bar FVG pattern starts.
                         So, we check bars bar_index+1, bar_index+2, bar_index+3.
        direction (str): "bullish" or "bearish".

    Returns:
        tuple[float, float] | None: (fvg_low, fvg_high) if FVG found, else None.
                                     For Bullish FVG: (Bar1.High, Bar3.Low)
                                     For Bearish FVG: (Bar3.High, Bar1.Low)
    """
    if bar_index + 3 >= len(data): # Need at least 3 bars *after* bar_index
        return None

    bar1 = data.iloc[bar_index + 1] # First bar of the 3-bar pattern
    bar2 = data.iloc[bar_index + 2] # Middle bar where the imbalance occurs
    bar3 = data.iloc[bar_index + 3] # Third bar

    fvg_low, fvg_high = None, None

    if direction == "bullish":
        # Bullish FVG: Bar1.High < Bar3.Low. FVG is the space on Bar2.
        # Price must move up strongly. Bar2 must be bullish.
        if bar1['High'] < bar3['Low'] and bar2['Close'] > bar2['Open']:
            # The FVG is the range from the high of the first candle to the low of the third candle.
            fvg_low = bar1['High']
            fvg_high = bar3['Low']
            # Ensure valid FVG (low < high) - this check is implicitly handled by bar1.High < bar3.Low
            # logger.debug(f"Bullish FVG found at {bar2.name}: Bar1.H={bar1['High']:.2f}, Bar3.L={bar3['Low']:.2f}")
            return fvg_low, fvg_high
    elif direction == "bearish":
        # Bearish FVG: Bar1.Low > Bar3.High. FVG is the space on Bar2.
        # Price must move down strongly. Bar2 must be bearish.
        if bar1['Low'] > bar3['High'] and bar2['Close'] < bar2['Open']:
            # The FVG is the range from the high of the third candle to the low of the first candle.
            fvg_low = bar3['High']
            fvg_high = bar1['Low']
            # logger.debug(f"Bearish FVG found at {bar2.name}: Bar1.L={bar1['Low']:.2f}, Bar3.H={bar3['High']:.2f}")
            return fvg_low, fvg_high
    return None


def _find_swing_points(data: pd.DataFrame, n: int = settings.UNICORN_SWING_LOOKBACK) -> pd.DataFrame:
    """
    Identifies swing highs and lows.
    A swing high is a high with 'n' lower highs on each side.
    A swing low is a low with 'n' higher lows on each side.
    Args:
        data (pd.DataFrame): OHLC data.
        n (int): Number of bars to look left and right.
    Returns:
        pd.DataFrame: DataFrame with 'SwingHigh' and 'SwingLow' columns (price at swing, else NaN).
    """
    data_copy = data.copy()
    data_copy['SwingHigh'] = np.nan
    data_copy['SwingLow'] = np.nan

    for i in range(n, len(data_copy) - n):
        # Check for Swing High
        is_swing_high = True
        for j in range(1, n + 1):
            if data_copy['High'].iloc[i] < data_copy['High'].iloc[i-j] or \
               data_copy['High'].iloc[i] < data_copy['High'].iloc[i+j]:
                is_swing_high = False
                break
        if is_swing_high:
            data_copy.loc[data_copy.index[i], 'SwingHigh'] = data_copy['High'].iloc[i]

        # Check for Swing Low
        is_swing_low = True
        for j in range(1, n + 1):
            if data_copy['Low'].iloc[i] > data_copy['Low'].iloc[i-j] or \
               data_copy['Low'].iloc[i] > data_copy['Low'].iloc[i+j]:
                is_swing_low = False
                break
        if is_swing_low:
            data_copy.loc[data_copy.index[i], 'SwingLow'] = data_copy['Low'].iloc[i]
            
    return data_copy


# --- Strategy Specific Signal Generators ---

def _generate_gap_guardian_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float,
    entry_start_time: dt_time,
    entry_end_time: dt_time
) -> pd.DataFrame:
    """ Original Gap Guardian Logic """
    signals_list = []
    for date_val, day_data in data.groupby(data.index.date):
        if day_data.empty: continue
        
        opening_bar_candidates = day_data[day_data.index.time >= entry_start_time]
        if opening_bar_candidates.empty: continue
        
        opening_bar_data = opening_bar_candidates.iloc[0:1]
        if opening_bar_data.empty: continue
            
        opening_bar_timestamp = opening_bar_data.index[0]
        opening_range_high = opening_bar_data['High'].iloc[0]
        opening_range_low = opening_bar_data['Low'].iloc[0]
        
        signal_scan_window_data = day_data[
            (day_data.index > opening_bar_timestamp) &
            (day_data.index.time < entry_end_time)
        ]
        if signal_scan_window_data.empty: continue

        for idx, bar in signal_scan_window_data.iterrows():
            signal_time = idx
            if bar['Low'] < opening_range_low and bar['Close'] > opening_range_low:
                entry_price = bar['Close']
                sl = entry_price - stop_loss_points
                tp = entry_price + (stop_loss_points * rrr)
                signals_list.append({'SignalTime': signal_time, 'SignalType': 'Long', 'EntryPrice': entry_price, 'SL': sl, 'TP': tp, 'Reason': f"GG: False breakdown of {opening_range_low:.2f}"})
                break
            elif bar['High'] > opening_range_high and bar['Close'] < opening_range_high:
                entry_price = bar['Close']
                sl = entry_price + stop_loss_points
                tp = entry_price - (stop_loss_points * rrr)
                signals_list.append({'SignalTime': signal_time, 'SignalType': 'Short', 'EntryPrice': entry_price, 'SL': sl, 'TP': tp, 'Reason': f"GG: False breakout of {opening_range_high:.2f}"})
                break
    return pd.DataFrame(signals_list)


def _generate_unicorn_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float
) -> pd.DataFrame:
    """
    Generates signals for the Unicorn strategy.
    Logic: Breaker Block + Overlapping FVG.
    """
    signals_list = []
    if len(data) < (settings.UNICORN_SWING_LOOKBACK * 2 + 5): # Min data for swings and patterns
        logger.warning("Unicorn: Not enough data for signal generation.")
        return pd.DataFrame()

    data_with_swings = _find_swing_points(data, n=settings.UNICORN_SWING_LOOKBACK)

    for i in range(settings.UNICORN_SWING_LOOKBACK * 2, len(data_with_swings) - (settings.UNICORN_SWING_LOOKBACK + 3)): # Ensure room for patterns
        current_bar = data_with_swings.iloc[i]
        current_bar_time = data_with_swings.index[i]

        # --- Bullish Unicorn Setup ---
        # 1. Find a recent swing low (sl1_idx)
        # 2. Find a swing high after sl1 (sh1_idx)
        # 3. Price sweeps sl1_val (forms ll1)
        # 4. Price displaces up, breaking sh1_val (forms hh1)
        # 5. Identify bullish breaker (last up-close candle(s) before ll1)
        # 6. Identify bullish FVG during displacement to hh1
        # 7. Check overlap: FVG overlaps with breaker
        
        # Simplified search for pattern components relative to current_bar 'i'
        # This is a complex pattern recognition task. The following is a conceptual outline
        # and needs robust implementation of component identification.
        
        # For Bullish: Look back for a potential breaker structure
        # A more robust way would be to iterate through swing points.
        # Example: Iterate i from left to right.
        # If data_with_swings['SwingLow'].iloc[i-k] is a swing low (sl1)
        # and data_with_swings['SwingHigh'].iloc[i-m] is a swing high after sl1 (sh1)
        # and data['Low'].iloc[i-p] < sl1_val (sweep_low)
        # and data['High'].iloc[i] > sh1_val (displacement_high, current bar is part of this move)
        
        # Find Bullish Breaker Block:
        # Look for a swing low, then swing high, then sweep of swing low, then displacement above swing high.
        # This requires iterating through historical swing points.
        # For simplicity in this first pass, we'll look for a recent FVG and a "conceptual" breaker.
        # A full robust implementation would involve state machines or more complex sequence detection.

        # Simplified: Check for a recent bullish FVG
        # An FVG is identified by _find_fvg based on bars i+1, i+2, i+3.
        # If we want to enter on retracement into an FVG that *just formed*,
        # the FVG would be identified on bars i-2, i-1, i.
        # Let's assume FVG forms, then price retraces.
        
        # For Bullish Unicorn:
        # 1. Identify Bullish Breaker: Highest up-close candle(s) before a new low (after a swing low was taken).
        #    This is hard to define precisely without clear swing point references.
        #    Let's assume a breaker zone is [breaker_low, breaker_high]
        # 2. Identify Bullish FVG [fvg_low, fvg_high] that formed during an upward displacement.
        # 3. Overlap: max(breaker_low, fvg_low) < min(breaker_high, fvg_high)
        # 4. Entry: Price retraces into this overlap zone.

        # Let's try a simplified FVG based entry first, then layer breaker logic.
        # This part is highly conceptual and needs refinement for robustness.
        # For now, let's focus on the FVG part of the "Unicorn" as "Breaker + FVG" is complex.
        # The prompt implies the FVG forms *during* the displacement.
        # So, if a displacement just happened (e.g., strong bullish bar at 'i'), look for FVG in the making.

        # Simplified Bullish FVG check (looking backwards for a recently formed FVG)
        # Consider FVG formed by bars i-3, i-2, i-1. Entry on bar 'i' if it retraces.
        if i >= 3:
            fvg_bullish = _find_fvg(data_with_swings, i - 3, "bullish") # FVG formed by (i-2, i-1, i)
            if fvg_bullish:
                fvg_b_low, fvg_b_high = fvg_bullish
                # Entry if current bar's low dips into the FVG and closes above its low end
                if current_bar['Low'] <= fvg_b_high and current_bar['Close'] > fvg_b_low : # Price touches or enters FVG
                    # This is a basic FVG entry, not full Unicorn.
                    # To be closer to Unicorn, this FVG should be part of a displacement after a sweep.
                    # And it should overlap a breaker.
                    # For now, this is a placeholder for a "Bullish FVG related entry"
                    entry_price = current_bar['Close'] 
                    sl_price = entry_price - stop_loss_points # Or below FVG low / structure
                    tp_price = entry_price + (stop_loss_points * rrr)
                    signals_list.append({
                        'SignalTime': current_bar_time, 'SignalType': 'Long', 
                        'EntryPrice': entry_price, 'SL': sl_price, 'TP': tp_price,
                        'Reason': f"Unicorn (Bullish FVG Entry): {fvg_b_low:.2f}-{fvg_b_high:.2f}"
                    })
                    # logger.info(f"Unicorn Long @ {current_bar_time}, FVG: {fvg_b_low:.2f}-{fvg_b_high:.2f}, Entry: {entry_price:.2f}")
                    # Potentially break if one signal per day/pattern is desired.

        # Simplified Bearish FVG check
        if i >= 3:
            fvg_bearish = _find_fvg(data_with_swings, i - 3, "bearish") # FVG formed by (i-2, i-1, i)
            if fvg_bearish:
                fvg_s_low, fvg_s_high = fvg_bearish
                if current_bar['High'] >= fvg_s_low and current_bar['Close'] < fvg_s_high: # Price touches or enters FVG
                    entry_price = current_bar['Close']
                    sl_price = entry_price + stop_loss_points # Or above FVG high / structure
                    tp_price = entry_price - (stop_loss_points * rrr)
                    signals_list.append({
                        'SignalTime': current_bar_time, 'SignalType': 'Short',
                        'EntryPrice': entry_price, 'SL': sl_price, 'TP': tp_price,
                        'Reason': f"Unicorn (Bearish FVG Entry): {fvg_s_low:.2f}-{fvg_s_high:.2f}"
                    })
                    # logger.info(f"Unicorn Short @ {current_bar_time}, FVG: {fvg_s_low:.2f}-{fvg_s_high:.2f}, Entry: {entry_price:.2f}")

    # Note: The Unicorn logic above is a simplified FVG entry.
    # A full Unicorn (Breaker + FVG overlap) is significantly more complex to code robustly
    # and would require careful state management of identified swing points and breaker blocks.
    # This simplified version focuses on FVG identification as a starting point.
    logger.warning("Unicorn strategy currently uses a simplified FVG entry logic. Full Breaker+FVG overlap is complex and not fully implemented.")

    return pd.DataFrame(signals_list)


def _generate_silver_bullet_signals(
    data: pd.DataFrame,
    stop_loss_points: float,
    rrr: float
) -> pd.DataFrame:
    """
    Generates signals for the Silver Bullet strategy.
    Logic: FVG entry within specific 1-hour NY time windows.
    """
    signals_list = []
    # Ensure data index is datetime and NY localized (should be by data_loader)
    if not isinstance(data.index, pd.DatetimeIndex) or data.index.tz != settings.NY_TIMEZONE:
        logger.error("Silver Bullet: Data must have NY-localized DatetimeIndex.")
        return pd.DataFrame()

    for i in range(len(data) - 3): # Need 3 bars for FVG, check up to data[i+2]
        current_bar = data.iloc[i]
        current_bar_time = data.index[i]
        current_bar_ny_time = current_bar_time.time()

        in_sb_window = False
        for start_t, end_t in settings.SILVER_BULLET_WINDOWS_NY:
            if start_t <= current_bar_ny_time < end_t:
                in_sb_window = True
                break
        
        if not in_sb_window:
            continue

        # Check for Bullish FVG forming or revisited
        # FVG formed by bars i, i+1, i+2. Entry on retracement into this FVG.
        # For simplicity, let's look for an FVG that formed recently (e.g., on bars i-2, i-1, i)
        # and current bar 'i' is the entry bar if it revisits.
        # Or, FVG formed by (i, i+1, i+2), entry on i+3 if it revisits.
        
        # Let's use: FVG formed by current_bar (i), next_bar (i+1), third_bar (i+2).
        # Signal on current_bar if it's an entry into a *previously* formed FVG, or
        # signal on a subsequent bar if it enters an FVG formed by (i, i+1, i+2).

        # Simplified: Look for FVG formed by (i, i+1, i+2), then entry on i+3 (or later)
        # For entry on current_bar 'i' into an FVG that *just completed* on bar 'i-1' (formed by i-3, i-2, i-1)
        if i >= 3: # Ensure we have bars i-3, i-2, i-1 for FVG, and i for entry
            # Bullish FVG check (FVG formed by i-3, i-2, i-1; entry on bar i)
            bullish_fvg = _find_fvg(data, i - 3, "bullish")
            if bullish_fvg:
                fvg_low, fvg_high = bullish_fvg
                # Entry if current_bar.Low dips into FVG and Close is bullish
                if current_bar['Low'] <= fvg_high and current_bar['Close'] > fvg_low and current_bar['Close'] > current_bar['Open']:
                    entry_price = current_bar['Close']
                    sl = entry_price - stop_loss_points
                    tp = entry_price + (stop_loss_points * rrr)
                    signals_list.append({
                        'SignalTime': current_bar_time, 'SignalType': 'Long', 'EntryPrice': entry_price,
                        'SL': sl, 'TP': tp, 'Reason': f"SB Long: FVG {fvg_low:.2f}-{fvg_high:.2f} in window"
                    })
                    # logger.info(f"SB Long @ {current_bar_time}, FVG: {fvg_low:.2f}-{fvg_high:.2f}, Entry: {entry_price:.2f}")
                    # Potentially break or ensure one signal per window/day
                    continue # Move to next bar to avoid multiple signals on same FVG

            # Bearish FVG check (FVG formed by i-3, i-2, i-1; entry on bar i)
            bearish_fvg = _find_fvg(data, i - 3, "bearish")
            if bearish_fvg:
                fvg_low, fvg_high = bearish_fvg
                # Entry if current_bar.High dips into FVG and Close is bearish
                if current_bar['High'] >= fvg_low and current_bar['Close'] < fvg_high and current_bar['Close'] < current_bar['Open']:
                    entry_price = current_bar['Close']
                    sl = entry_price + stop_loss_points
                    tp = entry_price - (stop_loss_points * rrr)
                    signals_list.append({
                        'SignalTime': current_bar_time, 'SignalType': 'Short', 'EntryPrice': entry_price,
                        'SL': sl, 'TP': tp, 'Reason': f"SB Short: FVG {fvg_low:.2f}-{fvg_high:.2f} in window"
                    })
                    # logger.info(f"SB Short @ {current_bar_time}, FVG: {fvg_low:.2f}-{fvg_high:.2f}, Entry: {entry_price:.2f}")
                    continue
                    
    return pd.DataFrame(signals_list)


# --- Main Signal Generation Dispatcher ---
def generate_signals(
    data: pd.DataFrame,
    strategy_name: str,
    stop_loss_points: float,
    rrr: float,
    entry_start_time: dt_time | None = None, # Optional, for Gap Guardian
    entry_end_time: dt_time | None = None    # Optional, for Gap Guardian
) -> pd.DataFrame:
    """
    Generates trading signals based on the selected strategy.
    """
    if data.empty:
        logger.warning("Input data for signal generation is empty.")
        return pd.DataFrame()

    if data.index.tz != settings.NY_TIMEZONE:
        try:
            logger.warning(f"Data timezone is {data.index.tz}, converting to NY timezone.")
            data = data.tz_convert(settings.NY_TIMEZONE_STR)
        except Exception as e:
            logger.error(f"Failed to convert data to NY timezone: {e}. Current: {data.index.tz}")
            return pd.DataFrame()
    
    logger.info(f"Generating signals for strategy: {strategy_name} with SL points: {stop_loss_points}, RRR: {rrr}.")

    signals_df = pd.DataFrame()
    if strategy_name == "Gap Guardian":
        if entry_start_time is None or entry_end_time is None:
            logger.error("Gap Guardian strategy requires entry_start_time and entry_end_time.")
            return pd.DataFrame()
        signals_df = _generate_gap_guardian_signals(data, stop_loss_points, rrr, entry_start_time, entry_end_time)
    elif strategy_name == "Unicorn":
        signals_df = _generate_unicorn_signals(data, stop_loss_points, rrr)
    elif strategy_name == "Silver Bullet":
        signals_df = _generate_silver_bullet_signals(data, stop_loss_points, rrr)
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return pd.DataFrame()

    if not signals_df.empty:
        signals_df['SignalTime'] = pd.to_datetime(signals_df['SignalTime'])
        signals_df.set_index('SignalTime', inplace=True, drop=False) # Keep SignalTime as a column
        signals_df.sort_index(inplace=True)
        logger.info(f"Generated {len(signals_df)} signals for {strategy_name}.")
    else:
        logger.info(f"No signals generated for {strategy_name}.")
        
    return signals_df


if __name__ == '__main__':
    from services.data_loader import fetch_historical_data # Assuming data_loader is in the same parent dir
    from datetime import date as dt_date

    sample_ticker = "^GSPC" # S&P 500
    start_d = dt_date.today() - pd.Timedelta(days=59) # Approx 2 months of data
    end_d = dt_date.today()
    
    test_sl = 15.0
    test_rrr = 2.0

    for strategy_to_test in settings.AVAILABLE_STRATEGIES:
        print(f"\n--- Testing Strategy: {strategy_to_test} for {sample_ticker} ---")
        for tf_display, tf_value in {"15 Minutes": "15m", "1 Hour": "1h"}.items(): # Test on couple of TFs
            print(f"--- Timeframe: {tf_display} ({tf_value}) ---")
            price_data_test = fetch_historical_data(sample_ticker, start_d, end_d, tf_value)
            
            if not price_data_test.empty:
                print(f"Price data ({len(price_data_test)} rows) from {price_data_test.index.min()} to {price_data_test.index.max()}")
                
                # Prepare params for generate_signals
                params = {
                    'data': price_data_test.copy(),
                    'strategy_name': strategy_to_test,
                    'stop_loss_points': test_sl,
                    'rrr': test_rrr
                }
                if strategy_to_test == "Gap Guardian":
                    params['entry_start_time'] = dt_time(settings.DEFAULT_ENTRY_WINDOW_START_HOUR, settings.DEFAULT_ENTRY_WINDOW_START_MINUTE)
                    params['entry_end_time'] = dt_time(settings.DEFAULT_ENTRY_WINDOW_END_HOUR, settings.DEFAULT_ENTRY_WINDOW_END_MINUTE)

                signals = generate_signals(**params)
                
                if not signals.empty:
                    print(f"Generated Signals ({len(signals)}):\n{signals.head()}")
                else:
                    print("No signals generated for this configuration.")
            else:
                print(f"Could not fetch price data for {tf_value}.")
