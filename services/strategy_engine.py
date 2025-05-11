# services/strategy_engine.py
"""
Implements the core logic for the Gap Guardian trading strategy.
Identifies entry signals based on price action within a specified time window.
"""
import pandas as pd
import numpy as np
from datetime import time
from config.settings import (
    NY_TIMEZONE, ENTRY_WINDOW_START_HOUR, ENTRY_WINDOW_START_MINUTE,
    ENTRY_WINDOW_END_HOUR, ENTRY_WINDOW_END_MINUTE
)
from utils.logger import get_logger

logger = get_logger(__name__)

def generate_signals(data: pd.DataFrame, stop_loss_points: float, rrr: float) -> pd.DataFrame:
    """
    Generates trading signals based on the Gap Guardian strategy.

    Args:
        data (pd.DataFrame): Price data (OHLC) indexed by Datetime (must be NY timezone).
        stop_loss_points (float): Stop loss distance in price points.
        rrr (float): Risk-Reward Ratio.

    Returns:
        pd.DataFrame: DataFrame containing trade signals with columns:
                      'SignalTime', 'SignalType' ('Long'/'Short'), 'EntryPrice', 'SL', 'TP'.
                      Returns an empty DataFrame if no signals are generated.
    """
    if data.empty:
        logger.warning("Input data for signal generation is empty.")
        return pd.DataFrame()

    if data.index.tz != NY_TIMEZONE:
        logger.error(f"Data must be in New York timezone ({NY_TIMEZONE}). Current: {data.index.tz}")
        # Attempt to convert, or raise error. For now, log and return empty.
        # data = data.tz_convert(NY_TIMEZONE) # This should be done in data_loader
        return pd.DataFrame()

    logger.info(f"Generating signals with SL points: {stop_loss_points}, RRR: {rrr}")

    signals_list = []
    
    # Define the entry window times
    entry_window_start = time(ENTRY_WINDOW_START_HOUR, ENTRY_WINDOW_START_MINUTE)
    entry_window_end = time(ENTRY_WINDOW_END_HOUR, ENTRY_WINDOW_END_MINUTE)

    # Group data by date to process one day at a time (for "1 trade per day" rule)
    for date_val, day_data in data.groupby(data.index.date):
        if day_data.empty:
            continue
        
        # Filter data for the entry window
        # The first bar of the window is at or after entry_window_start
        # The last bar to consider for entry is the one starting *before* entry_window_end
        # (e.g. if window ends 11:00, 10:45 bar is last one to check for entry on its close at 11:00)
        
        # Get the 9:30 AM bar (opening range bar)
        # This assumes the data interval aligns with the start time (e.g., 15min data, 9:30 is a bar start)
        opening_bar_time = pd.Timestamp(date_val, tz=NY_TIMEZONE) + pd.Timedelta(hours=entry_window_start.hour, minutes=entry_window_start.minute)
        
        if opening_bar_time not in day_data.index:
            # logger.debug(f"No data for opening bar at {opening_bar_time} on {date_val}. Skipping day.")
            # This can happen if market is closed or data is missing.
            # Or if the first bar of the day is later than 9:30.
            # Try to get the first bar at or after 9:30
            potential_opening_bars = day_data[day_data.index.time >= entry_window_start]
            if potential_opening_bars.empty:
                logger.debug(f"No data at or after {entry_window_start} on {date_val}. Skipping day.")
                continue
            opening_bar_data = potential_opening_bars.iloc[0:1] # Get the first bar
            if opening_bar_data.empty: # Should not happen if potential_opening_bars is not empty
                 logger.debug(f"Could not determine opening bar for {date_val}. Skipping day.")
                 continue
        else:
            opening_bar_data = day_data.loc[[opening_bar_time]]

        if opening_bar_data.empty:
            logger.warning(f"Could not find opening bar data for {date_val} at {opening_bar_time}.")
            continue
            
        window_open_high = opening_bar_data['High'].iloc[0]
        window_open_low = opening_bar_data['Low'].iloc[0]
        
        # Iterate through bars within the entry window (after the opening bar)
        # Strategy: check for false break of the *opening bar's* range
        
        # Bars to check for entry signals: from the bar *after* the opening_bar_data up to entry_window_end
        # The signal is generated on the close of the bar that confirms the false break.
        # The entry_window_end means the *close* of a bar can be at 11:00. So a bar starting at 10:45 for 15m interval.
        
        # Consider bars whose *start time* is within [opening_bar_time + interval, entry_window_end - interval]
        # Or more simply, iterate bars whose time is within the window.
        
        # Bars within the active trading window for signal generation
        # Exclude the opening bar itself for forming the false break, but include its data for H/L
        signal_scan_window_data = day_data[
            (day_data.index.time > opening_bar_data.index[0].time()) & # After the opening bar
            (day_data.index.time < entry_window_end) # Bar starts before window ends
        ]

        for idx, bar in signal_scan_window_data.iterrows():
            # Current bar's time (idx.time()) must be within the allowed window
            # The bar.name (idx) is the start time of the bar.
            # The signal occurs at the close of this bar, so idx + interval.
            
            signal_time = idx # Signal occurs at the close of this bar, entry is on this close
            
            # Check for Long Signal (False Breakdown)
            if bar['Low'] < window_open_low and bar['Close'] > window_open_low:
                entry_price = bar['Close']
                sl = entry_price - stop_loss_points
                tp = entry_price + (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': signal_time, # Time of the bar close where signal is confirmed
                    'SignalType': 'Long',
                    'EntryPrice': entry_price,
                    'SL': sl,
                    'TP': tp,
                    'Reason': f"False breakdown of {window_open_low:.2f}"
                })
                logger.info(f"Long signal on {date_val} at {signal_time}: Entry={entry_price:.2f}, SL={sl:.2f}, TP={tp:.2f}")
                break # Max one trade per day

            # Check for Short Signal (False Breakout)
            elif bar['High'] > window_open_high and bar['Close'] < window_open_high:
                entry_price = bar['Close']
                sl = entry_price + stop_loss_points
                tp = entry_price - (stop_loss_points * rrr)
                signals_list.append({
                    'SignalTime': signal_time,
                    'SignalType': 'Short',
                    'EntryPrice': entry_price,
                    'SL': sl,
                    'TP': tp,
                    'Reason': f"False breakout of {window_open_high:.2f}"
                })
                logger.info(f"Short signal on {date_val} at {signal_time}: Entry={entry_price:.2f}, SL={sl:.2f}, TP={tp:.2f}")
                break # Max one trade per day
        
    if not signals_list:
        logger.info("No signals generated.")
        return pd.DataFrame()
        
    signals_df = pd.DataFrame(signals_list)
    # Ensure SignalTime is DatetimeIndex for consistency if it became Timestamp objects
    signals_df['SignalTime'] = pd.to_datetime(signals_df['SignalTime'])
    signals_df.set_index('SignalTime', inplace=True, drop=False) # Keep SignalTime as a column too
    signals_df.sort_index(inplace=True)
    
    logger.info(f"Generated {len(signals_df)} signals.")
    return signals_df


if __name__ == '__main__':
    # Example Usage (requires data_loader and running from project root)
    from services.data_loader import fetch_historical_data
    from datetime import date, datetime, time

    # Fetch sample data
    sample_ticker = "^GSPC" # S&P 500
    # Ensure enough data for testing, e.g., a few days
    # For 15m data, yfinance usually gives last 60 days.
    start_dt = date(2024, 4, 1) # A date where market was open
    end_dt = date(2024, 4, 5)
    
    price_data = fetch_historical_data(sample_ticker, start_dt, end_dt, "15m")

    if not price_data.empty:
        print(f"Price data for {sample_ticker} (first 5 rows):")
        print(price_data.head())
        print(f"Timezone: {price_data.index.tz}")

        # Define strategy parameters
        stop_loss = 10.0  # 10 points for SPX
        risk_reward_ratio = 3.0

        # Generate signals
        signals = generate_signals(price_data, stop_loss, risk_reward_ratio)

        if not signals.empty:
            print("\nGenerated Signals:")
            print(signals)
        else:
            print("\nNo signals generated for the sample data.")
            # Try a wider date range if no signals
            start_dt_wider = date.today() - pd.Timedelta(days=59)
            end_dt_wider = date.today()
            price_data_wider = fetch_historical_data(sample_ticker, start_dt_wider, end_dt_wider, "15m")
            if not price_data_wider.empty:
                print(f"\nTrying with wider date range: {start_dt_wider} to {end_dt_wider}")
                signals_wider = generate_signals(price_data_wider, stop_loss, risk_reward_ratio)
                if not signals_wider.empty:
                    print("\nGenerated Signals (wider range):")
                    print(signals_wider)
                else:
                    print("\nNo signals generated even for wider range.")
            else:
                print("\nFailed to fetch wider range data.")


    else:
        print(f"Could not fetch price data for {sample_ticker} to test signal generation.")

