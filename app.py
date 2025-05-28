# app.py
"""
Main Streamlit application file for the Multi-Strategy Backtester.
Handles UI, user inputs, and orchestrates the backtesting process.
Allows selection of different trading strategies and displays their logic.
"""
import sys
import os

# --- sys.path modification and diagnostics ---
# Assuming app.py is in the project root directory (e.g., 'gap-guardian')
# which contains 'services', 'utils', 'config' as subdirectories.
APP_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(APP_FILE_PATH)

print(f"--- [DEBUG app.py] ---")
print(f"APP_FILE_PATH: {APP_FILE_PATH}")
print(f"Calculated PROJECT_ROOT: {PROJECT_ROOT}")
print(f"Original sys.path: {sys.path}")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"PROJECT_ROOT ('{PROJECT_ROOT}') was not in sys.path. Added it to the beginning.")
else:
    # If already in sys.path, ensure it's at the beginning for priority
    if sys.path[0] != PROJECT_ROOT:
        sys.path.remove(PROJECT_ROOT)
        sys.path.insert(0, PROJECT_ROOT)
        print(f"PROJECT_ROOT ('{PROJECT_ROOT}') was in sys.path but not at index 0. Moved it to the beginning.")
    else:
        print(f"PROJECT_ROOT ('{PROJECT_ROOT}') is already the first item in sys.path.")

print(f"Final sys.path for app.py: {sys.path}")
print(f"--- [END DEBUG app.py] ---")
# --- end of sys.path modification and diagnostics ---

# Now, proceed with other imports
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime, time as dt_time

# These imports depend on the sys.path being correctly set up.
from config import settings # This should now find config/settings.py
from utils import plotting, logger as app_logger # This should now find utils/
# The following import is where the error was occurring
from services import data_loader, strategy_engine, backtester, optimizer # This is line 24 in your latest traceback

logger = app_logger.get_logger(__name__)

st.set_page_config(page_title=settings.APP_TITLE, page_icon="🛡️📈", layout="wide", initial_sidebar_state="expanded")

def load_custom_css(css_file_path):
    """Loads custom CSS from a file and applies it."""
    try:
        # Construct path relative to this app.py file if style.css is in 'static' subdir
        # For Streamlit sharing, paths are usually relative to the main app script.
        full_css_path = os.path.join(os.path.dirname(__file__), css_file_path)
        if not os.path.exists(full_css_path): # Fallback for different execution contexts
            full_css_path = css_file_path

        with open(full_css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            logger.info(f"Successfully loaded CSS from: {full_css_path}")
    except FileNotFoundError:
        logger.warning(f"CSS file not found. Tried path: {full_css_path} and original path: {css_file_path}")
        st.warning(f"CSS file not found at path: {css_file_path}")
    except Exception as e:
        logger.warning(f"Error loading CSS file '{css_file_path}': {e}")
        st.warning(f"Error loading CSS file '{css_file_path}': {e}")

load_custom_css("static/style.css")


# Strategy Explanations (as previously defined)
STRATEGY_EXPLANATIONS = {
    "Gap Guardian": """
    **Concept:** Aims to capitalize on false breakouts/breakdowns of an initial opening range during a specific morning session.
    - **Time Frame:** Typically 15-minutes, but adaptable.
    - **Entry Window (NY Time):** e.g., 9:30 AM - 11:00 AM.
    - **Opening Range:** Defined by the high and low of the first bar occurring at or after the `Entry Start Time`.
    - **Long Entry:**
        1. Price breaks *below* the opening range low.
        2. Price then *closes back above* the opening range low.
        3. All within the specified `Entry Window`.
    - **Short Entry:**
        1. Price breaks *above* the opening range high.
        2. Price then *closes back below* the opening range high.
        3. All within the specified `Entry Window`.
    - **Risk Management:** Uses a fixed stop-loss (in points) and a risk-reward ratio (RRR) to determine the take profit.
    - **Frequency:** Typically aims for one trade per day if a setup occurs.
    """,
    "Unicorn": """
    **Concept:** A high-probability setup that combines a "Breaker Block" with an overlapping "Fair Value Gap (FVG)" for precision entries.
    *(Note: Current implementation uses a simplified FVG entry. Full Breaker+FVG logic is complex.)*
    - **Bullish Unicorn (Long Position):**
        1.  **Identify Structure (Bullish Breaker Pattern):**
            * A swing low forms.
            * A swing high forms after the swing low.
            * Price sweeps *below* the initial swing low (liquidity grab), creating a lower low.
            * Price then displaces *upwards strongly*, breaking *above* the swing high and creating a higher high. This upward move should ideally leave an FVG.
        2.  **Identify Bullish Breaker Block:** Mark out the highest up-close (bullish) candle or series of candles that occurred *just before* the price made the lower low (the liquidity sweep). This zone is the breaker.
        3.  **Identify Bullish FVG:** Look for a Bullish Fair Value Gap (a 3-candle pattern where the middle candle has a gap between the wicks of the first and third candles) that formed during the strong upward displacement move (the one that created the higher high).
        4.  **Confirm Overlap:** The key is that this Bullish FVG *overlaps* with the identified Bullish Breaker Block zone. This overlapping area is the high-probability entry zone.
        5.  **Entry:** Look to enter a long position when the price retraces *down into* this overlapping Breaker + FVG zone.
        6.  **Stop Loss:** Typically placed below the recent low (the lower low of the sweep) or more aggressively below the Breaker/FVG zone.
        7.  **Target:** Aims for a draw on liquidity, such as nearby swing highs or other areas of interest.
    - **Bearish Unicorn (Short Position):**
        1.  **Identify Structure (Bearish Breaker Pattern):**
            * A swing high forms.
            * A swing low forms after the swing high.
            * Price sweeps *above* the initial swing high (liquidity grab), creating a higher high.
            * Price then displaces *downwards strongly*, breaking *below* the swing low and creating a lower low. This downward move should ideally leave an FVG.
        2.  **Identify Bearish Breaker Block:** Mark out the lowest down-close (bearish) candle or series of candles that occurred *just before* the price made the higher high (the liquidity sweep).
        3.  **Identify Bearish FVG:** Look for a Bearish FVG that formed during the strong downward displacement move.
        4.  **Confirm Overlap:** Ensure this Bearish FVG *overlaps* with the Bearish Breaker Block.
        5.  **Entry:** Look to enter a short position when the price retraces *up into* this overlapping Breaker + FVG zone.
        6.  **Stop Loss:** Typically placed above the recent high (the higher high of the sweep) or more aggressively above the Breaker/FVG zone.
        7.  **Target:** Aims for a draw on liquidity, such as nearby swing lows.
    """,
    "Silver Bullet": """
    **Concept:** A time-based strategy that looks for entries into Fair Value Gaps (FVGs) during specific 1-hour windows in the New York trading session, aligning with anticipated liquidity draws.
    - **Time Windows (NY Time):**
        * 3:00 AM - 4:00 AM
        * 10:00 AM - 11:00 AM
        * 2:00 PM - 3:00 PM
    - **Logic for Long Positions:**
        1.  **Context (Draw on Liquidity):** Identify a reason for the price to move *higher*. This means the anticipated draw on liquidity (e.g., an old high, a bearish FVG/SIBI above current price) is *above* the market.
        2.  **Time:** Wait for one of the three specific 1-hour windows.
        3.  **Setup:**
            * Look for price showing bullish intent or having recently shifted market structure to the upside (e.g., breaking a recent swing high).
            * A **Bullish FVG** (a gap created during an upward move) forms or is revisited *during the specified time window*.
        4.  **Entry:** Enter a long position when the price dips *down into* this Bullish FVG within one of the active 1-hour windows.
    - **Logic for Short Positions:**
        1.  **Context (Draw on Liquidity):** Identify a reason for the price to move *lower*. This means the anticipated draw on liquidity (e.g., an old low, a bullish FVG/BISI below current price) is *below* the market.
        2.  **Time:** Wait for one of the three specific 1-hour windows.
        3.  **Setup:**
            * Look for price showing bearish intent or having recently shifted market structure to the downside (e.g., breaking a recent swing low).
            * A **Bearish FVG** (a gap created during a downward move) forms or is revisited *during the specified time window*.
        4.  **Entry:** Enter a short position when the price rallies *up into* this Bearish FVG within one of the active 1-hour windows.
    - **Risk Management:** Typically uses a configurable stop-loss (in points) and a risk-reward ratio (RRR).
    """
}


def initialize_app_session_state():
    """Initializes session state variables with default values if they don't exist."""
    defaults = {
        'backtest_results': None,
        'optimization_results_df': pd.DataFrame(),
        'price_data': pd.DataFrame(),
        'signals': pd.DataFrame(),
        'best_params_from_opt': None,
        'wfo_results': None,
        'selected_timeframe_value': settings.DEFAULT_STRATEGY_TIMEFRAME,
        'run_analysis_clicked_count': 0, # Tracks run button clicks to refresh tabs
        'wfo_isd_ui_val': settings.DEFAULT_WFO_IN_SAMPLE_DAYS,
        'wfo_oosd_ui_val': settings.DEFAULT_WFO_OUT_OF_SAMPLE_DAYS,
        'wfo_sd_ui_val': settings.DEFAULT_WFO_STEP_DAYS,
        'selected_strategy': settings.DEFAULT_STRATEGY, # Persist selected strategy
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_app_session_state()

# --- Sidebar Inputs ---
st.sidebar.header("Backtest Configuration")

# Strategy Selector
st.session_state.selected_strategy = st.sidebar.selectbox(
    "Select Strategy:",
    options=settings.AVAILABLE_STRATEGIES,
    index=settings.AVAILABLE_STRATEGIES.index(st.session_state.selected_strategy), # Use persisted value
    key="strategy_selector_v2" # Unique key
)
selected_strategy_name = st.session_state.selected_strategy

# Ticker Selector
selected_ticker_name = st.sidebar.selectbox(
    "Select Symbol:",
    options=list(settings.DEFAULT_TICKERS.keys()),
    index=0, # Default to first ticker
    key="ticker_sel_v12"
)
ticker_symbol = settings.DEFAULT_TICKERS[selected_ticker_name]

# Timeframe Selector
current_tf_value_in_state = st.session_state.selected_timeframe_value
default_tf_display_index = 0
if current_tf_value_in_state in settings.AVAILABLE_TIMEFRAMES.values():
    default_tf_display_index = list(settings.AVAILABLE_TIMEFRAMES.values()).index(current_tf_value_in_state)

selected_timeframe_display = st.sidebar.selectbox(
    "Select Timeframe:",
    options=list(settings.AVAILABLE_TIMEFRAMES.keys()),
    index=default_tf_display_index,
    key="timeframe_selector_ui_main_v12"
)
st.session_state.selected_timeframe_value = settings.AVAILABLE_TIMEFRAMES[selected_timeframe_display]
ui_current_interval = st.session_state.selected_timeframe_value


# Date Range Inputs
today = date.today()
max_history_limit_days = None
if ui_current_interval in settings.YFINANCE_SHORT_INTRADAY_INTERVALS:
    max_history_limit_days = settings.MAX_SHORT_INTRADAY_DAYS
elif ui_current_interval in settings.YFINANCE_HOURLY_INTERVALS:
    max_history_limit_days = settings.MAX_HOURLY_INTRADAY_DAYS

min_allowable_start_date_for_ui = (today - timedelta(days=max_history_limit_days - 1)) if max_history_limit_days else (today - timedelta(days=365 * 10)) # 10 years for daily/weekly
date_input_help_suffix = f"Data for {ui_current_interval} is limited to ~{max_history_limit_days} days." if max_history_limit_days else "Select historical period."

# Dynamic default start date based on interval
if ui_current_interval in settings.YFINANCE_SHORT_INTRADAY_INTERVALS + settings.YFINANCE_HOURLY_INTERVALS:
    default_start_days_ago = min(settings.MAX_SHORT_INTRADAY_DAYS - 5 if settings.MAX_SHORT_INTRADAY_DAYS else 15,
                                 max_history_limit_days - 1 if max_history_limit_days else 15)
else: # Longer timeframes like 1d, 1wk
    default_start_days_ago = 365 if ui_current_interval != "1wk" else 30*7 # 1 year for daily, 7 months for weekly

default_start_date_value = today - timedelta(days=default_start_days_ago)
if default_start_date_value < min_allowable_start_date_for_ui:
    default_start_date_value = min_allowable_start_date_for_ui
max_possible_start_date = today - timedelta(days=1) # Start date cannot be today
if default_start_date_value > max_possible_start_date:
    default_start_date_value = max_possible_start_date
if default_start_date_value < min_allowable_start_date_for_ui: # Final check
     default_start_date_value = min_allowable_start_date_for_ui


start_date_ui = st.sidebar.date_input(
    "Start Date:",
    value=default_start_date_value,
    min_value=min_allowable_start_date_for_ui,
    max_value=max_possible_start_date,
    key=f"start_date_widget_{ui_current_interval}_v12", # Key depends on interval to re-render if interval changes
    help=f"Start date for historical data. {date_input_help_suffix}"
)

min_end_date_value_ui = start_date_ui + timedelta(days=1) if start_date_ui else min_allowable_start_date_for_ui + timedelta(days=1)
default_end_date_value_ui = today
if default_end_date_value_ui < min_end_date_value_ui: default_end_date_value_ui = min_end_date_value_ui
if default_end_date_value_ui > today: default_end_date_value_ui = today


end_date_ui = st.sidebar.date_input(
    "End Date:",
    value=default_end_date_value_ui,
    min_value=min_end_date_value_ui,
    max_value=today,
    key=f"end_date_widget_{ui_current_interval}_v12",
    help=f"End date for historical data. {date_input_help_suffix}"
)


# Financial Parameters
initial_capital_ui = st.sidebar.number_input("Initial Capital ($):", min_value=1000.0, value=settings.DEFAULT_INITIAL_CAPITAL, step=1000.0, format="%.2f")
risk_per_trade_percent_ui = st.sidebar.number_input("Risk per Trade (%):", min_value=0.1, max_value=10.0, value=settings.DEFAULT_RISK_PER_TRADE_PERCENT, step=0.1, format="%.1f")

# Common Strategy Parameters (Manual Run)
st.sidebar.subheader("Common Strategy Parameters")
sl_points_single_ui = st.sidebar.number_input("SL (points):", min_value=0.1, value=settings.DEFAULT_STOP_LOSS_POINTS, step=0.1, format="%.2f", key="sl_s_man_v12")
rrr_single_ui = st.sidebar.number_input("RRR:", min_value=0.1, value=settings.DEFAULT_RRR, step=0.1, format="%.1f", key="rrr_s_man_v12")

# Strategy-Specific Parameters (Manual Run)
entry_start_hour_single_ui = settings.DEFAULT_ENTRY_WINDOW_START_HOUR
entry_start_minute_single_ui = settings.DEFAULT_ENTRY_WINDOW_START_MINUTE
entry_end_hour_single_ui = settings.DEFAULT_ENTRY_WINDOW_END_HOUR
entry_end_minute_single_ui = settings.DEFAULT_ENTRY_WINDOW_END_MINUTE

if selected_strategy_name == "Gap Guardian":
    st.sidebar.markdown("**Entry Window (NY Time - Manual Run):**")
    col1_entry, col2_entry = st.sidebar.columns(2)
    entry_start_hour_single_ui = col1_entry.number_input("Start Hr", 0, 23, settings.DEFAULT_ENTRY_WINDOW_START_HOUR, 1, key="esh_s_man_v12")
    entry_start_minute_single_ui = col2_entry.number_input("Start Min", 0, 59, settings.DEFAULT_ENTRY_WINDOW_START_MINUTE, 15, key="esm_s_man_v12") # Step 15 for minutes
    
    col1_exit, col2_exit = st.sidebar.columns(2)
    entry_end_hour_single_ui = col1_exit.number_input("End Hr", 0, 23, settings.DEFAULT_ENTRY_WINDOW_END_HOUR, 1, key="eeh_s_man_v12")
    entry_end_minute_single_ui = col2_exit.number_input("End Min", 0, 59, settings.DEFAULT_ENTRY_WINDOW_END_MINUTE, 15, key="eem_s_man_v12", help="Usually 00 for end of hour.")
elif selected_strategy_name == "Unicorn":
    st.sidebar.caption("Unicorn strategy uses SL/RRR. Entry is pattern-based (Breaker + FVG).")
elif selected_strategy_name == "Silver Bullet":
    st.sidebar.caption(f"Silver Bullet uses SL/RRR. Entry is FVG-based within fixed NY time windows: "
                       f"{', '.join([f'{s.strftime('%H:%M')}-{e.strftime('%H:%M')}' for s, e in settings.SILVER_BULLET_WINDOWS_NY])}.")


# Analysis Mode Selector
analysis_mode_ui = st.sidebar.radio(
    "Analysis Type:",
    ("Single Backtest", "Parameter Optimization", "Walk-Forward Optimization"),
    index=0, # Default to Single Backtest
    key="analysis_mode_v12"
)

# --- Optimization Parameters (Conditional Display) ---
# Initialize default optimization parameters
opt_algo_ui = settings.DEFAULT_OPTIMIZATION_ALGORITHM
sl_min_opt_ui, sl_max_opt_ui, sl_steps_opt_ui = settings.DEFAULT_SL_POINTS_OPTIMIZATION_RANGE.values()
rrr_min_opt_ui, rrr_max_opt_ui, rrr_steps_opt_ui = settings.DEFAULT_RRR_OPTIMIZATION_RANGE.values()
esh_min_opt_ui, esh_max_opt_ui, esh_steps_opt_ui = settings.DEFAULT_ENTRY_START_HOUR_OPTIMIZATION_RANGE.values()
esm_vals_opt_ui = list(settings.DEFAULT_ENTRY_START_MINUTE_OPTIMIZATION_VALUES) # Ensure it's a list for multiselect
eeh_min_opt_ui, eeh_max_opt_ui, eeh_steps_opt_ui = settings.DEFAULT_ENTRY_END_HOUR_OPTIMIZATION_RANGE.values()
rand_iters_ui = settings.DEFAULT_RANDOM_SEARCH_ITERATIONS
opt_metric_ui = settings.DEFAULT_OPTIMIZATION_METRIC


if analysis_mode_ui != "Single Backtest":
    st.sidebar.markdown("##### In-Sample Optimization Settings")
    opt_algo_ui = st.sidebar.selectbox("Algorithm:", settings.OPTIMIZATION_ALGORITHMS, index=settings.OPTIMIZATION_ALGORITHMS.index(opt_algo_ui), key="opt_algo_v12")
    opt_metric_ui = st.sidebar.selectbox("Optimize Metric:", settings.OPTIMIZATION_METRICS, index=settings.OPTIMIZATION_METRICS.index(opt_metric_ui), key="opt_metric_v12")
    
    st.sidebar.markdown("**SL Range (points):**")
    c1,c2,c3 = st.sidebar.columns(3)
    sl_min_opt_ui = c1.number_input("Min", value=sl_min_opt_ui, step=0.1, format="%.1f", key="slmin_o12", min_value=0.1)
    sl_max_opt_ui = c2.number_input("Max", value=sl_max_opt_ui, step=0.1, format="%.1f", key="slmax_o12", min_value=sl_min_opt_ui + 0.1)
    if opt_algo_ui == "Grid Search":
        sl_steps_opt_ui = c3.number_input("Steps", min_value=2, max_value=20, value=int(sl_steps_opt_ui), step=1, key="slsteps_o12")
    
    st.sidebar.markdown("**RRR Range:**")
    c1,c2,c3 = st.sidebar.columns(3)
    rrr_min_opt_ui = c1.number_input("Min", value=rrr_min_opt_ui, step=0.1, format="%.1f", key="rrrmin_o12", min_value=0.1)
    rrr_max_opt_ui = c2.number_input("Max", value=rrr_max_opt_ui, step=0.1, format="%.1f", key="rrrmax_o12", min_value=rrr_min_opt_ui + 0.1)
    if opt_algo_ui == "Grid Search":
        rrr_steps_opt_ui = c3.number_input("Steps", min_value=2, max_value=20, value=int(rrr_steps_opt_ui), step=1, key="rrrsteps_o12")

    if selected_strategy_name == "Gap Guardian":
        st.sidebar.markdown("**Entry Start Hr Range (NY):**")
        c1,c2,c3 = st.sidebar.columns(3)
        esh_min_opt_ui = c1.number_input("Min Hr", value=esh_min_opt_ui, min_value=0, max_value=23, step=1, key="eshmin_o12")
        esh_max_opt_ui = c2.number_input("Max Hr", value=esh_max_opt_ui, min_value=esh_min_opt_ui, max_value=23, step=1, key="eshmax_o12")
        if opt_algo_ui == "Grid Search":
            esh_steps_opt_ui = c3.number_input("Hr Steps", min_value=1, max_value=10, value=int(esh_steps_opt_ui), step=1, key="eshsteps_o12")
        
        esm_vals_opt_ui = st.sidebar.multiselect("Entry Start Min(s) (NY):", [0,15,30,45,50], default=esm_vals_opt_ui, key="esmvals_o12")
        if not esm_vals_opt_ui: esm_vals_opt_ui = [settings.DEFAULT_ENTRY_WINDOW_START_MINUTE] # Ensure at least one value

        st.sidebar.markdown("**Entry End Hr Range (NY):**")
        c1,c2,c3 = st.sidebar.columns(3)
        eeh_min_opt_ui = c1.number_input("Min Hr", value=eeh_min_opt_ui, min_value=0, max_value=23, step=1, key="eehmin_o12")
        eeh_max_opt_ui = c2.number_input("Max Hr", value=eeh_max_opt_ui, min_value=eeh_min_opt_ui, max_value=23, step=1, key="eehmax_o12")
        if opt_algo_ui == "Grid Search":
            eeh_steps_opt_ui = c3.number_input("Hr Steps", min_value=1, max_value=10, value=int(eeh_steps_opt_ui), step=1, key="eehsteps_o12")
        # Entry End Minute is usually fixed at 00 for Gap Guardian, so not optimized by default.
    
    if opt_algo_ui == "Random Search":
        rand_iters_ui = st.sidebar.number_input("Random Iterations:", min_value=10, max_value=1000, value=rand_iters_ui, step=10, key="randiter_o12")
    
    # Display estimated combinations for Grid Search
    if opt_algo_ui == "Grid Search":
        grid_combs = int(sl_steps_opt_ui * rrr_steps_opt_ui)
        if selected_strategy_name == "Gap Guardian":
            grid_combs *= int(esh_steps_opt_ui * len(esm_vals_opt_ui) * eeh_steps_opt_ui)
        st.sidebar.caption(f"Estimated Grid Combinations: {grid_combs}")
    else: # Random Search
        st.sidebar.caption(f"Random Iterations: {rand_iters_ui}")


# --- Walk-Forward Optimization Parameters (Conditional Display) ---
if analysis_mode_ui == "Walk-Forward Optimization":
    st.sidebar.markdown("##### WFO Settings (Calendar Days)")
    total_available_days_for_wfo = (end_date_ui - start_date_ui).days + 1
    MIN_WFO_IS_DAYS, MIN_WFO_OOS_DAYS, MIN_WFO_STEP_DAYS = 30, 10, 10 # Minimums

    # Suggest WFO parameters based on total available days
    calculated_isd, calculated_oosd, calculated_stepd = st.session_state.wfo_isd_ui_val, st.session_state.wfo_oosd_ui_val, st.session_state.wfo_sd_ui_val
    if total_available_days_for_wfo >= MIN_WFO_IS_DAYS + MIN_WFO_OOS_DAYS: # Enough data for at least one fold
        # Heuristic: OOS is ~25-33% of IS, Step is often same as OOS
        tentative_oosd = max(MIN_WFO_OOS_DAYS, total_available_days_for_wfo // 5) # e.g., 20% for OOS
        tentative_isd = max(MIN_WFO_IS_DAYS, total_available_days_for_wfo - (tentative_oosd * 2)) # Ensure IS is substantial
        
        if tentative_isd + tentative_oosd > total_available_days_for_wfo: # If sum exceeds total, adjust
            calculated_isd = max(MIN_WFO_IS_DAYS, int(total_available_days_for_wfo * 0.7))
            calculated_oosd = max(MIN_WFO_OOS_DAYS, total_available_days_for_wfo - calculated_isd)
        else:
            calculated_isd, calculated_oosd = tentative_isd, tentative_oosd
        
        calculated_stepd = max(MIN_WFO_STEP_DAYS, calculated_oosd) # Step at least OOS
        
        # Final sanity checks
        calculated_isd = max(MIN_WFO_IS_DAYS, calculated_isd)
        calculated_oosd = max(MIN_WFO_OOS_DAYS, calculated_oosd)
        if calculated_isd + calculated_oosd > total_available_days_for_wfo: # If still too large after adjustments
            calculated_oosd = max(MIN_WFO_OOS_DAYS, int(total_available_days_for_wfo * 0.25))
            calculated_isd = max(MIN_WFO_IS_DAYS, total_available_days_for_wfo - calculated_oosd)
            calculated_stepd = max(MIN_WFO_STEP_DAYS, calculated_oosd)

        st.session_state.wfo_isd_ui_val, st.session_state.wfo_oosd_ui_val, st.session_state.wfo_sd_ui_val = calculated_isd, calculated_oosd, calculated_stepd
        st.sidebar.caption(f"Suggested WFO: IS={calculated_isd}d, OOS={calculated_oosd}d, Step={calculated_stepd}d for {total_available_days_for_wfo}d total.")
    else:
        st.sidebar.caption(f"Total period ({total_available_days_for_wfo}d) is short for meaningful WFO. Minimum {MIN_WFO_IS_DAYS+MIN_WFO_OOS_DAYS}d recommended.")

    wfo_isd_ui = st.sidebar.number_input("In-Sample (Days):", min_value=MIN_WFO_IS_DAYS, value=st.session_state.wfo_isd_ui_val, step=10, key="wfoisd_v12")
    wfo_oosd_ui = st.sidebar.number_input("Out-of-Sample (Days):", min_value=MIN_WFO_OOS_DAYS, value=st.session_state.wfo_oosd_ui_val, step=5, key="wfoosd_v12")
    wfo_sd_ui = st.sidebar.number_input("Step (Days):", min_value=max(MIN_WFO_STEP_DAYS, wfo_oosd_ui), value=max(st.session_state.wfo_sd_ui_val, wfo_oosd_ui), step=5, key="wfosd_v12", help="Step must be >= Out-of-Sample days.")
    
    # Update session state from UI inputs for WFO
    st.session_state.wfo_isd_ui_val, st.session_state.wfo_oosd_ui_val, st.session_state.wfo_sd_ui_val = wfo_isd_ui, wfo_oosd_ui, wfo_sd_ui


# --- Main Application Area ---
st.title(f"🛡️📈 {settings.APP_TITLE}")
strategy_info_md = f"Selected Strategy: **{selected_strategy_name}** | Symbol: **{selected_ticker_name}** | Timeframe: **{selected_timeframe_display}** ({st.session_state.selected_timeframe_value})"
if selected_strategy_name == "Gap Guardian":
    strategy_info_md += f" | Default Manual Entry: {settings.DEFAULT_ENTRY_WINDOW_START_HOUR:02d}:{settings.DEFAULT_ENTRY_WINDOW_START_MINUTE:02d}-{settings.DEFAULT_ENTRY_WINDOW_END_HOUR:02d}:{settings.DEFAULT_ENTRY_WINDOW_END_MINUTE:02d} NYT"
elif selected_strategy_name == "Silver Bullet":
     strategy_info_md += f" | Fixed NYT Entry Windows: {', '.join([f'{s.strftime('%H:%M')}-{e.strftime('%H:%M')}' for s, e in settings.SILVER_BULLET_WINDOWS_NY])}"
st.markdown(strategy_info_md)

# Strategy Explanation Expander
with st.expander(f"Understanding the '{selected_strategy_name}' Strategy", expanded=False):
    st.markdown(STRATEGY_EXPLANATIONS.get(selected_strategy_name, "Explanation not available for this strategy."))


# --- Run Analysis Button and Logic ---
if st.sidebar.button("Run Analysis", type="primary", use_container_width=True, key="run_main_v12"):
    st.session_state.run_analysis_clicked_count += 1 # Increment to manage tab state
    logger.info(f"Run Analysis clicked. Strategy: {selected_strategy_name}, Mode: {analysis_mode_ui}, TF: {st.session_state.selected_timeframe_value}, Symbol: {ticker_symbol}")
    
    # Clear previous results from session state
    st.session_state.backtest_results = None
    st.session_state.optimization_results_df = pd.DataFrame()
    st.session_state.wfo_results = None
    st.session_state.price_data = pd.DataFrame()
    st.session_state.signals = pd.DataFrame()
    st.session_state.best_params_from_opt = None
    
    interval_for_this_run = st.session_state.selected_timeframe_value # Use interval from session state

    # Validate date range
    if start_date_ui >= end_date_ui:
        st.error(f"Error: Start date ({start_date_ui}) must be before end date ({end_date_ui}).")
    else:
        # Specific validation for WFO duration
        if analysis_mode_ui == "Walk-Forward Optimization":
            total_data_duration_days_for_check = (end_date_ui - start_date_ui).days + 1
            min_required_wfo_duration = st.session_state.wfo_isd_ui_val + st.session_state.wfo_oosd_ui_val 
            if total_data_duration_days_for_check < min_required_wfo_duration:
                st.error(f"Insufficient data for WFO. Selected range: {total_data_duration_days_for_check}d, "
                           f"WFO requires at least: {min_required_wfo_duration}d (IS={st.session_state.wfo_isd_ui_val}d + OOS={st.session_state.wfo_oosd_ui_val}d). "
                           f"Please adjust date range or WFO period lengths.")
                st.stop() # Halt execution if WFO validation fails

        with st.spinner(f"Fetching data for {ticker_symbol}..."):
            try:
                price_data_df = data_loader.fetch_historical_data(ticker_symbol, start_date_ui, end_date_ui, interval_for_this_run)
                st.session_state.price_data = price_data_df
            except Exception as e:
                logger.error(f"Failed to fetch data for {ticker_symbol}: {e}", exc_info=True)
                st.error(f"Data fetching failed for {ticker_symbol}. Error: {e}")
                price_data_df = pd.DataFrame() # Ensure it's a DataFrame
                st.session_state.price_data = price_data_df


        if price_data_df.empty:
            st.warning(f"No price data retrieved for {selected_ticker_name} ({ticker_symbol}) for the period {start_date_ui} to {end_date_ui} with interval {interval_for_this_run}. Cannot proceed with analysis.")
        else:
            # Prepare strategy parameters for the engine
            strategy_params_for_engine = {
                'strategy_name': selected_strategy_name,
                'stop_loss_points': sl_points_single_ui,
                'rrr': rrr_single_ui,
            }
            if selected_strategy_name == "Gap Guardian": # Add Gap Guardian specific params
                strategy_params_for_engine['entry_start_time'] = dt_time(entry_start_hour_single_ui, entry_start_minute_single_ui)
                strategy_params_for_engine['entry_end_time'] = dt_time(entry_end_hour_single_ui, entry_end_minute_single_ui)
            
            prog_bar_container = None # Initialize progress bar container

            # --- Single Backtest Logic ---
            if analysis_mode_ui == "Single Backtest":
                st.subheader(f"Single Backtest Run ({selected_strategy_name})")
                with st.spinner("Running single backtest..."):
                    try:
                        signals = strategy_engine.generate_signals(data=price_data_df.copy(), **strategy_params_for_engine)
                        st.session_state.signals = signals
                        trades, equity, perf = backtester.run_backtest(
                            price_data_df.copy(), signals, initial_capital_ui,
                            risk_per_trade_percent_ui, sl_points_single_ui, interval_for_this_run
                        )
                        
                        # Prepare display parameters for results summary
                        param_display_str = f"SL: {sl_points_single_ui:.2f} pts, RRR: {rrr_single_ui:.1f}"
                        if selected_strategy_name == "Gap Guardian":
                            param_display_str += f", Entry: {strategy_params_for_engine['entry_start_time']:%H:%M}-{strategy_params_for_engine['entry_end_time']:%H:%M} NYT"
                        
                        st.session_state.backtest_results = {
                            "trades": trades, "equity_curve": equity, "performance": perf,
                            "params": {
                                "Strategy": selected_strategy_name, "Symbol": selected_ticker_name,
                                "SL": sl_points_single_ui, "RRR": rrr_single_ui,
                                "TF": interval_for_this_run, "EntryDisplay": param_display_str, "src": "Manual"
                            }
                        }
                        st.success("Single backtest complete!")
                    except Exception as e:
                        logger.error(f"Error during single backtest for {selected_strategy_name}: {e}", exc_info=True)
                        st.error(f"An error occurred during the single backtest: {e}")

            # --- Parameter Optimization or Walk-Forward Optimization Logic ---
            elif analysis_mode_ui in ["Parameter Optimization", "Walk-Forward Optimization"]:
                prog_bar_container = st.empty() # Placeholder for progress bar
                prog_bar_container.progress(0, text="Initializing optimization...")

                def optimization_progress_callback(progress_fraction, message_text):
                    """Updates the Streamlit progress bar."""
                    prog_bar_container.progress(min(1.0, progress_fraction), text=f"{message_text}: {int(min(1.0, progress_fraction)*100)}% complete")

                # Define parameter ranges for optimization based on UI inputs
                actual_params_to_optimize_config = {
                    'sl_points': np.linspace(sl_min_opt_ui, sl_max_opt_ui, int(sl_steps_opt_ui)) if opt_algo_ui == "Grid Search" else (sl_min_opt_ui, sl_max_opt_ui),
                    'rrr': np.linspace(rrr_min_opt_ui, rrr_max_opt_ui, int(rrr_steps_opt_ui)) if opt_algo_ui == "Grid Search" else (rrr_min_opt_ui, rrr_max_opt_ui),
                }
                if selected_strategy_name == "Gap Guardian": # Add Gap Guardian specific optimization params
                    actual_params_to_optimize_config.update({
                        'entry_start_hour': [int(h) for h in np.linspace(esh_min_opt_ui, esh_max_opt_ui, int(esh_steps_opt_ui))] if opt_algo_ui == "Grid Search" else (esh_min_opt_ui, esh_max_opt_ui),
                        'entry_start_minute': esm_vals_opt_ui, # This is already a list from multiselect
                        'entry_end_hour': [int(h) for h in np.linspace(eeh_min_opt_ui, eeh_max_opt_ui, int(eeh_steps_opt_ui))] if opt_algo_ui == "Grid Search" else (eeh_min_opt_ui, eeh_max_opt_ui),
                        'entry_end_minute': settings.DEFAULT_ENTRY_END_MINUTE_OPTIMIZATION_VALUES # Typically fixed for GG
                    })
                
                # Control parameters for the optimizer function
                optimizer_control_params = {'metric_to_optimize': opt_metric_ui, 'strategy_name': selected_strategy_name}
                if opt_algo_ui == "Random Search":
                    optimizer_control_params['iterations'] = rand_iters_ui

                # --- Parameter Optimization (Full Period) ---
                if analysis_mode_ui == "Parameter Optimization":
                    st.subheader(f"Parameter Optimization ({opt_algo_ui} - {selected_strategy_name} - Full Period)")
                    with st.spinner(f"Running {opt_algo_ui} optimization... This may take some time."):
                        try:
                            if opt_algo_ui == "Grid Search":
                                opt_df = optimizer.run_grid_search(price_data_df, initial_capital_ui, risk_per_trade_percent_ui, actual_params_to_optimize_config, interval_for_this_run, optimizer_control_params, optimization_progress_callback)
                            else: # Random Search
                                opt_df = optimizer.run_random_search(price_data_df, initial_capital_ui, risk_per_trade_percent_ui, actual_params_to_optimize_config, interval_for_this_run, optimizer_control_params, optimization_progress_callback)
                            
                            st.session_state.optimization_results_df = opt_df
                            if not opt_df.empty:
                                st.success("Full period optimization finished!")
                                valid_opt_results = opt_df.dropna(subset=[opt_metric_ui]) # Consider only rows where the metric is valid
                                if not valid_opt_results.empty:
                                    # Find best parameters based on the chosen metric
                                    if opt_metric_ui == "Max Drawdown (%)": # Minimize (less negative is better)
                                        best_row_from_opt = valid_opt_results.loc[valid_opt_results[opt_metric_ui].idxmax()] # Max of negative values
                                    else: # Maximize other metrics
                                        best_row_from_opt = valid_opt_results.loc[valid_opt_results[opt_metric_ui].idxmax()]
                                    
                                    st.session_state.best_params_from_opt = best_row_from_opt.to_dict()
                                    
                                    # Prepare parameters for backtesting with these best found params
                                    best_params_for_bt_run = {'strategy_name': selected_strategy_name,
                                                              'stop_loss_points': best_row_from_opt["SL Points"],
                                                              'rrr': best_row_from_opt["RRR"]}
                                    entry_display_opt_str = f"SL: {best_row_from_opt['SL Points']:.2f}, RRR: {best_row_from_opt['RRR']:.1f}"
                                    if selected_strategy_name == "Gap Guardian":
                                        best_entry_start_t = dt_time(int(best_row_from_opt["EntryStartHour"]), int(best_row_from_opt["EntryStartMinute"]))
                                        best_entry_end_t = dt_time(int(best_row_from_opt["EntryEndHour"]), int(best_row_from_opt.get("EntryEndMinute", settings.DEFAULT_ENTRY_WINDOW_END_MINUTE))) # Use default if not in opt results
                                        best_params_for_bt_run['entry_start_time'] = best_entry_start_t
                                        best_params_for_bt_run['entry_end_time'] = best_entry_end_t
                                        entry_display_opt_str += f", Entry: {best_entry_start_t:%H:%M}-{best_entry_end_t:%H:%M} NYT"
                                    
                                    st.info(f"Best parameters for '{opt_metric_ui}': {entry_display_opt_str} (Metric Value: {best_row_from_opt[opt_metric_ui]:.2f})")
                                    
                                    # Run a final backtest with these best parameters
                                    with st.spinner("Running backtest with best optimized parameters..."):
                                        signals_best_opt = strategy_engine.generate_signals(data=price_data_df.copy(), **best_params_for_bt_run)
                                        st.session_state.signals = signals_best_opt # Save signals from this run
                                        trades_bo, equity_bo, perf_bo = backtester.run_backtest(price_data_df.copy(), signals_best_opt, initial_capital_ui, risk_per_trade_percent_ui, best_row_from_opt["SL Points"], interval_for_this_run)
                                        st.session_state.backtest_results = {
                                            "trades": trades_bo, "equity_curve": equity_bo, "performance": perf_bo,
                                            "params": {"Strategy": selected_strategy_name, "Symbol": selected_ticker_name,
                                                       "SL": best_row_from_opt["SL Points"], "RRR": best_row_from_opt["RRR"],
                                                       "TF": interval_for_this_run, "EntryDisplay": entry_display_opt_str,
                                                       "src": f"Opt ({opt_algo_ui})"}
                                        }
                                else:
                                    st.warning(f"No valid results found for metric '{opt_metric_ui}' in the optimization data. Cannot determine best parameters or run final backtest.")
                            else:
                                st.error("Optimization process yielded no results. Please check parameters and data.")
                        except Exception as e:
                            logger.error(f"Error during {analysis_mode_ui} for {selected_strategy_name}: {e}", exc_info=True)
                            st.error(f"An error occurred during {analysis_mode_ui}: {e}")
                        finally:
                             if prog_bar_container: prog_bar_container.progress(1.0, text="Optimization Complete!")


                # --- Walk-Forward Optimization ---
                elif analysis_mode_ui == "Walk-Forward Optimization":
                    st.subheader(f"Walk-Forward Optimization Run ({opt_algo_ui} - {selected_strategy_name})")
                    wfo_parameters_for_run = {
                        'in_sample_days': st.session_state.wfo_isd_ui_val,
                        'out_of_sample_days': st.session_state.wfo_oosd_ui_val,
                        'step_days': st.session_state.wfo_sd_ui_val
                    }
                    # Combine parameter ranges with control parameters for WFO's inner optimizer
                    wfo_optimizer_config_and_control = {**actual_params_to_optimize_config, **optimizer_control_params}
                    
                    with st.spinner(f"Running Walk-Forward Optimization with {opt_algo_ui}... This will take considerable time."):
                        try:
                            wfo_log_df, oos_trades_df, oos_equity_series, oos_performance_summary = optimizer.run_walk_forward_optimization(
                                price_data_df, initial_capital_ui, risk_per_trade_percent_ui,
                                wfo_parameters_for_run, opt_algo_ui, wfo_optimizer_config_and_control,
                                interval_for_this_run, optimization_progress_callback
                            )
                            st.session_state.wfo_results = {
                                "log": wfo_log_df,
                                "oos_trades": oos_trades_df,
                                "oos_equity_curve": oos_equity_series,
                                "oos_performance": oos_performance_summary
                            }
                            st.success("Walk-Forward Optimization finished!")
                            # For display consistency, populate backtest_results with WFO aggregated results
                            st.session_state.backtest_results = {
                                "trades": oos_trades_df,
                                "equity_curve": oos_equity_series,
                                "performance": oos_performance_summary,
                                "params": {"Strategy": selected_strategy_name, "Symbol": selected_ticker_name,
                                           "TF": interval_for_this_run, "src": "WFO Aggregated OOS"}
                            }
                        except Exception as e:
                            logger.error(f"Error during {analysis_mode_ui} for {selected_strategy_name}: {e}", exc_info=True)
                            st.error(f"An error occurred during {analysis_mode_ui}: {e}")
                        finally:
                            if prog_bar_container: prog_bar_container.progress(1.0, text="WFO Complete!")
            
            if prog_bar_container is not None: prog_bar_container.empty() # Clear progress bar area

# --- Display Area for Results ---
# Determine which tabs to create based on available results
main_tabs_to_display_names = []
if st.session_state.backtest_results:
    main_tabs_to_display_names.append("📊 Backtest Performance")
if not st.session_state.optimization_results_df.empty and analysis_mode_ui == "Parameter Optimization":
    main_tabs_to_display_names.append("⚙️ Optimization Results (Full Period)")
if st.session_state.wfo_results and analysis_mode_ui == "Walk-Forward Optimization":
    main_tabs_to_display_names.append("🚶 Walk-Forward Analysis")

if main_tabs_to_display_names:
    # Create a unique key for the tabs based on content and run count to force re-render if necessary
    tabs_key_string = "_".join(main_tabs_to_display_names) + f"_{st.session_state.run_analysis_clicked_count}_{selected_strategy_name}_{analysis_mode_ui}"
    created_tabs = st.tabs(main_tabs_to_display_names) 
    tab_map = dict(zip(main_tabs_to_display_names, created_tabs))

    # Tab 1: Backtest Performance (Single Run, Best Opt, or WFO Aggregated)
    if "📊 Backtest Performance" in tab_map:
        with tab_map["📊 Backtest Performance"]:
            if st.session_state.backtest_results:
                results_to_display = st.session_state.backtest_results
                performance_summary = results_to_display["performance"]
                trades_df_display = results_to_display["trades"]
                equity_curve_display = results_to_display["equity_curve"]
                
                run_params_info = results_to_display.get("params", {})
                run_source_info = run_params_info.get("src", "N/A")
                tf_display_info = run_params_info.get("TF", st.session_state.selected_timeframe_value)
                strat_display_info = run_params_info.get("Strategy", "N/A")
                symbol_display_info = run_params_info.get("Symbol", selected_ticker_name)

                param_info_header = f" (Strategy: {strat_display_info} | Symbol: {symbol_display_info} | Source: {run_source_info} | TF: {tf_display_info}"
                entry_display_val_info = run_params_info.get("EntryDisplay", "")
                if entry_display_val_info:
                    param_info_header += f" | {entry_display_val_info}"
                elif run_params_info.get("SL") is not None and run_params_info.get("RRR") is not None: # Fallback if EntryDisplay is missing
                    param_info_header += f" | SL: {float(run_params_info.get('SL')):.2f}, RRR: {float(run_params_info.get('RRR')):.1f}"
                param_info_header += ")"
                
                st.markdown(f"#### Performance Summary{param_info_header}")

                # Metric display function
                POSITIVE_COLOR, NEGATIVE_COLOR, NEUTRAL_COLOR = settings.POSITIVE_METRIC_COLOR, settings.NEGATIVE_METRIC_COLOR, settings.NEUTRAL_METRIC_COLOR
                def format_metric_value(value, precision=2, is_currency=True, is_percentage=False):
                    if pd.isna(value) or value is None: return "N/A"
                    if is_currency: return f"${value:,.{precision}f}"
                    if is_percentage: return f"{value:.{precision}f}%"
                    return f"{value:,.{precision}f}" if isinstance(value, (float, np.floating)) else str(value)

                def display_styled_metric_card(column, label, value_raw, is_currency=True, is_percentage=False, precision=2, profit_factor_logic=False, mdd_logic=False):
                    formatted_value = format_metric_value(value_raw, precision, is_currency, is_percentage)
                    color_style = NEUTRAL_COLOR # Default color for text (will be overridden by theme anyway)
                    
                    # Specific color logic for metrics
                    if not (pd.isna(value_raw) or value_raw is None):
                        if profit_factor_logic: # Profit Factor: >1 good, <1 bad
                            if value_raw > 1: color_style = POSITIVE_COLOR
                            elif value_raw < 1 and value_raw != 0 : color_style = NEGATIVE_COLOR
                            elif value_raw == 0 and performance_summary.get('Gross Profit', 0) == 0 and performance_summary.get('Gross Loss', 0) == 0: color_style = NEUTRAL_COLOR
                            elif value_raw == 0 : color_style = NEGATIVE_COLOR 
                        elif mdd_logic: # Max Drawdown: Always negative or zero. More negative is worse.
                            if value_raw < 0: color_style = NEGATIVE_COLOR
                            elif value_raw == 0: color_style = NEUTRAL_COLOR 
                        else: # General P&L like metrics
                            if value_raw > 0: color_style = POSITIVE_COLOR
                            elif value_raw < 0: color_style = NEGATIVE_COLOR
                    
                    column.markdown(f"""<div class="metric-card">
                                            <div class="metric-label">{label}</div>
                                            <div class="metric-value" style="color: {color_style};">{formatted_value}</div>
                                        </div>""", unsafe_allow_html=True)
                
                # Display metrics in columns
                col1_metrics, col2_metrics, col3_metrics = st.columns(3)
                with col1_metrics:
                    display_styled_metric_card(col1_metrics, "Total P&L", performance_summary.get('Total P&L'), is_currency=True)
                    display_styled_metric_card(col1_metrics, "Final Capital", performance_summary.get('Final Capital', initial_capital_ui), is_currency=True)
                    display_styled_metric_card(col1_metrics, "Max Drawdown", performance_summary.get('Max Drawdown (%)'), is_currency=False, is_percentage=True, mdd_logic=True)
                with col2_metrics:
                    display_styled_metric_card(col2_metrics, "Total Trades", int(performance_summary.get('Total Trades', 0)), is_currency=False, is_percentage=False)
                    display_styled_metric_card(col2_metrics, "Win Rate", performance_summary.get('Win Rate', 0), is_currency=False, is_percentage=True)
                    display_styled_metric_card(col2_metrics, "Profit Factor", performance_summary.get('Profit Factor', 0), is_currency=False, precision=2, profit_factor_logic=True)
                with col3_metrics:
                    display_styled_metric_card(col3_metrics, "Avg. Trade P&L", performance_summary.get('Average Trade P&L'), is_currency=True)
                    display_styled_metric_card(col3_metrics, "Avg. Winning Trade", performance_summary.get('Average Winning Trade'), is_currency=True)
                    display_styled_metric_card(col3_metrics, "Avg. Losing Trade", performance_summary.get('Average Losing Trade'), is_currency=True)
                
                # Detail Tabs within Performance Tab
                detail_tabs_list_names = ["📈 Equity Curve", "📊 Trades on Price", "📋 Trade Log"]
                # Conditionally add Signals tab if not WFO (WFO signals are per-fold, not aggregated for this view)
                if not st.session_state.signals.empty and analysis_mode_ui != "Walk-Forward Optimization":
                    detail_tabs_list_names.append("🔍 Generated Signals (Last Run)")
                detail_tabs_list_names.append("💾 Raw Price Data (Full Period)")
                
                detail_tabs_created = st.tabs(detail_tabs_list_names)
                
                with detail_tabs_created[0]: # Equity Curve
                    plot_title_equity = "Equity Curve"
                    plot_function_equity = plotting.plot_equity_curve
                    if analysis_mode_ui == "Walk-Forward Optimization" and "oos_equity_curve" in results_to_display:
                        plot_title_equity = "WFO: Aggregated Out-of-Sample Equity"
                        plot_function_equity = plotting.plot_wfo_equity_curve # Use WFO specific plotter if available
                        equity_curve_display = results_to_display["oos_equity_curve"] # Ensure correct equity curve for WFO
                    
                    if not equity_curve_display.empty:
                        st.plotly_chart(plot_function_equity(equity_curve_display, title=plot_title_equity), use_container_width=True)
                    else: st.info("Equity curve data is not available.")

                with detail_tabs_created[1]: # Trades on Price
                    if not st.session_state.price_data.empty and not trades_df_display.empty:
                        st.plotly_chart(plotting.plot_trades_on_price(st.session_state.price_data, trades_df_display, selected_ticker_name), use_container_width=True)
                    else: st.info("Price data or trade data not available for plotting trades on price.")

                with detail_tabs_created[2]: # Trade Log
                    if not trades_df_display.empty:
                        st.dataframe(trades_df_display.style.format({col: '{:.2f}' for col in trades_df_display.select_dtypes(include='float').columns}), height=300, use_container_width=True)
                    else: st.info("No trades were executed in this run.")

                idx_offset_details = 0 # To manage index for conditionally added tab
                if "🔍 Generated Signals (Last Run)" in detail_tabs_list_names:
                    with detail_tabs_created[3]: # Generated Signals
                        if not st.session_state.signals.empty:
                            st.dataframe(st.session_state.signals.style.format({col: '{:.2f}' for col in st.session_state.signals.select_dtypes(include='float').columns}), height=300, use_container_width=True)
                        else: st.info("No signals were generated for the last single backtest or optimization run.")
                    idx_offset_details = 1
                
                with detail_tabs_created[3 + idx_offset_details]: # Raw Price Data
                    if not st.session_state.price_data.empty:
                        st.markdown(f"Full period OHLCV data for **{selected_ticker_name} ({ticker_symbol})** ({len(st.session_state.price_data)} rows). Displaying first 100 rows.")
                        st.dataframe(st.session_state.price_data.head(100), height=300, use_container_width=True)
                        try:
                            csv_data_raw = st.session_state.price_data.to_csv(index=True).encode('utf-8')
                            st.download_button("Download Full Price Data CSV", csv_data_raw, f"{ticker_symbol}_price_data_{start_date_ui}_to_{end_date_ui}.csv", 'text/csv', key='dl_raw_price_main_v12')
                        except Exception as e_csv:
                            logger.error(f"Error generating CSV for raw price data: {e_csv}", exc_info=True)
                            st.warning("Could not prepare raw price data for download.")
                    else: st.info("Raw price data is not available for this run.")
            else:
                st.info("Run an analysis to see performance details. If an analysis was run, results might be empty if no trades occurred or data was insufficient.")

    # Tab 2: Optimization Results (Full Period) - Only if Parameter Optimization was run
    opt_tab_idx = main_tabs_to_display_names.index("⚙️ Optimization Results (Full Period)") if "⚙️ Optimization Results (Full Period)" in main_tabs_to_display_names else -1
    if opt_tab_idx != -1:
        with tab_map["⚙️ Optimization Results (Full Period)"]:
            opt_df_to_display = st.session_state.optimization_results_df
            if not opt_df_to_display.empty:
                st.markdown(f"#### Optimization Results Table ({selected_strategy_name} - Full Period - {opt_algo_ui})")
                # Format float columns for better readability
                float_cols_opt_table = [col for col in opt_df_to_display.columns if opt_df_to_display[col].dtype == 'float64']
                st.dataframe(opt_df_to_display.style.format({col: '{:.2f}' for col in float_cols_opt_table}), height=400, use_container_width=True)
                try:
                    csv_opt_results = opt_df_to_display.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Optimization Results CSV", csv_opt_results, f"{ticker_symbol}_{selected_strategy_name}_opt_results.csv", 'text/csv', key='dl_opt_csv_main_v12')
                except Exception as e_csv_opt:
                    logger.error(f"Error generating CSV for optimization results: {e_csv_opt}", exc_info=True)
                    st.warning("Could not prepare optimization results for download.")

                # Heatmap for Grid Search (if SL and RRR were optimized)
                if opt_algo_ui == "Grid Search" and 'SL Points' in opt_df_to_display.columns and 'RRR' in opt_df_to_display.columns:
                    st.markdown(f"#### Optimization Heatmap: {opt_metric_ui} (SL vs RRR - Full Period)")
                    try:
                        heatmap_fig = plotting.plot_optimization_heatmap(opt_df_to_display, 'SL Points', 'RRR', opt_metric_ui)
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                    except Exception as e_hm:
                        logger.error(f"Error generating optimization heatmap: {e_hm}", exc_info=True)
                        st.warning(f"Could not generate heatmap. Error: {e_hm}")
                elif opt_algo_ui == "Grid Search":
                     st.info("Heatmap for SL vs RRR requires 'SL Points' and 'RRR' to be part of the grid search parameters.")
                else: # Random Search
                    st.info("Heatmap is typically generated for Grid Search results. Review the table for Random Search optimization details.")
            else:
                st.info("No full-period optimization results available. Ensure 'Parameter Optimization' mode was run and completed successfully.")

    # Tab 3: Walk-Forward Analysis - Only if WFO was run
    wfo_tab_idx = main_tabs_to_display_names.index("🚶 Walk-Forward Analysis") if "🚶 Walk-Forward Analysis" in main_tabs_to_display_names else -1
    if wfo_tab_idx != -1:
        with tab_map["🚶 Walk-Forward Analysis"]:
            if st.session_state.wfo_results:
                wfo_results_display = st.session_state.wfo_results
                wfo_log_df_display = wfo_results_display.get("log", pd.DataFrame())
                wfo_oos_trades_df_display = wfo_results_display.get("oos_trades", pd.DataFrame())

                st.markdown(f"#### Walk-Forward Optimization Log ({selected_strategy_name} - {opt_algo_ui} for inner optimization)")
                if not wfo_log_df_display.empty:
                    float_cols_wfo_log = [col for col in wfo_log_df_display.columns if wfo_log_df_display[col].dtype == 'float64']
                    st.dataframe(wfo_log_df_display.style.format({col: '{:.2f}' for col in float_cols_wfo_log}), height=300, use_container_width=True)
                    try:
                        csv_wfo_log = wfo_log_df_display.to_csv(index=False).encode('utf-8')
                        st.download_button("Download WFO Log CSV", csv_wfo_log, f"{ticker_symbol}_{selected_strategy_name}_wfo_log.csv", 'text/csv', key='dl_wfo_log_main_v12')
                    except Exception as e_csv_wfo_log:
                        logger.error(f"Error generating CSV for WFO log: {e_csv_wfo_log}", exc_info=True)
                        st.warning("Could not prepare WFO log for download.")
                else:
                    st.info("WFO log is empty.")

                st.markdown("#### Aggregated Out-of-Sample (OOS) Trades from WFO")
                if not wfo_oos_trades_df_display.empty:
                    st.dataframe(wfo_oos_trades_df_display.style.format({col: '{:.2f}' for col in wfo_oos_trades_df_display.select_dtypes(include='float').columns}), height=300, use_container_width=True)
                    try:
                        csv_wfo_oos_trades = wfo_oos_trades_df_display.to_csv(index=False).encode('utf-8')
                        st.download_button("Download WFO OOS Trades CSV", csv_wfo_oos_trades, f"{ticker_symbol}_{selected_strategy_name}_wfo_oos_trades.csv", 'text/csv', key='dl_wfo_trades_main_v12')
                    except Exception as e_csv_wfo_trades:
                        logger.error(f"Error generating CSV for WFO OOS trades: {e_csv_wfo_trades}", exc_info=True)
                        st.warning("Could not prepare WFO OOS trades for download.")
                else:
                    st.info("No out-of-sample trades were generated during the Walk-Forward Optimization.")
                # Note: The aggregated OOS performance and equity curve are shown in the "Backtest Performance" tab when WFO is run.
            else:
                st.info("No Walk-Forward Optimization results available. Ensure 'Walk-Forward Optimization' mode was run and completed successfully.")

elif st.session_state.run_analysis_clicked_count > 0 : # If analysis was run but no tabs generated (e.g., all results empty)
    st.info("Analysis was run. If results are not displayed, it might be due to no trades or data for the selected parameters, or an error during processing. Check logs if errors are suspected.")
else: # Initial state before any analysis run
    if not any([st.session_state.backtest_results, not st.session_state.optimization_results_df.empty, st.session_state.wfo_results]):
        st.info("Configure parameters in the sidebar and click 'Run Analysis' to view results.")

# --- Footer and Disclaimer ---
st.sidebar.markdown("---")
st.sidebar.info(f"App Version: 0.6.2 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}") # Version increment
st.sidebar.caption("Disclaimer: This is a financial modeling tool for educational and research purposes. Past performance and optimization results are not indicative of future results and can be subject to overfitting. Always practice responsible risk management.")

