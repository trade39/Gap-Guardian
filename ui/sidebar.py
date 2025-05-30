# ui/sidebar.py
"""
Handles rendering of the Streamlit sidebar inputs and controls.
"""
import streamlit as st
from datetime import date, timedelta, time as dt_time
import numpy as np
from config import settings # Assuming settings.py is in a 'config' directory accessible via sys.path

# Strategy Explanations - Moved here as it's related to strategy selection in sidebar
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

def render_sidebar():
    """Renders all sidebar inputs and returns their current values and a run trigger."""
    st.sidebar.header("Backtest Configuration")

    # Strategy Selector
    selected_strategy_name = st.sidebar.selectbox(
        "Select Strategy:",
        options=settings.AVAILABLE_STRATEGIES,
        index=settings.AVAILABLE_STRATEGIES.index(st.session_state.selected_strategy),
        key="strategy_selector_sidebar_v1"
    )
    st.session_state.selected_strategy = selected_strategy_name # Persist selection

    # Ticker Selector
    selected_ticker_name = st.sidebar.selectbox(
        "Select Symbol:",
        options=list(settings.DEFAULT_TICKERS.keys()),
        index=0,
        key="ticker_sel_sidebar_v1"
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
        key="timeframe_selector_sidebar_v1"
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

    min_allowable_start_date_for_ui = (today - timedelta(days=max_history_limit_days - 1)) if max_history_limit_days else (today - timedelta(days=365 * 10))
    date_input_help_suffix = f"Data for {ui_current_interval} is limited to ~{max_history_limit_days} days." if max_history_limit_days else "Select historical period."

    if ui_current_interval in settings.YFINANCE_SHORT_INTRADAY_INTERVALS + settings.YFINANCE_HOURLY_INTERVALS:
        default_start_days_ago = min(settings.MAX_SHORT_INTRADAY_DAYS - 5 if settings.MAX_SHORT_INTRADAY_DAYS else 15,
                                     max_history_limit_days - 1 if max_history_limit_days else 15)
    else:
        default_start_days_ago = 365 if ui_current_interval != "1wk" else 30*7

    default_start_date_value = today - timedelta(days=default_start_days_ago)
    if default_start_date_value < min_allowable_start_date_for_ui:
        default_start_date_value = min_allowable_start_date_for_ui
    max_possible_start_date = today - timedelta(days=1)
    if default_start_date_value > max_possible_start_date:
        default_start_date_value = max_possible_start_date
    if default_start_date_value < min_allowable_start_date_for_ui:
         default_start_date_value = min_allowable_start_date_for_ui

    start_date_ui = st.sidebar.date_input(
        "Start Date:", value=default_start_date_value, min_value=min_allowable_start_date_for_ui,
        max_value=max_possible_start_date, key=f"start_date_sidebar_{ui_current_interval}_v1",
        help=f"Start date for historical data. {date_input_help_suffix}"
    )

    min_end_date_value_ui = start_date_ui + timedelta(days=1) if start_date_ui else min_allowable_start_date_for_ui + timedelta(days=1)
    default_end_date_value_ui = today
    if default_end_date_value_ui < min_end_date_value_ui: default_end_date_value_ui = min_end_date_value_ui
    if default_end_date_value_ui > today: default_end_date_value_ui = today

    end_date_ui = st.sidebar.date_input(
        "End Date:", value=default_end_date_value_ui, min_value=min_end_date_value_ui,
        max_value=today, key=f"end_date_sidebar_{ui_current_interval}_v1",
        help=f"End date for historical data. {date_input_help_suffix}"
    )

    # Financial Parameters
    initial_capital_ui = st.sidebar.number_input("Initial Capital ($):", min_value=1000.0, value=settings.DEFAULT_INITIAL_CAPITAL, step=1000.0, format="%.2f", key="ic_sidebar_v1")
    risk_per_trade_percent_ui = st.sidebar.number_input("Risk per Trade (%):", min_value=0.1, max_value=10.0, value=settings.DEFAULT_RISK_PER_TRADE_PERCENT, step=0.1, format="%.1f", key="rpt_sidebar_v1")

    # Common Strategy Parameters (Manual Run)
    st.sidebar.subheader("Common Strategy Parameters")
    sl_points_single_ui = st.sidebar.number_input("SL (points):", min_value=0.1, value=settings.DEFAULT_STOP_LOSS_POINTS, step=0.1, format="%.2f", key="sl_s_man_sidebar_v1")
    rrr_single_ui = st.sidebar.number_input("RRR:", min_value=0.1, value=settings.DEFAULT_RRR, step=0.1, format="%.1f", key="rrr_s_man_sidebar_v1")

    # Strategy-Specific Parameters (Manual Run)
    entry_start_hour_single_ui = settings.DEFAULT_ENTRY_WINDOW_START_HOUR
    entry_start_minute_single_ui = settings.DEFAULT_ENTRY_WINDOW_START_MINUTE
    entry_end_hour_single_ui = settings.DEFAULT_ENTRY_WINDOW_END_HOUR
    entry_end_minute_single_ui = settings.DEFAULT_ENTRY_WINDOW_END_MINUTE

    if selected_strategy_name == "Gap Guardian":
        st.sidebar.markdown("**Entry Window (NY Time - Manual Run):**")
        col1_entry, col2_entry = st.sidebar.columns(2)
        entry_start_hour_single_ui = col1_entry.number_input("Start Hr", 0, 23, settings.DEFAULT_ENTRY_WINDOW_START_HOUR, 1, key="esh_s_man_sidebar_v1")
        entry_start_minute_single_ui = col2_entry.number_input("Start Min", 0, 59, settings.DEFAULT_ENTRY_WINDOW_START_MINUTE, 15, key="esm_s_man_sidebar_v1")
        col1_exit, col2_exit = st.sidebar.columns(2)
        entry_end_hour_single_ui = col1_exit.number_input("End Hr", 0, 23, settings.DEFAULT_ENTRY_WINDOW_END_HOUR, 1, key="eeh_s_man_sidebar_v1")
        entry_end_minute_single_ui = col2_exit.number_input("End Min", 0, 59, settings.DEFAULT_ENTRY_WINDOW_END_MINUTE, 15, key="eem_s_man_sidebar_v1", help="Usually 00 for end of hour.")
    elif selected_strategy_name == "Unicorn":
        st.sidebar.caption("Unicorn strategy uses SL/RRR. Entry is pattern-based (Breaker + FVG).")
    elif selected_strategy_name == "Silver Bullet":
        st.sidebar.caption(f"Silver Bullet uses SL/RRR. Entry is FVG-based within fixed NY time windows: "
                           f"{', '.join([f'{s.strftime('%H:%M')}-{e.strftime('%H:%M')}' for s, e in settings.SILVER_BULLET_WINDOWS_NY])}.")

    # Analysis Mode Selector
    analysis_mode_ui = st.sidebar.radio(
        "Analysis Type:",
        ("Single Backtest", "Parameter Optimization", "Walk-Forward Optimization"),
        index=0, key="analysis_mode_sidebar_v1"
    )

    # Optimization Parameters
    opt_algo_ui = settings.DEFAULT_OPTIMIZATION_ALGORITHM
    sl_min_opt_ui, sl_max_opt_ui, sl_steps_opt_ui = settings.DEFAULT_SL_POINTS_OPTIMIZATION_RANGE.values()
    rrr_min_opt_ui, rrr_max_opt_ui, rrr_steps_opt_ui = settings.DEFAULT_RRR_OPTIMIZATION_RANGE.values()
    esh_min_opt_ui, esh_max_opt_ui, esh_steps_opt_ui = settings.DEFAULT_ENTRY_START_HOUR_OPTIMIZATION_RANGE.values()
    esm_vals_opt_ui = list(settings.DEFAULT_ENTRY_START_MINUTE_OPTIMIZATION_VALUES)
    eeh_min_opt_ui, eeh_max_opt_ui, eeh_steps_opt_ui = settings.DEFAULT_ENTRY_END_HOUR_OPTIMIZATION_RANGE.values()
    rand_iters_ui = settings.DEFAULT_RANDOM_SEARCH_ITERATIONS
    opt_metric_ui = settings.DEFAULT_OPTIMIZATION_METRIC

    if analysis_mode_ui != "Single Backtest":
        st.sidebar.markdown("##### In-Sample Optimization Settings")
        opt_algo_ui = st.sidebar.selectbox("Algorithm:", settings.OPTIMIZATION_ALGORITHMS, index=settings.OPTIMIZATION_ALGORITHMS.index(opt_algo_ui), key="opt_algo_sidebar_v1")
        opt_metric_ui = st.sidebar.selectbox("Optimize Metric:", settings.OPTIMIZATION_METRICS, index=settings.OPTIMIZATION_METRICS.index(opt_metric_ui), key="opt_metric_sidebar_v1")
        
        st.sidebar.markdown("**SL Range (points):**")
        c1,c2,c3 = st.sidebar.columns(3)
        sl_min_opt_ui = c1.number_input("Min", value=sl_min_opt_ui, step=0.1, format="%.1f", key="slmin_o_sidebar_v1", min_value=0.1)
        sl_max_opt_ui = c2.number_input("Max", value=sl_max_opt_ui, step=0.1, format="%.1f", key="slmax_o_sidebar_v1", min_value=sl_min_opt_ui + 0.1)
        if opt_algo_ui == "Grid Search":
            sl_steps_opt_ui = c3.number_input("Steps", min_value=2, max_value=20, value=int(sl_steps_opt_ui), step=1, key="slsteps_o_sidebar_v1")
        
        st.sidebar.markdown("**RRR Range:**")
        c1,c2,c3 = st.sidebar.columns(3)
        rrr_min_opt_ui = c1.number_input("Min", value=rrr_min_opt_ui, step=0.1, format="%.1f", key="rrrmin_o_sidebar_v1", min_value=0.1)
        rrr_max_opt_ui = c2.number_input("Max", value=rrr_max_opt_ui, step=0.1, format="%.1f", key="rrrmax_o_sidebar_v1", min_value=rrr_min_opt_ui + 0.1)
        if opt_algo_ui == "Grid Search":
            rrr_steps_opt_ui = c3.number_input("Steps", min_value=2, max_value=20, value=int(rrr_steps_opt_ui), step=1, key="rrrsteps_o_sidebar_v1")

        if selected_strategy_name == "Gap Guardian":
            st.sidebar.markdown("**Entry Start Hr Range (NY):**")
            c1,c2,c3 = st.sidebar.columns(3)
            esh_min_opt_ui = c1.number_input("Min Hr", value=esh_min_opt_ui, min_value=0, max_value=23, step=1, key="eshmin_o_sidebar_v1")
            esh_max_opt_ui = c2.number_input("Max Hr", value=esh_max_opt_ui, min_value=esh_min_opt_ui, max_value=23, step=1, key="eshmax_o_sidebar_v1")
            if opt_algo_ui == "Grid Search":
                esh_steps_opt_ui = c3.number_input("Hr Steps", min_value=1, max_value=10, value=int(esh_steps_opt_ui), step=1, key="eshsteps_o_sidebar_v1")
            
            esm_vals_opt_ui = st.sidebar.multiselect("Entry Start Min(s) (NY):", [0,15,30,45,50], default=esm_vals_opt_ui, key="esmvals_o_sidebar_v1")
            if not esm_vals_opt_ui: esm_vals_opt_ui = [settings.DEFAULT_ENTRY_WINDOW_START_MINUTE]

            st.sidebar.markdown("**Entry End Hr Range (NY):**")
            c1,c2,c3 = st.sidebar.columns(3)
            eeh_min_opt_ui = c1.number_input("Min Hr", value=eeh_min_opt_ui, min_value=0, max_value=23, step=1, key="eehmin_o_sidebar_v1")
            eeh_max_opt_ui = c2.number_input("Max Hr", value=eeh_max_opt_ui, min_value=eeh_min_opt_ui, max_value=23, step=1, key="eehmax_o_sidebar_v1")
            if opt_algo_ui == "Grid Search":
                eeh_steps_opt_ui = c3.number_input("Hr Steps", min_value=1, max_value=10, value=int(eeh_steps_opt_ui), step=1, key="eehsteps_o_sidebar_v1")
        
        if opt_algo_ui == "Random Search":
            rand_iters_ui = st.sidebar.number_input("Random Iterations:", min_value=10, max_value=1000, value=rand_iters_ui, step=10, key="randiter_o_sidebar_v1")
        
        if opt_algo_ui == "Grid Search":
            grid_combs = int(sl_steps_opt_ui * rrr_steps_opt_ui)
            if selected_strategy_name == "Gap Guardian":
                grid_combs *= int(esh_steps_opt_ui * len(esm_vals_opt_ui) * eeh_steps_opt_ui)
            st.sidebar.caption(f"Estimated Grid Combinations: {grid_combs}")
        else:
            st.sidebar.caption(f"Random Iterations: {rand_iters_ui}")

    # WFO Parameters
    wfo_isd_ui, wfo_oosd_ui, wfo_sd_ui = st.session_state.wfo_isd_ui_val, st.session_state.wfo_oosd_ui_val, st.session_state.wfo_sd_ui_val
    if analysis_mode_ui == "Walk-Forward Optimization":
        st.sidebar.markdown("##### WFO Settings (Calendar Days)")
        total_available_days_for_wfo = (end_date_ui - start_date_ui).days + 1
        MIN_WFO_IS_DAYS, MIN_WFO_OOS_DAYS, MIN_WFO_STEP_DAYS = 30, 10, 10

        calculated_isd, calculated_oosd, calculated_stepd = wfo_isd_ui, wfo_oosd_ui, wfo_sd_ui
        if total_available_days_for_wfo >= MIN_WFO_IS_DAYS + MIN_WFO_OOS_DAYS:
            tentative_oosd = max(MIN_WFO_OOS_DAYS, total_available_days_for_wfo // 5)
            tentative_isd = max(MIN_WFO_IS_DAYS, total_available_days_for_wfo - (tentative_oosd * 2))
            if tentative_isd + tentative_oosd > total_available_days_for_wfo:
                calculated_isd = max(MIN_WFO_IS_DAYS, int(total_available_days_for_wfo * 0.7))
                calculated_oosd = max(MIN_WFO_OOS_DAYS, total_available_days_for_wfo - calculated_isd)
            else:
                calculated_isd, calculated_oosd = tentative_isd, tentative_oosd
            calculated_stepd = max(MIN_WFO_STEP_DAYS, calculated_oosd)
            calculated_isd = max(MIN_WFO_IS_DAYS, calculated_isd)
            calculated_oosd = max(MIN_WFO_OOS_DAYS, calculated_oosd)
            if calculated_isd + calculated_oosd > total_available_days_for_wfo:
                calculated_oosd = max(MIN_WFO_OOS_DAYS, int(total_available_days_for_wfo * 0.25))
                calculated_isd = max(MIN_WFO_IS_DAYS, total_available_days_for_wfo - calculated_oosd)
                calculated_stepd = max(MIN_WFO_STEP_DAYS, calculated_oosd)
            st.session_state.wfo_isd_ui_val, st.session_state.wfo_oosd_ui_val, st.session_state.wfo_sd_ui_val = calculated_isd, calculated_oosd, calculated_stepd
            st.sidebar.caption(f"Suggested WFO: IS={calculated_isd}d, OOS={calculated_oosd}d, Step={calculated_stepd}d for {total_available_days_for_wfo}d total.")
        else:
            st.sidebar.caption(f"Total period ({total_available_days_for_wfo}d) is short. Min {MIN_WFO_IS_DAYS+MIN_WFO_OOS_DAYS}d recommended.")

        wfo_isd_ui = st.sidebar.number_input("In-Sample (Days):", min_value=MIN_WFO_IS_DAYS, value=st.session_state.wfo_isd_ui_val, step=10, key="wfoisd_sidebar_v1")
        wfo_oosd_ui = st.sidebar.number_input("Out-of-Sample (Days):", min_value=MIN_WFO_OOS_DAYS, value=st.session_state.wfo_oosd_ui_val, step=5, key="wfoosd_sidebar_v1")
        wfo_sd_ui = st.sidebar.number_input("Step (Days):", min_value=max(MIN_WFO_STEP_DAYS, wfo_oosd_ui), value=max(st.session_state.wfo_sd_ui_val, wfo_oosd_ui), step=5, key="wfosd_sidebar_v1", help="Step must be >= Out-of-Sample days.")
        st.session_state.wfo_isd_ui_val, st.session_state.wfo_oosd_ui_val, st.session_state.wfo_sd_ui_val = wfo_isd_ui, wfo_oosd_ui, wfo_sd_ui
    
    run_button_clicked = st.sidebar.button("Run Analysis", type="primary", use_container_width=True, key="run_main_sidebar_v1")

    # Package all sidebar inputs into a dictionary
    sidebar_inputs = {
        "selected_strategy_name": selected_strategy_name,
        "selected_ticker_name": selected_ticker_name,
        "ticker_symbol": ticker_symbol,
        "ui_current_interval": ui_current_interval,
        "start_date_ui": start_date_ui,
        "end_date_ui": end_date_ui,
        "initial_capital_ui": initial_capital_ui,
        "risk_per_trade_percent_ui": risk_per_trade_percent_ui,
        "sl_points_single_ui": sl_points_single_ui,
        "rrr_single_ui": rrr_single_ui,
        "entry_start_hour_single_ui": entry_start_hour_single_ui,
        "entry_start_minute_single_ui": entry_start_minute_single_ui,
        "entry_end_hour_single_ui": entry_end_hour_single_ui,
        "entry_end_minute_single_ui": entry_end_minute_single_ui,
        "analysis_mode_ui": analysis_mode_ui,
        "opt_algo_ui": opt_algo_ui,
        "sl_min_opt_ui": sl_min_opt_ui, "sl_max_opt_ui": sl_max_opt_ui, "sl_steps_opt_ui": sl_steps_opt_ui,
        "rrr_min_opt_ui": rrr_min_opt_ui, "rrr_max_opt_ui": rrr_max_opt_ui, "rrr_steps_opt_ui": rrr_steps_opt_ui,
        "esh_min_opt_ui": esh_min_opt_ui, "esh_max_opt_ui": esh_max_opt_ui, "esh_steps_opt_ui": esh_steps_opt_ui,
        "esm_vals_opt_ui": esm_vals_opt_ui,
        "eeh_min_opt_ui": eeh_min_opt_ui, "eeh_max_opt_ui": eeh_max_opt_ui, "eeh_steps_opt_ui": eeh_steps_opt_ui,
        "rand_iters_ui": rand_iters_ui,
        "opt_metric_ui": opt_metric_ui,
        "wfo_isd_ui": wfo_isd_ui, "wfo_oosd_ui": wfo_oosd_ui, "wfo_sd_ui": wfo_sd_ui,
        "run_button_clicked": run_button_clicked
    }
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"App Version: {settings.APP_VERSION} | Last Updated: {settings.APP_LAST_UPDATED}")
    st.sidebar.caption(settings.APP_DISCLAIMER)
    
    return sidebar_inputs

def display_main_content_header(sidebar_inputs):
    """Displays the main content header based on sidebar inputs."""
    selected_strategy_name = sidebar_inputs["selected_strategy_name"]
    selected_ticker_name = sidebar_inputs["selected_ticker_name"]
    selected_timeframe_display = [k for k, v in settings.AVAILABLE_TIMEFRAMES.items() if v == sidebar_inputs["ui_current_interval"]][0]

    st.title(f"🛡️📈 {settings.APP_TITLE}")
    strategy_info_md = f"Selected Strategy: **{selected_strategy_name}** | Symbol: **{selected_ticker_name}** | Timeframe: **{selected_timeframe_display}** ({sidebar_inputs['ui_current_interval']})"
    if selected_strategy_name == "Gap Guardian":
        strategy_info_md += f" | Default Manual Entry: {settings.DEFAULT_ENTRY_WINDOW_START_HOUR:02d}:{settings.DEFAULT_ENTRY_WINDOW_START_MINUTE:02d}-{settings.DEFAULT_ENTRY_WINDOW_END_HOUR:02d}:{settings.DEFAULT_ENTRY_WINDOW_END_MINUTE:02d} NYT"
    elif selected_strategy_name == "Silver Bullet":
         strategy_info_md += f" | Fixed NYT Entry Windows: {', '.join([f'{s.strftime('%H:%M')}-{e.strftime('%H:%M')}' for s, e in settings.SILVER_BULLET_WINDOWS_NY])}"
    st.markdown(strategy_info_md)

    with st.expander(f"Understanding the '{selected_strategy_name}' Strategy", expanded=False):
        st.markdown(STRATEGY_EXPLANATIONS.get(selected_strategy_name, "Explanation not available for this strategy."))

