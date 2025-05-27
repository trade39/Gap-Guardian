# app.py
"""
Main Streamlit application file for the Gap Guardian Strategy Backtester.
Handles UI, user inputs, and orchestrates the backtesting process,
including optimization of SL, RRR, and Entry Window Times, and Walk-Forward Optimization (WFO).
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime, time as dt_time

from config import settings
from services import data_loader, strategy_engine, backtester, optimizer
from utils import plotting, logger as app_logger

logger = app_logger.get_logger(__name__)

st.set_page_config(page_title=settings.APP_TITLE, page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
def load_custom_css(css_file_path):
    try:
        with open(css_file_path) as f: st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e: st.warning(f"CSS file not found or error: {e}")
load_custom_css("static/style.css")

def init_session_state():
    defaults = {
        'backtest_results': None, 'optimization_results_df': pd.DataFrame(),
        'price_data': pd.DataFrame(), 'signals': pd.DataFrame(),
        'best_params_from_opt': None, 'wfo_results': None,
        'selected_timeframe': settings.DEFAULT_STRATEGY_TIMEFRAME,
        'current_analysis_mode_for_tabs': None, # To help manage tab recreation
        'current_timeframe_for_tabs': None
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
init_session_state()

# --- Sidebar Inputs ---
st.sidebar.header("Backtest Configuration")
selected_ticker_name = st.sidebar.selectbox("Select Symbol:", options=list(settings.DEFAULT_TICKERS.keys()), index=0, key="ticker_sel")
ticker_symbol = settings.DEFAULT_TICKERS[selected_ticker_name]

selected_timeframe_display = st.sidebar.selectbox(
    "Select Timeframe:", options=list(settings.AVAILABLE_TIMEFRAMES.keys()),
    index=list(settings.AVAILABLE_TIMEFRAMES.values()).index(st.session_state.selected_timeframe),
    key="timeframe_selector_ui_main",
    help="Select data timeframe. Shorter timeframes have limited historical data."
)
st.session_state.selected_timeframe = settings.AVAILABLE_TIMEFRAMES[selected_timeframe_display]
current_interval = st.session_state.selected_timeframe

today = date.today()
max_history_limit_days = None
if current_interval in settings.YFINANCE_SHORT_INTRADAY_INTERVALS: max_history_limit_days = settings.MAX_SHORT_INTRADAY_DAYS
elif current_interval in settings.YFINANCE_HOURLY_INTERVALS: max_history_limit_days = settings.MAX_HOURLY_INTRADAY_DAYS

if max_history_limit_days:
    min_allowable_start_date_for_ui = today - timedelta(days=max_history_limit_days -1)
    date_input_help_suffix = f"Data for {current_interval} is limited to ~{max_history_limit_days} days."
else:
    min_allowable_start_date_for_ui = today - timedelta(days=365 * 10)
    date_input_help_suffix = "Select historical period."

default_start_offset = 30
if current_interval in ["1m", "5m", "15m", "30m", "1h", "60m", "90m"]:
    default_start_date_value = today - timedelta(days=min(15, max_history_limit_days -1 if max_history_limit_days else 15))
else:
    default_start_date_value = today - timedelta(days=default_start_offset * 7 if current_interval == "1wk" else default_start_offset)

if default_start_date_value < min_allowable_start_date_for_ui: default_start_date_value = min_allowable_start_date_for_ui
max_possible_start_date = today - timedelta(days=1)
if default_start_date_value > max_possible_start_date: default_start_date_value = max_possible_start_date
if default_start_date_value < min_allowable_start_date_for_ui: default_start_date_value = min_allowable_start_date_for_ui

start_date = st.sidebar.date_input("Start Date:", value=default_start_date_value, min_value=min_allowable_start_date_for_ui, max_value=max_possible_start_date, key=f"start_date_{current_interval}_v2", help=f"Start date. {date_input_help_suffix}")
min_end_date_value = start_date + timedelta(days=1) if start_date else min_allowable_start_date_for_ui + timedelta(days=1)
default_end_date_value = today
if default_end_date_value < min_end_date_value: default_end_date_value = min_end_date_value
if default_end_date_value > today: default_end_date_value = today
end_date = st.sidebar.date_input("End Date:", value=default_end_date_value, min_value=min_end_date_value, max_value=today, key=f"end_date_{current_interval}_v2", help=f"End date. {date_input_help_suffix}")

initial_capital = st.sidebar.number_input("Initial Capital ($):", min_value=1000.0, value=settings.DEFAULT_INITIAL_CAPITAL, step=1000.0, format="%.2f")
risk_per_trade_percent = st.sidebar.number_input("Risk per Trade (%):", min_value=0.1, max_value=10.0, value=settings.DEFAULT_RISK_PER_TRADE_PERCENT, step=0.1, format="%.1f")

st.sidebar.subheader("Strategy Parameters (Manual / Optimization Base)")
sl_points_single = st.sidebar.number_input("Stop Loss (points):", min_value=0.1, value=settings.DEFAULT_STOP_LOSS_POINTS, step=0.1, format="%.2f", key="sl_single_man")
rrr_single = st.sidebar.number_input("Risk/Reward Ratio:", min_value=0.1, value=settings.DEFAULT_RRR, step=0.1, format="%.1f", key="rrr_single_man")
st.sidebar.markdown("**Entry Window (NY Time - Manual Run):**")
col_esh, col_esm = st.sidebar.columns(2)
entry_start_hour_single = col_esh.number_input("Start Hour", min_value=0, max_value=23, value=settings.DEFAULT_ENTRY_WINDOW_START_HOUR, step=1, key="entry_sh_man")
entry_start_minute_single = col_esm.number_input("Start Minute", min_value=0, max_value=59, value=settings.DEFAULT_ENTRY_WINDOW_START_MINUTE, step=15, key="entry_sm_man")
col_eeh, col_eem = st.sidebar.columns(2)
entry_end_hour_single = col_eeh.number_input("End Hour", min_value=0, max_value=23, value=settings.DEFAULT_ENTRY_WINDOW_END_HOUR, step=1, key="entry_eh_man")
entry_end_minute_single = col_eem.number_input("End Minute", min_value=0, max_value=59, value=settings.DEFAULT_ENTRY_WINDOW_END_MINUTE, step=15, key="entry_em_man", help="Usually 00 for end of hour.")

analysis_mode = st.sidebar.radio("Select Analysis Type:", ("Single Backtest", "Parameter Optimization", "Walk-Forward Optimization"), index=0, key="analysis_mode_radio_main")

# --- Optimization Settings ---
opt_algo = settings.DEFAULT_OPTIMIZATION_ALGORITHM
sl_min_opt, sl_max_opt, sl_steps_opt = settings.DEFAULT_SL_POINTS_OPTIMIZATION_RANGE.values()
rrr_min_opt, rrr_max_opt, rrr_steps_opt = settings.DEFAULT_RRR_OPTIMIZATION_RANGE.values()
esh_min_opt, esh_max_opt, esh_steps_opt = settings.DEFAULT_ENTRY_START_HOUR_OPTIMIZATION_RANGE.values()
esm_values_opt = settings.DEFAULT_ENTRY_START_MINUTE_OPTIMIZATION_VALUES
eeh_min_opt, eeh_max_opt, eeh_steps_opt = settings.DEFAULT_ENTRY_END_HOUR_OPTIMIZATION_RANGE.values()
eem_values_opt = settings.DEFAULT_ENTRY_END_MINUTE_OPTIMIZATION_VALUES # Usually fixed [0]

random_iterations = settings.DEFAULT_RANDOM_SEARCH_ITERATIONS
optimization_metric = settings.DEFAULT_OPTIMIZATION_METRIC

if analysis_mode == "Parameter Optimization" or analysis_mode == "Walk-Forward Optimization":
    st.sidebar.markdown("##### In-Sample Optimization Settings")
    opt_algo = st.sidebar.selectbox("Optimization Algorithm:", settings.OPTIMIZATION_ALGORITHMS, index=settings.OPTIMIZATION_ALGORITHMS.index(opt_algo))
    optimization_metric = st.sidebar.selectbox("Optimize for Metric:", options=settings.OPTIMIZATION_METRICS, index=settings.OPTIMIZATION_METRICS.index(optimization_metric))

    st.sidebar.markdown("**Stop Loss (SL) Points Range:**")
    c1,c2,c3 = st.sidebar.columns(3); sl_min_opt = c1.number_input("Min", value=sl_min_opt, step=0.1, format="%.1f", key="sl_min_o")
    sl_max_opt = c2.number_input("Max", value=sl_max_opt, step=0.1, format="%.1f", key="sl_max_o")
    if opt_algo == "Grid Search": sl_steps_opt = c3.number_input("Steps", min_value=2, max_value=10, value=int(sl_steps_opt), step=1, key="sl_steps_o")
    
    st.sidebar.markdown("**Risk/Reward Ratio (RRR) Range:**")
    c1,c2,c3 = st.sidebar.columns(3); rrr_min_opt = c1.number_input("Min", value=rrr_min_opt, step=0.1, format="%.1f", key="rrr_min_o")
    rrr_max_opt = c2.number_input("Max", value=rrr_max_opt, step=0.1, format="%.1f", key="rrr_max_o")
    if opt_algo == "Grid Search": rrr_steps_opt = c3.number_input("Steps", min_value=2, max_value=10, value=int(rrr_steps_opt), step=1, key="rrr_steps_o")

    st.sidebar.markdown("**Entry Start Hour (NY) Range:**")
    c1,c2,c3 = st.sidebar.columns(3); esh_min_opt = c1.number_input("Min Hour", value=esh_min_opt, min_value=0, max_value=23, step=1, key="esh_min_o")
    esh_max_opt = c2.number_input("Max Hour", value=esh_max_opt, min_value=0, max_value=23, step=1, key="esh_max_o")
    if opt_algo == "Grid Search": esh_steps_opt = c3.number_input("Hour Steps", min_value=2, max_value=5, value=int(esh_steps_opt), step=1, key="esh_steps_o")
    
    esm_values_opt = st.sidebar.multiselect("Entry Start Minute(s) (NY):", options=[0, 15, 30, 45, 50], default=settings.DEFAULT_ENTRY_START_MINUTE_OPTIMIZATION_VALUES, key="esm_vals_o", help="Select one or more minutes for Grid Search, or range for Random (not yet implemented for random minute range).")
    if not esm_values_opt: esm_values_opt = [settings.DEFAULT_ENTRY_WINDOW_START_MINUTE] # Ensure at least one value

    st.sidebar.markdown("**Entry End Hour (NY) Range:**")
    c1,c2,c3 = st.sidebar.columns(3); eeh_min_opt = c1.number_input("Min Hour", value=eeh_min_opt, min_value=0, max_value=23, step=1, key="eeh_min_o")
    eeh_max_opt = c2.number_input("Max Hour", value=eeh_max_opt, min_value=0, max_value=23, step=1, key="eeh_max_o")
    if opt_algo == "Grid Search": eeh_steps_opt = c3.number_input("Hour Steps", min_value=2, max_value=5, value=int(eeh_steps_opt), step=1, key="eeh_steps_o")
    # Entry End Minute is typically fixed, e.g., 00. Not adding UI for its optimization for now to keep it simpler.

    if opt_algo == "Random Search":
        random_iterations = st.sidebar.number_input("Random Search Iterations:", min_value=10, max_value=500, value=random_iterations, step=10)
        st.sidebar.caption(f"Random Iterations: {random_iterations}")
    elif opt_algo == "Grid Search":
        total_grid_combs = int(sl_steps_opt * rrr_steps_opt * esh_steps_opt * len(esm_values_opt) * eeh_steps_opt)
        st.sidebar.caption(f"Grid Combinations: {total_grid_combs}")


wfo_in_sample_days, wfo_oos_days, wfo_step_days = settings.DEFAULT_WFO_IN_SAMPLE_DAYS, settings.DEFAULT_WFO_OUT_OF_SAMPLE_DAYS, settings.DEFAULT_WFO_STEP_DAYS
if analysis_mode == "Walk-Forward Optimization":
    st.sidebar.markdown("##### Walk-Forward Settings (Calendar Days)")
    wfo_in_sample_days = st.sidebar.number_input("In-Sample Period (Days):", min_value=30, value=wfo_in_sample_days, step=10)
    wfo_oos_days = st.sidebar.number_input("Out-of-Sample Period (Days):", min_value=10, value=wfo_oos_days, step=5)
    wfo_step_days = st.sidebar.number_input("Step / Re-optimize Every (Days):", min_value=wfo_oos_days, value=wfo_step_days, step=5)

# --- Main App Area ---
st.title(f"🛡️ {settings.APP_TITLE}")
st.markdown(f"Strategy: Gap Guardian | Timeframe: **{selected_timeframe_display}** ({current_interval}) | Default Entry Window: {settings.DEFAULT_ENTRY_WINDOW_START_HOUR:02d}:{settings.DEFAULT_ENTRY_WINDOW_START_MINUTE:02d}-{settings.DEFAULT_ENTRY_WINDOW_END_HOUR:02d}:{settings.DEFAULT_ENTRY_WINDOW_END_MINUTE:02d} NYT")

if st.sidebar.button("Run Analysis", type="primary", use_container_width=True, key="run_button_phase3"):
    init_session_state(); st.session_state.selected_timeframe = current_interval

    if start_date >= end_date: st.error(f"Error: Start date ({start_date}) must be before end date ({end_date}).")
    else:
        with st.spinner("Fetching data..."):
            price_data_df = data_loader.fetch_historical_data(ticker_symbol, start_date, end_date, current_interval)
            st.session_state.price_data = price_data_df
        if price_data_df.empty: st.warning(f"No price data for {selected_ticker_name} ({current_interval}). Cannot proceed.")
        else:
            # Prepare parameters for strategy engine and optimizer
            manual_entry_start_time = dt_time(entry_start_hour_single, entry_start_minute_single)
            manual_entry_end_time = dt_time(entry_end_hour_single, entry_end_minute_single)

            if analysis_mode == "Single Backtest":
                st.subheader("Single Backtest Run")
                with st.spinner("Running..."):
                    signals = strategy_engine.generate_signals(price_data_df.copy(), sl_points_single, rrr_single, manual_entry_start_time, manual_entry_end_time)
                    st.session_state.signals = signals
                    trades, equity, perf = backtester.run_backtest(price_data_df.copy(), signals, initial_capital, risk_per_trade_percent, sl_points_single)
                    st.session_state.backtest_results = {"trades": trades, "equity_curve": equity, "performance": perf, "params": {"SL": sl_points_single, "RRR": rrr_single, "TF": current_interval, "Entry": f"{manual_entry_start_time:%H:%M}-{manual_entry_end_time:%H:%M}", "src": "Manual"}}
                    st.success("Single backtest complete!")
            
            elif analysis_mode in ["Parameter Optimization", "Walk-Forward Optimization"]:
                prog_bar = st.progress(0, text="Initializing optimization...")
                def opt_prog_cb(p, stage): prog_bar.progress(p, text=f"{stage}: {int(p*100)}% complete")

                param_config_for_opt = {'metric_to_optimize': optimization_metric}
                if opt_algo == "Grid Search":
                    param_config_for_opt['sl_values'] = np.linspace(sl_min_opt, sl_max_opt, int(sl_steps_opt))
                    param_config_for_opt['rrr_values'] = np.linspace(rrr_min_opt, rrr_max_opt, int(rrr_steps_opt))
                    param_config_for_opt['entry_start_hour_values'] = [int(h) for h in np.linspace(esh_min_opt, esh_max_opt, int(esh_steps_opt))]
                    param_config_for_opt['entry_start_minute_values'] = esm_values_opt # Already a list of ints
                    param_config_for_opt['entry_end_hour_values'] = [int(h) for h in np.linspace(eeh_min_opt, eeh_max_opt, int(eeh_steps_opt))]
                    # Entry end minute kept fixed from settings for now
                    param_config_for_opt['entry_end_minute_values'] = settings.DEFAULT_ENTRY_END_MINUTE_OPTIMIZATION_VALUES

                else: # Random Search
                    param_config_for_opt['sl_range'] = (sl_min_opt, sl_max_opt)
                    param_config_for_opt['rrr_range'] = (rrr_min_opt, rrr_max_opt)
                    param_config_for_opt['entry_start_hour_range'] = (esh_min_opt, esh_max_opt)
                    param_config_for_opt['entry_start_minute_values'] = esm_values_opt # Sample from this list
                    param_config_for_opt['entry_end_hour_range'] = (eeh_min_opt, eeh_max_opt)
                    param_config_for_opt['entry_end_minute_values'] = settings.DEFAULT_ENTRY_END_MINUTE_OPTIMIZATION_VALUES
                    param_config_for_opt['iterations'] = random_iterations
                
                if analysis_mode == "Parameter Optimization":
                    st.subheader(f"Parameter Optimization ({opt_algo} - Full Period)")
                    with st.spinner(f"Running {opt_algo}..."):
                        if opt_algo == "Grid Search":
                            opt_df = optimizer.run_grid_search(price_data_df, initial_capital, risk_per_trade_percent, param_config_for_opt, lambda p,s: opt_prog_cb(p,s))
                        else: # Random
                            opt_df = optimizer.run_random_search(price_data_df, initial_capital, risk_per_trade_percent, param_config_for_opt, random_iterations, lambda p,s: opt_prog_cb(p,s))
                        st.session_state.optimization_results_df = opt_df
                        prog_bar.progress(1.0, text="Optimization Complete!")
                        if not opt_df.empty:
                            st.success("Full period optimization finished!")
                            valid_opt = opt_df.dropna(subset=[optimization_metric])
                            if not valid_opt.empty:
                                best_r = valid_opt.loc[valid_opt[optimization_metric].idxmin()] if optimization_metric == "Max Drawdown (%)" else valid_opt.loc[valid_opt[optimization_metric].idxmax()]
                                st.session_state.best_params_from_opt = {"SL": best_r["SL Points"], "RRR": best_r["RRR"], "ES_H": best_r["EntryStartHour"], "ES_M": best_r["EntryStartMinute"], "EE_H": best_r["EntryEndHour"], "EE_M": best_r.get("EntryEndMinute",0), "Metric": best_r[optimization_metric]}
                                st.info(f"Best params for '{optimization_metric}': SL={best_r['SL Points']:.2f}, RRR={best_r['RRR']:.1f}, EntryStart={int(best_r['EntryStartHour']):02d}:{int(best_r['EntryStartMinute']):02d}, EntryEnd={int(best_r['EntryEndHour']):02d}:{int(best_r.get('EntryEndMinute',0)):02d} (Val: {best_r[optimization_metric]:.2f})")
                                # Run backtest with best params
                                best_entry_start = dt_time(int(best_r["EntryStartHour"]), int(best_r["EntryStartMinute"]))
                                best_entry_end = dt_time(int(best_r["EntryEndHour"]), int(best_r.get("EntryEndMinute",0))) # Use default if not in results
                                signals_b = strategy_engine.generate_signals(price_data_df.copy(), best_r["SL Points"], best_r["RRR"], best_entry_start, best_entry_end)
                                st.session_state.signals = signals_b
                                trades_b, equity_b, perf_b = backtester.run_backtest(price_data_df.copy(), signals_b, initial_capital, risk_per_trade_percent, best_r["SL Points"])
                                st.session_state.backtest_results = {"trades": trades_b, "equity_curve": equity_b, "performance": perf_b, "params": {"SL": best_r["SL Points"], "RRR": best_r["RRR"], "TF": current_interval, "Entry": f"{best_entry_start:%H:%M}-{best_entry_end:%H:%M}", "src": f"Opt ({opt_algo})"}}
                            else: st.warning(f"No valid results for '{optimization_metric}' in optimization.")
                        else: st.error("Optimization yielded no results.")

                elif analysis_mode == "Walk-Forward Optimization":
                    st.subheader(f"Walk-Forward Optimization Run ({opt_algo})")
                    wfo_p = {'in_sample_days': wfo_in_sample_days, 'out_of_sample_days': wfo_oos_days, 'step_days': wfo_step_days}
                    with st.spinner(f"Running WFO with {opt_algo}... This will take considerable time."):
                        wfo_log, oos_trades, oos_equity, oos_perf = optimizer.run_walk_forward_optimization(price_data_df, initial_capital, risk_per_trade_percent, wfo_p, opt_algo, param_config_for_opt, lambda p,s: opt_prog_cb(p,s))
                        st.session_state.wfo_results = {"log": wfo_log, "oos_trades": oos_trades, "oos_equity_curve": oos_equity, "oos_performance": oos_perf}
                        prog_bar.progress(1.0, text="WFO Complete!")
                        st.success("Walk-Forward Optimization finished!")
                        st.session_state.backtest_results = {"trades": oos_trades, "equity_curve": oos_equity, "performance": oos_perf, "params": {"TF": current_interval, "src": "WFO Aggregated"}}

# --- Display Area ---
# ... (Display logic from previous version, ensure it handles new param info in titles/summaries) ...
main_tabs_list = ["📊 Backtest Performance"]
if not st.session_state.optimization_results_df.empty and analysis_mode == "Parameter Optimization": main_tabs_list.append("⚙️ Optimization Results (Full Period)")
if st.session_state.wfo_results and analysis_mode == "Walk-Forward Optimization": main_tabs_list.append("🚶 Walk-Forward Analysis")

if st.session_state.backtest_results or not st.session_state.optimization_results_df.empty or st.session_state.wfo_results:
    if 'main_display_tabs' not in st.session_state or st.session_state.get('current_analysis_mode_for_tabs') != analysis_mode or st.session_state.get('current_timeframe_for_tabs') != current_interval:
        st.session_state.main_display_tabs = st.tabs(main_tabs_list); st.session_state.current_analysis_mode_for_tabs = analysis_mode; st.session_state.current_timeframe_for_tabs = current_interval
    active_tabs = st.session_state.main_display_tabs
    
    with active_tabs[0]: # Backtest Performance
        if st.session_state.backtest_results:
            results = st.session_state.backtest_results; performance = results["performance"]; trades = results["trades"]; equity_curve = results["equity_curve"]
            run_params = results.get("params", {}); run_source = run_params.get("src", "N/A"); tf_disp = run_params.get("TF", current_interval)
            param_info = f" (Source: {run_source} | TF: {tf_disp}"
            if "SL" in run_params: param_info += f" | SL: {run_params['SL']:.2f}"
            if "RRR" in run_params: param_info += f" | RRR: {run_params['RRR']:.1f}"
            if "Entry" in run_params: param_info += f" | Entry: {run_params['Entry']}"
            param_info += ")"
            st.markdown(f"#### Performance Summary{param_info}")
            # ... (display_styled_metric calls as before) ...
            POSITIVE_COLOR = settings.POSITIVE_METRIC_COLOR; NEGATIVE_COLOR = settings.NEGATIVE_METRIC_COLOR; NEUTRAL_COLOR = settings.NEUTRAL_METRIC_COLOR
            def format_metric_display(v, p=2, c=True, pct=False):
                if pd.isna(v) or v is None: return "N/A"
                if c: return f"${v:,.{p}f}"
                if pct: return f"{v:.{p}f}%"
                return f"{v:,.{p}f}" if isinstance(v, float) else str(v)
            def display_styled_metric(col, lbl, val, raw, c=True, pct=False, p=2, pf_logic=False, mdd_logic=False):
                fmt_val = format_metric_display(val, p, c, pct); clr = NEUTRAL_COLOR
                if not (pd.isna(raw) or raw is None):
                    if pf_logic:
                        if raw > 1: clr = POSITIVE_COLOR
                        elif raw < 1 and raw != 0 : clr = NEGATIVE_COLOR
                        elif raw == 0 and performance.get('Gross Profit', 0) == 0 and performance.get('Gross Loss', 0) == 0: clr = NEUTRAL_COLOR
                        elif raw == 0 : clr = NEGATIVE_COLOR
                    elif mdd_logic:
                        if raw < 0: clr = NEGATIVE_COLOR
                        elif raw == 0: clr = NEUTRAL_COLOR
                    else:
                        if raw > 0: clr = POSITIVE_COLOR
                        elif raw < 0: clr = NEGATIVE_COLOR
                col.markdown(f"""<div class="metric-card"><div class="metric-label">{lbl}</div><div class="metric-value" style="color: {clr};">{fmt_val}</div></div>""", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1: display_styled_metric(col1, "Total P&L", performance.get('Total P&L'), performance.get('Total P&L')); display_styled_metric(col1, "Final Capital", performance.get('Final Capital', initial_capital), performance.get('Final Capital', initial_capital), c=True); display_styled_metric(col1, "Max Drawdown", performance.get('Max Drawdown (%)'), performance.get('Max Drawdown (%)'), c=False, pct=True, mdd_logic=True)
            with col2: display_styled_metric(col2, "Total Trades", int(performance.get('Total Trades', 0)), int(performance.get('Total Trades', 0)), c=False, pct=False); display_styled_metric(col2, "Win Rate", performance.get('Win Rate', 0), performance.get('Win Rate', 0), c=False, pct=True); display_styled_metric(col2, "Profit Factor", performance.get('Profit Factor', 0), performance.get('Profit Factor', 0), c=False, p=2, pf_logic=True)
            with col3: display_styled_metric(col3, "Avg. Trade P&L", performance.get('Average Trade P&L'), performance.get('Average Trade P&L')); display_styled_metric(col3, "Avg. Winning Trade", performance.get('Average Winning Trade'), performance.get('Average Winning Trade')); display_styled_metric(col3, "Avg. Losing Trade", performance.get('Average Losing Trade'), performance.get('Average Losing Trade'))
            # Detail Tabs
            detail_tabs_list = ["📈 Equity Curve", "📊 Trades on Price", "📋 Trade Log"]
            if not st.session_state.signals.empty and analysis_mode != "Walk-Forward Optimization": detail_tabs_list.append("🔍 Generated Signals (Last Run)")
            detail_tabs_list.append("💾 Raw Price Data (Full Period)")
            detail_tabs = st.tabs(detail_tabs_list)
            with detail_tabs[0]:
                plot_title = "Equity Curve" if analysis_mode != "Walk-Forward Optimization" else "WFO: Aggregated Out-of-Sample Equity"
                plot_func = plotting.plot_equity_curve if analysis_mode != "Walk-Forward Optimization" else plotting.plot_wfo_equity_curve
                if not equity_curve.empty: st.plotly_chart(plot_func(equity_curve, title=plot_title), use_container_width=True)
                else: st.info("Equity curve is not available.")
            with detail_tabs[1]:
                if not st.session_state.price_data.empty and not trades.empty : st.plotly_chart(plotting.plot_trades_on_price(st.session_state.price_data, trades, selected_ticker_name), use_container_width=True)
                else: st.info("Price/trade data not available for plotting.")
            with detail_tabs[2]:
                if not trades.empty: st.dataframe(trades.style.format({col: '{:.2f}' for col in trades.select_dtypes(include='float').columns}), height=300, use_container_width=True)
                else: st.info("No trades were executed.")
            idx_offset = 0
            if "🔍 Generated Signals (Last Run)" in detail_tabs_list:
                with detail_tabs[3]:
                    if not st.session_state.signals.empty: st.dataframe(st.session_state.signals.style.format({col: '{:.2f}' for col in st.session_state.signals.select_dtypes(include='float').columns}), height=300, use_container_width=True)
                    else: st.info("No signals generated for the last single/optimization run.")
                idx_offset = 1
            with detail_tabs[3+idx_offset]:
                if not st.session_state.price_data.empty:
                    st.markdown(f"Full period OHLCV data for **{selected_ticker_name}** ({len(st.session_state.price_data)} rows).")
                    st.dataframe(st.session_state.price_data.head(), height=300, use_container_width=True)
                    csv_data = st.session_state.price_data.to_csv(index=True).encode('utf-8'); st.download_button("Download Full Price Data CSV", csv_data, f"{ticker_symbol}_price_data.csv", 'text/csv', key='dl_raw_price_main_v2')
                else: st.info("Raw price data is not available.")
        else: st.info("Run an analysis to see performance details.")

    # Tab 2: Optimization Results
    opt_tab_idx = main_tabs_list.index("⚙️ Optimization Results (Full Period)") if "⚙️ Optimization Results (Full Period)" in main_tabs_list else -1
    if opt_tab_idx != -1 and len(active_tabs) > opt_tab_idx:
        with active_tabs[opt_tab_idx]:
            opt_df = st.session_state.optimization_results_df
            if not opt_df.empty:
                st.markdown("#### Grid/Random Search Results (Full Period)")
                # Ensure all expected columns are formatted, handle missing ones gracefully
                float_cols_opt = [col for col in opt_df.columns if opt_df[col].dtype == 'float64']
                st.dataframe(opt_df.style.format({col: '{:.2f}' for col in float_cols_opt}), height=300)
                csv_opt = opt_df.to_csv(index=False).encode('utf-8'); st.download_button("Download Optimization CSV", csv_opt, f"{ticker_symbol}_opt_results.csv", 'text/csv', key='dl_opt_csv_main_v2')
                st.markdown("#### Optimization Heatmap (SL vs RRR - Full Period)")
                opt_metric_hm = optimization_metric if analysis_mode == "Parameter Optimization" else settings.DEFAULT_OPTIMIZATION_METRIC
                if opt_algo == "Grid Search" and 'SL Points' in opt_df.columns and 'RRR' in opt_df.columns: # Heatmap only for Grid Search & if SL/RRR were varied
                    heatmap_fig = plotting.plot_optimization_heatmap(opt_df, 'SL Points', 'RRR', opt_metric_hm)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                else: st.info("Heatmap for SL vs RRR is generated for Grid Search when these are optimized. For other parameters or Random Search, review the table.")
            else: st.info("No full-period optimization results. Run 'Parameter Optimization' mode.")

    # Tab 3: Walk-Forward Analysis
    wfo_tab_idx = main_tabs_list.index("🚶 Walk-Forward Analysis") if "🚶 Walk-Forward Analysis" in main_tabs_list else -1
    if wfo_tab_idx != -1 and len(active_tabs) > wfo_tab_idx:
        with active_tabs[wfo_tab_idx]:
            if st.session_state.wfo_results:
                wfo_res = st.session_state.wfo_results
                st.markdown("#### Walk-Forward Optimization Log"); st.dataframe(wfo_res["log"].style.format({col: '{:.2f}' for col in wfo_res["log"].select_dtypes(include='float').columns if col in wfo_res["log"]}), height=300) # Check col exists
                csv_wfo_log = wfo_res["log"].to_csv(index=False).encode('utf-8'); st.download_button("Download WFO Log CSV", csv_wfo_log, f"{ticker_symbol}_wfo_log.csv", 'text/csv', key='dl_wfo_log_main_v2')
                st.markdown("#### Aggregated Out-of-Sample Trades")
                if not wfo_res["oos_trades"].empty:
                    st.dataframe(wfo_res["oos_trades"].style.format({col: '{:.2f}' for col in wfo_res["oos_trades"].select_dtypes(include='float').columns}), height=300)
                    csv_wfo_trades = wfo_res["oos_trades"].to_csv(index=False).encode('utf-8'); st.download_button("Download WFO OOS Trades CSV", csv_wfo_trades, f"{ticker_symbol}_wfo_oos_trades.csv", 'text/csv', key='dl_wfo_trades_main_v2')
                else: st.info("No out-of-sample trades generated during WFO.")
            else: st.info("No WFO results. Run 'Walk-Forward Optimization' mode.")

elif st.sidebar.button("Clear Results", use_container_width=True, key="clear_button_main_page_tf_v2"):
    init_session_state(); st.info("Results cleared."); st.experimental_rerun()
else:
    if not any([st.session_state.backtest_results, not st.session_state.optimization_results_df.empty, st.session_state.wfo_results]):
        st.info("Configure parameters and click 'Run Analysis'.")

st.sidebar.markdown("---")
st.sidebar.info(f"App Version: 0.3.1 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.sidebar.caption("Disclaimer: Financial modeling tool. Past performance and optimization results are not indicative of future results and can be overfit.")

