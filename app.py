# app.py
"""
Main Streamlit application file for the Gap Guardian Strategy Backtester.
Handles UI, user inputs, and orchestrates the backtesting process,
including optimization of SL, RRR, and Entry Window Times, and Walk-Forward Optimization (WFO).
The selected data timeframe (current_interval) is now passed to all relevant backend functions.
Corrected parameter key handling for optimizer.
Improved state reset and tab handling for robustness on re-runs.
Refined WFO progress bar and overall robustness.
Corrected prog_bar_container handling.
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

def initialize_app_session_state():
    defaults = {
        'backtest_results': None, 'optimization_results_df': pd.DataFrame(),
        'price_data': pd.DataFrame(), 'signals': pd.DataFrame(),
        'best_params_from_opt': None, 'wfo_results': None,
        'selected_timeframe_value': settings.DEFAULT_STRATEGY_TIMEFRAME,
        'run_analysis_clicked_count': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
initialize_app_session_state()

# --- Sidebar Inputs ---
st.sidebar.header("Backtest Configuration")
selected_ticker_name = st.sidebar.selectbox("Select Symbol:", options=list(settings.DEFAULT_TICKERS.keys()), index=0, key="ticker_sel_v8") # Incr
ticker_symbol = settings.DEFAULT_TICKERS[selected_ticker_name]

current_tf_value_in_state = st.session_state.selected_timeframe_value
default_tf_display_index = 0
if current_tf_value_in_state in settings.AVAILABLE_TIMEFRAMES.values():
    default_tf_display_index = list(settings.AVAILABLE_TIMEFRAMES.values()).index(current_tf_value_in_state)
selected_timeframe_display = st.sidebar.selectbox("Select Timeframe:", options=list(settings.AVAILABLE_TIMEFRAMES.keys()), index=default_tf_display_index, key="timeframe_selector_ui_main_v8") # Incr
st.session_state.selected_timeframe_value = settings.AVAILABLE_TIMEFRAMES[selected_timeframe_display]
ui_current_interval = st.session_state.selected_timeframe_value

today = date.today()
max_history_limit_days = None
if ui_current_interval in settings.YFINANCE_SHORT_INTRADAY_INTERVALS: max_history_limit_days = settings.MAX_SHORT_INTRADAY_DAYS
elif ui_current_interval in settings.YFINANCE_HOURLY_INTERVALS: max_history_limit_days = settings.MAX_HOURLY_INTRADAY_DAYS
min_allowable_start_date_for_ui = (today - timedelta(days=max_history_limit_days -1)) if max_history_limit_days else (today - timedelta(days=365 * 10))
date_input_help_suffix = f"Data for {ui_current_interval} is limited to ~{max_history_limit_days} days." if max_history_limit_days else "Select historical period."
default_start_date_value = (today - timedelta(days=min(15, max_history_limit_days -1 if max_history_limit_days else 15))) if ui_current_interval in ["1m","5m","15m","30m","1h","60m","90m"] else (today - timedelta(days=30*7 if ui_current_interval=="1wk" else 30))
if default_start_date_value < min_allowable_start_date_for_ui: default_start_date_value = min_allowable_start_date_for_ui
max_possible_start_date = today - timedelta(days=1)
if default_start_date_value > max_possible_start_date: default_start_date_value = max_possible_start_date
if default_start_date_value < min_allowable_start_date_for_ui: default_start_date_value = min_allowable_start_date_for_ui

start_date_ui = st.sidebar.date_input("Start Date:", value=default_start_date_value, min_value=min_allowable_start_date_for_ui, max_value=max_possible_start_date, key=f"start_date_widget_{ui_current_interval}_v8", help=f"Start date. {date_input_help_suffix}") # Incr
min_end_date_value_ui = start_date_ui + timedelta(days=1) if start_date_ui else min_allowable_start_date_for_ui + timedelta(days=1)
default_end_date_value_ui = today
if default_end_date_value_ui < min_end_date_value_ui: default_end_date_value_ui = min_end_date_value_ui
if default_end_date_value_ui > today: default_end_date_value_ui = today
end_date_ui = st.sidebar.date_input("End Date:", value=default_end_date_value_ui, min_value=min_end_date_value_ui, max_value=today, key=f"end_date_widget_{ui_current_interval}_v8", help=f"End date. {date_input_help_suffix}") # Incr

initial_capital_ui = st.sidebar.number_input("Initial Capital ($):", 1000.0, value=settings.DEFAULT_INITIAL_CAPITAL, step=1000.0, format="%.2f")
risk_per_trade_percent_ui = st.sidebar.number_input("Risk per Trade (%):", 0.1, 10.0, value=settings.DEFAULT_RISK_PER_TRADE_PERCENT, step=0.1, format="%.1f")

st.sidebar.subheader("Strategy Parameters (Manual / Optimization Base)")
sl_points_single_ui = st.sidebar.number_input("SL (points):", 0.1, value=settings.DEFAULT_STOP_LOSS_POINTS, step=0.1, format="%.2f", key="sl_s_man_v8") # Incr
rrr_single_ui = st.sidebar.number_input("RRR:", 0.1, value=settings.DEFAULT_RRR, step=0.1, format="%.1f", key="rrr_s_man_v8") # Incr
st.sidebar.markdown("**Entry Window (NY Time - Manual Run):**")
c1,c2=st.sidebar.columns(2); entry_start_hour_single_ui = c1.number_input("Start Hr",0,23,settings.DEFAULT_ENTRY_WINDOW_START_HOUR,1,key="esh_s_man_v8") # Incr
entry_start_minute_single_ui = c2.number_input("Start Min",0,59,settings.DEFAULT_ENTRY_WINDOW_START_MINUTE,15,key="esm_s_man_v8") # Incr
c1,c2=st.sidebar.columns(2); entry_end_hour_single_ui = c1.number_input("End Hr",0,23,settings.DEFAULT_ENTRY_WINDOW_END_HOUR,1,key="eeh_s_man_v8") # Incr
entry_end_minute_single_ui = c2.number_input("End Min",0,59,settings.DEFAULT_ENTRY_WINDOW_END_MINUTE,15,key="eem_s_man_v8", help="Usually 00.") # Incr

analysis_mode_ui = st.sidebar.radio("Analysis Type:", ("Single Backtest", "Parameter Optimization", "Walk-Forward Optimization"), 0, key="analysis_mode_v8") # Incr

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
    opt_algo_ui = st.sidebar.selectbox("Algorithm:", settings.OPTIMIZATION_ALGORITHMS, settings.OPTIMIZATION_ALGORITHMS.index(opt_algo_ui), key="opt_algo_v8") # Incr
    opt_metric_ui = st.sidebar.selectbox("Optimize Metric:", settings.OPTIMIZATION_METRICS, settings.OPTIMIZATION_METRICS.index(opt_metric_ui), key="opt_metric_v8") # Incr
    st.sidebar.markdown("**SL Range:**"); c1,c2,c3=st.sidebar.columns(3); sl_min_opt_ui=c1.number_input("Min",value=sl_min_opt_ui,step=0.1,format="%.1f",key="slmin_o8") # Incr
    sl_max_opt_ui=c2.number_input("Max",value=sl_max_opt_ui,step=0.1,format="%.1f",key="slmax_o8") # Incr
    if opt_algo_ui=="Grid Search": sl_steps_opt_ui=c3.number_input("Steps",2,10,int(sl_steps_opt_ui),1,key="slsteps_o8") # Incr
    st.sidebar.markdown("**RRR Range:**"); c1,c2,c3=st.sidebar.columns(3); rrr_min_opt_ui=c1.number_input("Min",value=rrr_min_opt_ui,step=0.1,format="%.1f",key="rrrmin_o8") # Incr
    rrr_max_opt_ui=c2.number_input("Max",value=rrr_max_opt_ui,step=0.1,format="%.1f",key="rrrmax_o8") # Incr
    if opt_algo_ui=="Grid Search": rrr_steps_opt_ui=c3.number_input("Steps",2,10,int(rrr_steps_opt_ui),1,key="rrrsteps_o8") # Incr
    st.sidebar.markdown("**Entry Start Hr Range:**"); c1,c2,c3=st.sidebar.columns(3); esh_min_opt_ui=c1.number_input("Min Hr",value=esh_min_opt_ui,min_value=0,max_value=23,step=1,key="eshmin_o8") # Incr
    esh_max_opt_ui=c2.number_input("Max Hr",value=esh_max_opt_ui,min_value=0,max_value=23,step=1,key="eshmax_o8") # Incr
    if opt_algo_ui=="Grid Search": esh_steps_opt_ui=c3.number_input("Hr Steps",2,5,int(esh_steps_opt_ui),1,key="eshsteps_o8") # Incr
    esm_vals_opt_ui=st.sidebar.multiselect("Entry Start Min(s):", [0,15,30,45,50], default=esm_vals_opt_ui, key="esmvals_o8") # Incr
    if not esm_vals_opt_ui: esm_vals_opt_ui = [settings.DEFAULT_ENTRY_WINDOW_START_MINUTE]
    st.sidebar.markdown("**Entry End Hr Range:**"); c1,c2,c3=st.sidebar.columns(3); eeh_min_opt_ui=c1.number_input("Min Hr",value=eeh_min_opt_ui,min_value=0,max_value=23,step=1,key="eehmin_o8") # Incr
    eeh_max_opt_ui=c2.number_input("Max Hr",value=eeh_max_opt_ui,min_value=0,max_value=23,step=1,key="eehmax_o8") # Incr
    if opt_algo_ui=="Grid Search": eeh_steps_opt_ui=c3.number_input("Hr Steps",2,5,int(eeh_steps_opt_ui),1,key="eehsteps_o8") # Incr
    if opt_algo_ui=="Random Search": rand_iters_ui=st.sidebar.number_input("Random Iterations:",10,500,rand_iters_ui,10,key="randiter_o8") # Incr
    if opt_algo_ui=="Grid Search": st.sidebar.caption(f"Grid Combs: {int(sl_steps_opt_ui*rrr_steps_opt_ui*esh_steps_opt_ui*len(esm_vals_opt_ui)*eeh_steps_opt_ui)}")
    else: st.sidebar.caption(f"Random Iterations: {rand_iters_ui}")

wfo_isd_ui,wfo_oosd_ui,wfo_sd_ui = settings.DEFAULT_WFO_IN_SAMPLE_DAYS, settings.DEFAULT_WFO_OUT_OF_SAMPLE_DAYS, settings.DEFAULT_WFO_STEP_DAYS
if analysis_mode_ui == "Walk-Forward Optimization":
    st.sidebar.markdown("##### WFO Settings (Days)"); wfo_isd_ui=st.sidebar.number_input("In-Sample:",30,value=wfo_isd_ui,step=10,key="wfoisd_v8") # Incr
    wfo_oosd_ui=st.sidebar.number_input("Out-of-Sample:",10,value=wfo_oosd_ui,step=5,key="wfoosd_v8") # Incr
    wfo_sd_ui=st.sidebar.number_input("Step:",min_value=wfo_oosd_ui,value=wfo_sd_ui,step=5,key="wfosd_v8") # Incr

st.title(f"🛡️ {settings.APP_TITLE}")
st.markdown(f"Strategy: Gap Guardian | TF: **{selected_timeframe_display}** ({st.session_state.selected_timeframe_value}) | Default Entry: {settings.DEFAULT_ENTRY_WINDOW_START_HOUR:02d}:{settings.DEFAULT_ENTRY_WINDOW_START_MINUTE:02d}-{settings.DEFAULT_ENTRY_WINDOW_END_HOUR:02d}:{settings.DEFAULT_ENTRY_WINDOW_END_MINUTE:02d} NYT")

if st.sidebar.button("Run Analysis", type="primary", use_container_width=True, key="run_main_v8"): # Incr
    st.session_state.run_analysis_clicked_count += 1
    logger.info(f"Run Analysis button clicked (Count: {st.session_state.run_analysis_clicked_count}). Mode: {analysis_mode_ui}, TF: {st.session_state.selected_timeframe_value}")
    st.session_state.backtest_results = None; st.session_state.optimization_results_df = pd.DataFrame(); st.session_state.wfo_results = None
    st.session_state.price_data = pd.DataFrame(); st.session_state.signals = pd.DataFrame(); st.session_state.best_params_from_opt = None
    
    interval_for_this_run = st.session_state.selected_timeframe_value

    if start_date_ui >= end_date_ui: st.error(f"Error: Start date ({start_date_ui}) must be before end date ({end_date_ui}).")
    else:
        with st.spinner("Fetching data..."):
            price_data_df = data_loader.fetch_historical_data(ticker_symbol, start_date_ui, end_date_ui, interval_for_this_run)
            st.session_state.price_data = price_data_df
        if price_data_df.empty: st.warning(f"No price data for {selected_ticker_name} ({interval_for_this_run}). Cannot proceed.")
        else:
            manual_entry_start_t = dt_time(entry_start_hour_single_ui, entry_start_minute_single_ui)
            manual_entry_end_t = dt_time(entry_end_hour_single_ui, entry_end_minute_single_ui)
            
            prog_bar_container = None # Initialize to None

            if analysis_mode_ui == "Single Backtest":
                st.subheader("Single Backtest Run")
                with st.spinner("Running..."):
                    signals = strategy_engine.generate_signals(price_data_df.copy(), sl_points_single_ui, rrr_single_ui, manual_entry_start_t, manual_entry_end_t)
                    st.session_state.signals = signals
                    trades, equity, perf = backtester.run_backtest(price_data_df.copy(), signals, initial_capital_ui, risk_per_trade_percent_ui, sl_points_single_ui, interval_for_this_run)
                    st.session_state.backtest_results = {"trades":trades,"equity_curve":equity,"performance":perf,"params":{"SL": sl_points_single_ui,"RRR": rrr_single_ui,"TF":interval_for_this_run,"Entry":f"{manual_entry_start_t:%H:%M}-{manual_entry_end_t:%H:%M}","src":"Manual"}}
                    st.success("Single backtest complete!")
            
            elif analysis_mode_ui in ["Parameter Optimization", "Walk-Forward Optimization"]:
                prog_bar_container = st.empty() # Assign st.empty() object here
                prog_bar_container.progress(0, text="Initializing optimization...")
                def opt_cb(p,s): prog_bar_container.progress(p, text=f"{s}: {int(p*100)}% complete")
                
                actual_params_to_optimize_config = {
                    'sl_points': np.linspace(sl_min_opt_ui, sl_max_opt_ui, int(sl_steps_opt_ui)) if opt_algo_ui == "Grid Search" else (sl_min_opt_ui, sl_max_opt_ui),
                    'rrr': np.linspace(rrr_min_opt_ui, rrr_max_opt_ui, int(rrr_steps_opt_ui)) if opt_algo_ui == "Grid Search" else (rrr_min_opt_ui, rrr_max_opt_ui),
                    'entry_start_hour': [int(h) for h in np.linspace(esh_min_opt_ui, esh_max_opt_ui, int(esh_steps_opt_ui))] if opt_algo_ui == "Grid Search" else (esh_min_opt_ui, esh_max_opt_ui),
                    'entry_start_minute': esm_vals_opt_ui,
                    'entry_end_hour': [int(h) for h in np.linspace(eeh_min_opt_ui, eeh_max_opt_ui, int(eeh_steps_opt_ui))] if opt_algo_ui == "Grid Search" else (eeh_min_opt_ui, eeh_max_opt_ui),
                    'entry_end_minute': settings.DEFAULT_ENTRY_END_MINUTE_OPTIMIZATION_VALUES
                }
                optimizer_control_params = {'metric_to_optimize': opt_metric_ui}
                if opt_algo_ui == "Random Search": optimizer_control_params['iterations'] = rand_iters_ui

                if analysis_mode_ui == "Parameter Optimization":
                    st.subheader(f"Parameter Optimization ({opt_algo_ui} - Full Period)")
                    with st.spinner(f"Running {opt_algo_ui}..."):
                        if opt_algo_ui == "Grid Search": opt_df = optimizer.run_grid_search(price_data_df, initial_capital_ui, risk_per_trade_percent_ui, actual_params_to_optimize_config, interval_for_this_run, lambda p,s: opt_cb(p,s))
                        else: opt_df = optimizer.run_random_search(price_data_df, initial_capital_ui, risk_per_trade_percent_ui, actual_params_to_optimize_config, rand_iters_ui, interval_for_this_run, lambda p,s: opt_cb(p,s))
                        st.session_state.optimization_results_df = opt_df
                        # prog_bar_container.progress(1.0, text="Optimization Complete!") # Moved after results processing
                        if not opt_df.empty:
                            st.success("Full period optimization finished!")
                            valid_opt = opt_df.dropna(subset=[opt_metric_ui])
                            if not valid_opt.empty:
                                best_r = valid_opt.loc[valid_opt[opt_metric_ui].idxmin()] if opt_metric_ui=="Max Drawdown (%)" else valid_opt.loc[valid_opt[opt_metric_ui].idxmax()]
                                st.session_state.best_params_from_opt = {"SL":best_r["SL Points"],"RRR":best_r["RRR"],"ES_H":best_r["EntryStartHour"],"ES_M":best_r["EntryStartMinute"],"EE_H":best_r["EntryEndHour"],"EE_M":best_r.get("EntryEndMinute",0),"Metric":best_r[opt_metric_ui]}
                                best_es_t = dt_time(int(best_r["EntryStartHour"]),int(best_r["EntryStartMinute"]))
                                best_ee_t = dt_time(int(best_r["EntryEndHour"]),int(best_r.get("EntryEndMinute",0)))
                                st.info(f"Best for '{opt_metric_ui}': SL={best_r['SL Points']:.2f}, RRR={best_r['RRR']:.1f}, Entry={best_es_t:%H:%M}-{best_ee_t:%H:%M} (Val: {best_r[opt_metric_ui]:.2f})")
                                with st.spinner("Running backtest with best parameters..."):
                                    signals_b = strategy_engine.generate_signals(price_data_df.copy(),best_r["SL Points"],best_r["RRR"],best_es_t,best_ee_t)
                                    st.session_state.signals = signals_b
                                    trades_b,equity_b,perf_b = backtester.run_backtest(price_data_df.copy(),signals_b,initial_capital_ui,risk_per_trade_percent_ui,best_r["SL Points"],interval_for_this_run)
                                    st.session_state.backtest_results = {"trades":trades_b,"equity_curve":equity_b,"performance":perf_b,"params":{"SL":best_r["SL Points"],"RRR":best_r["RRR"],"TF":interval_for_this_run,"Entry":f"{best_es_t:%H:%M}-{best_ee_t:%H:%M}","src":f"Opt ({opt_algo_ui})"}}
                            else: st.warning(f"No valid results for '{opt_metric_ui}' in optimization.")
                        else: st.error("Optimization yielded no results.")
                        prog_bar_container.progress(1.0, text="Optimization Complete!") # Now after processing
                
                elif analysis_mode_ui == "Walk-Forward Optimization":
                    st.subheader(f"Walk-Forward Optimization Run ({opt_algo_ui})")
                    wfo_p = {'in_sample_days':wfo_isd_ui,'out_of_sample_days':wfo_oosd_ui,'step_days':wfo_sd_ui}
                    wfo_optimizer_config = {**optimizer_control_params, **actual_params_to_optimize_config}
                    with st.spinner(f"Running WFO with {opt_algo_ui}... This will take considerable time."):
                        wfo_log,oos_trades,oos_equity,oos_perf = optimizer.run_walk_forward_optimization(
                            price_data_df,initial_capital_ui,risk_per_trade_percent_ui,
                            wfo_p, opt_algo_ui, wfo_optimizer_config,
                            interval_for_this_run,lambda p,s: opt_cb(p,s)
                        )
                        st.session_state.wfo_results = {"log":wfo_log,"oos_trades":oos_trades,"oos_equity_curve":oos_equity,"oos_performance":oos_perf}
                        # prog_bar_container.progress(1.0, text="WFO Complete!") # Moved after results processing
                        st.success("Walk-Forward Optimization finished!")
                        st.session_state.backtest_results = {"trades":oos_trades,"equity_curve":oos_equity,"performance":oos_perf,"params":{"TF":interval_for_this_run,"src":"WFO Aggregated"}}
                        prog_bar_container.progress(1.0, text="WFO Complete!") # Now after processing
            
            if prog_bar_container is not None: # Check if it was created before trying to empty
                prog_bar_container.empty() 

# --- Display Area ---
# ... (Display logic from previous version)
main_tabs_to_display_names = []
if st.session_state.backtest_results: main_tabs_to_display_names.append("📊 Backtest Performance")
if not st.session_state.optimization_results_df.empty and analysis_mode_ui == "Parameter Optimization": main_tabs_to_display_names.append("⚙️ Optimization Results (Full Period)")
if st.session_state.wfo_results and analysis_mode_ui == "Walk-Forward Optimization": main_tabs_to_display_names.append("🚶 Walk-Forward Analysis")

if main_tabs_to_display_names:
    tabs_key_string = "_".join(main_tabs_to_display_names) + f"_{st.session_state.run_analysis_clicked_count}"
    created_tabs = st.tabs(main_tabs_to_display_names) # Removed key for simplicity, relying on list change
    tab_map = dict(zip(main_tabs_to_display_names, created_tabs))

    if "📊 Backtest Performance" in tab_map:
        with tab_map["📊 Backtest Performance"]:
            if st.session_state.backtest_results:
                results = st.session_state.backtest_results; performance = results["performance"]; trades = results["trades"]; equity_curve = results["equity_curve"]
                run_params = results.get("params", {}); run_source = run_params.get("src", "N/A"); tf_disp = run_params.get("TF", st.session_state.selected_timeframe_value)
                param_info = f" (Source: {run_source} | TF: {tf_disp}"
                sl_val = run_params.get("SL", run_params.get("SL Points")) 
                rrr_val = run_params.get("RRR")
                entry_val = run_params.get("Entry")
                if sl_val is not None: param_info += f" | SL: {float(sl_val):.2f}"
                if rrr_val is not None: param_info += f" | RRR: {float(rrr_val):.1f}"
                if entry_val is not None: param_info += f" | Entry: {entry_val}"
                param_info += ")"
                st.markdown(f"#### Performance Summary{param_info}")
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
                col1,col2,col3=st.columns(3)
                with col1: display_styled_metric(col1,"Total P&L",performance.get('Total P&L'),performance.get('Total P&L')); display_styled_metric(col1,"Final Capital",performance.get('Final Capital',initial_capital_ui),performance.get('Final Capital',initial_capital_ui),c=True); display_styled_metric(col1,"Max Drawdown",performance.get('Max Drawdown (%)'),performance.get('Max Drawdown (%)'),c=False,pct=True,mdd_logic=True)
                with col2: display_styled_metric(col2,"Total Trades",int(performance.get('Total Trades',0)),int(performance.get('Total Trades',0)),c=False,pct=False); display_styled_metric(col2,"Win Rate",performance.get('Win Rate',0),performance.get('Win Rate',0),c=False,pct=True); display_styled_metric(col2,"Profit Factor",performance.get('Profit Factor',0),performance.get('Profit Factor',0),c=False,p=2,pf_logic=True)
                with col3: display_styled_metric(col3,"Avg. Trade P&L",performance.get('Average Trade P&L'),performance.get('Average Trade P&L')); display_styled_metric(col3,"Avg. Winning Trade",performance.get('Average Winning Trade'),performance.get('Average Winning Trade')); display_styled_metric(col3,"Avg. Losing Trade",performance.get('Average Losing Trade'),performance.get('Average Losing Trade'))
                detail_tabs_list = ["📈 Equity Curve","📊 Trades on Price","📋 Trade Log"]
                if not st.session_state.signals.empty and analysis_mode_ui != "Walk-Forward Optimization": detail_tabs_list.append("🔍 Generated Signals (Last Run)")
                detail_tabs_list.append("💾 Raw Price Data (Full Period)")
                detail_tabs = st.tabs(detail_tabs_list)
                with detail_tabs[0]:
                    plot_title = "Equity Curve" if analysis_mode_ui != "Walk-Forward Optimization" else "WFO: Aggregated Out-of-Sample Equity"
                    plot_func = plotting.plot_equity_curve if analysis_mode_ui != "Walk-Forward Optimization" else plotting.plot_wfo_equity_curve
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
                        csv_data = st.session_state.price_data.to_csv(index=True).encode('utf-8'); st.download_button("Download Full Price Data CSV", csv_data, f"{ticker_symbol}_price_data.csv", 'text/csv', key='dl_raw_price_main_v8') # Incr
                    else: st.info("Raw price data is not available.")
            else: st.info("Run an analysis to see performance details.")

    opt_tab_idx = main_tabs_to_display_names.index("⚙️ Optimization Results (Full Period)") if "⚙️ Optimization Results (Full Period)" in main_tabs_to_display_names else -1
    if opt_tab_idx != -1:
        with tab_map["⚙️ Optimization Results (Full Period)"]:
            opt_df_display = st.session_state.optimization_results_df
            if not opt_df_display.empty:
                st.markdown("#### Grid/Random Search Results (Full Period)")
                float_cols_opt_disp = [col for col in opt_df_display.columns if opt_df_display[col].dtype == 'float64']
                st.dataframe(opt_df_display.style.format({col: '{:.2f}' for col in float_cols_opt_disp}), height=300)
                csv_opt_disp = opt_df_display.to_csv(index=False).encode('utf-8'); st.download_button("Download Optimization CSV", csv_opt_disp, f"{ticker_symbol}_opt_results.csv", 'text/csv', key='dl_opt_csv_main_v8') # Incr
                st.markdown("#### Optimization Heatmap (SL vs RRR - Full Period)")
                opt_metric_hm_disp = opt_metric_ui if analysis_mode_ui == "Parameter Optimization" else settings.DEFAULT_OPTIMIZATION_METRIC
                if opt_algo_ui == "Grid Search" and 'SL Points' in opt_df_display.columns and 'RRR' in opt_df_display.columns:
                    heatmap_fig_disp = plotting.plot_optimization_heatmap(opt_df_display, 'SL Points', 'RRR', opt_metric_hm_disp)
                    st.plotly_chart(heatmap_fig_disp, use_container_width=True)
                else: st.info("Heatmap for SL vs RRR is generated for Grid Search. For other parameters or Random Search, review the table.")
            else: st.info("No full-period optimization results. Run 'Parameter Optimization' mode.")

    wfo_tab_idx = main_tabs_to_display_names.index("🚶 Walk-Forward Analysis") if "🚶 Walk-Forward Analysis" in main_tabs_to_display_names else -1
    if wfo_tab_idx != -1:
        with tab_map["🚶 Walk-Forward Analysis"]:
            if st.session_state.wfo_results:
                wfo_res_disp = st.session_state.wfo_results
                st.markdown("#### Walk-Forward Optimization Log"); st.dataframe(wfo_res_disp["log"].style.format({col: '{:.2f}' for col in wfo_res_disp["log"].select_dtypes(include='float').columns if col in wfo_res_disp["log"]}), height=300)
                csv_wfo_log_disp = wfo_res_disp["log"].to_csv(index=False).encode('utf-8'); st.download_button("Download WFO Log CSV", csv_wfo_log_disp, f"{ticker_symbol}_wfo_log.csv", 'text/csv', key='dl_wfo_log_main_v8') # Incr
                st.markdown("#### Aggregated Out-of-Sample Trades")
                if not wfo_res_disp["oos_trades"].empty:
                    st.dataframe(wfo_res_disp["oos_trades"].style.format({col: '{:.2f}' for col in wfo_res_disp["oos_trades"].select_dtypes(include='float').columns}), height=300)
                    csv_wfo_trades_disp = wfo_res_disp["oos_trades"].to_csv(index=False).encode('utf-8'); st.download_button("Download WFO OOS Trades CSV", csv_wfo_trades_disp, f"{ticker_symbol}_wfo_oos_trades.csv", 'text/csv', key='dl_wfo_trades_main_v8') # Incr
                else: st.info("No out-of-sample trades generated during WFO.")
            else: st.info("No WFO results. Run 'Walk-Forward Optimization' mode.")

elif st.session_state.run_analysis_clicked_count > 0 :
    st.info("Analysis was run. If results are not displayed, it might be due to no trades or data for the selected parameters. Check logs if errors are suspected.")
else:
    if not any([st.session_state.backtest_results, not st.session_state.optimization_results_df.empty, st.session_state.wfo_results]):
        st.info("Configure parameters in the sidebar and click 'Run Analysis'.")

st.sidebar.markdown("---")
st.sidebar.info(f"App Version: 0.4.2 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.sidebar.caption("Disclaimer: Financial modeling tool. Past performance and optimization results are not indicative of future results and can be overfit.")

