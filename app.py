# app.py
"""
Main Streamlit application file for the Gap Guardian Strategy Backtester.
Handles UI, user inputs, and orchestrates the backtesting process,
including optimization and Walk-Forward Optimization (WFO).
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime

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

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'backtest_results': None, 'optimization_results_df': pd.DataFrame(),
        'price_data': pd.DataFrame(), 'signals': pd.DataFrame(),
        'best_params_from_opt': None, 'wfo_results': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
init_session_state()

# --- Sidebar Inputs ---
st.sidebar.header("Backtest Configuration")
selected_ticker_name = st.sidebar.selectbox("Select Symbol:", options=list(settings.DEFAULT_TICKERS.keys()), index=0)
ticker_symbol = settings.DEFAULT_TICKERS[selected_ticker_name]

today = date.today()
# ... (Date input logic from previous version, ensure it's robust) ...
is_intraday_strategy = settings.STRATEGY_TIME_FRAME not in ["1d", "1wk", "1mo"]
if is_intraday_strategy:
    min_allowable_start_date_for_ui = today - timedelta(days=settings.MAX_INTRADAY_DAYS - 1)
    default_start_date_value = min_allowable_start_date_for_ui
    date_input_help_suffix = f"Intraday data is limited to the last ~{settings.MAX_INTRADAY_DAYS} days."
    if default_start_date_value >= (today - timedelta(days=1)): default_start_date_value = today - timedelta(days=max(1, settings.MAX_INTRADAY_DAYS - (today - min_allowable_start_date_for_ui).days)) if (today - timedelta(days=1)) > min_allowable_start_date_for_ui else min_allowable_start_date_for_ui
else:
    min_allowable_start_date_for_ui = today - timedelta(days=365 * 5); default_start_date_value = today - timedelta(days=30); date_input_help_suffix = "Select historical period."
if default_start_date_value < min_allowable_start_date_for_ui: default_start_date_value = min_allowable_start_date_for_ui
if default_start_date_value >= (today - timedelta(days=1)): default_start_date_value = (today - timedelta(days=1)) if (today - timedelta(days=1)) > min_allowable_start_date_for_ui else min_allowable_start_date_for_ui

start_date = st.sidebar.date_input("Start Date:", value=default_start_date_value, min_value=min_allowable_start_date_for_ui, max_value=today - timedelta(days=1), help=f"Start date. {date_input_help_suffix}")
min_end_date_value = start_date + timedelta(days=1) if start_date else min_allowable_start_date_for_ui + timedelta(days=1)
default_end_date_value = today
if default_end_date_value < min_end_date_value: default_end_date_value = min_end_date_value
if default_end_date_value > today: default_end_date_value = today
end_date = st.sidebar.date_input("End Date:", value=default_end_date_value, min_value=min_end_date_value, max_value=today, help=f"End date. {date_input_help_suffix}")

initial_capital = st.sidebar.number_input("Initial Capital ($):", min_value=1000.0, value=settings.DEFAULT_INITIAL_CAPITAL, step=1000.0, format="%.2f")
risk_per_trade_percent = st.sidebar.number_input("Risk per Trade (%):", min_value=0.1, max_value=10.0, value=settings.DEFAULT_RISK_PER_TRADE_PERCENT, step=0.1, format="%.1f")

st.sidebar.subheader("Strategy Parameters (Manual / Optimization Fallback)")
sl_points_single = st.sidebar.number_input("Stop Loss (points):", min_value=0.1, value=settings.DEFAULT_STOP_LOSS_POINTS, step=0.1, format="%.2f", key="sl_single")
rrr_single = st.sidebar.number_input("Risk/Reward Ratio:", min_value=0.1, value=settings.DEFAULT_RRR, step=0.1, format="%.1f", key="rrr_single")

# --- Analysis Mode Selection ---
st.sidebar.subheader("Analysis Mode")
analysis_mode = st.sidebar.radio("Select Analysis Type:", ("Single Backtest", "Parameter Optimization", "Walk-Forward Optimization"), index=0, key="analysis_mode_radio")

# --- Optimization Settings (Conditional) ---
opt_algo = settings.DEFAULT_OPTIMIZATION_ALGORITHM
sl_min, sl_max, sl_steps = 0.0,0.0,0 # Initialize
rrr_min, rrr_max, rrr_steps = 0.0,0.0,0
random_iterations = settings.DEFAULT_RANDOM_SEARCH_ITERATIONS
optimization_metric = settings.DEFAULT_OPTIMIZATION_METRIC

if analysis_mode == "Parameter Optimization" or analysis_mode == "Walk-Forward Optimization":
    st.sidebar.markdown("##### In-Sample Optimization Settings")
    opt_algo = st.sidebar.selectbox("Optimization Algorithm:", settings.OPTIMIZATION_ALGORITHMS, index=settings.OPTIMIZATION_ALGORITHMS.index(settings.DEFAULT_OPTIMIZATION_ALGORITHM))
    
    opt_sl_range_cfg = settings.DEFAULT_SL_POINTS_OPTIMIZATION_RANGE
    opt_rrr_range_cfg = settings.DEFAULT_RRR_OPTIMIZATION_RANGE

    col_sl1, col_sl2, col_sl3 = st.sidebar.columns(3)
    sl_min = col_sl1.number_input("SL Min", value=opt_sl_range_cfg["min"], step=0.1, format="%.1f", key="sl_min_opt")
    sl_max = col_sl2.number_input("SL Max", value=opt_sl_range_cfg["max"], step=0.1, format="%.1f", key="sl_max_opt")
    if opt_algo == "Grid Search":
        sl_steps = col_sl3.number_input("SL Steps", min_value=2, max_value=20, value=opt_sl_range_cfg["steps"], step=1, key="sl_steps_opt")
    
    col_rrr1, col_rrr2, col_rrr3 = st.sidebar.columns(3)
    rrr_min = col_rrr1.number_input("RRR Min", value=opt_rrr_range_cfg["min"], step=0.1, format="%.1f", key="rrr_min_opt")
    rrr_max = col_rrr2.number_input("RRR Max", value=opt_rrr_range_cfg["max"], step=0.1, format="%.1f", key="rrr_max_opt")
    if opt_algo == "Grid Search":
        rrr_steps = col_rrr3.number_input("RRR Steps", min_value=2, max_value=20, value=opt_rrr_range_cfg["steps"], step=1, key="rrr_steps_opt")

    if opt_algo == "Random Search":
        random_iterations = st.sidebar.number_input("Random Search Iterations:", min_value=5, max_value=1000, value=settings.DEFAULT_RANDOM_SEARCH_ITERATIONS, step=5)
        sl_steps = rrr_steps = 0 # Not used for random search display logic for combinations

    optimization_metric = st.sidebar.selectbox("Optimize for Metric:", options=settings.OPTIMIZATION_METRICS, index=settings.OPTIMIZATION_METRICS.index(settings.DEFAULT_OPTIMIZATION_METRIC))
    
    if opt_algo == "Grid Search":
        st.sidebar.caption(f"Grid Search Combinations: {int(sl_steps * rrr_steps)}")
    elif opt_algo == "Random Search":
        st.sidebar.caption(f"Random Search Iterations: {random_iterations}")


# --- WFO Settings (Conditional) ---
wfo_in_sample_days, wfo_oos_days, wfo_step_days = 0,0,0 # Initialize
if analysis_mode == "Walk-Forward Optimization":
    st.sidebar.markdown("##### Walk-Forward Settings (Calendar Days)")
    wfo_in_sample_days = st.sidebar.number_input("In-Sample Period (Days):", min_value=30, max_value=365*2, value=settings.DEFAULT_WFO_IN_SAMPLE_DAYS, step=10)
    wfo_oos_days = st.sidebar.number_input("Out-of-Sample Period (Days):", min_value=10, max_value=180, value=settings.DEFAULT_WFO_OUT_OF_SAMPLE_DAYS, step=5)
    wfo_step_days = st.sidebar.number_input("Step / Re-optimize Every (Days):", min_value=wfo_oos_days, max_value=180, value=settings.DEFAULT_WFO_STEP_DAYS, step=5)
    st.sidebar.warning("WFO can be very time-consuming.")


# --- Main Application Area ---
st.title(f"🛡️ {settings.APP_TITLE}")
st.markdown(f"Strategy: Gap Guardian | Interval: {settings.STRATEGY_TIME_FRAME} | Entry: 9:30-11:00 AM NY False Break | Max 1 trade/day.")

if st.sidebar.button("Run Analysis", type="primary", use_container_width=True, key="run_button_main"):
    st.session_state.backtest_results = None; st.session_state.optimization_results_df = pd.DataFrame(); st.session_state.wfo_results = None
    st.session_state.price_data = pd.DataFrame(); st.session_state.signals = pd.DataFrame(); st.session_state.best_params_from_opt = None

    if start_date >= end_date:
        st.error(f"Error: Start date ({start_date}) must be before end date ({end_date}).")
    else:
        with st.spinner("Fetching data..."):
            price_data_df = data_loader.fetch_historical_data(ticker_symbol, start_date, end_date, settings.STRATEGY_TIME_FRAME)
            st.session_state.price_data = price_data_df

        if price_data_df.empty:
            st.warning(f"No price data for {selected_ticker_name} for the period. Cannot proceed.")
        else:
            # --- SINGLE BACKTEST ---
            if analysis_mode == "Single Backtest":
                st.subheader("Single Backtest Run")
                with st.spinner("Running single backtest..."):
                    signals_df = strategy_engine.generate_signals(price_data_df.copy(), sl_points_single, rrr_single)
                    st.session_state.signals = signals_df
                    trades_df, equity, perf = backtester.run_backtest(price_data_df.copy(), signals_df, initial_capital, risk_per_trade_percent, sl_points_single)
                    st.session_state.backtest_results = {"trades": trades_df, "equity_curve": equity, "performance": perf, "params": {"SL Points": sl_points_single, "RRR": rrr_single, "source": "Manual"}}
                    st.success("Single backtest complete!")

            # --- PARAMETER OPTIMIZATION (FULL PERIOD) ---
            elif analysis_mode == "Parameter Optimization":
                st.subheader("Parameter Optimization Run (Full Period)")
                num_combs_text = f"{int(sl_steps * rrr_steps)} combinations" if opt_algo == "Grid Search" else f"{random_iterations} iterations"
                with st.spinner(f"Running {opt_algo} ({num_combs_text})... This may take time."):
                    opt_progress_bar = st.progress(0, text="Optimization in progress...")
                    def opt_prog_cb(p, algo_name): opt_progress_bar.progress(p, text=f"{algo_name}: {int(p*100)}% complete")

                    if opt_algo == "Grid Search":
                        sl_vals = np.linspace(sl_min, sl_max, int(sl_steps))
                        rrr_vals = np.linspace(rrr_min, rrr_max, int(rrr_steps))
                        optimization_df = optimizer.run_grid_search(price_data_df, initial_capital, risk_per_trade_percent, sl_vals, rrr_vals, lambda p, an: opt_prog_cb(p, an))
                    else: # Random Search
                        optimization_df = optimizer.run_random_search(price_data_df, initial_capital, risk_per_trade_percent, (sl_min, sl_max), (rrr_min, rrr_max), random_iterations, lambda p, an: opt_prog_cb(p, an))
                    
                    st.session_state.optimization_results_df = optimization_df
                    opt_progress_bar.progress(1.0, text="Optimization complete!")

                if not optimization_df.empty:
                    st.success("Full period optimization finished!")
                    valid_opt_df = optimization_df.dropna(subset=[optimization_metric])
                    if not valid_opt_df.empty:
                        best_row = valid_opt_df.loc[valid_opt_df[optimization_metric].idxmin()] if optimization_metric == "Max Drawdown (%)" else valid_opt_df.loc[valid_opt_df[optimization_metric].idxmax()]
                        st.session_state.best_params_from_opt = {"SL Points": best_row["SL Points"], "RRR": best_row["RRR"], "MetricValue": best_row[optimization_metric]}
                        st.info(f"Best parameters on full period for '{optimization_metric}': SL={best_row['SL Points']:.2f}, RRR={best_row['RRR']:.1f} (Value: {best_row[optimization_metric]:.2f})")
                        # Run backtest with these best params
                        st.markdown("--- \n ### Backtest with Best Full-Period Optimized Parameters")
                        with st.spinner("Running backtest with best parameters..."):
                            best_sl, best_rrr = best_row["SL Points"], best_row["RRR"]
                            signals_best = strategy_engine.generate_signals(price_data_df.copy(), best_sl, best_rrr)
                            st.session_state.signals = signals_best
                            trades_best, equity_best, perf_best = backtester.run_backtest(price_data_df.copy(), signals_best, initial_capital, risk_per_trade_percent, best_sl)
                            st.session_state.backtest_results = {"trades": trades_best, "equity_curve": equity_best, "performance": perf_best, "params": {"SL Points": best_sl, "RRR": best_rrr, "source": f"Optimization ({opt_algo})"}}
                            st.success("Backtest with best full-period parameters complete!")
                    else: st.warning(f"No valid results found for metric '{optimization_metric}' in optimization.")
                else: st.error("Optimization did not yield any results.")
            
            # --- WALK-FORWARD OPTIMIZATION ---
            elif analysis_mode == "Walk-Forward Optimization":
                st.subheader("Walk-Forward Optimization Run")
                wfo_p = {'in_sample_days': wfo_in_sample_days, 'out_of_sample_days': wfo_oos_days, 'step_days': wfo_step_days}
                opt_p_config = {'metric_to_optimize': optimization_metric}
                if opt_algo == "Grid Search":
                    opt_p_config['sl_values'] = np.linspace(sl_min, sl_max, int(sl_steps))
                    opt_p_config['rrr_values'] = np.linspace(rrr_min, rrr_max, int(rrr_steps))
                else: # Random Search
                    opt_p_config['sl_range'] = (sl_min, sl_max)
                    opt_p_config['rrr_range'] = (rrr_min, rrr_max)
                    opt_p_config['iterations'] = random_iterations
                
                with st.spinner(f"Running Walk-Forward Optimization using {opt_algo}... This will take significant time."):
                    wfo_progress_bar = st.progress(0, text="WFO in progress...")
                    def wfo_prog_cb(p, stage_text): wfo_progress_bar.progress(p, text=f"{stage_text}: {int(p*100)}% complete")

                    wfo_log_df, oos_trades, oos_equity, oos_perf = optimizer.run_walk_forward_optimization(
                        price_data_df, initial_capital, risk_per_trade_percent, wfo_p, opt_algo, opt_p_config,
                        progress_callback=lambda p, st_txt: wfo_prog_cb(p,st_txt)
                    )
                    st.session_state.wfo_results = {
                        "log": wfo_log_df, "oos_trades": oos_trades,
                        "oos_equity_curve": oos_equity, "oos_performance": oos_perf
                    }
                    wfo_progress_bar.progress(1.0, text="WFO complete!")
                    st.success("Walk-Forward Optimization finished!")
                    # Set backtest_results to the aggregated OOS performance for display
                    st.session_state.backtest_results = {
                        "trades": oos_trades, "equity_curve": oos_equity, "performance": oos_perf,
                        "params": {"source": "Walk-Forward OOS Aggregated"}
                    }


# --- Display Area ---
main_tabs_list = ["📊 Backtest Performance"]
if not st.session_state.optimization_results_df.empty and analysis_mode == "Parameter Optimization": # Show opt results only if that mode was run
    main_tabs_list.append("⚙️ Optimization Results (Full Period)")
if st.session_state.wfo_results and analysis_mode == "Walk-Forward Optimization":
    main_tabs_list.append("🚶 Walk-Forward Analysis")

if st.session_state.backtest_results or not st.session_state.optimization_results_df.empty or st.session_state.wfo_results:
    # Ensure tabs are created only once after results are available
    if 'main_display_tabs' not in st.session_state or st.session_state.get('current_analysis_mode_for_tabs') != analysis_mode:
        st.session_state.main_display_tabs = st.tabs(main_tabs_list)
        st.session_state.current_analysis_mode_for_tabs = analysis_mode
    
    active_tabs = st.session_state.main_display_tabs

    # Tab 1: Backtest Performance (Single Run or Aggregated WFO OOS)
    with active_tabs[0]:
        if st.session_state.backtest_results:
            # ... (Existing styled metric display logic - ensure it uses st.session_state.backtest_results) ...
            results = st.session_state.backtest_results; performance = results["performance"]; trades = results["trades"]; equity_curve = results["equity_curve"]
            run_params = results.get("params", {}); run_source = run_params.get("source", "N/A")
            sl_disp = f"{run_params.get('SL Points', 'N/A'):.2f}" if isinstance(run_params.get('SL Points'), (int, float)) else "N/A"
            rrr_disp = f"{run_params.get('RRR', 'N/A'):.1f}" if isinstance(run_params.get('RRR'), (int, float)) else "N/A"
            param_info = f" (Source: {run_source}"
            if sl_disp != "N/A": param_info += f" | SL: {sl_disp}"
            if rrr_disp != "N/A": param_info += f" | RRR: {rrr_disp}"
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
                fmt_val = format_metric_display(val, p, c, pct)
                clr = NEUTRAL_COLOR
                if not (pd.isna(raw) or raw is None):
                    if pf_logic: # Profit Factor
                        if raw > 1: clr = POSITIVE_COLOR
                        elif raw < 1 and raw != 0 : clr = NEGATIVE_COLOR
                        elif raw == 0 and performance.get('Gross Profit', 0) == 0 and performance.get('Gross Loss', 0) == 0: clr = NEUTRAL_COLOR # No trades
                        elif raw == 0 : clr = NEGATIVE_COLOR # No profit, some loss
                    elif mdd_logic: # Max Drawdown
                        if raw < 0: clr = NEGATIVE_COLOR
                        elif raw == 0: clr = NEUTRAL_COLOR # Or POSITIVE_COLOR if 0 MDD is exceptionally good
                    else: # Standard P&L like metrics
                        if raw > 0: clr = POSITIVE_COLOR
                        elif raw < 0: clr = NEGATIVE_COLOR
                col.markdown(f"""<div class="metric-card"><div class="metric-label">{lbl}</div><div class="metric-value" style="color: {clr};">{fmt_val}</div></div>""", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3) # Performance Metrics Display
            with col1:
                display_styled_metric(col1, "Total P&L", performance.get('Total P&L'), performance.get('Total P&L'))
                display_styled_metric(col1, "Final Capital", performance.get('Final Capital', initial_capital), performance.get('Final Capital', initial_capital), c=True)
                display_styled_metric(col1, "Max Drawdown", performance.get('Max Drawdown (%)'), performance.get('Max Drawdown (%)'), c=False, pct=True, mdd_logic=True)
            with col2:
                display_styled_metric(col2, "Total Trades", int(performance.get('Total Trades', 0)), int(performance.get('Total Trades', 0)), c=False, pct=False)
                display_styled_metric(col2, "Win Rate", performance.get('Win Rate', 0), performance.get('Win Rate', 0), c=False, pct=True)
                display_styled_metric(col2, "Profit Factor", performance.get('Profit Factor', 0), performance.get('Profit Factor', 0), c=False, p=2, pf_logic=True)
            with col3:
                display_styled_metric(col3, "Avg. Trade P&L", performance.get('Average Trade P&L'), performance.get('Average Trade P&L'))
                display_styled_metric(col3, "Avg. Winning Trade", performance.get('Average Winning Trade'), performance.get('Average Winning Trade'))
                display_styled_metric(col3, "Avg. Losing Trade", performance.get('Average Losing Trade'), performance.get('Average Losing Trade'))

            # Detail Tabs for Equity, Trades, etc.
            detail_tabs_list = ["📈 Equity Curve", "📊 Trades on Price", "📋 Trade Log"]
            if not st.session_state.signals.empty and analysis_mode != "Walk-Forward Optimization": # Signals for single/opt run
                 detail_tabs_list.append("🔍 Generated Signals (Last Run)")
            detail_tabs_list.append("💾 Raw Price Data (Full Period)")
            
            detail_tabs = st.tabs(detail_tabs_list)
            with detail_tabs[0]: # Equity
                if not equity_curve.empty:
                    plot_title = "Equity Curve"
                    if analysis_mode == "Walk-Forward Optimization": plot_title = "WFO: Aggregated Out-of-Sample Equity"
                    st.plotly_chart(plotting.plot_wfo_equity_curve(equity_curve, title=plot_title) if analysis_mode == "Walk-Forward Optimization" else plotting.plot_equity_curve(equity_curve, title=plot_title), use_container_width=True)
                else: st.info("Equity curve is not available.")
            with detail_tabs[1]: # Trades on Price (shows trades on the full price data, useful for WFO to see all OOS trades)
                if not st.session_state.price_data.empty and not trades.empty : st.plotly_chart(plotting.plot_trades_on_price(st.session_state.price_data, trades, selected_ticker_name), use_container_width=True)
                else: st.info("Price/trade data not available for plotting.")
            with detail_tabs[2]: # Trade Log
                if not trades.empty:
                    st.dataframe(trades.style.format({col: '{:.2f}' for col in trades.select_dtypes(include='float').columns}), height=300, use_container_width=True)
                else: st.info("No trades were executed.")
            
            idx_offset = 0
            if "🔍 Generated Signals (Last Run)" in detail_tabs_list:
                with detail_tabs[3]:
                    if not st.session_state.signals.empty: st.dataframe(st.session_state.signals.style.format({col: '{:.2f}' for col in st.session_state.signals.select_dtypes(include='float').columns}), height=300, use_container_width=True)
                    else: st.info("No signals generated for the last single/optimization run.")
                idx_offset = 1
            with detail_tabs[3+idx_offset]: # Raw Price Data
                if not st.session_state.price_data.empty:
                    st.markdown(f"Full period OHLCV data for **{selected_ticker_name}** ({len(st.session_state.price_data)} rows).")
                    st.dataframe(st.session_state.price_data.head(), height=300, use_container_width=True)
                    csv_data = st.session_state.price_data.to_csv(index=True).encode('utf-8')
                    st.download_button("Download Full Price Data CSV", csv_data, f"{ticker_symbol}_price_data.csv", 'text/csv', key='dl_raw_price')
                else: st.info("Raw price data is not available.")
        else:
            st.info("Run an analysis to see performance details.")

    # Tab 2: Optimization Results (Full Period)
    if "⚙️ Optimization Results (Full Period)" in main_tabs_list and len(active_tabs) > 1:
        with active_tabs[1]:
            opt_df = st.session_state.optimization_results_df
            if not opt_df.empty:
                st.markdown("#### Grid/Random Search Results (Full Period)")
                st.dataframe(opt_df.style.format({col: '{:.2f}' for col in opt_df.select_dtypes(include='float').columns}), height=300)
                csv_opt = opt_df.to_csv(index=False).encode('utf-8'); st.download_button("Download Optimization CSV", csv_opt, f"{ticker_symbol}_opt_results.csv", 'text/csv', key='dl_opt_csv')
                
                st.markdown("#### Optimization Heatmap (Full Period)")
                opt_metric_hm = optimization_metric if analysis_mode == "Parameter Optimization" else settings.DEFAULT_OPTIMIZATION_METRIC
                if opt_algo == "Grid Search": # Heatmap only for Grid Search
                    heatmap_fig = plotting.plot_optimization_heatmap(opt_df, 'SL Points', 'RRR', opt_metric_hm)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                else: st.info("Heatmap is typically generated for Grid Search. For Random Search, review the table above.")
            else: st.info("No full-period optimization results. Run 'Parameter Optimization' mode.")

    # Tab 3: Walk-Forward Analysis
    if "🚶 Walk-Forward Analysis" in main_tabs_list and len(active_tabs) > (1 if "⚙️ Optimization Results (Full Period)" not in main_tabs_list else 2) :
        wfo_tab_index = main_tabs_list.index("🚶 Walk-Forward Analysis")
        with active_tabs[wfo_tab_index]:
            if st.session_state.wfo_results:
                wfo_res = st.session_state.wfo_results
                st.markdown("#### Walk-Forward Optimization Log")
                st.markdown("This table shows the best parameters chosen in each in-sample period and the corresponding out-of-sample (OOS) performance for that fold.")
                st.dataframe(wfo_res["log"].style.format({col: '{:.2f}' for col in wfo_res["log"].select_dtypes(include='float').columns}), height=300)
                csv_wfo_log = wfo_res["log"].to_csv(index=False).encode('utf-8'); st.download_button("Download WFO Log CSV", csv_wfo_log, f"{ticker_symbol}_wfo_log.csv", 'text/csv', key='dl_wfo_log')

                st.markdown("#### Aggregated Out-of-Sample Trades")
                if not wfo_res["oos_trades"].empty:
                    st.dataframe(wfo_res["oos_trades"].style.format({col: '{:.2f}' for col in wfo_res["oos_trades"].select_dtypes(include='float').columns}), height=300)
                    csv_wfo_trades = wfo_res["oos_trades"].to_csv(index=False).encode('utf-8'); st.download_button("Download WFO OOS Trades CSV", csv_wfo_trades, f"{ticker_symbol}_wfo_oos_trades.csv", 'text/csv', key='dl_wfo_trades')
                else: st.info("No out-of-sample trades generated during WFO.")
            else: st.info("No WFO results. Run 'Walk-Forward Optimization' mode.")


elif st.sidebar.button("Clear Results", use_container_width=True, key="clear_button_main_page"):
    init_session_state() # Reset all relevant states
    st.info("Results cleared.")
    st.experimental_rerun()
else:
    if not any([st.session_state.backtest_results, not st.session_state.optimization_results_df.empty, st.session_state.wfo_results]):
        st.info("Configure parameters and click 'Run Analysis'.")

st.sidebar.markdown("---")
st.sidebar.info(f"App Version: 0.2.0 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.sidebar.caption("Disclaimer: Financial modeling tool. Past performance and optimization results (especially without robust WFO) are not indicative of future results and can be overfit.")

