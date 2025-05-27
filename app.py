# app.py
"""
Main Streamlit application file for the Gap Guardian Strategy Backtester.
Handles UI, user inputs, and orchestrates the backtesting process, including optimization.
"""
import streamlit as st
import pandas as pd
import numpy as np # For linspace
from datetime import date, timedelta, datetime

# Project imports
from config import settings
from services import data_loader, strategy_engine, backtester, optimizer # Added optimizer
from utils import plotting, logger as app_logger

# Initialize logger
logger = app_logger.get_logger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Custom CSS ---
def load_custom_css(css_file_path):
    try:
        with open(css_file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found at {css_file_path}. Styles may not be applied.")
    except Exception as e: st.error(f"Error loading CSS: {e}")
load_custom_css("static/style.css")

# --- Application State ---
if 'backtest_results' not in st.session_state: st.session_state.backtest_results = None
if 'optimization_results_df' not in st.session_state: st.session_state.optimization_results_df = pd.DataFrame()
if 'price_data' not in st.session_state: st.session_state.price_data = pd.DataFrame()
if 'signals' not in st.session_state: st.session_state.signals = pd.DataFrame() # For single backtest
if 'best_params_from_opt' not in st.session_state: st.session_state.best_params_from_opt = None


# --- Sidebar for Inputs ---
st.sidebar.header("Backtest Configuration")
# ... (Symbol, Date, Capital, Risk % inputs remain the same as previous version) ...
selected_ticker_name = st.sidebar.selectbox("Select Symbol:", options=list(settings.DEFAULT_TICKERS.keys()), index=0)
ticker_symbol = settings.DEFAULT_TICKERS[selected_ticker_name]

today = date.today()
is_intraday_strategy = settings.STRATEGY_TIME_FRAME not in ["1d", "1wk", "1mo"]
if is_intraday_strategy:
    min_allowable_start_date_for_ui = today - timedelta(days=settings.MAX_INTRADAY_DAYS - 1)
    default_start_date_value = min_allowable_start_date_for_ui
    date_input_help_suffix = f"Intraday data is limited to the last ~{settings.MAX_INTRADAY_DAYS} days."
    if default_start_date_value >= (today - timedelta(days=1)): default_start_date_value = today - timedelta(days=1 if (today - timedelta(days=1)) > min_allowable_start_date_for_ui else 0)
else:
    min_allowable_start_date_for_ui = today - timedelta(days=365 * 5)
    default_start_date_value = today - timedelta(days=30)
    date_input_help_suffix = "Select historical period."

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

# --- Standard Backtest Parameters (used if optimization is off, or as defaults) ---
st.sidebar.subheader("Strategy Parameters (for Single Run)")
sl_points_single = st.sidebar.number_input("Stop Loss (points):", min_value=0.1, value=settings.DEFAULT_STOP_LOSS_POINTS, step=0.1, format="%.2f", key="sl_single")
rrr_single = st.sidebar.number_input("Risk/Reward Ratio:", min_value=0.1, value=settings.DEFAULT_RRR, step=0.1, format="%.1f", key="rrr_single")


# --- Optimization Section ---
st.sidebar.subheader("Parameter Optimization")
enable_optimization = st.sidebar.checkbox("Enable Parameter Optimization", value=False)

opt_sl_range = settings.DEFAULT_SL_POINTS_OPTIMIZATION_RANGE
opt_rrr_range = settings.DEFAULT_RRR_OPTIMIZATION_RANGE

if enable_optimization:
    st.sidebar.caption("Define ranges for Stop Loss points and RRR to optimize.")
    col_sl1, col_sl2, col_sl3 = st.sidebar.columns(3)
    sl_min = col_sl1.number_input("SL Min", value=opt_sl_range["min"], step=0.1, format="%.1f", key="sl_min")
    sl_max = col_sl2.number_input("SL Max", value=opt_sl_range["max"], step=0.1, format="%.1f", key="sl_max")
    sl_steps = col_sl3.number_input("SL Steps", min_value=2, max_value=20, value=opt_sl_range["steps"], step=1, key="sl_steps")

    col_rrr1, col_rrr2, col_rrr3 = st.sidebar.columns(3)
    rrr_min = col_rrr1.number_input("RRR Min", value=opt_rrr_range["min"], step=0.1, format="%.1f", key="rrr_min")
    rrr_max = col_rrr2.number_input("RRR Max", value=opt_rrr_range["max"], step=0.1, format="%.1f", key="rrr_max")
    rrr_steps = col_rrr3.number_input("RRR Steps", min_value=2, max_value=20, value=opt_rrr_range["steps"], step=1, key="rrr_steps")
    
    optimization_metric = st.sidebar.selectbox(
        "Optimize for Metric:",
        options=settings.OPTIMIZATION_METRICS,
        index=settings.OPTIMIZATION_METRICS.index(settings.DEFAULT_OPTIMIZATION_METRIC),
        help="Select the performance metric to maximize/minimize during optimization."
    )
    st.sidebar.warning("Optimization can be time-consuming depending on the number of parameter combinations.")


# --- Main Application Area ---
st.title(f"🛡️ {settings.APP_TITLE}")
st.markdown(f"Strategy: Gap Guardian | Interval: {settings.STRATEGY_TIME_FRAME} | Entry: 9:30-11:00 AM NY False Break | Max 1 trade/day.")


if st.sidebar.button("Run Analysis", type="primary", use_container_width=True, key="run_button"):
    # Clear previous results
    st.session_state.backtest_results = None
    st.session_state.optimization_results_df = pd.DataFrame()
    st.session_state.price_data = pd.DataFrame()
    st.session_state.signals = pd.DataFrame()
    st.session_state.best_params_from_opt = None


    if start_date >= end_date:
        st.error(f"Error: Start date ({start_date}) must be before end date ({end_date}).")
    else:
        with st.spinner("Fetching data..."):
            price_data_df = data_loader.fetch_historical_data(
                ticker_symbol, start_date, end_date, settings.STRATEGY_TIME_FRAME
            )
            st.session_state.price_data = price_data_df

        if price_data_df.empty:
            st.warning(f"No price data found for {selected_ticker_name} for the selected period. Cannot proceed.")
        else:
            if enable_optimization:
                st.subheader("Optimization Run")
                with st.spinner(f"Running Grid Search Optimization ({sl_steps * rrr_steps} combinations)... This may take a while."):
                    opt_progress_bar = st.progress(0, text="Optimization in progress...")
                    
                    sl_points_to_test = np.linspace(sl_min, sl_max, int(sl_steps))
                    rrr_values_to_test = np.linspace(rrr_min, rrr_max, int(rrr_steps))

                    optimization_df = optimizer.run_grid_search(
                        price_data_df,
                        initial_capital,
                        risk_per_trade_percent,
                        sl_points_to_test,
                        rrr_values_to_test,
                        optimization_metric,
                        progress_callback=lambda p: opt_progress_bar.progress(p, text=f"Optimization: {int(p*100)}% complete")
                    )
                    st.session_state.optimization_results_df = optimization_df
                    opt_progress_bar.progress(1.0, text="Optimization complete!")

                if not optimization_df.empty:
                    st.success("Optimization finished!")
                    # Determine best parameters
                    # Note: For Max Drawdown, lower is better. For others, higher is better.
                    if optimization_metric == "Max Drawdown (%)": # Lower is better
                        best_row = optimization_df.loc[optimization_df[optimization_metric].idxmin()]
                    else: # Higher is better
                        best_row = optimization_df.loc[optimization_df[optimization_metric].idxmax()]
                    
                    st.session_state.best_params_from_opt = {
                        "SL Points": best_row["SL Points"],
                        "RRR": best_row["RRR"],
                        "MetricValue": best_row[optimization_metric]
                    }
                    st.info(f"Best parameters found for '{optimization_metric}': "
                            f"SL Points = {best_row['SL Points']:.2f}, RRR = {best_row['RRR']:.1f} "
                            f"(Value: {best_row[optimization_metric]:.2f})")
                    
                    # Automatically run a single backtest with the best parameters
                    st.markdown("--- \n ### Backtest with Best Optimized Parameters")
                    with st.spinner("Running backtest with best parameters..."):
                        best_sl = best_row["SL Points"]
                        best_rrr = best_row["RRR"]
                        signals_df_best = strategy_engine.generate_signals(price_data_df.copy(), best_sl, best_rrr)
                        st.session_state.signals = signals_df_best # Store signals for this run
                        
                        trades_df_best, equity_series_best, performance_metrics_best = backtester.run_backtest(
                            price_data_df.copy(), signals_df_best, initial_capital,
                            risk_per_trade_percent, best_sl
                        )
                        st.session_state.backtest_results = {
                            "trades": trades_df_best, "equity_curve": equity_series_best, "performance": performance_metrics_best,
                            "params": {"SL Points": best_sl, "RRR": best_rrr, "source": "Optimization"}
                        }
                        st.success("Backtest with best parameters complete!")
                else:
                    st.error("Optimization did not yield any results.")

            else: # Single backtest run
                st.subheader("Single Backtest Run")
                with st.spinner("Running single backtest..."):
                    logger.info(f"Running single backtest with SL: {sl_points_single}, RRR: {rrr_single}")
                    signals_df = strategy_engine.generate_signals(price_data_df.copy(), sl_points_single, rrr_single)
                    st.session_state.signals = signals_df # Store signals for this run

                    trades_df, equity_series, performance_metrics = backtester.run_backtest(
                        price_data_df.copy(), signals_df, initial_capital,
                        risk_per_trade_percent, sl_points_single
                    )
                    st.session_state.backtest_results = {
                        "trades": trades_df, "equity_curve": equity_series, "performance": performance_metrics,
                        "params": {"SL Points": sl_points_single, "RRR": rrr_single, "source": "Manual"}
                    }
                    st.success("Single backtest complete!")


# --- Display Results ---
# This section will now display either single backtest results or optimization results + best param backtest

# Tabs for different views
main_tabs_list = ["📊 Backtest Performance"]
if not st.session_state.optimization_results_df.empty:
    main_tabs_list.append("⚙️ Optimization Results")

if st.session_state.backtest_results or not st.session_state.optimization_results_df.empty:
    selected_main_tab = st.tabs(main_tabs_list)

    with selected_main_tab[0]: # Backtest Performance Tab
        if st.session_state.backtest_results:
            results = st.session_state.backtest_results
            performance = results["performance"]
            trades = results["trades"]
            equity_curve = results["equity_curve"]
            # price_data_display = st.session_state.price_data # Already available globally in session_state
            # signals_display = st.session_state.signals # Already available globally in session_state
            run_params = results.get("params", {})
            run_source = run_params.get("source", "N/A")
            run_sl = run_params.get("SL Points", "N/A")
            run_rrr = run_params.get("RRR", "N/A")

            st.markdown(f"#### Performance Summary (Source: {run_source} | SL: {run_sl:.2f} | RRR: {run_rrr:.1f})")

            # ... (Existing styled metric display logic using display_styled_metric) ...
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


            detail_tabs_list = ["📈 Equity Curve", "📊 Trades on Price", "📋 Trade Log"]
            if not st.session_state.signals.empty: # Only show signals if they exist for current run
                 detail_tabs_list.append("🔍 Generated Signals")
            detail_tabs_list.append("💾 Raw Price Data")
            
            detail_tabs = st.tabs(detail_tabs_list)
            with detail_tabs[0]: # Equity
                if not equity_curve.empty: st.plotly_chart(plotting.plot_equity_curve(equity_curve), use_container_width=True)
                else: st.info("Equity curve is not available.")
            with detail_tabs[1]: # Trades on Price
                if not st.session_state.price_data.empty and not trades.empty : st.plotly_chart(plotting.plot_trades_on_price(st.session_state.price_data, trades, selected_ticker_name), use_container_width=True)
                else: st.info("Price/trade data not available for plotting.")
            with detail_tabs[2]: # Trade Log
                if not trades.empty:
                    float_cols = trades.select_dtypes(include='float').columns; format_dict = {col: '{:.2f}' for col in float_cols}
                    st.dataframe(trades.style.format(format_dict), use_container_width=True)
                else: st.info("No trades were executed.")
            
            idx_offset = 0
            if "🔍 Generated Signals" in detail_tabs_list:
                with detail_tabs[3]: # Signals Log
                    if not st.session_state.signals.empty:
                        st.markdown("Raw signals generated by the strategy engine for this run."); float_cols_signals = st.session_state.signals.select_dtypes(include='float').columns; format_dict_signals = {col: '{:.2f}' for col in float_cols_signals}
                        st.dataframe(st.session_state.signals.style.format(format_dict_signals), use_container_width=True)
                    else: st.info("No signals were generated for this run.")
                idx_offset = 1
            
            with detail_tabs[3+idx_offset]: # Raw Price Data
                if not st.session_state.price_data.empty:
                    st.markdown(f"Raw OHLCV price data for **{selected_ticker_name}** ({len(st.session_state.price_data)} rows).")
                    st.dataframe(st.session_state.price_data.head(), height=300, use_container_width=True)
                    # ... (download button logic) ...
                    csv_data = st.session_state.price_data.to_csv(index=True).encode('utf-8')
                    st.download_button(label="Download Full Price Data as CSV", data=csv_data, file_name=f"{ticker_symbol}_price_data_{start_date}_to_{end_date}.csv", mime='text/csv', key='download-csv-main')
                else: st.info("Raw price data is not available.")
        else:
            st.info("Run a backtest or optimization to see performance details here.")


    if "⚙️ Optimization Results" in main_tabs_list:
        with selected_main_tab[1]: # Optimization Results Tab
            opt_df = st.session_state.optimization_results_df
            if not opt_df.empty:
                st.markdown("#### Grid Search Optimization Results")
                st.dataframe(opt_df.style.format({col: '{:.2f}' for col in opt_df.select_dtypes(include='float').columns}), height=300)

                # Download Optimization Results
                csv_opt_data = opt_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Optimization Results as CSV",
                    data=csv_opt_data,
                    file_name=f"{ticker_symbol}_optimization_results.csv",
                    mime='text/csv',
                    key='download-opt-csv'
                )
                
                st.markdown("#### Optimization Heatmap")
                if st.session_state.best_params_from_opt: # Use the metric that was optimized for
                     opt_metric_for_heatmap = optimization_metric # From sidebar selection
                else: # Fallback if somehow best_params not set but opt_df exists
                     opt_metric_for_heatmap = settings.DEFAULT_OPTIMIZATION_METRIC

                heatmap_fig = plotting.plot_optimization_heatmap(
                    opt_df,
                    param1_name='SL Points',
                    param2_name='RRR',
                    metric_name=opt_metric_for_heatmap
                )
                st.plotly_chart(heatmap_fig, use_container_width=True)
                st.caption(f"Heatmap shows '{opt_metric_for_heatmap}'. Darker green (or higher values for some color scales) indicates better performance for that metric. "
                           "Red/lower values indicate poorer performance. Note: For 'Max Drawdown (%)', lower (more negative) is typically worse, but the color scale might represent magnitude.")

            else:
                st.info("No optimization results to display. Enable and run optimization from the sidebar.")


elif st.sidebar.button("Clear Results", use_container_width=True, key="clear_results_button_main"):
    st.session_state.backtest_results = None
    st.session_state.optimization_results_df = pd.DataFrame()
    st.session_state.price_data = pd.DataFrame()
    st.session_state.signals = pd.DataFrame()
    st.session_state.best_params_from_opt = None
    st.info("Results cleared. Configure and run a new analysis.")
    st.experimental_rerun()
else:
    if 'backtest_results' not in st.session_state and st.session_state.optimization_results_df.empty :
        st.info("Configure parameters in the sidebar and click 'Run Analysis' to see results.")

st.sidebar.markdown("---")
st.sidebar.info(f"App Version: 0.1.4 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}") # Incremented version
st.sidebar.caption("Disclaimer: This is a financial modeling tool. Past performance is not indicative of future results. Optimization results can be overfit to historical data.")

