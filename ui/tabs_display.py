# ui/tabs_display.py
"""
Handles rendering of the main results tabs.
"""
import streamlit as st
import pandas as pd
import numpy as np
from config import settings # Assuming settings.py is in a 'config' directory
from utils import plotting # Assuming plotting.py is in 'utils'
from utils.logger import get_logger

logger = get_logger(__name__)

def format_metric_value(value, precision=2, is_currency=True, is_percentage=False):
    """Formats metric values for display."""
    if pd.isna(value) or value is None: return "N/A"
    if is_currency: return f"${value:,.{precision}f}"
    if is_percentage: return f"{value:.{precision}f}%"
    return f"{value:,.{precision}f}" if isinstance(value, (float, np.floating)) else str(value)

def display_styled_metric_card(column, label, value_raw, is_currency=True, is_percentage=False, precision=2, profit_factor_logic=False, mdd_logic=False):
    """Displays a single styled metric card."""
    formatted_value = format_metric_value(value_raw, precision, is_currency, is_percentage)
    color_style_str = "" 
    
    if not (pd.isna(value_raw) or value_raw is None):
        if profit_factor_logic: 
            if value_raw > 1: color_style_str = f"color: {settings.POSITIVE_METRIC_COLOR};"
            elif value_raw < 1 and value_raw != 0 : color_style_str = f"color: {settings.NEGATIVE_METRIC_COLOR};"
        elif mdd_logic: 
            if value_raw < 0: color_style_str = f"color: {settings.NEGATIVE_METRIC_COLOR};"
            elif value_raw == 0: color_style_str = f"color: {settings.NEUTRAL_METRIC_COLOR};"
            else: color_style_str = f"color: {settings.POSITIVE_METRIC_COLOR};"
        else: 
            if value_raw > 0: color_style_str = f"color: {settings.POSITIVE_METRIC_COLOR};"
            elif value_raw < 0: color_style_str = f"color: {settings.NEGATIVE_METRIC_COLOR};"
    
    column.markdown(f"""<div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value" style="{color_style_str}">{formatted_value}</div>
                        </div>""", unsafe_allow_html=True)

def render_performance_tab(analysis_mode_ui, initial_capital_ui, selected_ticker_name, ticker_symbol, start_date_ui, end_date_ui):
    """Renders the content of the 'Backtest Performance' tab."""
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

        st.subheader("Performance Summary")
        summary_details_md = f"""
        <div class='performance-summary-details'>
            <span class='summary-parameter-detail'>Strategy: {strat_display_info}</span>
            <span class='summary-parameter-detail'>Symbol: {symbol_display_info}</span>
            <span class='summary-parameter-detail'>Source: {run_source_info}</span>
            <span class='summary-parameter-detail'>Timeframe: {tf_display_info}</span>
        """
        entry_display_val_info = run_params_info.get("EntryDisplay", "")
        if entry_display_val_info:
            summary_details_md += f"<span class='summary-parameter-detail'>Parameters: {entry_display_val_info}</span>"
        elif run_params_info.get("SL") is not None and run_params_info.get("RRR") is not None:
            summary_details_md += f"<span class='summary-parameter-detail'>Parameters: SL: {float(run_params_info.get('SL')):.2f}, RRR: {float(run_params_info.get('RRR')):.1f}</span>"
        summary_details_md += "</div>"
        st.markdown(summary_details_md, unsafe_allow_html=True)
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            display_styled_metric_card(col1, "Total P&L", performance_summary.get('Total P&L'), is_currency=True)
            display_styled_metric_card(col1, "Final Capital", performance_summary.get('Final Capital', initial_capital_ui), is_currency=True)
            display_styled_metric_card(col1, "Max Drawdown", performance_summary.get('Max Drawdown (%)'), is_currency=False, is_percentage=True, mdd_logic=True)
        with col2:
            display_styled_metric_card(col2, "Total Trades", int(performance_summary.get('Total Trades', 0)), is_currency=False, is_percentage=False)
            display_styled_metric_card(col2, "Win Rate", performance_summary.get('Win Rate', 0), is_currency=False, is_percentage=True)
            display_styled_metric_card(col2, "Profit Factor", performance_summary.get('Profit Factor', 0), is_currency=False, precision=2, profit_factor_logic=True)
        with col3:
            display_styled_metric_card(col3, "Avg. Trade P&L", performance_summary.get('Average Trade P&L'), is_currency=True)
            display_styled_metric_card(col3, "Avg. Winning Trade", performance_summary.get('Average Winning Trade'), is_currency=True)
            display_styled_metric_card(col3, "Avg. Losing Trade", performance_summary.get('Average Losing Trade'), is_currency=True)
        
        detail_tabs_names = ["📈 Equity Curve", "📊 Trades on Price", "📋 Trade Log"]
        if not st.session_state.signals.empty and analysis_mode_ui != "Walk-Forward Optimization":
            detail_tabs_names.append("🔍 Generated Signals (Last Run)")
        detail_tabs_names.append("💾 Raw Price Data (Full Period)")
        
        detail_tabs = st.tabs(detail_tabs_names)
        
        with detail_tabs[0]: # Equity Curve
            plot_title = "Equity Curve"
            plot_func = plotting.plot_equity_curve
            if analysis_mode_ui == "Walk-Forward Optimization" and "oos_equity_curve" in results_to_display:
                plot_title = "WFO: Aggregated Out-of-Sample Equity"
                plot_func = plotting.plot_wfo_equity_curve
                equity_curve_display = results_to_display["oos_equity_curve"]
            
            if not equity_curve_display.empty:
                st.plotly_chart(plot_func(equity_curve_display, title=plot_title), use_container_width=True)
            else: st.info("Equity curve data not available.")

        with detail_tabs[1]: # Trades on Price
            if not st.session_state.price_data.empty and not trades_df_display.empty:
                st.plotly_chart(plotting.plot_trades_on_price(st.session_state.price_data, trades_df_display, selected_ticker_name), use_container_width=True)
            else: st.info("Price/trade data not available for plotting trades on price.")

        with detail_tabs[2]: # Trade Log
            if not trades_df_display.empty:
                st.dataframe(trades_df_display.style.format({col: '{:.2f}' for col in trades_df_display.select_dtypes(include='float').columns}), height=300, use_container_width=True)
            else: st.info("No trades executed.")

        idx_offset = 0
        if "🔍 Generated Signals (Last Run)" in detail_tabs_names:
            with detail_tabs[3]:
                if not st.session_state.signals.empty:
                    st.dataframe(st.session_state.signals.style.format({col: '{:.2f}' for col in st.session_state.signals.select_dtypes(include='float').columns}), height=300, use_container_width=True)
                else: st.info("No signals generated for the last run.")
            idx_offset = 1
        
        with detail_tabs[3 + idx_offset]: # Raw Price Data
            if not st.session_state.price_data.empty:
                st.markdown(f"Full OHLCV data for **{selected_ticker_name} ({ticker_symbol})** ({len(st.session_state.price_data)} rows). Displaying first 100.")
                st.dataframe(st.session_state.price_data.head(100), height=300, use_container_width=True)
                try:
                    csv_raw = st.session_state.price_data.to_csv(index=True).encode('utf-8')
                    st.download_button("Download Full Price Data CSV", csv_raw, f"{ticker_symbol}_price_data_{start_date_ui}_to_{end_date_ui}.csv", 'text/csv', key='dl_raw_price_tabs_v1')
                except Exception as e:
                    logger.error(f"Error generating CSV for raw price data: {e}", exc_info=True)
                    st.warning("Could not prepare raw price data for download.")
            else: st.info("Raw price data not available.")
    else:
        st.info("Run analysis to see performance. Results might be empty if no trades or insufficient data.")

def render_optimization_tab(selected_strategy_name, opt_algo_ui, opt_metric_ui, ticker_symbol):
    """Renders the 'Optimization Results (Full Period)' tab."""
    opt_df_display = st.session_state.optimization_results_df
    if not opt_df_display.empty:
        st.subheader(f"Optimization Results ({selected_strategy_name} - Full Period - {opt_algo_ui})")
        float_cols = [col for col in opt_df_display.columns if opt_df_display[col].dtype == 'float64']
        st.dataframe(opt_df_display.style.format({col: '{:.2f}' for col in float_cols}), height=400, use_container_width=True)
        try:
            csv_opt = opt_df_display.to_csv(index=False).encode('utf-8')
            st.download_button("Download Optimization Results CSV", csv_opt, f"{ticker_symbol}_{selected_strategy_name}_opt_results.csv", 'text/csv', key='dl_opt_csv_tabs_v1')
        except Exception as e:
            logger.error(f"Error generating CSV for optimization results: {e}", exc_info=True)
            st.warning("Could not prepare optimization results for download.")

        if opt_algo_ui == "Grid Search" and 'SL Points' in opt_df_display.columns and 'RRR' in opt_df_display.columns:
            st.markdown(f"##### Optimization Heatmap: {opt_metric_ui} (SL vs RRR - Full Period)")
            try:
                heatmap = plotting.plot_optimization_heatmap(opt_df_display, 'SL Points', 'RRR', opt_metric_ui)
                st.plotly_chart(heatmap, use_container_width=True)
            except Exception as e:
                logger.error(f"Error generating optimization heatmap: {e}", exc_info=True)
                st.warning(f"Could not generate heatmap. Error: {e}")
        elif opt_algo_ui == "Grid Search":
            st.info("Heatmap for SL vs RRR requires 'SL Points' and 'RRR' in grid search.")
        else:
            st.info("Heatmap typically for Grid Search. Review table for Random Search details.")
    else:
        st.info("No full-period optimization results. Ensure 'Parameter Optimization' was run.")

def render_wfo_tab(selected_strategy_name, opt_algo_ui, ticker_symbol):
    """Renders the 'Walk-Forward Analysis' tab."""
    if st.session_state.wfo_results:
        wfo_res = st.session_state.wfo_results
        wfo_log = wfo_res.get("log", pd.DataFrame())
        wfo_oos_trades = wfo_res.get("oos_trades", pd.DataFrame())

        st.subheader(f"Walk-Forward Optimization Log ({selected_strategy_name} - {opt_algo_ui} for inner opt)")
        if not wfo_log.empty:
            float_cols = [col for col in wfo_log.columns if wfo_log[col].dtype == 'float64']
            st.dataframe(wfo_log.style.format({col: '{:.2f}' for col in float_cols}), height=300, use_container_width=True)
            try:
                csv_log = wfo_log.to_csv(index=False).encode('utf-8')
                st.download_button("Download WFO Log CSV", csv_log, f"{ticker_symbol}_{selected_strategy_name}_wfo_log.csv", 'text/csv', key='dl_wfo_log_tabs_v1')
            except Exception as e:
                logger.error(f"Error generating CSV for WFO log: {e}", exc_info=True)
                st.warning("Could not prepare WFO log for download.")
        else:
            st.info("WFO log is empty.")

        st.markdown("##### Aggregated Out-of-Sample (OOS) Trades from WFO")
        if not wfo_oos_trades.empty:
            st.dataframe(wfo_oos_trades.style.format({col: '{:.2f}' for col in wfo_oos_trades.select_dtypes(include='float').columns}), height=300, use_container_width=True)
            try:
                csv_trades = wfo_oos_trades.to_csv(index=False).encode('utf-8')
                st.download_button("Download WFO OOS Trades CSV", csv_trades, f"{ticker_symbol}_{selected_strategy_name}_wfo_oos_trades.csv", 'text/csv', key='dl_wfo_trades_tabs_v1')
            except Exception as e:
                logger.error(f"Error generating CSV for WFO OOS trades: {e}", exc_info=True)
                st.warning("Could not prepare WFO OOS trades for download.")
        else:
            st.info("No OOS trades generated during WFO.")
    else:
        st.info("No WFO results. Ensure 'Walk-Forward Optimization' was run.")


def render_results_tabs(sidebar_inputs):
    """Determines which main tabs to display and calls their rendering functions."""
    analysis_mode_ui = sidebar_inputs["analysis_mode_ui"]
    
    tabs_to_display = []
    if st.session_state.backtest_results:
        tabs_to_display.append("📊 Backtest Performance")
    if not st.session_state.optimization_results_df.empty and analysis_mode_ui == "Parameter Optimization":
        tabs_to_display.append("⚙️ Optimization Results (Full Period)")
    if st.session_state.wfo_results and analysis_mode_ui == "Walk-Forward Optimization":
        tabs_to_display.append("🚶 Walk-Forward Analysis")

    if tabs_to_display:
        # Use a dynamic key for tabs to force re-render if the set of available tabs changes
        tabs_key = "_".join(tabs_to_display) + f"_{st.session_state.run_analysis_clicked_count}"
        created_tabs = st.tabs(tabs_to_display)
        
        tab_map = dict(zip(tabs_to_display, created_tabs))

        if "📊 Backtest Performance" in tab_map:
            with tab_map["📊 Backtest Performance"]:
                render_performance_tab(
                    analysis_mode_ui,
                    sidebar_inputs["initial_capital_ui"],
                    sidebar_inputs["selected_ticker_name"],
                    sidebar_inputs["ticker_symbol"],
                    sidebar_inputs["start_date_ui"],
                    sidebar_inputs["end_date_ui"]
                )
        
        if "⚙️ Optimization Results (Full Period)" in tab_map:
            with tab_map["⚙️ Optimization Results (Full Period)"]:
                render_optimization_tab(
                    sidebar_inputs["selected_strategy_name"],
                    sidebar_inputs["opt_algo_ui"],
                    sidebar_inputs["opt_metric_ui"],
                    sidebar_inputs["ticker_symbol"]
                )

        if "🚶 Walk-Forward Analysis" in tab_map:
            with tab_map["🚶 Walk-Forward Analysis"]:
                render_wfo_tab(
                    sidebar_inputs["selected_strategy_name"],
                    sidebar_inputs["opt_algo_ui"],
                    sidebar_inputs["ticker_symbol"]
                )
    elif st.session_state.run_analysis_clicked_count > 0:
        st.info("Analysis was run. If results are not displayed, it might be due to no trades or data for the selected parameters, or an error during processing. Check logs if errors are suspected.")
    else:
        if not any([st.session_state.backtest_results, not st.session_state.optimization_results_df.empty, st.session_state.wfo_results]):
            st.info("Configure parameters in the sidebar and click 'Run Analysis' to view results.")

