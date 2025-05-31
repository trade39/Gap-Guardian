# ui/tabs_display.py
"""
Handles rendering of the main results tabs, including new performance metrics
and CSV report export functionality.
"""
import streamlit as st
import pandas as pd
import numpy as np
import io # For CSV in-memory buffer
import csv # For writing CSV data
from datetime import timedelta

from config import settings 
from utils import plotting 
from utils.logger import get_logger

logger = get_logger(__name__)

def format_metric_value(value, precision=2, is_currency=True, is_percentage=False, is_ratio=False):
    if pd.isna(value) or value is None or (isinstance(value, float) and (np.isinf(value))):
        return "N/A" if not (isinstance(value, float) and np.isinf(value)) else ("Inf" if value > 0 else "-Inf")
    if is_currency: return f"${value:,.{precision}f}"
    if is_percentage: return f"{value:.{precision}f}%"
    if is_ratio: return f"{value:.{precision}f}" 
    return f"{value:,.{precision}f}" if isinstance(value, (float, np.floating)) else str(value)

def display_styled_metric_card(column, label, value_raw, is_currency=True, is_percentage=False, is_ratio=False, precision=2, 
                               profit_factor_logic=False, mdd_logic=False, sharpe_logic=False, recovery_factor_logic=False, expected_value_logic=False):
    formatted_value = format_metric_value(value_raw, precision, is_currency, is_percentage, is_ratio)
    color_style_str = "" 
    if not (pd.isna(value_raw) or value_raw is None or (isinstance(value_raw, float) and np.isinf(value_raw))):
        base_value_for_coloring = float(value_raw)
        if profit_factor_logic: 
            if base_value_for_coloring > 1: color_style_str = f"color: {settings.POSITIVE_METRIC_COLOR};"
            elif base_value_for_coloring < 1 and base_value_for_coloring != 0 : color_style_str = f"color: {settings.NEGATIVE_METRIC_COLOR};"
            else: color_style_str = f"color: {settings.NEUTRAL_METRIC_COLOR};"
        elif mdd_logic: 
            if base_value_for_coloring < -1: color_style_str = f"color: {settings.NEGATIVE_METRIC_COLOR};"
            elif base_value_for_coloring < 0 : color_style_str = f"color: {settings.NEGATIVE_METRIC_COLOR};" 
            else: color_style_str = f"color: {settings.NEUTRAL_METRIC_COLOR};" 
        elif sharpe_logic:
            if base_value_for_coloring > 1: color_style_str = f"color: {settings.POSITIVE_METRIC_COLOR};"
            elif base_value_for_coloring > 0: color_style_str = f"color: orange;" 
            else: color_style_str = f"color: {settings.NEGATIVE_METRIC_COLOR};"       
        elif recovery_factor_logic:
             if base_value_for_coloring > 2: color_style_str = f"color: {settings.POSITIVE_METRIC_COLOR};"
             elif base_value_for_coloring > 1: color_style_str = f"color: orange;"
             else: color_style_str = f"color: {settings.NEGATIVE_METRIC_COLOR};"
        elif expected_value_logic: 
            if base_value_for_coloring > 0: color_style_str = f"color: {settings.POSITIVE_METRIC_COLOR};"
            elif base_value_for_coloring < 0: color_style_str = f"color: {settings.NEGATIVE_METRIC_COLOR};"
            else: color_style_str = f"color: {settings.NEUTRAL_METRIC_COLOR};"
        else: 
            if base_value_for_coloring > 0: color_style_str = f"color: {settings.POSITIVE_METRIC_COLOR};"
            elif base_value_for_coloring < 0: color_style_str = f"color: {settings.NEGATIVE_METRIC_COLOR};"
            else: color_style_str = f"color: {settings.NEUTRAL_METRIC_COLOR};"
    
    column.markdown(f"""<div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value" style="{color_style_str}">{formatted_value}</div>
                        </div>""", unsafe_allow_html=True)

def generate_strategy_report_csv_string(backtest_results: dict, sidebar_inputs: dict, price_data_len: int) -> str:
    """Generates a CSV string formatted similarly to a MetaTrader Strategy Tester Report."""
    output = io.StringIO()
    writer = csv.writer(output)

    perf = backtest_results.get("performance", {})
    run_params = backtest_results.get("params", {})
    trades_df = backtest_results.get("trades", pd.DataFrame())

    writer.writerow(["Strategy Tester Report"])
    writer.writerow([f"{settings.APP_TITLE} (Build {settings.APP_VERSION})"]) # App name and version
    writer.writerow([]) # Empty line

    writer.writerow(["Settings"])
    writer.writerow(["Expert:", "", "", run_params.get("Strategy", "N/A")])
    writer.writerow(["Symbol:", "", "", sidebar_inputs.get("ticker_symbol", "N/A")])
    period_str = f"{sidebar_inputs.get('ui_current_interval', 'N/A')} ({sidebar_inputs.get('start_date_ui', 'N/A')} - {sidebar_inputs.get('end_date_ui', 'N/A')})"
    writer.writerow(["Period:", "", "", period_str])
    writer.writerow([])

    writer.writerow(["Inputs:", "", "", "="]) # Placeholder for equal sign
    if run_params.get("Strategy") == "Gap Guardian" and "EntryDisplay" in run_params:
        # Attempt to parse EntryDisplay for individual components if possible, or use the string
        # This part is simplified; robust parsing would be more complex.
        # Example: "SL: 10.00, RRR: 2.0, Entry: 09:30-11:00 NYT"
        sl_val = run_params.get("SL", sidebar_inputs.get("sl_points_single_ui"))
        rrr_val = run_params.get("RRR", sidebar_inputs.get("rrr_single_ui"))
        writer.writerow(["", "", "", f"EntryStartHour={sidebar_inputs.get('entry_start_hour_single_ui','N/A')}"])
        writer.writerow(["", "", "", f"EntryStartMinute={sidebar_inputs.get('entry_start_minute_single_ui','N/A')}"])
        writer.writerow(["", "", "", f"EntryEndHour={sidebar_inputs.get('entry_end_hour_single_ui','N/A')}"])
        writer.writerow(["", "", "", f"EntryEndMinute={sidebar_inputs.get('entry_end_minute_single_ui','N/A')}"])
    else: # For other strategies or if EntryDisplay is not detailed
        sl_val = run_params.get("SL", sidebar_inputs.get("sl_points_single_ui"))
        rrr_val = run_params.get("RRR", sidebar_inputs.get("rrr_single_ui"))
    
    writer.writerow(["", "", "", f"RiskPercent={sidebar_inputs.get('risk_per_trade_percent_ui','N/A')}"])
    writer.writerow(["", "", "", f"StopLossPoints={sl_val}"]) # Assuming SL from params or sidebar
    writer.writerow(["", "", "", f"RiskRewardRatio={rrr_val}"]) # Assuming RRR from params or sidebar
    writer.writerow(["", "", "", f"CommissionType={run_params.get('CommissionType','None')}"])
    writer.writerow(["", "", "", f"CommissionRate={run_params.get('CommissionRate',0.0)}"]) # Rate as decimal
    writer.writerow(["", "", "", f"SlippagePoints={run_params.get('SlippagePoints',0.0)}"])
    writer.writerow([])

    writer.writerow(["Company:", "", "", "N/A"]) # Placeholder
    writer.writerow(["Currency:", "", "", "USD (assumed)"]) # Placeholder
    writer.writerow(["Initial Deposit:", "", "", perf.get("Initial Capital", sidebar_inputs.get("initial_capital_ui"))])
    writer.writerow(["Leverage:", "", "", "N/A"]) # Placeholder
    writer.writerow([])

    writer.writerow(["Results"])
    writer.writerow(["History Quality:", "", "", "N/A (yfinance data)"])
    writer.writerow(["Bars:", "", "", price_data_len, "Ticks:", "", "", "N/A", "Symbols:", "", "", 1])
    
    # Using .get with default to handle missing keys gracefully
    writer.writerow(["Total Net Profit:", "", "", f"{perf.get('Total Net P&L', 0.0):.2f}", "Balance Drawdown Absolute:", "", "", f"{perf.get('Max Drawdown ($)', 0.0):.2f}"])
    writer.writerow(["Gross Profit:", "", "", f"{perf.get('Gross Profit', 0.0):.2f}", "Balance Drawdown Maximal:", "", "", f"{perf.get('Max Drawdown ($)', 0.0):.2f} ({perf.get('Max Drawdown (%)', 0.0):.2f}%)"])
    writer.writerow(["Gross Loss:", "", "", f"{perf.get('Gross Loss', 0.0):.2f}", "Balance Drawdown Relative:", "", "", f"{perf.get('Max Drawdown (%)', 0.0):.2f}% ({perf.get('Max Drawdown ($)', 0.0):.2f})"]) # MT has this swapped
    writer.writerow([])
    writer.writerow(["Profit Factor:", "", "", f"{perf.get('Profit Factor', 0.0):.4f}", "Expected Payoff:", "", "", f"{perf.get('Expected Value', 0.0):.4f}"])
    writer.writerow(["Recovery Factor:", "", "", f"{perf.get('Recovery Factor', 0.0):.4f}", "Sharpe Ratio:", "", "", f"{perf.get('Sharpe Ratio (Annualized)', 0.0):.4f}"])
    writer.writerow([])

    writer.writerow(["Minimal position holding time:", "", "", str(perf.get('Min Position Holding Time', pd.NaT)).split('.')[0]]) # Remove ms
    writer.writerow(["Maximal position holding time:", "", "", str(perf.get('Max Position Holding Time', pd.NaT)).split('.')[0]])
    writer.writerow(["Average position holding time:", "", "", str(timedelta(seconds=perf.get('Avg Position Holding Time', pd.Timedelta(0)).total_seconds() if pd.notna(perf.get('Avg Position Holding Time')) else 0)).split('.')[0]])
    writer.writerow([])

    writer.writerow(["Total Trades:", "", "", perf.get('Total Trades', 0), 
                     "Short Trades (won %):", "", "", f"{perf.get('Short Trades',0)} ({perf.get('Short Trades Won',0)/perf.get('Short Trades',1)*100:.2f}%)" if perf.get('Short Trades',0) > 0 else "0 (0.00%)",
                     "Long Trades (won %):", "", "", f"{perf.get('Long Trades',0)} ({perf.get('Long Trades Won',0)/perf.get('Long Trades',1)*100:.2f}%)" if perf.get('Long Trades',0) > 0 else "0 (0.00%)"])
    
    profit_trades_pct = (perf.get('Winning Trades',0) / perf.get('Total Trades',1) * 100) if perf.get('Total Trades',0) > 0 else 0.0
    loss_trades_pct = (perf.get('Losing Trades',0) / perf.get('Total Trades',1) * 100) if perf.get('Total Trades',0) > 0 else 0.0
    writer.writerow(["Profit Trades (% of total):", "", "", f"{perf.get('Winning Trades',0)} ({profit_trades_pct:.2f}%)", 
                     "Loss Trades (% of total):", "", "", f"{perf.get('Losing Trades',0)} ({loss_trades_pct:.2f}%)"])
    writer.writerow([])
    writer.writerow(["", "Largest profit trade:", "", "", f"{perf.get('Largest Profit Trade',0.0):.2f}", "Largest loss trade:", "", "", f"{perf.get('Largest Loss Trade',0.0):.2f}"])
    writer.writerow(["", "Average profit trade:", "", "", f"{perf.get('Average Winning Trade',0.0):.2f}", "Average loss trade:", "", "", f"{perf.get('Average Losing Trade',0.0):.2f}"])
    writer.writerow(["", "Maximum consecutive wins ($):", "", "", f"{perf.get('Max Consecutive Wins',0)} ({perf.get('Max Consecutive Wins ($)',0.0):.2f})", 
                     "Maximum consecutive losses ($):", "", "", f"{perf.get('Max Consecutive Losses',0)} ({perf.get('Max Consecutive Losses ($)',0.0):.2f})"])
    writer.writerow(["", "Maximal consecutive profit (count):", "", "", f"{perf.get('Max Consecutive Wins ($)',0.0):.2f} ({perf.get('Max Consecutive Wins',0)})", # Swapped order for MT
                     "Maximal consecutive loss (count):", "", "", f"{perf.get('Max Consecutive Losses ($)',0.0):.2f} ({perf.get('Max Consecutive Losses',0)})"])
    writer.writerow(["", "Average consecutive wins:", "", "", perf.get('Avg Consecutive Wins',0), "Average consecutive losses:", "", "", perf.get('Avg Consecutive Losses',0)])
    writer.writerow([])

    # Simplified Trades List
    writer.writerow(["Trades List"])
    if not trades_df.empty:
        trade_cols = ['EntryTime', 'Type', 'ActualEntryPrice', 'SL', 'TP', 'ExitTime', 'ExitPrice', 'P&L', 'TotalCommission', 'ExitReason']
        writer.writerow(trade_cols) # Header
        for _, trade_row in trades_df.iterrows():
            row_to_write = [
                trade_row.get('EntryTime', pd.NaT).strftime('%Y.%m.%d %H:%M:%S') if pd.notna(trade_row.get('EntryTime')) else 'N/A',
                trade_row.get('Type', 'N/A'),
                f"{trade_row.get('ActualEntryPrice', 0.0):.2f}",
                f"{trade_row.get('SL', 0.0):.2f}",
                f"{trade_row.get('TP', 0.0):.2f}",
                trade_row.get('ExitTime', pd.NaT).strftime('%Y.%m.%d %H:%M:%S') if pd.notna(trade_row.get('ExitTime')) else 'N/A',
                f"{trade_row.get('ExitPrice', 0.0):.2f}",
                f"{trade_row.get('P&L', 0.0):.2f}",
                f"{trade_row.get('TotalCommission', 0.0):.2f}",
                trade_row.get('ExitReason', 'N/A')
            ]
            writer.writerow(row_to_write)
    else:
        writer.writerow(["No trades executed."])

    return output.getvalue()


def render_performance_tab(analysis_mode_ui, initial_capital_ui, selected_ticker_name, ticker_symbol, start_date_ui, end_date_ui):
    if st.session_state.backtest_results:
        results_to_display = st.session_state.backtest_results
        performance_summary = results_to_display["performance"]
        trades_df_display = results_to_display["trades"] 
        equity_curve_display = results_to_display["equity_curve"]
        
        run_params_info = results_to_display.get("params", {})
        # ... (rest of the variable assignments) ...
        run_source_info = run_params_info.get("src", "N/A")
        tf_display_info = run_params_info.get("TF", st.session_state.selected_timeframe_value)
        strat_display_info = run_params_info.get("Strategy", "N/A")
        symbol_display_info = run_params_info.get("Symbol", selected_ticker_name)


        st.subheader("Performance Summary")
        # Add download button here
        try:
            csv_report_str = generate_strategy_report_csv_string(
                st.session_state.backtest_results,
                st.session_state.get('last_sidebar_inputs', {}), # Assumes sidebar_inputs is stored in session_state
                len(st.session_state.price_data) if not st.session_state.price_data.empty else 0
            )
            st.download_button(
                label="📥 Download Full Report (CSV)",
                data=csv_report_str,
                file_name=f"{ticker_symbol}_{strat_display_info}_report_{start_date_ui}_to_{end_date_ui}.csv",
                mime="text/csv",
                key="download_full_report_csv_v1"
            )
        except Exception as e_csv:
            logger.error(f"Error generating CSV report: {e_csv}", exc_info=True)
            st.warning(f"Could not generate CSV report: {e_csv}")


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
            sl_val = run_params_info.get('SL')
            rrr_val = run_params_info.get('RRR')
            sl_str = f"{float(sl_val):.2f}" if isinstance(sl_val, (int, float)) else str(sl_val)
            rrr_str = f"{float(rrr_val):.1f}" if isinstance(rrr_val, (int, float)) else str(rrr_val)
            summary_details_md += f"<span class='summary-parameter-detail'>Parameters: SL: {sl_str}, RRR: {rrr_str}</span>"
        
        if "CommissionType" in run_params_info and run_params_info["CommissionType"] != "None":
            comm_type = run_params_info["CommissionType"]
            comm_rate = run_params_info["CommissionRate"] # This is now decimal from sidebar
            comm_rate_display = f"${comm_rate:.2f}/side" if comm_type == "Fixed per Trade" else f"{comm_rate*100:.3f}%/side"
            summary_details_md += f"<span class='summary-parameter-detail'>Commission: {comm_type} ({comm_rate_display})</span>"
        if "SlippagePoints" in run_params_info and run_params_info["SlippagePoints"] > 0:
            summary_details_md += f"<span class='summary-parameter-detail'>Slippage: {run_params_info['SlippagePoints']:.2f} pts/side</span>"
        
        summary_details_md += "</div>"
        st.markdown(summary_details_md, unsafe_allow_html=True)
        st.markdown("---")

        st.subheader("Key Performance Indicators (KPIs)")
        col1, col2, col3 = st.columns(3)
        with col1:
            display_styled_metric_card(col1, "Total Net P&L", performance_summary.get('Total Net P&L'), is_currency=True)
            display_styled_metric_card(col1, "Expected Value/Trade", performance_summary.get('Expected Value'), is_currency=True, expected_value_logic=True)
        with col2:
            display_styled_metric_card(col2, "Total Trades", int(performance_summary.get('Total Trades', 0)), is_currency=False, is_percentage=False)
            display_styled_metric_card(col2, "Win Rate", performance_summary.get('Win Rate', 0), is_currency=False, is_percentage=True)
        with col3:
            display_styled_metric_card(col3, "Profit Factor", performance_summary.get('Profit Factor', 0.0), is_currency=False, is_ratio=True, precision=2, profit_factor_logic=True)
            display_styled_metric_card(col3, "Max Drawdown", performance_summary.get('Max Drawdown (%)'), is_currency=False, is_percentage=True, mdd_logic=True)

        st.markdown("---") 
        col4, col5, col6 = st.columns(3)
        with col4:
            display_styled_metric_card(col4, "Avg. Trade P&L", performance_summary.get('Average Trade P&L'), is_currency=True)
            display_styled_metric_card(col4, "Sharpe Ratio (Ann.)", performance_summary.get('Sharpe Ratio (Annualized)'), is_currency=False, is_ratio=True, precision=2, sharpe_logic=True)
        with col5:
            display_styled_metric_card(col5, "Avg. Winning Trade", performance_summary.get('Average Winning Trade'), is_currency=True)
            display_styled_metric_card(col5, "Recovery Factor", performance_summary.get('Recovery Factor'), is_currency=False, is_ratio=True, precision=2, recovery_factor_logic=True)
        with col6:
            display_styled_metric_card(col6, "Avg. Losing Trade", performance_summary.get('Average Losing Trade'), is_currency=True) 
            display_styled_metric_card(col6, "Final Capital", performance_summary.get('Final Capital', initial_capital_ui), is_currency=True)

        if performance_summary.get('Total Commissions Paid', 0) > 0 or abs(performance_summary.get('Total Slippage Impact (approx)', 0)) > 0.001:
            st.markdown("---")
            cost_col1, cost_col2 = st.columns(2)
            with cost_col1:
                display_styled_metric_card(cost_col1, "Total Commissions", performance_summary.get('Total Commissions Paid', 0.0), is_currency=True, precision=2)
            with cost_col2:
                display_styled_metric_card(cost_col2, "Total Slippage Impact", performance_summary.get('Total Slippage Impact (approx)', 0.0), is_currency=True, precision=2)
        
        if not trades_df_display.empty:
            st.markdown("---") 
            st.subheader("P&L Temporal Analysis")
            with st.expander("View P&L by Time Periods", expanded=False): # Default to collapsed
                plot_col1, plot_col2, plot_col3 = st.columns(3)
                with plot_col1:
                    fig_pnl_hour = plotting.plot_pnl_by_hour(trades_df_display.copy(), title="P&L by Hour")
                    st.plotly_chart(fig_pnl_hour, use_container_width=True)
                with plot_col2:
                    fig_pnl_day = plotting.plot_pnl_by_day_of_week(trades_df_display.copy(), title="P&L by Day")
                    st.plotly_chart(fig_pnl_day, use_container_width=True)
                with plot_col3:
                    fig_pnl_month = plotting.plot_pnl_by_month(trades_df_display.copy(), title="P&L by Month")
                    st.plotly_chart(fig_pnl_month, use_container_width=True)
        else:
            st.info("No trade data available for P&L temporal analysis.")
        st.markdown("---")

        detail_tabs_names = ["📈 Equity Curve", "📊 Trades on Price", "📋 Trade Log"]
        if not st.session_state.signals.empty and analysis_mode_ui != "Walk-Forward Optimization":
            detail_tabs_names.append("🔍 Generated Signals (Last Run)")
        detail_tabs_names.append("💾 Raw Price Data (Full Period)")
        
        detail_tabs = st.tabs(detail_tabs_names)
        # ... (rest of the detail_tabs content remains the same)
        with detail_tabs[0]: 
            plot_title = "Equity Curve"
            plot_func = plotting.plot_equity_curve
            if analysis_mode_ui == "Walk-Forward Optimization" and "oos_equity_curve" in results_to_display:
                plot_title = "WFO: Aggregated Out-of-Sample Equity"
                plot_func = plotting.plot_wfo_equity_curve
                equity_curve_display = results_to_display["oos_equity_curve"]
            
            if not equity_curve_display.empty:
                st.plotly_chart(plot_func(equity_curve_display, title=plot_title), use_container_width=True)
            else: st.info("Equity curve data not available.")

        with detail_tabs[1]: 
            if not st.session_state.price_data.empty and not trades_df_display.empty:
                st.plotly_chart(plotting.plot_trades_on_price(st.session_state.price_data, trades_df_display.copy(), selected_ticker_name), use_container_width=True)
            else: st.info("Price/trade data not available for plotting trades on price.")

        with detail_tabs[2]: 
            if not trades_df_display.empty:
                float_cols_trades = trades_df_display.select_dtypes(include=['float', 'float64', 'float32']).columns
                format_dict_trades = {col: '{:.2f}' for col in float_cols_trades}
                st.dataframe(trades_df_display.style.format(format_dict_trades), height=300, use_container_width=True)
            else: st.info("No trades executed.")

        idx_offset = 0
        if "🔍 Generated Signals (Last Run)" in detail_tabs_names:
            with detail_tabs[3]:
                if not st.session_state.signals.empty:
                    float_cols_signals = st.session_state.signals.select_dtypes(include=['float', 'float64', 'float32']).columns
                    format_dict_signals = {col: '{:.2f}' for col in float_cols_signals}
                    st.dataframe(st.session_state.signals.style.format(format_dict_signals), height=300, use_container_width=True)
                else: st.info("No signals generated for the last run.")
            idx_offset = 1
        
        with detail_tabs[3 + idx_offset]: 
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

# ... (render_optimization_tab and render_wfo_tab remain largely the same, ensure formatting is applied to dataframes if new metrics propagate there)

def render_optimization_tab(selected_strategy_name, opt_algo_ui, opt_metric_ui, ticker_symbol):
    opt_df_display = st.session_state.optimization_results_df
    if not opt_df_display.empty:
        st.subheader(f"Optimization Results ({selected_strategy_name} - Full Period - {opt_algo_ui})")
        format_dict_opt = {col: '{:.2f}' for col in opt_df_display.select_dtypes(include=['float', 'float64', 'float32']).columns}
        for col in ['Total Trades', 'Max Consecutive Wins', 'Max Consecutive Losses', 'Avg Consecutive Wins', 'Avg Consecutive Losses']: 
            if col in opt_df_display.columns:
                format_dict_opt[col] = '{:.0f}'
        st.dataframe(opt_df_display.style.format(format_dict_opt), height=400, use_container_width=True)
        # ... (rest of the function)
        try:
            csv_opt = opt_df_display.to_csv(index=False).encode('utf-8')
            st.download_button("Download Optimization Results CSV", csv_opt, f"{ticker_symbol}_{selected_strategy_name}_opt_results.csv", 'text/csv', key='dl_opt_csv_tabs_v1')
        except Exception as e:
            logger.error(f"Error generating CSV for optimization results: {e}", exc_info=True)
            st.warning("Could not prepare optimization results for download.")

        if opt_algo_ui == "Grid Search" and 'SL Points' in opt_df_display.columns and 'RRR' in opt_df_display.columns:
            st.markdown(f"##### Optimization Heatmap: {opt_metric_ui} (SL vs RRR - Full Period)")
            try:
                heatmap = plotting.plot_optimization_heatmap(opt_df_display.copy(), 'SL Points', 'RRR', opt_metric_ui)
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
    if st.session_state.wfo_results:
        wfo_res = st.session_state.wfo_results
        wfo_log = wfo_res.get("log", pd.DataFrame())
        wfo_oos_trades = wfo_res.get("oos_trades", pd.DataFrame())

        st.subheader(f"Walk-Forward Optimization Log ({selected_strategy_name} - {opt_algo_ui} for inner opt)")
        if not wfo_log.empty:
            float_cols_wfo = wfo_log.select_dtypes(include=['float', 'float64', 'float32']).columns
            format_dict_wfo = {col: '{:.2f}' for col in float_cols_wfo}
            st.dataframe(wfo_log.style.format(format_dict_wfo), height=300, use_container_width=True)
            # ... (rest of the function)
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
            float_cols_wfo_trades = wfo_oos_trades.select_dtypes(include=['float', 'float64', 'float32']).columns
            format_dict_wfo_trades = {col: '{:.2f}' for col in float_cols_wfo_trades}
            st.dataframe(wfo_oos_trades.style.format(format_dict_wfo_trades), height=300, use_container_width=True)
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
    # Store sidebar_inputs in session_state for the download button if analysis has run
    if st.session_state.backtest_results or st.session_state.run_analysis_clicked_count > 0:
        st.session_state.last_sidebar_inputs = sidebar_inputs.copy()

    analysis_mode_ui = sidebar_inputs["analysis_mode_ui"]
    tabs_to_display = []
    if st.session_state.backtest_results or st.session_state.run_analysis_clicked_count > 0 :
         tabs_to_display.append("📊 Backtest Performance")

    if not st.session_state.optimization_results_df.empty and analysis_mode_ui == "Parameter Optimization":
        tabs_to_display.append("⚙️ Optimization Results (Full Period)")
    if st.session_state.wfo_results and analysis_mode_ui == "Walk-Forward Optimization":
        tabs_to_display.append("🚶 Walk-Forward Analysis")

    if not tabs_to_display and st.session_state.run_analysis_clicked_count == 0:
        st.info("Configure parameters in the sidebar and click 'Run Analysis' to view results.")
        return 

    if not tabs_to_display and st.session_state.run_analysis_clicked_count > 0:
        st.warning("Analysis was run, but no results were generated to display. This could be due to no trades, data issues, or an error. Check logs if errors are suspected.")
        return

    if tabs_to_display:
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
    elif st.session_state.run_analysis_clicked_count > 0 : 
        st.info("Analysis was run. If results are not displayed, it might be due to no trades or data for the selected parameters, or an error during processing. Check logs if errors are suspected.")

