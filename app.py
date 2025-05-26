# app.py
"""
Main Streamlit application file for the Gap Guardian Strategy Backtester.
Handles UI, user inputs, and orchestrates the backtesting process.
"""
import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime

# Project imports
from config import settings
from services import data_loader, strategy_engine, backtester
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
        logger.warning(f"CSS file not found at {css_file_path}")
    except Exception as e:
        st.error(f"Error loading CSS: {e}")
        logger.error(f"Error loading CSS from {css_file_path}: {e}", exc_info=True)

load_custom_css("static/style.css")

# --- Application State ---
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'price_data' not in st.session_state:
    st.session_state.price_data = pd.DataFrame()
if 'signals' not in st.session_state:
    st.session_state.signals = pd.DataFrame()

# --- Sidebar for Inputs ---
st.sidebar.header("Backtest Configuration")

selected_ticker_name = st.sidebar.selectbox(
    "Select Symbol:",
    options=list(settings.DEFAULT_TICKERS.keys()),
    index=0,
    help="Choose the financial instrument for backtesting."
)
ticker_symbol = settings.DEFAULT_TICKERS[selected_ticker_name]

# Date input configurations
today = date.today()
# For intraday strategies (like the current "15m" default), limit history
# STRATEGY_TIME_FRAME is fixed in settings, so we can directly apply this logic
is_intraday_strategy = settings.STRATEGY_TIME_FRAME not in ["1d", "1wk", "1mo"]

if is_intraday_strategy:
    min_allowable_start_date_for_ui = today - timedelta(days=settings.MAX_INTRADAY_DAYS - 1)
    default_start_date_value = min_allowable_start_date_for_ui
    date_input_help_suffix = f"Intraday data is limited to the last ~{settings.MAX_INTRADAY_DAYS} days."
    # Ensure default_start_date_value is not after today - 1 day (min possible end date)
    if default_start_date_value >= today:
        default_start_date_value = today - timedelta(days=1) if today > date(1,1,1) else today # Handle edge case
else: # For daily, weekly, monthly - allow more history
    # Example: Allow up to 5 years of history for daily+ strategies
    min_allowable_start_date_for_ui = today - timedelta(days=365 * 5)
    default_start_date_value = today - timedelta(days=30) # Default to last 30 days for daily
    date_input_help_suffix = "Select the historical period for backtesting."


# Ensure default_start_date_value is not in the future relative to today or a sensible past date
if default_start_date_value >= today :
    default_start_date_value = today - timedelta(days=1 if today > min_allowable_start_date_for_ui else 0)
if default_start_date_value < min_allowable_start_date_for_ui:
     default_start_date_value = min_allowable_start_date_for_ui


start_date = st.sidebar.date_input(
    "Start Date:",
    value=default_start_date_value,
    min_value=min_allowable_start_date_for_ui,
    max_value=today - timedelta(days=1), # Start date cannot be today or in the future
    help=f"Start date for historical data. {date_input_help_suffix}"
)

# Ensure end_date is after start_date and not in the future
# Default end_date to today, but its min_value must be after start_date
min_end_date_value = start_date + timedelta(days=1) if start_date else today
default_end_date_value = today
if default_end_date_value < min_end_date_value:
    default_end_date_value = min_end_date_value
if default_end_date_value > today: # Should not happen if min_end_date_value logic is correct
    default_end_date_value = today


end_date = st.sidebar.date_input(
    "End Date:",
    value=default_end_date_value,
    min_value=min_end_date_value,
    max_value=today,
    help=f"End date for historical data. {date_input_help_suffix}"
)


initial_capital = st.sidebar.number_input(
    "Initial Capital ($):",
    min_value=1000.0,
    value=settings.DEFAULT_INITIAL_CAPITAL,
    step=1000.0,
    format="%.2f",
    help="Starting account balance for the backtest."
)

risk_per_trade_percent = st.sidebar.number_input(
    "Risk per Trade (%):",
    min_value=0.1,
    max_value=10.0,
    value=settings.DEFAULT_RISK_PER_TRADE_PERCENT,
    step=0.1,
    format="%.1f",
    help="Percentage of current capital to risk on a single trade."
)

stop_loss_points = st.sidebar.number_input(
    "Stop Loss (points/price):",
    min_value=0.1,
    value=settings.DEFAULT_STOP_LOSS_POINTS,
    step=0.1,
    format="%.2f",
    help="Stop loss distance from entry price in asset's price points."
)

rrr = settings.DEFAULT_RRR

# --- Main Application Area ---
st.title(f"🛡️ {settings.APP_TITLE}")
st.markdown("Backtest the Gap Guardian intraday trading strategy. Configure parameters in the sidebar and click 'Run Backtest'.")
st.markdown(f"**Strategy Rules:** Interval: {settings.STRATEGY_TIME_FRAME}. Enter on false break of 9:30 AM NY bar's range during 9:30-11:00 AM NY. Max 1 trade/day. Exit on SL or 1:{int(rrr)} TP.")

if st.sidebar.button("Run Backtest", type="primary", use_container_width=True):
    st.session_state.backtest_results = None
    st.session_state.price_data = pd.DataFrame()
    st.session_state.signals = pd.DataFrame()

    # Final validation before running, though UI should prevent most issues
    if start_date >= end_date:
        st.error(f"Error: Start date ({start_date}) must be before end date ({end_date}). Please correct the selection.")
        logger.error(f"UI Validation Error: Start date {start_date} is not before end date {end_date}.")
    else:
        with st.spinner("Running backtest... Please wait."):
            progress_bar = st.progress(0, text="Initializing backtest...")
            try:
                progress_bar.progress(10, text=f"Fetching data for {selected_ticker_name} ({ticker_symbol})...")
                logger.info(f"Attempting to fetch data for {ticker_symbol} from {start_date} to {end_date} for interval {settings.STRATEGY_TIME_FRAME}")
                price_data_df = data_loader.fetch_historical_data(
                    ticker_symbol, start_date, end_date, settings.STRATEGY_TIME_FRAME
                )
                st.session_state.price_data = price_data_df

                if price_data_df.empty:
                    # data_loader.py now shows st.warning, so we might not need another one here unless it's a different condition
                    # st.warning(f"No price data found for {selected_ticker_name} for the selected period. Cannot proceed.")
                    progress_bar.progress(100, text="Backtest process ended: No data.")
                else:
                    progress_bar.progress(30, text="Data fetched. Generating signals...")
                    logger.info(f"Data fetched successfully: {len(price_data_df)} rows.")
                    signals_df = strategy_engine.generate_signals(
                        price_data_df, stop_loss_points, rrr
                    )
                    st.session_state.signals = signals_df
                    if signals_df.empty:
                        logger.info("No signals generated by the strategy.")
                        st.info("No trading signals were generated for the selected parameters and period.")
                    else:
                        logger.info(f"Generated {len(signals_df)} signals.")
                    progress_bar.progress(60, text="Signals generated. Running backtest simulation...")
                    trades_df, equity_series, performance_metrics = backtester.run_backtest(
                        price_data_df, signals_df, initial_capital,
                        risk_per_trade_percent, stop_loss_points
                    )
                    st.session_state.backtest_results = {
                        "trades": trades_df, "equity_curve": equity_series, "performance": performance_metrics
                    }
                    progress_bar.progress(100, text="Backtest complete!")
                    st.success("Backtest finished successfully!")
            except Exception as e:
                logger.error(f"An error occurred during the backtest: {e}", exc_info=True)
                st.error(f"An critical error occurred during the backtest execution: {e}")
                if 'progress_bar' in locals(): progress_bar.progress(100, text="Backtest failed due to an error.")

# --- Display Results ---
if st.session_state.backtest_results:
    results = st.session_state.backtest_results
    performance = results["performance"]
    trades = results["trades"]
    equity_curve = results["equity_curve"]
    price_data_display = st.session_state.price_data
    signals_display = st.session_state.signals

    st.subheader("Backtest Performance Summary")

    POSITIVE_COLOR = settings.POSITIVE_METRIC_COLOR
    NEGATIVE_COLOR = settings.NEGATIVE_METRIC_COLOR
    NEUTRAL_COLOR = settings.NEUTRAL_METRIC_COLOR

    def format_metric_display(value, precision=2, is_currency=True, is_percentage=False):
        if pd.isna(value) or value is None: return "N/A"
        if is_currency: return f"${value:,.{precision}f}"
        if is_percentage: return f"{value:.{precision}f}%"
        return f"{value:,.{precision}f}" if isinstance(value, float) else str(value)

    def display_styled_metric(column, label, value, raw_value_for_coloring,
                              is_currency=True, is_percentage=False, precision=2,
                              profit_factor_logic=False, max_drawdown_logic=False):
        formatted_value = format_metric_display(value, precision, is_currency, is_percentage)
        color = NEUTRAL_COLOR
        if not (pd.isna(raw_value_for_coloring) or raw_value_for_coloring is None):
            if profit_factor_logic:
                if raw_value_for_coloring > 1: color = POSITIVE_COLOR
                elif raw_value_for_coloring < 1 and raw_value_for_coloring != 0 : color = NEGATIVE_COLOR
                elif raw_value_for_coloring == 0 and performance.get('Gross Profit', 0) == 0 and performance.get('Gross Loss', 0) == 0: color = NEUTRAL_COLOR
                elif raw_value_for_coloring == 0 : color = NEGATIVE_COLOR
            elif max_drawdown_logic:
                if raw_value_for_coloring < 0: color = NEGATIVE_COLOR
                elif raw_value_for_coloring == 0: color = NEUTRAL_COLOR
            else:
                if raw_value_for_coloring > 0: color = POSITIVE_COLOR
                elif raw_value_for_coloring < 0: color = NEGATIVE_COLOR
        column.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="color: {color};">{formatted_value}</div></div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        display_styled_metric(col1, "Total P&L", performance.get('Total P&L'), performance.get('Total P&L'))
        display_styled_metric(col1, "Final Capital", performance.get('Final Capital', initial_capital), performance.get('Final Capital', initial_capital), is_currency=True)
        display_styled_metric(col1, "Max Drawdown", performance.get('Max Drawdown (%)'), performance.get('Max Drawdown (%)'), is_currency=False, is_percentage=True, max_drawdown_logic=True)
    with col2:
        display_styled_metric(col2, "Total Trades", int(performance.get('Total Trades', 0)), int(performance.get('Total Trades', 0)), is_currency=False, is_percentage=False)
        display_styled_metric(col2, "Win Rate", performance.get('Win Rate', 0), performance.get('Win Rate', 0), is_currency=False, is_percentage=True)
        display_styled_metric(col2, "Profit Factor", performance.get('Profit Factor', 0), performance.get('Profit Factor', 0), is_currency=False, precision=2, profit_factor_logic=True)
    with col3:
        display_styled_metric(col3, "Avg. Trade P&L", performance.get('Average Trade P&L'), performance.get('Average Trade P&L'))
        display_styled_metric(col3, "Avg. Winning Trade", performance.get('Average Winning Trade'), performance.get('Average Winning Trade'))
        display_styled_metric(col3, "Avg. Losing Trade", performance.get('Average Losing Trade'), performance.get('Average Losing Trade'))

    tab_equity, tab_trades_chart, tab_trades_log, tab_signals_log, tab_raw_data = st.tabs(["📈 Equity Curve", "📊 Trades on Price", "📋 Trade Log", "🔍 Generated Signals", "💾 Raw Price Data"])
    with tab_equity:
        if not equity_curve.empty: st.plotly_chart(plotting.plot_equity_curve(equity_curve), use_container_width=True)
        else: st.info("Equity curve is not available.")
    with tab_trades_chart:
        if not price_data_display.empty and not trades.empty : st.plotly_chart(plotting.plot_trades_on_price(price_data_display, trades, selected_ticker_name), use_container_width=True)
        elif trades.empty and not price_data_display.empty: st.info("No trades were executed to plot.")
        elif price_data_display.empty: st.info("Price data is not available for plotting.")
        else: st.info("Price data or trade data not available for plotting.")
    with tab_trades_log:
        if not trades.empty:
            float_cols = trades.select_dtypes(include='float').columns
            format_dict = {col: '{:.2f}' for col in float_cols}
            st.dataframe(trades.style.format(format_dict), use_container_width=True)
        else: st.info("No trades were executed.")
    with tab_signals_log:
        if not signals_display.empty:
            st.markdown("These are raw signals *before* backtesting simulation.")
            float_cols_signals = signals_display.select_dtypes(include='float').columns
            format_dict_signals = {col: '{:.2f}' for col in float_cols_signals}
            st.dataframe(signals_display.style.format(format_dict_signals), use_container_width=True)
        else: st.info("No signals were generated by the strategy engine.")
    with tab_raw_data:
        if not price_data_display.empty:
            st.markdown(f"Raw OHLCV price data for **{selected_ticker_name}** ({len(price_data_display)} rows).")
            st.dataframe(price_data_display.head(), height=300, use_container_width=True)
            csv_data = price_data_display.to_csv(index=True).encode('utf-8')
            st.download_button(label="Download Full Price Data as CSV", data=csv_data, file_name=f"{ticker_symbol}_price_data_{start_date}_to_{end_date}.csv", mime='text/csv', key='download-csv')
        else: st.info("Raw price data is not available.")

elif st.sidebar.button("Clear Results", use_container_width=True, key="clear_results_button"):
    st.session_state.backtest_results = None
    st.session_state.price_data = pd.DataFrame()
    st.session_state.signals = pd.DataFrame()
    st.info("Results cleared. Configure and run a new backtest.")
    st.experimental_rerun()
else:
    if 'backtest_results' not in st.session_state or st.session_state.backtest_results is None:
        st.info("Configure parameters in the sidebar and click 'Run Backtest' to see results.")

st.sidebar.markdown("---")
st.sidebar.info(f"App Version: 0.1.3 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.sidebar.caption("Disclaimer: This is a financial modeling tool. Past performance is not indicative of future results.")

