# app.py
"""
Main Streamlit application file for the Gap Guardian Strategy Backtester.
Handles UI, user inputs, and orchestrates the backtesting process.
"""
import streamlit as st
import pandas as pd
from datetime import date, timedelta, datetime

# Project imports (ensure PYTHONPATH is set up if running locally outside an IDE that handles it)
from config import settings
from services import data_loader, strategy_engine, backtester
from utils import plotting, logger as app_logger # Renamed to avoid conflict with streamlit.logger
# Removed incorrect import: from static.style import load_css

# Initialize logger
logger = app_logger.get_logger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="🛡️", # Shield icon for "Guardian"
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Custom CSS ---
# Function to load CSS (can be moved to a utils/ui.py or similar)
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
# Use st.session_state to store results and prevent re-computation on every interaction
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
    index=0, # Default to the first ticker
    help="Choose the financial instrument for backtesting."
)
ticker_symbol = settings.DEFAULT_TICKERS[selected_ticker_name]

# Date inputs
max_end_date = date.today()
# Ensure default_start_date does not precede min_start_date for yfinance intraday limits
default_start_date_candidate = max_end_date - timedelta(days=settings.MAX_INTRADAY_DAYS - 1)

start_date = st.sidebar.date_input(
    "Start Date:",
    value=default_start_date_candidate,
    # min_value can be further in the past, data_loader will adjust for intraday if necessary
    min_value=max_end_date - timedelta(days=365*2), 
    max_value=max_end_date - timedelta(days=1),
    help=f"Start date for historical data. Note: Intraday data (e.g., 15-min) is typically limited by yfinance to the last {settings.MAX_INTRADAY_DAYS} days. The data loader will adjust if an older start date is selected for intraday data."
)
end_date = st.sidebar.date_input(
    "End Date:",
    value=max_end_date,
    min_value=start_date + timedelta(days=1) if start_date else max_end_date - timedelta(days=settings.MAX_INTRADAY_DAYS - 2), # Ensure end_date is after start_date
    max_value=max_end_date,
    help="End date for historical data."
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
    max_value=10.0, # Increased sensible max risk slightly
    value=settings.DEFAULT_RISK_PER_TRADE_PERCENT,
    step=0.1,
    format="%.1f", # Adjusted format for typical risk percentages
    help="Percentage of current capital to risk on a single trade."
)

stop_loss_points = st.sidebar.number_input(
    "Stop Loss (points/price):",
    min_value=0.1, # Allow smaller SL for certain assets
    value=settings.DEFAULT_STOP_LOSS_POINTS, 
    step=0.1, # Finer step for precision
    format="%.2f", 
    help="Stop loss distance from entry price in asset's price points (e.g., 15 for S&P 500 points, or 0.0015 for EUR/USD)."
)

rrr = settings.DEFAULT_RRR


# --- Main Application Area ---
st.title(f"🛡️ {settings.APP_TITLE}")
st.markdown("Backtest the Gap Guardian intraday trading strategy. Configure parameters in the sidebar and click 'Run Backtest'.")
st.markdown(f"**Strategy Rules:** Enter on false break of 9:30 AM NY bar's range during 9:30-11:00 AM NY. Max 1 trade/day. Exit on SL or 1:{int(rrr)} TP.")


if st.sidebar.button("Run Backtest", type="primary", use_container_width=True):
    st.session_state.backtest_results = None 
    st.session_state.price_data = pd.DataFrame()
    st.session_state.signals = pd.DataFrame()
    
    if start_date >= end_date:
        st.error("Error: Start date must be before end date.")
        logger.error(f"Validation Error: Start date {start_date} is not before end date {end_date}.")
    else:
        # Display a spinner during execution
        with st.spinner("Running backtest... Please wait."):
            progress_bar = st.progress(0, text="Initializing backtest...")
            try:
                # 1. Fetch Data
                progress_bar.progress(10, text=f"Fetching data for {selected_ticker_name} ({ticker_symbol})...")
                logger.info(f"Attempting to fetch data for {ticker_symbol} from {start_date} to {end_date}")
                
                price_data_df = data_loader.fetch_historical_data(
                    ticker_symbol,
                    start_date,
                    end_date,
                    interval=settings.STRATEGY_TIME_FRAME
                )
                st.session_state.price_data = price_data_df

                if price_data_df.empty:
                    st.warning(f"No price data found for {selected_ticker_name} for the selected period. Cannot proceed.")
                    progress_bar.progress(100, text="Backtest failed: No data.")
                    # Use st.stop() carefully as it halts script execution. Ensure UI is updated.
                    # For this flow, letting it proceed to show "no results" might be okay.
                else:
                    progress_bar.progress(30, text="Data fetched. Generating signals...")
                    logger.info(f"Data fetched successfully: {len(price_data_df)} rows.")

                    # 2. Generate Signals (only if data is available)
                    signals_df = strategy_engine.generate_signals(
                        price_data_df,
                        stop_loss_points,
                        rrr
                    )
                    st.session_state.signals = signals_df
                    
                    if signals_df.empty:
                        logger.info("No signals generated by the strategy.")
                        st.info("No trading signals were generated for the selected parameters and period.")
                    else:
                        logger.info(f"Generated {len(signals_df)} signals.")

                    progress_bar.progress(60, text="Signals generated. Running backtest simulation...")
                    
                    # 3. Run Backtest
                    trades_df, equity_series, performance_metrics = backtester.run_backtest(
                        price_data_df, # Pass even if empty, backtester handles it
                        signals_df,    # Pass even if empty
                        initial_capital,
                        risk_per_trade_percent,
                        stop_loss_points 
                    )
                    
                    st.session_state.backtest_results = {
                        "trades": trades_df,
                        "equity_curve": equity_series,
                        "performance": performance_metrics
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
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    # Helper to format metrics safely
    def format_metric(value, precision=2, is_currency=True, is_percentage=False):
        if pd.isna(value) or value is None:
            return "N/A"
        if is_currency:
            return f"${value:,.{precision}f}"
        if is_percentage:
            return f"{value:.{precision}f}%"
        return f"{value:,.{precision}f}" if isinstance(value, float) else str(value)

    with col1:
        st.metric("Total P&L", format_metric(performance.get('Total P&L')))
        st.metric("Final Capital", format_metric(performance.get('Final Capital', initial_capital)))
        st.metric("Max Drawdown", format_metric(performance.get('Max Drawdown (%)'), is_currency=False, is_percentage=True))
    with col2:
        st.metric("Total Trades", str(int(performance.get('Total Trades', 0))))
        st.metric("Win Rate", format_metric(performance.get('Win Rate', 0), is_currency=False, is_percentage=True))
        st.metric("Profit Factor", format_metric(performance.get('Profit Factor', 0), is_currency=False))
    with col3:
        st.metric("Avg. Trade P&L", format_metric(performance.get('Average Trade P&L')))
        st.metric("Avg. Winning Trade", format_metric(performance.get('Average Winning Trade')))
        st.metric("Avg. Losing Trade", format_metric(performance.get('Average Losing Trade'))) # Usually negative

    # Tabs for detailed results
    tab_equity, tab_trades_chart, tab_trades_log, tab_signals_log, tab_raw_data = st.tabs([
        "📈 Equity Curve", "📊 Trades on Price", "📋 Trade Log", "🔍 Generated Signals", "💾 Raw Price Data"
    ])

    with tab_equity:
        if not equity_curve.empty:
            st.plotly_chart(plotting.plot_equity_curve(equity_curve), use_container_width=True)
        else:
            st.info("Equity curve is not available. This can happen if no trades were made or if an error occurred during calculation.")

    with tab_trades_chart:
        if not price_data_display.empty and not trades.empty :
            st.plotly_chart(plotting.plot_trades_on_price(price_data_display, trades, selected_ticker_name), use_container_width=True)
        elif trades.empty and not price_data_display.empty:
            st.info("No trades were executed to plot. Price data is available.")
            # Optionally plot just the price data if no trades
            # fig_price_only = go.Figure()
            # fig_price_only.add_trace(go.Candlestick(x=price_data_display.index, open=price_data_display['Open'], high=price_data_display['High'], low=price_data_display['Low'], close=price_data_display['Close'], name=f'{selected_ticker_name} Price'))
            # fig_price_only.update_layout(title=f'{selected_ticker_name} Price Data (No Trades)', xaxis_rangeslider_visible=False)
            # st.plotly_chart(fig_price_only, use_container_width=True)
        elif price_data_display.empty:
             st.info("Price data is not available for plotting.")
        else: # Should not be reached if logic above is correct
            st.info("Price data or trade data not available for plotting.")
            
    with tab_trades_log:
        if not trades.empty:
            # Improve display of float columns
            float_cols = trades.select_dtypes(include='float').columns
            format_dict = {col: '{:.2f}' for col in float_cols}
            st.dataframe(trades.style.format(format_dict), use_container_width=True)
        else:
            st.info("No trades were executed.")
            
    with tab_signals_log:
        if not signals_display.empty:
            st.markdown("These are the raw signals generated by the strategy engine *before* backtesting simulation (e.g., position sizing, exits).")
            float_cols_signals = signals_display.select_dtypes(include='float').columns
            format_dict_signals = {col: '{:.2f}' for col in float_cols_signals}
            st.dataframe(signals_display.style.format(format_dict_signals), use_container_width=True)
        else:
            st.info("No signals were generated by the strategy engine.")

    with tab_raw_data:
        if not price_data_display.empty:
            st.markdown(f"Displaying raw OHLCV price data for **{selected_ticker_name}** used in the backtest ({len(price_data_display)} rows).")
            st.dataframe(price_data_display.head(), height=300, use_container_width=True) # Show head for brevity
            
            csv_data = price_data_display.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="Download Full Price Data as CSV",
                data=csv_data,
                file_name=f"{ticker_symbol}_price_data_{start_date}_to_{end_date}.csv",
                mime='text/csv',
                key='download-csv' # Add a key for stability
            )
        else:
            st.info("Raw price data is not available.")

elif st.sidebar.button("Clear Results", use_container_width=True, key="clear_results_button"): # Button to clear results manually
    st.session_state.backtest_results = None
    st.session_state.price_data = pd.DataFrame()
    st.session_state.signals = pd.DataFrame()
    st.info("Results cleared. Configure and run a new backtest.")
    st.experimental_rerun() # Rerun to reflect cleared state immediately

else:
    if 'backtest_results' not in st.session_state or st.session_state.backtest_results is None:
        st.info("Configure parameters in the sidebar and click 'Run Backtest' to see results.")

st.sidebar.markdown("---")
st.sidebar.info(f"App Version: 0.1.1 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.sidebar.caption("Disclaimer: This is a financial modeling tool for educational and research purposes. Past performance is not indicative of future results. Always conduct your own due diligence.")
