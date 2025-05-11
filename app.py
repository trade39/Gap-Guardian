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
from static.style import load_css # Assuming you might create a style.py to load css

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
    except Exception as e:
        st.error(f"Error loading CSS: {e}")

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
# Max 60 days for 15m interval from yfinance. Set default range accordingly.
max_end_date = date.today()
default_start_date = max_end_date - timedelta(days=settings.MAX_INTRADAY_DAYS - 1) # e.g., 59 days before today
min_start_date = max_end_date - timedelta(days=settings.MAX_INTRADAY_DAYS -1) # To prevent requesting too old data

start_date = st.sidebar.date_input(
    "Start Date:",
    value=default_start_date,
    min_value=max_end_date - timedelta(days=365*2), # Allow selection further back, data_loader will adjust
    max_value=max_end_date - timedelta(days=1),
    help=f"Start date for historical data. Note: 15-min data is typically limited to the last {settings.MAX_INTRADAY_DAYS} days."
)
end_date = st.sidebar.date_input(
    "End Date:",
    value=max_end_date,
    min_value=start_date + timedelta(days=1), # End date must be after start date
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
    max_value=5.0, # Sensible max risk
    value=settings.DEFAULT_RISK_PER_TRADE_PERCENT,
    step=0.1,
    format="%.2f",
    help="Percentage of current capital to risk on a single trade."
)

stop_loss_points = st.sidebar.number_input(
    "Stop Loss (points/price):",
    min_value=1.0, # Minimum SL of 1 point/dollar
    value=settings.DEFAULT_STOP_LOSS_POINTS, # Default to 15 points
    step=1.0,
    format="%.2f", # Allow fractional points for assets like FX or specific commodities
    help="Stop loss distance from entry price in asset's price points (e.g., 15 for S&P 500 points, or $15 for Gold if 1 point = $1)."
)

# RRR is fixed at 1:3 by strategy, but can be made configurable if desired
rrr = settings.DEFAULT_RRR
# st.sidebar.caption(f"Risk/Reward Ratio: 1:{rrr} (Fixed by Strategy)")


# --- Main Application Area ---
st.title(f"🛡️ {settings.APP_TITLE}")
st.markdown("Backtest the Gap Guardian intraday trading strategy. Configure parameters in the sidebar and click 'Run Backtest'.")
st.markdown(f"**Strategy Rules:** Enter on false break of 9:30 AM NY bar's range during 9:30-11:00 AM NY. Max 1 trade/day. Exit on SL or 1:{int(rrr)} TP.")


if st.sidebar.button("Run Backtest", type="primary", use_container_width=True):
    st.session_state.backtest_results = None # Clear previous results
    st.session_state.price_data = pd.DataFrame()
    st.session_state.signals = pd.DataFrame()
    
    if start_date >= end_date:
        st.error("Error: Start date must be before end date.")
    else:
        progress_bar = st.progress(0, text="Initializing backtest...")
        try:
            # 1. Fetch Data
            progress_bar.progress(10, text=f"Fetching data for {selected_ticker_name} ({ticker_symbol})...")
            logger.info(f"Attempting to fetch data for {ticker_symbol} from {start_date} to {end_date}")
            
            # Convert date objects to datetime for yfinance if needed, though yf handles dates
            # Forcing time to start of day for start_date and end of day for end_date can be more robust with yf
            # However, data_loader handles yfinance's behavior with date objects.
            
            price_data_df = data_loader.fetch_historical_data(
                ticker_symbol,
                start_date, # datetime.combine(start_date, datetime.min.time()),
                end_date, # datetime.combine(end_date, datetime.max.time()),
                interval=settings.STRATEGY_TIME_FRAME
            )
            st.session_state.price_data = price_data_df

            if price_data_df.empty:
                st.warning(f"No price data found for {selected_ticker_name} for the selected period. Cannot proceed.")
                progress_bar.progress(100, text="Backtest failed: No data.")
                st.stop()

            progress_bar.progress(30, text="Data fetched. Generating signals...")
            logger.info(f"Data fetched successfully: {len(price_data_df)} rows.")

            # 2. Generate Signals
            signals_df = strategy_engine.generate_signals(
                price_data_df,
                stop_loss_points,
                rrr
            )
            st.session_state.signals = signals_df
            
            if signals_df.empty:
                logger.info("No signals generated by the strategy.")
                st.info("No trading signals were generated for the selected parameters and period.")
                # Still run backtester to get flat equity curve and zero metrics
                # This is handled by backtester if signals_df is empty
            else:
                 logger.info(f"Generated {len(signals_df)} signals.")

            progress_bar.progress(60, text="Signals generated. Running backtest simulation...")
            
            # 3. Run Backtest
            trades_df, equity_series, performance_metrics = backtester.run_backtest(
                price_data_df,
                signals_df,
                initial_capital,
                risk_per_trade_percent,
                stop_loss_points # Pass the configured SL distance
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
            st.error(f"An error occurred: {e}")
            if progress_bar: progress_bar.progress(100, text="Backtest failed.")


# --- Display Results ---
if st.session_state.backtest_results:
    results = st.session_state.backtest_results
    performance = results["performance"]
    trades = results["trades"]
    equity_curve = results["equity_curve"]
    price_data_display = st.session_state.price_data # For plotting trades on price
    signals_display = st.session_state.signals

    st.subheader("Backtest Performance Summary")
    
    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total P&L ($)", f"{performance.get('Total P&L', 0):,.2f}")
        st.metric("Final Capital ($)", f"{performance.get('Final Capital', initial_capital):,.2f}")
        st.metric("Max Drawdown (%)", f"{performance.get('Max Drawdown (%)', 0):.2f}%")
    with col2:
        st.metric("Total Trades", f"{performance.get('Total Trades', 0)}")
        st.metric("Win Rate (%)", f"{performance.get('Win Rate', 0):.2f}%")
        st.metric("Profit Factor", f"{performance.get('Profit Factor', 0):.2f}")
    with col3:
        st.metric("Avg. Trade P&L ($)", f"{performance.get('Average Trade P&L', 0):.2f}")
        st.metric("Avg. Winning Trade ($)", f"{performance.get('Average Winning Trade', 0):.2f}")
        st.metric("Avg. Losing Trade ($)", f"{performance.get('Average Losing Trade', 0):.2f}")

    # Tabs for detailed results
    tab_equity, tab_trades_chart, tab_trades_log, tab_signals_log, tab_raw_data = st.tabs([
        "📈 Equity Curve", "📊 Trades on Price", "📋 Trade Log", "🔍 Generated Signals", "💾 Raw Price Data"
    ])

    with tab_equity:
        if not equity_curve.empty:
            st.plotly_chart(plotting.plot_equity_curve(equity_curve), use_container_width=True)
        else:
            st.info("Equity curve is not available (e.g. no trades or error).")

    with tab_trades_chart:
        if not price_data_display.empty and not trades.empty :
            st.plotly_chart(plotting.plot_trades_on_price(price_data_display, trades, selected_ticker_name), use_container_width=True)
        elif trades.empty:
            st.info("No trades were executed to plot.")
        else:
            st.info("Price data or trade data not available for plotting.")
            
    with tab_trades_log:
        if not trades.empty:
            st.dataframe(trades, use_container_width=True)
        else:
            st.info("No trades were executed.")
            
    with tab_signals_log:
        if not signals_display.empty:
            st.markdown("These are the raw signals generated by the strategy engine *before* backtesting simulation (e.g., position sizing, exits).")
            st.dataframe(signals_display, use_container_width=True)
        else:
            st.info("No signals were generated by the strategy engine.")

    with tab_raw_data:
        if not price_data_display.empty:
            st.markdown(f"Displaying raw OHLCV price data for **{selected_ticker_name}** used in the backtest.")
            st.dataframe(price_data_display, height=300, use_container_width=True)
            
            # Option to download data
            csv_data = price_data_display.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="Download Price Data as CSV",
                data=csv_data,
                file_name=f"{ticker_symbol}_price_data_{start_date}_to_{end_date}.csv",
                mime='text/csv',
            )
        else:
            st.info("Raw price data is not available.")

else:
    st.info("Configure parameters in the sidebar and click 'Run Backtest' to see results.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Version 0.1.0 | {datetime.now().year}")
st.sidebar.caption("This is a financial modeling tool. Past performance is not indicative of future results.")

