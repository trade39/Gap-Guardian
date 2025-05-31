# utils/plotting.py
"""
Functions for creating visualizations using Plotly.
Handles duplicate entries in optimization results for heatmap generation.
Includes P&L by time period visualizations.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from config.settings import PLOTLY_TEMPLATE # Assuming PLOTLY_TEMPLATE is defined in settings
from utils.logger import get_logger

# Instantiate logger
logger = get_logger(__name__)

def plot_equity_curve(equity_curve: pd.Series, title: str = "Equity Curve") -> go.Figure:
    """
    Plots the equity curve.

    Args:
        equity_curve (pd.Series): Series containing equity values over time.
        title (str): Title of the plot.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = go.Figure()
    if not equity_curve.empty:
        fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Equity'))
    else:
        fig.add_annotation(text="No equity data to display.", showarrow=False, yshift=10)
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Equity", template=PLOTLY_TEMPLATE, height=500, hovermode="x unified")
    return fig

def plot_wfo_equity_curve(
    chained_oos_equity: pd.Series,
    title: str = "Walk-Forward Out-of-Sample Equity Curve"
) -> go.Figure:
    """
    Plots the chained out-of-sample equity curve from Walk-Forward Optimization.

    Args:
        chained_oos_equity (pd.Series): Series containing chained OOS equity values.
        title (str): Title of the plot.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = go.Figure()
    if not chained_oos_equity.empty:
        fig.add_trace(go.Scatter(x=chained_oos_equity.index, y=chained_oos_equity, mode='lines', name='WFO OOS Equity'))
    else:
        # Display a message if no data
        fig.add_annotation(text="No out-of-sample equity data to display.", showarrow=False, yshift=10)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Equity",
        template=PLOTLY_TEMPLATE,
        height=500,
        hovermode="x unified"
    )
    return fig


def plot_trades_on_price(price_data: pd.DataFrame, trades: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Plots trades (entry and exit points) overlaid on the price candlestick chart.
    Uses 'ActualEntryPrice' for plotting entries.

    Args:
        price_data (pd.DataFrame): OHLC price data.
        trades (pd.DataFrame): DataFrame of executed trades. Must contain 'ActualEntryPrice', 
                               'EntryTime', 'ExitTime', 'ExitPrice', 'Type'.
        symbol (str): The financial symbol being plotted.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    if not price_data.empty:
        fig.add_trace(go.Candlestick(x=price_data.index,
                                    open=price_data['Open'],
                                    high=price_data['High'],
                                    low=price_data['Low'],
                                    close=price_data['Close'],
                                    name=f'{symbol} Price'), row=1, col=1)
    else:
        fig.add_annotation(text="No price data to display for trades overlay.", showarrow=False, yshift=10)


    if not trades.empty:
        trades_plot_df = trades.copy()
        # Ensure 'EntryTime' and 'ExitTime' are datetime objects
        if 'EntryTime' in trades_plot_df.columns and not pd.api.types.is_datetime64_any_dtype(trades_plot_df['EntryTime']):
            trades_plot_df['EntryTime'] = pd.to_datetime(trades_plot_df['EntryTime'])
        if 'ExitTime' in trades_plot_df.columns and not pd.api.types.is_datetime64_any_dtype(trades_plot_df['ExitTime']):
            trades_plot_df['ExitTime'] = pd.to_datetime(trades_plot_df['ExitTime'])

        # Check for required columns for plotting trades
        required_trade_cols = ['EntryTime', 'ActualEntryPrice', 'Type', 'ExitTime', 'ExitPrice']
        missing_cols = [col for col in required_trade_cols if col not in trades_plot_df.columns]
        if missing_cols:
            logger.error(f"Plot Trades: Trades DataFrame is missing required columns: {missing_cols}. Cannot plot trades accurately.")
            fig.add_annotation(text=f"Trade data incomplete (missing: {', '.join(missing_cols)}). Cannot plot trades.", showarrow=False, yshift=-20)
            fig.update_layout(title=f'Trades for {symbol} (Data Incomplete)', xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False, template=PLOTLY_TEMPLATE, height=600, hovermode="x unified")
            return fig


        long_entries = trades_plot_df[trades_plot_df['Type'] == 'Long']
        if not long_entries.empty:
            fig.add_trace(go.Scatter(x=long_entries['EntryTime'], y=long_entries['ActualEntryPrice'], # Changed to ActualEntryPrice
                                     mode='markers', name='Long Entry',
                                     marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
        
        short_entries = trades_plot_df[trades_plot_df['Type'] == 'Short']
        if not short_entries.empty:
            fig.add_trace(go.Scatter(x=short_entries['EntryTime'], y=short_entries['ActualEntryPrice'], # Changed to ActualEntryPrice
                                     mode='markers', name='Short Entry',
                                     marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)
        
        # Exit markers (assuming 'ExitTime' and 'ExitPrice' are correctly populated)
        if 'ExitTime' in trades_plot_df.columns and 'ExitPrice' in trades_plot_df.columns:
             fig.add_trace(go.Scatter(x=trades_plot_df['ExitTime'], y=trades_plot_df['ExitPrice'],
                                     mode='markers', name='Exit',
                                     marker=dict(color='blue', size=8, symbol='square')), row=1, col=1)
    else:
        logger.info("Plot Trades: No trades to plot.")
        # Optionally add an annotation if no trades
        # fig.add_annotation(text="No trades executed.", showarrow=False, yshift=0)


    fig.update_layout(title=f'Trades for {symbol}', xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False, template=PLOTLY_TEMPLATE, height=600, hovermode="x unified")
    return fig


def plot_optimization_heatmap(
    optimization_results_df: pd.DataFrame,
    param1_name: str,
    param2_name: str,
    metric_name: str
) -> go.Figure:
    """
    Generates a heatmap for visualizing optimization results between two parameters.
    Handles duplicate index/column entries by averaging the metric.
    """
    fig = go.Figure() # Initialize fig to ensure it's always defined
    if optimization_results_df.empty or not all(p in optimization_results_df.columns for p in [param1_name, param2_name, metric_name]):
        logger.warning(f"Insufficient data or missing columns for heatmap. Params: {param1_name}, {param2_name}. Metric: {metric_name}. Columns available: {optimization_results_df.columns.tolist()}")
        fig.update_layout(title=f"Insufficient Data for Heatmap ({metric_name})", height=400, template=PLOTLY_TEMPLATE)
        fig.add_annotation(text="Not enough data or missing columns for heatmap generation.", showarrow=False)
        return fig

    try:
        logger.debug(f"Original optimization_results_df for heatmap (head):\n{optimization_results_df[[param1_name, param2_name, metric_name]].head()}")

        cleaned_df = optimization_results_df.dropna(subset=[param1_name, param2_name, metric_name])
        
        if cleaned_df.empty:
            logger.warning("DataFrame became empty after attempting to clean NaNs from grouping columns for heatmap.")
            fig.update_layout(title=f"No Valid Data for Heatmap ({metric_name}) after NaN cleaning", height=400, template=PLOTLY_TEMPLATE)
            fig.add_annotation(text="Data became empty after cleaning NaNs from axis parameters.", showarrow=False)
            return fig

        aggregated_df = cleaned_df.groupby([param2_name, param1_name], as_index=False)[metric_name].mean()
        logger.debug(f"Aggregated_df for heatmap (head):\n{aggregated_df.head()}")

        heatmap_data = aggregated_df.pivot(index=param2_name, columns=param1_name, values=metric_name)
        heatmap_data = heatmap_data.sort_index(ascending=False) 
        logger.debug(f"Pivoted heatmap_data (head):\n{heatmap_data.head()}")
        
        fig = px.imshow(heatmap_data, 
                        labels=dict(x=param1_name, y=param2_name, color=metric_name),
                        x=heatmap_data.columns, 
                        y=heatmap_data.index, 
                        aspect="auto",
                        color_continuous_scale=px.colors.diverging.RdYlGn if "P&L" in metric_name or "Ratio" in metric_name or "Factor" in metric_name else px.colors.sequential.Viridis,
                        origin='lower'
                       )
        
        fig.update_layout(title=f'Optimization Heatmap: {metric_name} vs. {param1_name} & {param2_name}',
                          xaxis_title=param1_name, yaxis_title=param2_name, height=600, template=PLOTLY_TEMPLATE)
        
        fig.update_xaxes(type='category', tickvals=heatmap_data.columns, ticktext=[f"{x:.2f}" if isinstance(x, float) else str(x) for x in heatmap_data.columns])
        fig.update_yaxes(type='category', tickvals=heatmap_data.index, ticktext=[f"{y:.1f}" if isinstance(y, float) else str(y) for y in heatmap_data.index])

    except Exception as e:
        logger.error(f"Error creating heatmap for metric '{metric_name}' with params '{param1_name}', '{param2_name}': {e}", exc_info=True)
        fig.update_layout(title=f"Error Generating Heatmap: Review Logs", height=400, template=PLOTLY_TEMPLATE)
        fig.add_annotation(text=f"Could not generate heatmap. Details: {str(e)}", showarrow=False)
    return fig

def plot_pnl_by_hour(trades_df: pd.DataFrame, title: str = "P&L by Hour of Day") -> go.Figure:
    """
    Plots the aggregated P&L by hour of the day.
    Args:
        trades_df (pd.DataFrame): DataFrame of executed trades with 'P&L' and 'EntryTime'.
                                  'EntryTime' is assumed to be timezone-aware (e.g., NY time).
        title (str): Title of the plot.
    Returns:
        go.Figure: Plotly bar chart figure object.
    """
    fig = go.Figure()
    if trades_df.empty or 'P&L' not in trades_df.columns or 'EntryTime' not in trades_df.columns:
        logger.warning("P&L by Hour: Trades data is empty or missing required columns (P&L, EntryTime).")
        fig.update_layout(title=title, height=300, template=PLOTLY_TEMPLATE)
        fig.add_annotation(text="No trade data available for P&L by Hour.", showarrow=False)
        return fig

    trades_plot_df = trades_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(trades_plot_df['EntryTime']):
        trades_plot_df['EntryTime'] = pd.to_datetime(trades_plot_df['EntryTime'])
    
    if trades_plot_df['EntryTime'].dt.tz is None:
        logger.info("P&L by Hour: 'EntryTime' is timezone-naive. Assuming UTC and converting to NY for display.")
        try:
            trades_plot_df['EntryTime'] = trades_plot_df['EntryTime'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        except Exception as e:
            logger.error(f"P&L by Hour: Error localizing/converting naive EntryTime: {e}. Proceeding with naive time.", exc_info=True)

    trades_plot_df['Hour'] = trades_plot_df['EntryTime'].dt.hour
    pnl_by_hour = trades_plot_df.groupby('Hour')['P&L'].sum().reset_index()
    
    all_hours = pd.DataFrame({'Hour': range(24)})
    pnl_by_hour = pd.merge(all_hours, pnl_by_hour, on='Hour', how='left')
    pnl_by_hour['P&L'] = pnl_by_hour['P&L'].fillna(0) # Fill P&L NaNs with 0
    pnl_by_hour = pnl_by_hour.sort_values(by='Hour')

    fig = px.bar(pnl_by_hour, x='Hour', y='P&L', title=title, labels={'Hour': 'Hour of Day (Entry Time)', 'P&L': 'Total P&L'})
    fig.update_layout(template=PLOTLY_TEMPLATE, height=400, xaxis_type='category')
    fig.update_traces(marker_color=['red' if p < 0 else 'green' for p in pnl_by_hour['P&L']])
    return fig

def plot_pnl_by_day_of_week(trades_df: pd.DataFrame, title: str = "P&L by Day of Week") -> go.Figure:
    """
    Plots the aggregated P&L by day of the week.
    Args:
        trades_df (pd.DataFrame): DataFrame of executed trades with 'P&L' and 'EntryTime'.
        title (str): Title of the plot.
    Returns:
        go.Figure: Plotly bar chart figure object.
    """
    fig = go.Figure()
    if trades_df.empty or 'P&L' not in trades_df.columns or 'EntryTime' not in trades_df.columns:
        logger.warning("P&L by Day: Trades data is empty or missing required columns (P&L, EntryTime).")
        fig.update_layout(title=title, height=300, template=PLOTLY_TEMPLATE)
        fig.add_annotation(text="No trade data available for P&L by Day of Week.", showarrow=False)
        return fig

    trades_plot_df = trades_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(trades_plot_df['EntryTime']):
        trades_plot_df['EntryTime'] = pd.to_datetime(trades_plot_df['EntryTime'])

    if trades_plot_df['EntryTime'].dt.tz is None:
        logger.info("P&L by Day: 'EntryTime' is timezone-naive. Assuming UTC and converting to NY for display.")
        try:
            trades_plot_df['EntryTime'] = trades_plot_df['EntryTime'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        except Exception as e:
            logger.error(f"P&L by Day: Error localizing/converting naive EntryTime: {e}. Proceeding with naive time.", exc_info=True)


    trades_plot_df['DayOfWeek'] = trades_plot_df['EntryTime'].dt.day_name()
    pnl_by_day_grouped = trades_plot_df.groupby('DayOfWeek')['P&L'].sum().reset_index()
    
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    all_days_df = pd.DataFrame({'DayOfWeek': pd.Categorical(days_order, categories=days_order, ordered=True)})
    
    pnl_by_day = pd.merge(all_days_df, pnl_by_day_grouped, on='DayOfWeek', how='left')
    
    pnl_by_day['P&L'] = pnl_by_day['P&L'].fillna(0)


    fig = px.bar(pnl_by_day, x='DayOfWeek', y='P&L', title=title, labels={'DayOfWeek': 'Day of Week (Entry Time)', 'P&L': 'Total P&L'})
    fig.update_layout(template=PLOTLY_TEMPLATE, height=400)
    fig.update_traces(marker_color=['red' if p < 0 else 'green' for p in pnl_by_day['P&L']])
    return fig

def plot_pnl_by_month(trades_df: pd.DataFrame, title: str = "P&L by Month") -> go.Figure:
    """
    Plots the aggregated P&L by month.
    Args:
        trades_df (pd.DataFrame): DataFrame of executed trades with 'P&L' and 'EntryTime'.
        title (str): Title of the plot.
    Returns:
        go.Figure: Plotly bar chart figure object.
    """
    fig = go.Figure()
    if trades_df.empty or 'P&L' not in trades_df.columns or 'EntryTime' not in trades_df.columns:
        logger.warning("P&L by Month: Trades data is empty or missing required columns (P&L, EntryTime).")
        fig.update_layout(title=title, height=300, template=PLOTLY_TEMPLATE)
        fig.add_annotation(text="No trade data available for P&L by Month.", showarrow=False)
        return fig

    trades_plot_df = trades_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(trades_plot_df['EntryTime']):
        trades_plot_df['EntryTime'] = pd.to_datetime(trades_plot_df['EntryTime'])

    if trades_plot_df['EntryTime'].dt.tz is None:
        logger.info("P&L by Month: 'EntryTime' is timezone-naive. Assuming UTC and converting to NY for display.")
        try:
            trades_plot_df['EntryTime'] = trades_plot_df['EntryTime'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        except Exception as e:
            logger.error(f"P&L by Month: Error localizing/converting naive EntryTime: {e}. Proceeding with naive time.", exc_info=True)

    trades_plot_df['MonthSort'] = trades_plot_df['EntryTime'].dt.strftime('%Y-%m') 
    trades_plot_df['MonthDisplay'] = trades_plot_df['EntryTime'].dt.strftime('%Y-%m (%B)') 
    
    pnl_by_month = trades_plot_df.groupby(['MonthSort', 'MonthDisplay'])['P&L'].sum().reset_index()
    pnl_by_month = pnl_by_month.sort_values(by='MonthSort')


    fig = px.bar(pnl_by_month, x='MonthDisplay', y='P&L', title=title, labels={'MonthDisplay': 'Month (Entry Time)', 'P&L': 'Total P&L'})
    fig.update_layout(template=PLOTLY_TEMPLATE, height=400, xaxis_tickangle=-45)
    if not pnl_by_month.empty:
        fig.update_traces(marker_color=['red' if p < 0 else 'green' for p in pnl_by_month['P&L']])
    return fig
