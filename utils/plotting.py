# utils/plotting.py
"""
Functions for creating visualizations using Plotly.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px # Added for heatmap
from config.settings import PLOTLY_TEMPLATE

def plot_equity_curve(equity_curve: pd.Series, title: str = "Equity Curve") -> go.Figure:
    """
    Plots the equity curve.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Equity'))
    
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
    Plots price data (OHLC) with trade entry and exit markers.
    """
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    fig.add_trace(go.Candlestick(x=price_data.index,
                                 open=price_data['Open'],
                                 high=price_data['High'],
                                 low=price_data['Low'],
                                 close=price_data['Close'],
                                 name=f'{symbol} Price'), row=1, col=1)

    if not trades.empty:
        long_entries = trades[trades['Type'] == 'Long']
        fig.add_trace(go.Scatter(x=long_entries['EntryTime'], y=long_entries['EntryPrice'],
                                 mode='markers', name='Long Entry',
                                 marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
        
        short_entries = trades[trades['Type'] == 'Short']
        fig.add_trace(go.Scatter(x=short_entries['EntryTime'], y=short_entries['EntryPrice'],
                                 mode='markers', name='Short Entry',
                                 marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)

        fig.add_trace(go.Scatter(x=trades['ExitTime'], y=trades['ExitPrice'],
                                 mode='markers', name='Exit',
                                 marker=dict(color='blue', size=8, symbol='square')), row=1, col=1)
        
        for _, trade in trades.iterrows():
            fig.add_shape(type="line",
                          x0=trade['EntryTime'], y0=trade['SL'], x1=trade['ExitTime'], y1=trade['SL'],
                          line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash"))
            fig.add_shape(type="line",
                          x0=trade['EntryTime'], y0=trade['TP'], x1=trade['ExitTime'], y1=trade['TP'],
                          line=dict(color="rgba(0,255,0,0.5)", width=1, dash="dash"))

    fig.update_layout(
        title=f'Trades for {symbol}',
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template=PLOTLY_TEMPLATE,
        height=600,
        hovermode="x unified"
    )
    return fig

def plot_optimization_heatmap(
    optimization_results_df: pd.DataFrame,
    param1_name: str, # e.g., 'SL Points'
    param2_name: str, # e.g., 'RRR'
    metric_name: str  # e.g., 'Total P&L'
) -> go.Figure:
    """
    Plots a heatmap of optimization results.

    Args:
        optimization_results_df (pd.DataFrame): DataFrame from run_grid_search.
        param1_name (str): Name of the first parameter (x-axis).
        param2_name (str): Name of the second parameter (y-axis).
        metric_name (str): Name of the performance metric to plot (color scale).

    Returns:
        go.Figure: Plotly figure object.
    """
    if optimization_results_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No Optimization Data for Heatmap", height=400, template=PLOTLY_TEMPLATE)
        return fig

    try:
        # Pivot the table to get it into the right shape for a heatmap
        heatmap_data = optimization_results_df.pivot(
            index=param2_name, # y-axis
            columns=param1_name, # x-axis
            values=metric_name
        )
        heatmap_data = heatmap_data.sort_index(ascending=False) # Typically RRR or similar on y-axis, higher values at top

        fig = px.imshow(
            heatmap_data,
            labels=dict(x=param1_name, y=param2_name, color=metric_name),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            aspect="auto", # Adjust aspect ratio
            color_continuous_scale=px.colors.diverging.RdYlGn if "P&L" in metric_name or "Ratio" in metric_name else px.colors.sequential.Viridis, # Example: Red-Yellow-Green for P&L
            origin='lower' # Or 'upper' depending on preference for y-axis direction
        )
        fig.update_layout(
            title=f'Optimization Heatmap: {metric_name} vs. {param1_name} & {param2_name}',
            xaxis_title=param1_name,
            yaxis_title=param2_name,
            height=600,
            template=PLOTLY_TEMPLATE
        )
        fig.update_xaxes(type='category') # Treat parameters as categories for axis ticks
        fig.update_yaxes(type='category')


    except Exception as e:
        # Handle cases where pivot might fail (e.g., duplicate param combinations, though unlikely with itertools.product)
        # Or if data is not numeric for the metric
        fig = go.Figure()
        fig.update_layout(title=f"Error Generating Heatmap: {e}", height=400, template=PLOTLY_TEMPLATE)
        # You might want to log this error as well
        # from utils.logger import get_logger
        # logger = get_logger(__name__)
        # logger.error(f"Error creating heatmap: {e}", exc_info=True)


    return fig
