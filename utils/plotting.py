# utils/plotting.py
"""
Functions for creating visualizations using Plotly.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config.settings import PLOTLY_TEMPLATE

def plot_equity_curve(equity_curve: pd.Series, title: str = "Equity Curve") -> go.Figure:
    """
    Plots the equity curve.

    Args:
        equity_curve (pd.Series): Series with datetime index and equity values.
        title (str): Title of the plot.

    Returns:
        go.Figure: Plotly figure object.
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

    Args:
        price_data (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' columns and DatetimeIndex.
        trades (pd.DataFrame): DataFrame with trade details including 'EntryTime', 'EntryPrice', 
                               'ExitTime', 'ExitPrice', 'Type' ('Long'/'Short'), 'SL', 'TP'.
        symbol (str): The symbol being plotted.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # Plot OHLC data
    fig.add_trace(go.Candlestick(x=price_data.index,
                                 open=price_data['Open'],
                                 high=price_data['High'],
                                 low=price_data['Low'],
                                 close=price_data['Close'],
                                 name=f'{symbol} Price'), row=1, col=1)

    # Plot trades
    if not trades.empty:
        # Long entries
        long_entries = trades[trades['Type'] == 'Long']
        fig.add_trace(go.Scatter(x=long_entries['EntryTime'], y=long_entries['EntryPrice'],
                                 mode='markers', name='Long Entry',
                                 marker=dict(color='green', size=10, symbol='triangle-up')), row=1, col=1)
        
        # Short entries
        short_entries = trades[trades['Type'] == 'Short']
        fig.add_trace(go.Scatter(x=short_entries['EntryTime'], y=short_entries['EntryPrice'],
                                 mode='markers', name='Short Entry',
                                 marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)

        # Exits (could be SL or TP)
        # For simplicity, mark all exits with one symbol. Can be differentiated further.
        fig.add_trace(go.Scatter(x=trades['ExitTime'], y=trades['ExitPrice'],
                                 mode='markers', name='Exit',
                                 marker=dict(color='blue', size=8, symbol='square')), row=1, col=1)
        
        # Plot SL and TP lines for each trade (optional, can make plot busy)
        for _, trade in trades.iterrows():
            # SL line
            fig.add_shape(type="line",
                          x0=trade['EntryTime'], y0=trade['SL'], x1=trade['ExitTime'], y1=trade['SL'],
                          line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash"),
                          name=f"SL {trade.name}")
            # TP line
            fig.add_shape(type="line",
                          x0=trade['EntryTime'], y0=trade['TP'], x1=trade['ExitTime'], y1=trade['TP'],
                          line=dict(color="rgba(0,255,0,0.5)", width=1, dash="dash"),
                          name=f"TP {trade.name}")


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
