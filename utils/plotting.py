# utils/plotting.py
"""
Functions for creating visualizations using Plotly.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from config.settings import PLOTLY_TEMPLATE

def plot_equity_curve(equity_curve: pd.Series, title: str = "Equity Curve") -> go.Figure:
    fig = go.Figure()
    if not equity_curve.empty:
        fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve, mode='lines', name='Equity'))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Equity", template=PLOTLY_TEMPLATE, height=500, hovermode="x unified")
    return fig

def plot_wfo_equity_curve(
    chained_oos_equity: pd.Series,
    title: str = "Walk-Forward Out-of-Sample Equity Curve"
) -> go.Figure:
    """Plots the chained out-of-sample equity curve from WFO."""
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
    # ... (existing code for plot_trades_on_price, no changes needed here for WFO directly) ...
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    if not price_data.empty:
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
        
        # SL/TP lines can make the plot very busy with many WFO trades, consider making this optional
        # for _, trade in trades.iterrows():
        #     fig.add_shape(type="line", x0=trade['EntryTime'], y0=trade['SL'], x1=trade['ExitTime'], y1=trade['SL'], line=dict(color="rgba(255,0,0,0.3)", width=1, dash="dash"))
        #     fig.add_shape(type="line", x0=trade['EntryTime'], y0=trade['TP'], x1=trade['ExitTime'], y1=trade['TP'], line=dict(color="rgba(0,255,0,0.3)", width=1, dash="dash"))

    fig.update_layout(title=f'Trades for {symbol}', xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False, template=PLOTLY_TEMPLATE, height=600, hovermode="x unified")
    return fig


def plot_optimization_heatmap(
    optimization_results_df: pd.DataFrame,
    param1_name: str, param2_name: str, metric_name: str
) -> go.Figure:
    # ... (existing code for plot_optimization_heatmap, minor robustness for empty data) ...
    if optimization_results_df.empty or not all(p in optimization_results_df.columns for p in [param1_name, param2_name, metric_name]):
        fig = go.Figure()
        fig.update_layout(title=f"Insufficient Data for Heatmap ({metric_name})", height=400, template=PLOTLY_TEMPLATE)
        fig.add_annotation(text="Not enough data or missing columns for heatmap.", showarrow=False)
        return fig
    try:
        heatmap_data = optimization_results_df.pivot(index=param2_name, columns=param1_name, values=metric_name)
        heatmap_data = heatmap_data.sort_index(ascending=False)
        fig = px.imshow(heatmap_data, labels=dict(x=param1_name, y=param2_name, color=metric_name),
                        x=heatmap_data.columns, y=heatmap_data.index, aspect="auto",
                        color_continuous_scale=px.colors.diverging.RdYlGn if "P&L" in metric_name or "Ratio" in metric_name or "Factor" in metric_name else px.colors.sequential.Viridis,
                        origin='lower')
        fig.update_layout(title=f'Optimization Heatmap: {metric_name} vs. {param1_name} & {param2_name}',
                          xaxis_title=param1_name, yaxis_title=param2_name, height=600, template=PLOTLY_TEMPLATE)
        fig.update_xaxes(type='category', tickvals=heatmap_data.columns, ticktext=[f"{x:.2f}" for x in heatmap_data.columns])
        fig.update_yaxes(type='category', tickvals=heatmap_data.index, ticktext=[f"{y:.1f}" for y in heatmap_data.index])
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Error Generating Heatmap: {e}", height=400, template=PLOTLY_TEMPLATE)
        logger.error(f"Error creating heatmap: {e}", exc_info=True)
    return fig
