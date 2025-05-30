# services/optimizer/metrics_calculator.py
"""
Functions for calculating performance metrics used during optimization.
"""
import pandas as pd
import numpy as np
from config import settings  # Assuming settings.py is accessible
from utils.logger import get_logger

logger = get_logger(__name__)

def _calculate_daily_returns(equity_series: pd.Series) -> pd.Series:
    """
    Calculates daily percentage returns from an equity series.
    Ensures the equity_series has a DatetimeIndex before resampling.
    """
    if equity_series.empty:
        logger.debug("_calculate_daily_returns: Input equity_series is empty.")
        return pd.Series(dtype=float)

    if not isinstance(equity_series.index, pd.DatetimeIndex):
        logger.warning(f"_calculate_daily_returns: equity_series.index is not a DatetimeIndex (type: {type(equity_series.index)}). Attempting conversion.")
        try:
            equity_series.index = pd.to_datetime(equity_series.index)
            if equity_series.index.tz is None: # If still naive, attempt to localize
                logger.warning("_calculate_daily_returns: Index converted to DatetimeIndex but is timezone-naive. Attempting to localize to NY as a default.")
                equity_series.index = equity_series.index.tz_localize(settings.NY_TIMEZONE_STR)
        except Exception as e:
            logger.error(f"_calculate_daily_returns: Failed to convert/localize equity_series.index. Error: {e}", exc_info=True)
            return pd.Series(dtype=float)

    if equity_series.index.tz is not None and equity_series.index.tz.zone != settings.NY_TIMEZONE.zone:
        try:
            equity_series.index = equity_series.index.tz_convert(settings.NY_TIMEZONE_STR)
        except Exception as e:
            logger.error(f"_calculate_daily_returns: Failed to convert equity_series.index to NY timezone. Error: {e}", exc_info=True)
            return pd.Series(dtype=float)

    if equity_series.nunique() <= 1:
        logger.debug("_calculate_daily_returns: Equity series has zero or one unique value, or is too short.")
        if len(equity_series) <= 1: return pd.Series(dtype=float)
        daily_index_for_zero_returns = pd.date_range(start=equity_series.index.min().normalize(), end=equity_series.index.max().normalize(), freq='B')
        if len(daily_index_for_zero_returns) > 1:
             return pd.Series(0.0, index=daily_index_for_zero_returns[1:], dtype=float)
        return pd.Series(dtype=float)

    try:
        daily_equity = equity_series.resample('D').last()
    except TypeError as e:
        logger.error(f"_calculate_daily_returns: TypeError during resample. Index type: {type(equity_series.index)}, Index head: {equity_series.index[:5]}. Error: {e}", exc_info=True)
        return pd.Series(dtype=float)

    if daily_equity.empty:
        logger.debug("_calculate_daily_returns: daily_equity is empty after resample.")
        return pd.Series(dtype=float)

    daily_equity = daily_equity.ffill()
    first_valid_idx = daily_equity.first_valid_index()
    if first_valid_idx is not None:
        daily_equity = daily_equity.loc[first_valid_idx:]
    else:
        logger.debug("_calculate_daily_returns: daily_equity is all NaN after resample and ffill.")
        return pd.Series(dtype=float)
    
    if daily_equity.empty or len(daily_equity) < 2 :
        logger.debug("_calculate_daily_returns: daily_equity became too short after ffill/loc for pct_change.")
        return pd.Series(dtype=float)

    daily_returns = daily_equity.pct_change().fillna(0)
    return daily_returns.iloc[1:]

def calculate_sharpe_ratio(returns_series: pd.Series, risk_free_rate: float = settings.RISK_FREE_RATE, trading_days_per_year: int = settings.TRADING_DAYS_PER_YEAR) -> float:
    """Calculates the annualized Sharpe ratio."""
    if returns_series.empty or len(returns_series) < max(2, settings.MIN_TRADES_FOR_METRICS) or returns_series.std() == 0:
        return np.nan
    excess_returns = returns_series - (risk_free_rate / trading_days_per_year)
    sharpe = excess_returns.mean() / excess_returns.std()
    return sharpe * np.sqrt(trading_days_per_year)

def calculate_sortino_ratio(returns_series: pd.Series, risk_free_rate: float = settings.RISK_FREE_RATE, trading_days_per_year: int = settings.TRADING_DAYS_PER_YEAR) -> float:
    """Calculates the annualized Sortino ratio."""
    if returns_series.empty or len(returns_series) < max(2, settings.MIN_TRADES_FOR_METRICS):
        return np.nan
    
    target_return = risk_free_rate / trading_days_per_year
    excess_returns = returns_series - target_return
    downside_returns = excess_returns[excess_returns < 0]

    if downside_returns.empty or downside_returns.std() == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0 if excess_returns.mean() == 0 else np.nan
        
    downside_std = downside_returns.std()
    if downside_std == 0 :
        return np.inf if excess_returns.mean() > 0 else 0.0 if excess_returns.mean() == 0 else np.nan

    sortino = excess_returns.mean() / downside_std
    return sortino * np.sqrt(trading_days_per_year)
