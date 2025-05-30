# services/optimizer/__init__.py
"""
Makes the 'optimizer' directory a Python package and exposes key optimization functions.
"""
from .metrics_calculator import calculate_sharpe_ratio, calculate_sortino_ratio
from .search_algorithms import run_grid_search, run_random_search
from .wfo_orchestrator import run_walk_forward_optimization

# Functions like _calculate_daily_returns and _run_single_backtest_for_optimization
# are typically kept internal to this package (not listed in __all__) unless
# explicitly needed by external modules.

__all__ = [
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "run_grid_search",
    "run_random_search",
    "run_walk_forward_optimization"
]

# This allows other modules to do:
# from services.optimizer import run_grid_search, etc.
