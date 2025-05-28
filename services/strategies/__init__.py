# services/strategies/__init__.py
# This file makes the 'strategies' directory a Python package.
# It also imports strategy functions for easier access by strategy_engine.py.

import sys
import os

# --- sys.path diagnostics for services/strategies/__init__.py ---
INIT_FILE_PATH = os.path.abspath(__file__)
print(f"--- [DEBUG __init__.py @ {os.path.dirname(INIT_FILE_PATH).split(os.sep)[-2:]}] ---") # Show 'services/strategies'
print(f"    INIT_FILE_PATH: {INIT_FILE_PATH}")
print(f"    sys.path as seen by services/strategies/__init__.py: {sys.path}")
# We expect the project root (e.g., '/mount/src/gap-guardian') to be in sys.path, added by app.py.
project_root_candidate_strat_init = None
for path_entry in sys.path:
    if os.path.isdir(os.path.join(path_entry, 'utils')) and \
       os.path.isdir(os.path.join(path_entry, 'config')) and \
       os.path.isdir(os.path.join(path_entry, 'services')):
        project_root_candidate_strat_init = path_entry
        break
print(f"    Attempted to find project root in sys.path: {project_root_candidate_strat_init if project_root_candidate_strat_init else 'Not found based on subdirs.'}")
print(f"--- [END DEBUG __init__.py @ {os.path.dirname(INIT_FILE_PATH).split(os.sep)[-2:]}] ---")
# --- end of sys.path diagnostics ---

print(f"--- [DEBUG __init__.py @ {os.path.dirname(INIT_FILE_PATH).split(os.sep)[-2:]}] Attempting to import strategy generation functions ---")
try:
    from .gap_guardian import generate_gap_guardian_signals
    print(f"--- [DEBUG __init__.py] Successfully imported generate_gap_guardian_signals ---")
    from .unicorn import generate_unicorn_signals
    print(f"--- [DEBUG __init__.py] Successfully imported generate_unicorn_signals ---")
    from .silver_bullet import generate_silver_bullet_signals
    print(f"--- [DEBUG __init__.py] Successfully imported generate_silver_bullet_signals ---")
    print(f"--- [DEBUG __init__.py @ {os.path.dirname(INIT_FILE_PATH).split(os.sep)[-2:]}] All strategy functions imported successfully. ---")
except ImportError as e:
    print(f"--- [CRITICAL ERROR __init__.py @ {os.path.dirname(INIT_FILE_PATH).split(os.sep)[-2:]}] Failed to import one or more strategy functions. This usually means the respective strategy file (e.g., gap_guardian.py) failed to load its own dependencies (like utils/config). Error: {e} ---")
    print(f"    Current sys.path during this error: {sys.path}")
    raise # Re-raise the error


__all__ = [
    "generate_gap_guardian_signals",
    "generate_unicorn_signals",
    "generate_silver_bullet_signals"
]
