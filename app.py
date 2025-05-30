# app.py
"""
Main Streamlit application file for the Multi-Strategy Backtester.
Orchestrates UI components and core logic modules.
"""
import sys
import os
import streamlit as st
from datetime import datetime

# --- sys.path modification ---
APP_FILE_PATH = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(APP_FILE_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"PROJECT_ROOT ('{PROJECT_ROOT}') added to sys.path.")
elif sys.path[0] != PROJECT_ROOT:
    sys.path.remove(PROJECT_ROOT)
    sys.path.insert(0, PROJECT_ROOT)
    print(f"PROJECT_ROOT ('{PROJECT_ROOT}') moved to beginning of sys.path.")
# --- End sys.path modification ---

# --- Module Imports ---
# It's crucial that these imports succeed. If they fail, it often indicates
# an issue with how PROJECT_ROOT is set relative to the module locations,
# or that __init__.py files are missing in subdirectories like 'config', 'utils', etc.
try:
    from config import settings
    from utils.logger import get_logger
    from ui import sidebar as ui_sidebar # Renamed to avoid conflict
    from ui import tabs_display
    from core import orchestration
except ImportError as e:
    # Provide more context if imports fail
    st.error(f"Critical Import Error: {e}. Check sys.path and module structure. Current sys.path: {sys.path}")
    print(f"CRITICAL ERROR in app.py: Failed to import modules. Sys.path: {sys.path}. Error: {e}")
    st.stop() # Stop execution if essential modules can't be loaded
# --- End Module Imports ---

logger = get_logger(__name__)

def load_custom_css(css_file_path):
    """Loads custom CSS from a file and applies it."""
    try:
        full_css_path = os.path.join(PROJECT_ROOT, css_file_path) # Assume css_file_path is relative to PROJECT_ROOT
        if not os.path.exists(full_css_path):
            logger.error(f"CSS file not found at: {full_css_path}. CSS will not be loaded.")
            st.warning(f"CSS file not found: {css_file_path}. Custom styles may not apply.")
            return
        with open(full_css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        logger.info(f"Successfully loaded CSS from: {full_css_path}")
    except Exception as e:
        logger.error(f"Error loading CSS file '{css_file_path}': {e}", exc_info=True)
        st.warning(f"Could not load custom CSS. Error: {e}")

def initialize_app_session_state():
    """Initializes session state variables with default values if they don't exist."""
    # These are application-level defaults for session state.
    # Specific UI component defaults (like for sidebar inputs) might be handled within those components
    # or by referencing settings.py directly where those components are defined.
    defaults = {
        'backtest_results': None,
        'optimization_results_df': pd.DataFrame(), # Ensure pandas is imported if used here
        'price_data': pd.DataFrame(),
        'signals': pd.DataFrame(),
        'best_params_from_opt': None,
        'wfo_results': None,
        'selected_timeframe_value': settings.DEFAULT_STRATEGY_TIMEFRAME,
        'run_analysis_clicked_count': 0,
        'wfo_isd_ui_val': settings.DEFAULT_WFO_IN_SAMPLE_DAYS, # UI state for WFO params
        'wfo_oosd_ui_val': settings.DEFAULT_WFO_OUT_OF_SAMPLE_DAYS,
        'wfo_sd_ui_val': settings.DEFAULT_WFO_STEP_DAYS,
        'selected_strategy': settings.DEFAULT_STRATEGY,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    logger.debug("Session state initialized/checked.")


def main():
    """Main application function."""
    st.set_page_config(
        page_title=settings.APP_TITLE,
        page_icon="🛡️📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load CSS - ensure the path is correct. If style.css is in "static/style.css"
    # and app.py is at the project root, this path should work.
    load_custom_css("static/style.css") 
    
    initialize_app_session_state()

    # Render sidebar and get inputs
    # The render_sidebar function now encapsulates all sidebar UI elements
    # and returns a dictionary of the user's selections.
    sidebar_inputs = ui_sidebar.render_sidebar()

    # Display main content header (title, strategy info, explanation expander)
    # This function is now in ui_sidebar.py as it's closely related to strategy selection
    ui_sidebar.display_main_content_header(sidebar_inputs)


    # Core logic: If "Run Analysis" button (now part of sidebar_inputs) was clicked
    if sidebar_inputs["run_button_clicked"]:
        st.session_state.run_analysis_clicked_count += 1
        # The orchestration module handles the entire analysis pipeline
        orchestration.run_analysis_pipeline(sidebar_inputs)
    
    # Display results tabs
    # The tabs_display module reads from st.session_state to show results
    tabs_display.render_results_tabs(sidebar_inputs)


if __name__ == "__main__":
    # Basic check for PROJECT_ROOT being set, as it's critical for module resolution
    if not PROJECT_ROOT or not os.path.isdir(PROJECT_ROOT):
        st.error("PROJECT_ROOT is not correctly defined or accessible. Application cannot start.")
        print("CRITICAL: PROJECT_ROOT not defined or not a directory. Exiting.")
    else:
        main()

