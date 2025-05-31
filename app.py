# app.py
"""
Main Streamlit application file for the Multi-Strategy Backtester.
Orchestrates UI components and core logic modules.
Handles asynchronous execution of the analysis pipeline.
"""
import sys
import os
import streamlit as st
from datetime import datetime
import pandas as pd
import threading # Added for asynchronous operations
import time # Added for periodic checks

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
try:
    from config import settings
    from utils.logger import get_logger
    from ui import sidebar as ui_sidebar
    from ui import tabs_display
    from core import orchestration
except ImportError as e:
    st.error(f"Critical Import Error: {e}. Check sys.path and module structure. Current sys.path: {sys.path}")
    print(f"CRITICAL ERROR in app.py: Failed to import modules. Sys.path: {sys.path}. Error: {e}")
    st.stop()
# --- End Module Imports ---

logger = get_logger(__name__)

def load_custom_css(css_file_path):
    """Loads custom CSS from a file and applies it."""
    try:
        full_css_path = os.path.join(PROJECT_ROOT, css_file_path)
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
    defaults = {
        'backtest_results': None,
        'optimization_results_df': pd.DataFrame(),
        'price_data': pd.DataFrame(),
        'signals': pd.DataFrame(),
        'best_params_from_opt': None,
        'wfo_results': None,
        'selected_timeframe_value': settings.DEFAULT_STRATEGY_TIMEFRAME,
        'run_analysis_clicked_count': 0,
        'wfo_isd_ui_val': settings.DEFAULT_WFO_IN_SAMPLE_DAYS,
        'wfo_oosd_ui_val': settings.DEFAULT_WFO_OUT_OF_SAMPLE_DAYS,
        'wfo_sd_ui_val': settings.DEFAULT_WFO_STEP_DAYS,
        'selected_strategy': settings.DEFAULT_STRATEGY,
        # New state variables for asynchronous operations
        'analysis_thread': None,
        'analysis_running': False,
        'analysis_error': None,
        'analysis_complete': False, # To track if analysis has finished to display results
        'current_progress_value': 0.0,
        'current_progress_message': "Initializing...",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    logger.debug("Session state initialized/checked.")

def analysis_thread_target(inputs_dict):
    """Target function for the analysis thread."""
    try:
        # This function will update session state directly with results or errors
        orchestration.run_analysis_pipeline(inputs_dict)
        st.session_state.analysis_error = None # Clear any previous error
    except Exception as e:
        logger.error(f"Exception in analysis thread: {e}", exc_info=True)
        st.session_state.analysis_error = f"An error occurred in the analysis thread: {str(e)}"
        # Clear potentially partial results from session state in case of error
        st.session_state.backtest_results = None
        st.session_state.optimization_results_df = pd.DataFrame()
        st.session_state.wfo_results = None
    finally:
        # Signal completion regardless of success or failure
        st.session_state.analysis_running = False
        st.session_state.analysis_complete = True # Mark as complete to trigger result display
        # Note: We don't call st.experimental_rerun() from the thread.
        # The main script will handle reruns based on state changes.

def main():
    """Main application function."""
    st.set_page_config(
        page_title=settings.APP_TITLE,
        page_icon="🛡️📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_custom_css("static/style.css") 
    initialize_app_session_state()

    sidebar_inputs = ui_sidebar.render_sidebar()
    ui_sidebar.display_main_content_header(sidebar_inputs)

    # Handle "Run Analysis" button click
    if sidebar_inputs["run_button_clicked"]:
        if not st.session_state.analysis_running:
            logger.info("Run Analysis button clicked. Starting new analysis.")
            st.session_state.run_analysis_clicked_count += 1
            st.session_state.analysis_running = True
            st.session_state.analysis_complete = False # Reset completion flag
            st.session_state.analysis_error = None
            st.session_state.backtest_results = None # Clear previous results
            st.session_state.optimization_results_df = pd.DataFrame()
            st.session_state.wfo_results = None
            st.session_state.current_progress_value = 0.0
            st.session_state.current_progress_message = "Initializing analysis..."
            
            # Start the analysis in a new thread
            # Pass a copy of sidebar_inputs to avoid issues if it's modified
            thread = threading.Thread(target=analysis_thread_target, args=(sidebar_inputs.copy(),))
            st.session_state.analysis_thread = thread
            thread.start()
            st.experimental_rerun() # Rerun to show spinner immediately
        else:
            logger.info("Run Analysis button clicked, but analysis is already running.")
            st.warning("Analysis is already in progress. Please wait for it to complete.")

    # UI updates based on analysis state
    if st.session_state.analysis_running:
        # Display progress bar and spinner
        st.progress(st.session_state.current_progress_value, text=st.session_state.current_progress_message)
        st.spinner(text=st.session_state.current_progress_message)
        
        # Periodically check thread status and request rerun to update progress
        # This creates a polling mechanism.
        # A more advanced solution might involve callbacks or queues if Streamlit fully supported them with threads.
        if st.session_state.analysis_thread and st.session_state.analysis_thread.is_alive():
            time.sleep(0.5) # Short sleep to allow other interactions and reduce busy-waiting
            st.experimental_rerun()
        else: # Thread finished (or wasn't started properly)
            st.session_state.analysis_running = False # Ensure flag is updated
            st.session_state.analysis_complete = True
            st.experimental_rerun() # Rerun to display results or error

    elif st.session_state.analysis_complete: # Analysis is not running, but has completed
        if st.session_state.analysis_error:
            st.error(st.session_state.analysis_error)
        # Display results whether there was an error or not (some results might still be there)
        tabs_display.render_results_tabs(sidebar_inputs)
        # Optionally, reset analysis_complete after displaying results if you want the "Run" button to clear them
        # For now, results persist until a new run.
    
    elif st.session_state.run_analysis_clicked_count == 0 : # Initial state, no run yet
         if not any([st.session_state.backtest_results, not st.session_state.optimization_results_df.empty, st.session_state.wfo_results]):
            st.info("Configure parameters in the sidebar and click 'Run Analysis' to view results.")


if __name__ == "__main__":
    if not PROJECT_ROOT or not os.path.isdir(PROJECT_ROOT):
        st.error("PROJECT_ROOT is not correctly defined or accessible. Application cannot start.")
        print("CRITICAL: PROJECT_ROOT not defined or not a directory. Exiting.")
    else:
        # This is crucial for multiprocessing to work correctly on all platforms,
        # especially Windows. It ensures child processes can import the main module.
        # It should be placed inside the `if __name__ == "__main__":` block.
        # However, given Streamlit's execution model, this specific placement might be
        # less critical than in a pure multiprocessing script, but it's good practice.
        # multiprocessing.freeze_support() # Consider if optimizer's multiprocessing needs it
        main()
