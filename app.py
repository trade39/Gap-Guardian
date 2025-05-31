# app.py
"""
Main Streamlit application file for the Multi-Strategy Backtester.
Orchestrates UI components and core logic modules.
Handles asynchronous execution of the analysis pipeline and result processing.
"""
import sys
import os
import streamlit as st
from datetime import datetime
import pandas as pd
import threading
import time

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

logger = get_logger(__name__)

def load_custom_css(css_file_path):
    """Loads custom CSS from a file and applies it."""
    try:
        full_css_path = os.path.join(PROJECT_ROOT, css_file_path)
        if not os.path.exists(full_css_path):
            logger.error(f"CSS file not found at: {full_css_path}. CSS will not be loaded.")
            # st.warning(f"CSS file not found: {css_file_path}. Custom styles may not apply.") # Removed to reduce UI clutter
            return
        with open(full_css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        logger.info(f"Successfully loaded CSS from: {full_css_path}")
    except Exception as e:
        logger.error(f"Error loading CSS file '{css_file_path}': {e}", exc_info=True)
        st.warning(f"Could not load custom CSS. Error: {e}")

def initialize_app_session_state():
    """Initializes session state variables."""
    defaults = {
        'backtest_results': None,
        'optimization_results_df': pd.DataFrame(),
        'price_data': pd.DataFrame(), # Will be populated by orchestration's return now
        'signals': pd.DataFrame(), # Will be populated by orchestration's return now
        'best_params_from_opt': None,
        'wfo_results': None,
        'selected_timeframe_value': settings.DEFAULT_STRATEGY_TIMEFRAME,
        'run_analysis_clicked_count': 0,
        'wfo_isd_ui_val': settings.DEFAULT_WFO_IN_SAMPLE_DAYS,
        'wfo_oosd_ui_val': settings.DEFAULT_WFO_OUT_OF_SAMPLE_DAYS,
        'wfo_sd_ui_val': settings.DEFAULT_WFO_STEP_DAYS,
        'selected_strategy': settings.DEFAULT_STRATEGY,
        'analysis_thread': None,
        'analysis_running': False,
        'analysis_error_message': None, # Specific for error messages from thread
        'analysis_complete_flag': False,
        'analysis_output_from_thread': None, # To store the direct output of the thread
        'current_progress_value': 0.0,
        'current_progress_message': "Initializing...",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    logger.debug("Session state initialized/checked.")

def analysis_thread_target(inputs_dict):
    """Target function for the analysis thread. It now stores results for app.py to process."""
    try:
        # Orchestration now returns results instead of setting session state directly for them
        results = orchestration.run_analysis_pipeline(inputs_dict)
        st.session_state.analysis_output_from_thread = results
        st.session_state.analysis_error_message = results.get('error_message') # Get error from results
    except Exception as e:
        logger.error(f"Unhandled exception directly in analysis_thread_target: {e}", exc_info=True)
        st.session_state.analysis_output_from_thread = {'error_message': f"Critical error in analysis thread: {str(e)}"}
        st.session_state.analysis_error_message = st.session_state.analysis_output_from_thread['error_message']
    finally:
        st.session_state.analysis_running = False
        st.session_state.analysis_complete_flag = True
        # Main app will handle rerun based on these flags

def process_analysis_results():
    """Processes results from the analysis_output_from_thread and updates main session state."""
    if st.session_state.analysis_output_from_thread:
        output = st.session_state.analysis_output_from_thread
        
        # Explicitly set main result states from the thread's output
        st.session_state.backtest_results = output.get('backtest_results')
        st.session_state.optimization_results_df = output.get('optimization_results_df', pd.DataFrame())
        st.session_state.wfo_results = output.get('wfo_results')
        st.session_state.price_data = output.get('price_data', pd.DataFrame()) # Get price_data
        st.session_state.signals = output.get('signals', pd.DataFrame()) # Get signals
        st.session_state.best_params_from_opt = output.get('best_params_from_opt')
        
        st.session_state.analysis_error_message = output.get('error_message') # Update error message
        
        if st.session_state.analysis_error_message:
            logger.error(f"Analysis completed with error: {st.session_state.analysis_error_message}")
        else:
            logger.info("Analysis completed successfully, results processed into session state.")
            
        # Clear the temporary holding state
        st.session_state.analysis_output_from_thread = None


def main():
    """Main application function."""
    st.set_page_config(
        page_title=settings.APP_TITLE,
        page_icon="🛡️📈",
        layout="wide",
        initial_sidebar_state="expanded",
        # theme=settings.DEFAULT_THEME # If you add default theme to settings
    )
    
    # Ensure .streamlit/config.toml has primaryColor set for dark theme components if needed
    # [theme]
    # base="dark" 
    # primaryColor="#1E88E5" # Your blue accent

    load_custom_css("static/style.css") 
    initialize_app_session_state()

    sidebar_inputs = ui_sidebar.render_sidebar()
    ui_sidebar.display_main_content_header(sidebar_inputs)

    if sidebar_inputs["run_button_clicked"]:
        if not st.session_state.analysis_running:
            logger.info("Run Analysis button clicked. Starting new analysis.")
            st.session_state.run_analysis_clicked_count += 1
            st.session_state.analysis_running = True
            st.session_state.analysis_complete_flag = False
            st.session_state.analysis_error_message = None
            st.session_state.analysis_output_from_thread = None # Clear previous output

            # Clear previous results from main session state before new run
            st.session_state.backtest_results = None
            st.session_state.optimization_results_df = pd.DataFrame()
            st.session_state.wfo_results = None
            st.session_state.price_data = pd.DataFrame()
            st.session_state.signals = pd.DataFrame()
            st.session_state.best_params_from_opt = None

            st.session_state.current_progress_value = 0.0
            st.session_state.current_progress_message = "Initializing analysis..."
            
            thread = threading.Thread(target=analysis_thread_target, args=(sidebar_inputs.copy(),))
            st.session_state.analysis_thread = thread
            thread.start()
            st.rerun() 
        else:
            logger.info("Run Analysis button clicked, but analysis is already running.")
            st.warning("Analysis is already in progress. Please wait for it to complete.")

    if st.session_state.analysis_running:
        st.progress(st.session_state.current_progress_value, text=st.session_state.current_progress_message)
        # st.spinner(text=st.session_state.current_progress_message) # Spinner can be redundant with progress
        
        if st.session_state.analysis_thread and st.session_state.analysis_thread.is_alive():
            time.sleep(0.5) 
            st.rerun()
        else: # Thread likely finished, flags will be set by thread_target
            # This case might be hit if thread finishes very quickly
            # The next rerun will catch analysis_complete_flag
            if not st.session_state.analysis_complete_flag: # If thread finished but flag not yet processed
                 st.session_state.analysis_running = False # Ensure running is false
                 st.session_state.analysis_complete_flag = True # Ensure complete is true
            st.rerun()

    elif st.session_state.analysis_complete_flag:
        process_analysis_results() # Process results from thread into main session state

        if st.session_state.analysis_error_message:
            st.error(st.session_state.analysis_error_message)
        
        # Always attempt to display tabs, it will show info message if no results
        tabs_display.render_results_tabs(sidebar_inputs)
        
        # Reset flags for next run, but keep results displayed
        # st.session_state.analysis_complete_flag = False # Keep true to show results until next run
        # st.session_state.analysis_error_message = None
    
    elif st.session_state.run_analysis_clicked_count == 0:
         if not any([st.session_state.backtest_results, 
                     not st.session_state.optimization_results_df.empty if st.session_state.optimization_results_df is not None else False, 
                     st.session_state.wfo_results]):
            st.info("Configure parameters in the sidebar and click 'Run Analysis' to view results.")

if __name__ == "__main__":
    if not PROJECT_ROOT or not os.path.isdir(PROJECT_ROOT):
        st.error("PROJECT_ROOT is not correctly defined or accessible. Application cannot start.")
        print("CRITICAL: PROJECT_ROOT not defined or not a directory. Exiting.")
    else:
        main()
