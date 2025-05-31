# core/orchestration.py
"""
Handles the core analysis pipeline logic.
Modified to update session state for progress and results when run in a thread.
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import time as dt_time

from config import settings
from services import data_loader, strategy_engine, backtester, optimizer
from utils.logger import get_logger

logger = get_logger(__name__)

def run_analysis_pipeline(inputs: dict):
    """
    Executes the main analysis pipeline based on user inputs.
    Updates st.session_state with progress, results, or errors.
    This function is intended to be run in a separate thread.
    """
    try:
        logger.info(f"Analysis Pipeline Thread Started. Strategy: {inputs['selected_strategy_name']}, Mode: {inputs['analysis_mode_ui']}")
        st.session_state.current_progress_value = 0.05
        st.session_state.current_progress_message = f"Starting {inputs['analysis_mode_ui']} for {inputs['selected_strategy_name']}..."

        # Clear previous results from session state (already done in app.py before starting thread, but good for safety)
        st.session_state.backtest_results = None
        st.session_state.optimization_results_df = pd.DataFrame()
        st.session_state.wfo_results = None
        st.session_state.price_data = pd.DataFrame()
        st.session_state.signals = pd.DataFrame()
        st.session_state.best_params_from_opt = None
        st.session_state.analysis_error = None # Clear any prior error

        # Validate date range
        if inputs['start_date_ui'] >= inputs['end_date_ui']:
            st.session_state.analysis_error = f"Error: Start date ({inputs['start_date_ui']}) must be before end date ({inputs['end_date_ui']})."
            logger.error(st.session_state.analysis_error)
            return

        # Specific validation for WFO duration
        if inputs['analysis_mode_ui'] == "Walk-Forward Optimization":
            total_data_duration_days_for_check = (inputs['end_date_ui'] - inputs['start_date_ui']).days + 1
            min_required_wfo_duration = inputs['wfo_isd_ui'] + inputs['wfo_oosd_ui']
            if total_data_duration_days_for_check < min_required_wfo_duration:
                st.session_state.analysis_error = (
                    f"Insufficient data for WFO. Selected range: {total_data_duration_days_for_check}d, "
                    f"WFO requires at least: {min_required_wfo_duration}d (IS={inputs['wfo_isd_ui']}d + OOS={inputs['wfo_oosd_ui']}d). "
                    f"Please adjust date range or WFO period lengths."
                )
                logger.error(st.session_state.analysis_error)
                return

        st.session_state.current_progress_value = 0.1
        st.session_state.current_progress_message = f"Fetching data for {inputs['ticker_symbol']}..."
        
        price_data_df = data_loader.fetch_historical_data(
            inputs['ticker_symbol'], inputs['start_date_ui'], inputs['end_date_ui'], inputs['ui_current_interval']
        )
        st.session_state.price_data = price_data_df # Store fetched data in session state

        if st.session_state.price_data.empty:
            st.session_state.analysis_error = (
                f"No price data retrieved for {inputs['selected_ticker_name']} ({inputs['ticker_symbol']}) "
                f"for the period {inputs['start_date_ui']} to {inputs['end_date_ui']} with interval {inputs['ui_current_interval']}. Cannot proceed."
            )
            logger.warning(st.session_state.analysis_error)
            return

        st.session_state.current_progress_value = 0.2
        st.session_state.current_progress_message = "Data fetched. Preparing analysis..."

        # Prepare strategy parameters for the engine
        strategy_params_for_engine = {
            'strategy_name': inputs['selected_strategy_name'],
            'stop_loss_points': inputs['sl_points_single_ui'],
            'rrr': inputs['rrr_single_ui'],
        }
        if inputs['selected_strategy_name'] == "Gap Guardian":
            strategy_params_for_engine['entry_start_time'] = dt_time(inputs['entry_start_hour_single_ui'], inputs['entry_start_minute_single_ui'])
            strategy_params_for_engine['entry_end_time'] = dt_time(inputs['entry_end_hour_single_ui'], inputs['entry_end_minute_single_ui'])
        
        # --- Single Backtest Logic ---
        if inputs['analysis_mode_ui'] == "Single Backtest":
            st.session_state.current_progress_message = "Running single backtest..."
            signals = strategy_engine.generate_signals(data=st.session_state.price_data.copy(), **strategy_params_for_engine)
            st.session_state.signals = signals
            st.session_state.current_progress_value = 0.5
            
            trades, equity, perf = backtester.run_backtest(
                st.session_state.price_data.copy(), signals, inputs['initial_capital_ui'],
                inputs['risk_per_trade_percent_ui'], inputs['sl_points_single_ui'], inputs['ui_current_interval']
            )
            param_display_str = f"SL: {inputs['sl_points_single_ui']:.2f} pts, RRR: {inputs['rrr_single_ui']:.1f}"
            if inputs['selected_strategy_name'] == "Gap Guardian":
                param_display_str += f", Entry: {strategy_params_for_engine['entry_start_time']:%H:%M}-{strategy_params_for_engine['entry_end_time']:%H:%M} NYT"
            
            st.session_state.backtest_results = {
                "trades": trades, "equity_curve": equity, "performance": perf,
                "params": {
                    "Strategy": inputs['selected_strategy_name'], "Symbol": inputs['selected_ticker_name'],
                    "SL": inputs['sl_points_single_ui'], "RRR": inputs['rrr_single_ui'],
                    "TF": inputs['ui_current_interval'], "EntryDisplay": param_display_str, "src": "Manual"
                }
            }
            st.session_state.current_progress_value = 1.0
            st.session_state.current_progress_message = "Single backtest complete!"
            logger.info("Single backtest complete.")

        # --- Parameter Optimization or Walk-Forward Optimization Logic ---
        elif inputs['analysis_mode_ui'] in ["Parameter Optimization", "Walk-Forward Optimization"]:
            
            def optimization_progress_callback_thread_safe(progress_fraction, message_text):
                st.session_state.current_progress_value = 0.2 + (progress_fraction * 0.7) # Scale progress from 20% to 90%
                st.session_state.current_progress_message = f"{message_text}: {int(progress_fraction*100)}% complete"

            actual_params_to_optimize_config = {
                'sl_points': np.linspace(inputs['sl_min_opt_ui'], inputs['sl_max_opt_ui'], int(inputs['sl_steps_opt_ui'])) if inputs['opt_algo_ui'] == "Grid Search" else (inputs['sl_min_opt_ui'], inputs['sl_max_opt_ui']),
                'rrr': np.linspace(inputs['rrr_min_opt_ui'], inputs['rrr_max_opt_ui'], int(inputs['rrr_steps_opt_ui'])) if inputs['opt_algo_ui'] == "Grid Search" else (inputs['rrr_min_opt_ui'], inputs['rrr_max_opt_ui']),
            }
            if inputs['selected_strategy_name'] == "Gap Guardian":
                actual_params_to_optimize_config.update({
                    'entry_start_hour': [int(h) for h in np.linspace(inputs['esh_min_opt_ui'], inputs['esh_max_opt_ui'], int(inputs['esh_steps_opt_ui']))] if inputs['opt_algo_ui'] == "Grid Search" else (inputs['esh_min_opt_ui'], inputs['esh_max_opt_ui']),
                    'entry_start_minute': inputs['esm_vals_opt_ui'],
                    'entry_end_hour': [int(h) for h in np.linspace(inputs['eeh_min_opt_ui'], inputs['eeh_max_opt_ui'], int(inputs['eeh_steps_opt_ui']))] if inputs['opt_algo_ui'] == "Grid Search" else (inputs['eeh_min_opt_ui'], inputs['eeh_max_opt_ui']),
                    'entry_end_minute': settings.DEFAULT_ENTRY_END_MINUTE_OPTIMIZATION_VALUES
                })
            
            optimizer_control_params = {'metric_to_optimize': inputs['opt_metric_ui'], 'strategy_name': inputs['selected_strategy_name']}
            if inputs['opt_algo_ui'] == "Random Search":
                optimizer_control_params['iterations'] = inputs['rand_iters_ui']

            if inputs['analysis_mode_ui'] == "Parameter Optimization":
                st.session_state.current_progress_message = f"Running {inputs['opt_algo_ui']} optimization..."
                
                if inputs['opt_algo_ui'] == "Grid Search":
                    opt_df = optimizer.run_grid_search(st.session_state.price_data.copy(), inputs['initial_capital_ui'], inputs['risk_per_trade_percent_ui'], actual_params_to_optimize_config, inputs['ui_current_interval'], optimizer_control_params, optimization_progress_callback_thread_safe)
                else: # Random Search
                    opt_df = optimizer.run_random_search(st.session_state.price_data.copy(), inputs['initial_capital_ui'], inputs['risk_per_trade_percent_ui'], actual_params_to_optimize_config, inputs['ui_current_interval'], optimizer_control_params, optimization_progress_callback_thread_safe)
                
                st.session_state.optimization_results_df = opt_df
                if not opt_df.empty:
                    logger.info("Full period optimization finished!")
                    valid_opt_results = opt_df.dropna(subset=[inputs['opt_metric_ui']])
                    if not valid_opt_results.empty:
                        best_row_from_opt = None
                        if inputs['opt_metric_ui'] == "Max Drawdown (%)": # Lower (less negative) is better
                            best_row_from_opt = valid_opt_results.loc[valid_opt_results[inputs['opt_metric_ui']].idxmax()] # idxmax for less negative
                        else: # Higher is better for other metrics
                            best_row_from_opt = valid_opt_results.loc[valid_opt_results[inputs['opt_metric_ui']].idxmax()]
                        
                        st.session_state.best_params_from_opt = best_row_from_opt.to_dict()
                        best_params_for_bt_run = {'strategy_name': inputs['selected_strategy_name'],
                                                  'stop_loss_points': best_row_from_opt["SL Points"],
                                                  'rrr': best_row_from_opt["RRR"]}
                        entry_display_opt_str = f"SL: {best_row_from_opt['SL Points']:.2f}, RRR: {best_row_from_opt['RRR']:.1f}"
                        if inputs['selected_strategy_name'] == "Gap Guardian":
                            best_entry_start_t = dt_time(int(best_row_from_opt["EntryStartHour"]), int(best_row_from_opt["EntryStartMinute"]))
                            best_entry_end_t = dt_time(int(best_row_from_opt["EntryEndHour"]), int(best_row_from_opt.get("EntryEndMinute", settings.DEFAULT_ENTRY_WINDOW_END_MINUTE)))
                            best_params_for_bt_run['entry_start_time'] = best_entry_start_t
                            best_params_for_bt_run['entry_end_time'] = best_entry_end_t
                            entry_display_opt_str += f", Entry: {best_entry_start_t:%H:%M}-{best_entry_end_t:%H:%M} NYT"
                        
                        logger.info(f"Best parameters for '{inputs['opt_metric_ui']}': {entry_display_opt_str} (Metric Value: {best_row_from_opt[inputs['opt_metric_ui']]:.2f})")
                        
                        st.session_state.current_progress_value = 0.9
                        st.session_state.current_progress_message = "Running backtest with best optimized parameters..."
                        signals_best_opt = strategy_engine.generate_signals(data=st.session_state.price_data.copy(), **best_params_for_bt_run)
                        st.session_state.signals = signals_best_opt
                        trades_bo, equity_bo, perf_bo = backtester.run_backtest(st.session_state.price_data.copy(), signals_best_opt, inputs['initial_capital_ui'], inputs['risk_per_trade_percent_ui'], best_row_from_opt["SL Points"], inputs['ui_current_interval'])
                        st.session_state.backtest_results = {
                            "trades": trades_bo, "equity_curve": equity_bo, "performance": perf_bo,
                            "params": {"Strategy": inputs['selected_strategy_name'], "Symbol": inputs['selected_ticker_name'],
                                       "SL": best_row_from_opt["SL Points"], "RRR": best_row_from_opt["RRR"],
                                       "TF": inputs['ui_current_interval'], "EntryDisplay": entry_display_opt_str,
                                       "src": f"Opt ({inputs['opt_algo_ui']})"}
                        }
                    else:
                        st.session_state.analysis_error = f"No valid results found for metric '{inputs['opt_metric_ui']}' in optimization. Cannot run final backtest."
                        logger.warning(st.session_state.analysis_error)
                else:
                    st.session_state.analysis_error = "Optimization yielded no results. Check parameters/data."
                    logger.error(st.session_state.analysis_error)
                st.session_state.current_progress_message = "Parameter Optimization Complete!"


            elif inputs['analysis_mode_ui'] == "Walk-Forward Optimization":
                st.session_state.current_progress_message = f"Running Walk-Forward Optimization with {inputs['opt_algo_ui']}..."
                wfo_parameters_for_run = {
                    'in_sample_days': inputs['wfo_isd_ui'],
                    'out_of_sample_days': inputs['wfo_oosd_ui'],
                    'step_days': inputs['wfo_sd_ui']
                }
                wfo_optimizer_config_and_control = {**actual_params_to_optimize_config, **optimizer_control_params}
                
                wfo_log_df, oos_trades_df, oos_equity_series, oos_performance_summary = optimizer.run_walk_forward_optimization(
                    st.session_state.price_data.copy(), inputs['initial_capital_ui'], inputs['risk_per_trade_percent_ui'],
                    wfo_parameters_for_run, inputs['opt_algo_ui'], wfo_optimizer_config_and_control,
                    inputs['ui_current_interval'], optimization_progress_callback_thread_safe
                )
                st.session_state.wfo_results = {
                    "log": wfo_log_df, "oos_trades": oos_trades_df,
                    "oos_equity_curve": oos_equity_series, "oos_performance": oos_performance_summary
                }
                st.session_state.backtest_results = { # For consistent display in performance tab
                    "trades": oos_trades_df, "equity_curve": oos_equity_series,
                    "performance": oos_performance_summary,
                    "params": {"Strategy": inputs['selected_strategy_name'], "Symbol": inputs['selected_ticker_name'],
                               "TF": inputs['ui_current_interval'], "src": "WFO Aggregated OOS"}
                }
                logger.info("Walk-Forward Optimization finished!")
                st.session_state.current_progress_message = "Walk-Forward Optimization Complete!"
        
        st.session_state.current_progress_value = 1.0

    except Exception as e:
        # This is the main exception handler for the entire pipeline within the thread
        error_message = f"Error during analysis pipeline: {e}"
        logger.error(error_message, exc_info=True)
        st.session_state.analysis_error = error_message
        # Ensure progress indicates completion with error
        st.session_state.current_progress_value = 1.0
        st.session_state.current_progress_message = "Analysis failed. Check error messages."

    finally:
        # This block ensures these states are set even if an error occurs mid-pipeline
        # The app.py will pick up these state changes on its next rerun.
        logger.info(f"Analysis Pipeline Thread Finished. Error: {st.session_state.analysis_error}")

