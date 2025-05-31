# Multi-Strategy ICT Backtester & Optimizer

This Streamlit application allows users to backtest, optimize, and analyze various trading strategies, particularly those inspired by Inner Circle Trader (ICT) concepts, on user-selected financial symbols using historical data from Yahoo Finance.

## Core Features

* **Multiple Strategies**: Implements several distinct trading strategies:
    * **Gap Guardian**: Focuses on morning session opening range false breakouts.
    * **Unicorn**: Aims to identify entries based on Breaker Blocks combined with Fair Value Gaps (FVG). (Currently simplified FVG entry).
    * **Silver Bullet**: Time-based FVG entries during specific New York session windows.
* **Interactive UI**: Configure backtesting parameters, strategy settings, and optimization controls via a user-friendly sidebar.
* **Dynamic Data Loading**: Fetches historical market data from Yahoo Finance for various symbols and timeframes.
* **Comprehensive Backtesting**:
    * Detailed trade log with entry/exit times, prices, P&L, and exit reasons.
    * Equity curve visualization.
    * Extensive performance metrics (Total P&L, Win Rate, Profit Factor, Max Drawdown, Sharpe Ratio, Sortino Ratio, etc.).
    * Consideration of transaction costs (commission and slippage).
* **Parameter Optimization**:
    * Optimize strategy parameters (e.g., Stop Loss, Risk/Reward Ratio, entry window times) using Grid Search or Random Search.
    * Visualize optimization results, including heatmaps for Grid Search.
* **Walk-Forward Optimization (WFO)**:
    * Perform robust out-of-sample testing by iteratively optimizing parameters on in-sample data and testing on subsequent out-of-sample periods.
    * View WFO logs, aggregated OOS trades, and chained OOS equity curves.
* **Reporting**:
    * Downloadable CSV reports for backtest performance, optimization results, and WFO logs, including a MetaTrader-style strategy report.
* **Customizable Theming**: Supports light and dark themes via CSS and Streamlit configuration.
* **Modular Architecture**: Code is organized into logical modules for easier maintenance and extension.

## Strategies Implemented

### 1. Gap Guardian
* **Concept**: Aims to capitalize on false breakouts/breakdowns of an initial opening range during a specific morning session.
* **Time Frame**: Typically 15-minutes, but adaptable.
* **Entry Window (NY Time)**: User-configurable (e.g., 9:30 AM - 11:00 AM).
* **Opening Range**: Defined by the high and low of the first bar occurring at or after the `Entry Start Time`.
* **Long Entry**: Price breaks *below* the opening range low, then *closes back above* it within the `Entry Window`.
* **Short Entry**: Price breaks *above* the opening range high, then *closes back below* it within the `Entry Window`.
* **Risk Management**: Uses a user-defined stop-loss (in points) and a risk-reward ratio (RRR).

### 2. Unicorn (Simplified FVG Entry)
* **Concept**: Aims for high-probability entries by combining a "Breaker Block" with an overlapping "Fair Value Gap (FVG)".
    * *Note: The current implementation focuses on a simplified FVG entry logic. Full Breaker + FVG overlap is more complex.*
* **Entry Logic**: Looks for a recently formed FVG and enters if the current bar retraces into this FVG and shows a reaction (e.g., bullish close for long after dipping into a bullish FVG).
* **Risk Management**: Uses a user-defined stop-loss (in points) and a risk-reward ratio (RRR).

### 3. Silver Bullet
* **Concept**: A time-based strategy looking for entries into Fair Value Gaps (FVGs) during specific 1-hour windows in the New York trading session.
* **Time Windows (NY Time)**:
    * 3:00 AM - 4:00 AM
    * 10:00 AM - 11:00 AM
    * 2:00 PM - 3:00 PM
* **Entry Logic**: Enters on retracement into a qualifying FVG that forms or is revisited during one of the specified 1-hour windows, assuming context (draw on liquidity) aligns.
* **Risk Management**: Uses a user-defined stop-loss (in points) and a risk-reward ratio (RRR).

## Project Structure


ict_strategies_backtester/
├── .streamlit/
│   └── config.toml         # Streamlit theme and app configuration
├── app.py                  # Main Streamlit application UI and flow
├── config/
│   ├── init.py
│   └── settings.py         # Application constants and default parameters
├── core/
│   ├── init.py
│   └── orchestration.py    # Core analysis pipeline logic
├── services/
│   ├── init.py
│   ├── data_loader.py      # Handles fetching and preparing market data
│   ├── strategy_engine.py  # Dispatches to strategy-specific signal generation
│   ├── backtester.py       # Executes backtests, simulates trades, calculates P&L
│   ├── strategies/         # Individual strategy logic modules
│   │   ├── init.py
│   │   ├── gap_guardian.py
│   │   ├── unicorn.py
│   │   └── silver_bullet.py
│   └── optimizer/          # Parameter optimization and WFO logic
│       ├── init.py
│       ├── search_algorithms.py
│       ├── optimization_utils.py
│       ├── metrics_calculator.py
│       └── wfo_orchestrator.py
├── ui/
│   ├── init.py
│   ├── sidebar.py          # Renders sidebar inputs and controls
│   └── tabs_display.py     # Renders main results tabs
├── utils/
│   ├── init.py
│   ├── logger.py           # Logging configuration
│   ├── plotting.py         # Functions for generating visualizations
│   └── technical_analysis.py # Common TA helper functions (FVG, Swings)
├── static/
│   └── style.css           # Custom CSS for styling the application
├── .env.example            # Example environment variables (if any needed)
├── .gitignore              # Specifies intentionally untracked files
├── README.md               # This file: project documentation
└── requirements.txt        # Python package dependencies


## Setup and Installation

1.  **Clone the repository (Optional for Streamlit Cloud):**
    ```bash
    git clone <repository_url>
    cd ict_strategies_backtester
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    *Recommended Python version: 3.9+*

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    Or, if your project root is one level above `app.py` and `app.py` is inside a package-like structure, you might run it as a module from the directory containing `app.py`'s parent folder:
    ```bash
    python -m streamlit run <your_project_directory_name>/app.py
    ```

## Usage

1.  Open the application in your browser (typically `http://localhost:8501`).
2.  Configure the analysis parameters in the sidebar:
    * Select a **Strategy**, **Symbol**, and **Timeframe**.
    * Set the **Start Date** and **End Date** for historical data.
        * **Note on Data Limits**: Yahoo Finance has limitations on historical intraday data. For short intervals (1m-30m), history is typically limited to ~60 days. For hourly intervals (1h, 90m), it's often up to ~730 days. The application will attempt to adjust the start date if it exceeds these limits for the selected interval.
    * Adjust **Financial Parameters** like Initial Capital and Risk per Trade %.
    * Configure **Transaction Cost Assumptions** (Commission and Slippage).
    * Choose an **Analysis Type**:
        * `Single Backtest`: Runs the selected strategy with manually set parameters.
        * `Parameter Optimization`: Optimizes strategy parameters over the full period.
        * `Walk-Forward Optimization`: Performs robust WFO.
    * Set strategy-specific parameters (e.g., SL/RRR, entry windows for Gap Guardian) or optimization ranges.
3.  Click the "**🚀 Run Analysis**" button.
4.  Review the results displayed in the main panel, including performance metrics, charts (Equity Curve, Trades on Price), trade logs, and optimization/WFO details if applicable.
5.  Use the "📥 Download Full Report (CSV)" button or other download buttons in relevant tabs to export data.

## Customization and Theming

* **Themes**: The application supports light and dark themes. Configure these in `.streamlit/config.toml`. The default is set to dark mode.
* **Styling**: Custom CSS is located in `static/style.css`. Modify this file to change the application's appearance.

## Future Enhancements

* Full implementation of advanced ICT concepts (e.g., full Breaker+FVG for Unicorn, SIBI/BISI context).
* Support for more asset classes and exchanges.
* Integration with a persistent database for storing results and configurations.
* More sophisticated signal filtering and context analysis.
* Unit and integration tests for increased robustness.
* AI/ML model integration for predictive analysis or adaptive parameter tuning.
* Enhanced interactive charting features.

## Disclaimer

This is a financial modeling tool for educational and research purposes. Trading financial markets involves substantial risk of loss and is not suitable for every investor. Past performance and optimization results are not indicative of future results and can be subject to overfitting. Always practice responsible risk management and consult with a qualified financial advisor before making any trading decisions.
