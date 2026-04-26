# InvestPortfolio Optimizer

A desktop application for automated portfolio optimisation using evolutionary algorithms, Reinforcement Learning (PPO), and classical mean-variance methods. Designed for academic research and demonstration, it enables users to construct optimal investment portfolios under cardinality constraints, backtest strategies, and visualize results through a modern PySide6 GUI.

---

## Features

- **Hybrid Evolutionary Optimizer:** Genetic algorithm with gradient ascent and SLSQP refinement for portfolio weights.
- **Reinforcement Learning (PPO):** Train a PPO agent (stable-baselines3) to allocate assets using a custom Gymnasium environment.
- **Classical Mean-Variance:** Markowitz optimization via PyPortfolioOpt.
- **Plugin System:** Easily add custom optimization algorithms as plugins.
- **Backtesting Engine:** Simulate historical performance with periodic rebalancing and detailed metrics.
- **S&P 500 Data:** Downloads and stores 30 years of weekly price data in a local SQLite database.
- **PySide6 GUI:** Modern, dark-themed interface with loading and main windows, interactive charts, and progress feedback.

---

## Architecture

- **Facade Pattern:** `PortfolioCore` is the single entry point for all subsystems (DB, optimization, RL, backtesting, plugins).
- **Repository Pattern:** `PortfolioRepository` abstracts all database operations.
- **Plugin System:** User-supplied optimizers auto-discovered via `PluginManager`.
- **QThread Workers:** All heavy operations run in background threads, communicating with the UI via Qt signals.
- **Deferred Imports:** RL/ML dependencies are imported only when needed.

**Main Layers:**
- UI Layer (PySide6)
- Facade Layer (`PortfolioCore`)
- Optimizer, RL, Backtesting, Plugin subsystems
- Data Layer (SQLite, SQLAlchemy)

---

## Project Structure

```
Diploma/
├── CLAUDE.md                # Project technical specification
├── README.md                # This file
├── requirements.txt         # Python dependencies (see below)
├── documentation.txt        # Auto-generated API docs
├── run_test.sh              # LSTM test script (legacy)
├── clean_styles.py          # Utility: removes hardcoded stylesheets
├── resources/db/portfolio.db# SQLite database (auto-created)
├── models/ppo_portfolio/    # Trained PPO models
├── app/
│   ├── main.py              # Entry point: GUI launcher
│   ├── core/                # Facade layer
│   ├── algorithms/          # Hybrid evolutionary optimizer
│   ├── ai/                  # PPO RL subsystem
│   ├── backtesting/         # Backtesting engine
│   ├── data/                # Data access and DB
│   ├── plugins/             # Plugin optimizers
│   ├── ui/                  # GUI and widgets
│   ├── tests/               # Pytest-based tests
│   └── ...                  # Additional scripts and modules
```

---

## Installation

### Prerequisites
- **Python 3.11 (optimal)** — due to matplotlib's compatibility with PySide6
- macOS (tested), Linux/Windows may work
- Virtual environment recommended

### Install Dependencies
```bash
cd /Users/macbook/Documents/Diploma
pip install -r requirements.txt
# For PPO functionality (not in requirements.txt):
pip install stable-baselines3 gymnasium
```

### Populate the Database
Download S&P 500 price data (30 years, weekly):
```bash
python app/run_downloader.py
```
The database is created at `resources/db/portfolio.db`.

---

## Usage

### Launch the GUI
```bash
python app/main.py
```

### Run Tests
```bash
pytest app/tests/ -v
```

### Run Optimizer Standalone
```bash
python app/algorithms/hybrid_evo_optimizer.py
```

### Run Algorithm Tournament (Markowitz vs Hybrid Evo)
```bash
python app/check_optimization.py
```

### Train PPO Agent
```bash
python app/train_ppo.py
# or for direct mode:
python app/train_ppo.py --mode direct
```

---

## Key Concepts
- **Penalised Sharpe Ratio:** Fitness = Sharpe − penalties (enforces sum-to-1, cardinality, no shorting)
- **Cardinality Constraint:** Max K assets per portfolio (default 15)
- **EWMA Returns & Ledoit-Wolf Covariance:** Robust statistics for optimizer
- **Variance Thermostat:** Adaptive mutation rates in GA
- **Curriculum Learning:** PPO trains on sub-windows before full data
- **Rebalancing:** Backtester supports periodic rebalancing (monthly, quarterly, or buy-and-hold)
- **Plugin Example:** See `app/plugins/inverse_volatility.py` for a custom optimizer

---

## Development & Contribution

- **Code Style:** English code, Ukrainian comments/UI, type hints, NumPy docstrings
- **Add a Plugin:**
  1. Create a `.py` file in `app/plugins/` inheriting from `BaseOptimizer`
  2. Implement `optimize(prices_df, config_dict) -> Dict[str, float]`
  3. Auto-discovered by `PluginManager`
- **Add a Subsystem:**
  1. Add method to `PortfolioCore` (facade)
  2. Place module under `app/`
  3. Use deferred imports for heavy dependencies
- **Add a UI Page:**
  1. Create widget in `app/ui/widget/`
  2. Register in `main_window.py`
  3. Add sidebar button and connect

---




---

## References & Resources
- [PyPortfolioOpt Documentation](https://pyportfolioopt.readthedocs.io/)
- [stable-baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Yahoo Finance](https://finance.yahoo.com/) (data source)
- [Wikipedia: S&P 500 Companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)

---

## License

© 2026. Academic/research use only. See `CLAUDE.md` for full details.
