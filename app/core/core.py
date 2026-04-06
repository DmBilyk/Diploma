"""
core.py
=======

Facade (GoF) for the portfolio-optimisation platform.

``PortfolioCore`` is the **single entry point** that the UI layer, CLI
scripts, and tests use to interact with every subsystem:

* **Database**      – asset / quote CRUD via ``PortfolioRepository``
* **Optimisation**  – evolutionary search via ``HybridEvoOptimizer``
* **AI / LSTM**     – neural-network-driven expected returns
* **Plugins**       – user-supplied optimisation algorithms
* **Backtesting**   – retrospective portfolio evaluation

A compound ``run_and_backtest()`` method implements the
**beta-testing** workflow: optimise → immediately backtest →
optionally persist the experiment.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from app.algorithms.hybrid_evo_optimizer import (
    HybridEvoOptimizer,
    OptimizationResult,
)
from app.backtesting.backtest_engine import (
    BacktestEngine,
    BacktestReport,
    PortfolioSpec,
)
from app.data.models import Experiment, init_db
from app.data.repository import PortfolioRepository
from app.plugins.base_optimizer import BaseOptimizer
from app.plugins.plugin_manager import PluginManager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  FACADE
# ═══════════════════════════════════════════════════════════════════════════


class PortfolioCore:
    """Unified facade for the portfolio-optimisation platform.

    All heavyweight objects (repository, plugin manager, …) are created
    **lazily** on first access so that instantiating ``PortfolioCore()``
    is always cheap.

    Parameters
    ----------
    repo : PortfolioRepository | None
        Inject a custom repository (useful for testing).
        *None* → created lazily via ``init_db()``.
    plugins_dir : str | None
        Override the directory scanned by ``PluginManager``.
    """

    def __init__(
        self,
        repo: Optional[PortfolioRepository] = None,
        plugins_dir: Optional[str] = None,
    ) -> None:
        self._repo = repo
        self._plugins_dir = plugins_dir
        self._plugin_manager: Optional[PluginManager] = None
        self._data_engine = None

    # ── lazy properties ────────────────────────────────────────────────

    @property
    def repo(self) -> PortfolioRepository:
        """Shared ``PortfolioRepository`` instance (created once)."""
        if self._repo is None:
            self._repo = PortfolioRepository()
        return self._repo

    @property
    def plugin_manager(self) -> PluginManager:
        """Shared ``PluginManager`` instance (created once)."""
        if self._plugin_manager is None:
            self._plugin_manager = PluginManager(plugins_dir=self._plugins_dir)
        return self._plugin_manager

    @property
    def data_engine(self):
        """Shared DataEngine instance (created lazily to keep Core lightweight)."""
        if self._data_engine is None:
            from app.data.data_engine import DataEngine
            self._data_engine = DataEngine()
        return self._data_engine

    # ══════════════════════════════════════════════════════════════════
    #  1. DATABASE ACCESS
    # ══════════════════════════════════════════════════════════════════

    def sync_market_data(self, progress_callback=None) -> dict:
        """
        Розумна синхронізація ринкових даних (дельта-оновлення або повне завантаження).

        Parameters
        ----------
        progress_callback : callable, optional
            Функція вигляду `func(percent: int, message: str)` для оновлення UI.
        """
        if progress_callback:
            progress_callback(5, "Отримання списку активів S&P 500...")

        tickers = self.data_engine.get_sp500_tickers()

        if progress_callback:
            progress_callback(10, "Перевірка стану бази даних...")

        latest_date = self.repo.get_latest_quote_date()

        # Визначаємо start_date для DataEngine
        start_date = None
        if latest_date:
            # Відкочуємося на 7 днів назад від останньої дати в базі для перекриття
            start_date = datetime.datetime.combine(latest_date, datetime.time.min) - datetime.timedelta(days=7)
            logger.info("Starting Delta Update from %s", start_date)
        else:
            logger.info("Starting Full 30-year database sync")

        # Передаємо callback у DataEngine (якщо він там підтримується)
        market_data = self.data_engine.download_market_data(
            tickers,
            start_date=start_date,
            progress_callback=progress_callback
        )

        if progress_callback:
            progress_callback(90, "Формування запитів до БД...")

        total_assets = len(market_data)
        for i, (ticker, df) in enumerate(market_data.items()):
            self.repo.add_asset(ticker, name=ticker, sector="Unknown")
            self.repo.save_quotes_bulk(ticker, df)


            if progress_callback and total_assets > 0:
                if i % max(1, total_assets // 10) == 0:
                    current_pct = 90 + int((i / total_assets) * 9)
                    progress_callback(current_pct, f"Запис у БД: {ticker}...")

        if progress_callback:
            progress_callback(100, "Синхронізація успішно завершена!")

        return {"status": "success", "assets_updated": total_assets}

    def get_tickers(self) -> List[str]:
        """Return every ticker stored in the database."""
        return self.repo.get_all_tickers()

    def get_assets(self) -> List[Dict[str, Any]]:
        """Return full asset metadata (ticker, name, sector)."""
        return self.repo.get_all_assets()

    def get_price_history(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Pivot table of adjusted-close prices from the database.

        Parameters
        ----------
        tickers : list[str]
        start_date, end_date : str | None   ISO-8601 date filters.

        Returns
        -------
        pd.DataFrame – columns = tickers, index = dates.
        """
        return self.repo.get_price_history(
            tickers, start_date=start_date, end_date=end_date,
        )

    def get_quotes(self, ticker: str) -> pd.DataFrame:
        """Single-ticker quote series (for UI charts)."""
        return self.repo.get_quotes(ticker)

    # ══════════════════════════════════════════════════════════════════
    #  2. EVOLUTIONARY OPTIMISATION
    # ══════════════════════════════════════════════════════════════════

    def run_optimization(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: int = 52,
        pop_size: int = 200,
        n_generations: int = 200,
        max_cardinality: int = 15,
        risk_free_rate: float = 0.02,
        n_elite: int = 5,
        top_m_refine: int = 10,
        seed: Optional[int] = None,
        ewma_span: int = 52,
    ) -> OptimizationResult:
        """Run the Hybrid Evolutionary Optimizer.

        Delegates to :class:`HybridEvoOptimizer` and passes the shared
        ``PortfolioRepository`` so that no duplicate DB connections are
        created.

        Returns
        -------
        OptimizationResult
        """
        optimizer = HybridEvoOptimizer(
            pop_size=pop_size,
            n_generations=n_generations,
            max_cardinality=max_cardinality,
            risk_free_rate=risk_free_rate,
            n_elite=n_elite,
            top_m_refine=top_m_refine,
            seed=seed,
            ewma_span=ewma_span,
        )
        result = optimizer.run(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            repo=self.repo,
        )
        logger.info("Evolutionary optimisation finished — Sharpe %.4f", result.sharpe_ratio)
        return result

    # ══════════════════════════════════════════════════════════════════
    #  3. AI / LSTM OPTIMISATION
    # ══════════════════════════════════════════════════════════════════

    def run_lstm_optimization(
        self,
        model_path: str,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        seq_length: int = 52,
        alpha: float = 0.5,
        pop_size: int = 200,
        n_generations: int = 200,
        max_cardinality: int = 15,
        risk_free_rate: float = 0.02,
        n_elite: int = 5,
        top_m_refine: int = 10,
        seed: Optional[int] = None,
    ) -> Tuple[OptimizationResult, Dict[str, float]]:
        """Run LSTM-driven optimisation.

        Loads the pre-trained model from *model_path*, predicts expected
        returns, blends them with historical estimates, and feeds the
        result into the evolutionary optimiser.

        Parameters
        ----------
        model_path : str
            Filesystem path to the saved ``PortfolioLSTMModel``.
        alpha : float
            Blending weight.  1.0 = pure LSTM,  0.0 = pure historical.

        Returns
        -------
        (OptimizationResult, dict[str, float])
            The optimisation result **and** the per-ticker blended μ vector.
        """
        # Deferred import — TensorFlow / Keras may not be installed in
        # every environment, and the facade should remain importable.
        from app.ai.lstm_model import PortfolioLSTMModel
        from app.ai.predictor import run_lstm_optimization as _run

        model = PortfolioLSTMModel.load(model_path)

        result, mu_dict = _run(
            model=model,
            repo=self.repo,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            seq_length=seq_length,
            alpha=alpha,
            pop_size=pop_size,
            n_generations=n_generations,
            max_cardinality=max_cardinality,
            risk_free_rate=risk_free_rate,
            n_elite=n_elite,
            top_m_refine=top_m_refine,
            seed=seed,
        )
        logger.info("LSTM optimisation finished — Sharpe %.4f", result.sharpe_ratio)
        return result, mu_dict

    # ══════════════════════════════════════════════════════════════════
    #  4. PLUGIN-BASED OPTIMISATION
    # ══════════════════════════════════════════════════════════════════

    def get_plugins(self) -> Dict[str, Type[BaseOptimizer]]:
        """Discover available plugin optimisers.

        Returns
        -------
        dict[str, Type[BaseOptimizer]]
            Mapping *class name → class*.
        """
        return self.plugin_manager.get_plugins()

    def run_plugin_optimization(
        self,
        plugin_name: str,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Run a user-supplied plugin optimiser by name.

        Parameters
        ----------
        plugin_name : str
            Registered class name (as returned by ``get_plugins()``).
        config : dict | None
            Arbitrary config forwarded to ``BaseOptimizer.optimize()``.

        Returns
        -------
        dict[str, float]
            Optimal weights ``{ticker: weight}``.

        Raises
        ------
        KeyError
            If *plugin_name* is not found among discovered plugins.
        """
        plugins = self.get_plugins()
        if plugin_name not in plugins:
            available = ", ".join(sorted(plugins)) or "(none)"
            raise KeyError(
                f"Plugin '{plugin_name}' not found. Available: {available}"
            )

        all_tickers = tickers or self.get_tickers()
        prices_df = self.get_price_history(
            all_tickers, start_date=start_date, end_date=end_date,
        )

        optimizer_cls = plugins[plugin_name]
        optimizer: BaseOptimizer = optimizer_cls()
        weights = optimizer.optimize(prices_df, config or {})

        logger.info(
            "Plugin '%s' finished — %d assets selected",
            plugin_name, len(weights),
        )
        return weights

    # ══════════════════════════════════════════════════════════════════
    #  5. BACKTESTING
    # ══════════════════════════════════════════════════════════════════

    def run_backtest(
        self,
        portfolios: List[PortfolioSpec],
        start_date: str,
        end_date: str,
        initial_capital: float = 100_000.0,
        risk_free_rate: float = 0.02,
        rebalance_every: Optional[int] = None,
    ) -> BacktestReport:
        """Backtest one or more portfolios over a historical window.

        Parameters
        ----------
        portfolios : list[PortfolioSpec]
            Each entry carries a name and a weight dictionary.
        start_date, end_date : str
            ISO-8601 date bounds for the simulation.
        initial_capital : float
            Dollar budget at the start.
        rebalance_every : int | None
            Rebalance to target weights every N rows.  *None* = buy-and-hold.
            Since data is weekly: 4 ≈ monthly, 13 ≈ quarterly.

        Returns
        -------
        BacktestReport
        """
        engine = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate,
            repo=self.repo,
            rebalance_every=rebalance_every,
        )
        report = engine.run(portfolios)
        logger.info(
            "Backtest finished — %d portfolio(s), %s → %s, rebalance_every=%s",
            len(portfolios), start_date, end_date, rebalance_every,
        )
        return report

    # ══════════════════════════════════════════════════════════════════
    #  6. EXPERIMENT PERSISTENCE
    # ══════════════════════════════════════════════════════════════════

    def save_experiment(
        self,
        name: str,
        algorithm: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> None:
        """Persist an experiment record to the database.

        Parameters
        ----------
        name : str        Human-readable label.
        algorithm : str   Algorithm identifier (e.g. ``"HybridEvo"``).
        parameters : dict Configuration used for the run.
        metrics : dict    Outcome metrics (Sharpe, return, risk, …).
        """
        Session = init_db()
        with Session() as session:
            experiment = Experiment(
                name=name,
                algorithm=algorithm,
                parameters=parameters,
                metrics=metrics,
            )
            session.add(experiment)
            session.commit()
            logger.info("Experiment '%s' saved (algorithm=%s)", name, algorithm)

    # ══════════════════════════════════════════════════════════════════
    #  7. BETA-TESTING — OPTIMISE + BACKTEST IN ONE CALL
    # ══════════════════════════════════════════════════════════════════

    def run_and_backtest(
        self,
        *,
        # Optimisation parameters
        method: str = "evo",
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: int = 52,
        pop_size: int = 200,
        n_generations: int = 200,
        max_cardinality: int = 15,
        risk_free_rate: float = 0.02,
        seed: Optional[int] = None,
        # LSTM-specific
        model_path: Optional[str] = None,
        alpha: float = 0.5,
        seq_length: int = 52,
        # Plugin-specific
        plugin_name: Optional[str] = None,
        plugin_config: Optional[Dict[str, Any]] = None,
        # Date split — prevents look-ahead bias
        train_end: Optional[str] = None,
        # Backtest parameters
        backtest_start: Optional[str] = None,
        backtest_end: Optional[str] = None,
        initial_capital: float = 100_000.0,
        rebalance_every: int = 4,
        # Persistence
        save: bool = False,
        experiment_name: Optional[str] = None,
    ) -> Tuple[OptimizationResult, BacktestReport]:
        """Optimise → backtest → (optionally) save.  Beta-testing workflow.

        This compound method executes the full beta-testing pipeline:

        1. **Optimise** using the chosen *method* (``"evo"``, ``"lstm"``,
           or ``"plugin"``) over ``[start_date, train_end]``.
        2. **Backtest** the resulting portfolio over ``[backtest_start, backtest_end]``
           (a **non-overlapping** window to avoid look-ahead bias).
        3. **Save** the experiment to the database (if *save=True*).

        Parameters
        ----------
        method : ``"evo"`` | ``"lstm"`` | ``"plugin"``
            Which optimisation back-end to use.
        train_end : str | None
            End of the training/optimisation window.  When provided,
            ``backtest_start`` defaults to ``train_end`` and
            ``backtest_end`` defaults to ``end_date``.
        backtest_start, backtest_end : str | None
            Explicit backtest date window.  Must not overlap with the
            optimisation window.
        rebalance_every : int
            Rebalance to target weights every N data rows.
            Default 4 (≈ monthly for weekly data).
        save : bool
            Whether to persist the experiment.
        experiment_name : str | None
            Label for the saved experiment.  Auto-generated if *None*.

        Returns
        -------
        (OptimizationResult, BacktestReport)

        Raises
        ------
        ValueError
            If *method* is unknown, required parameters are missing, or
            date windows are not specified.
        """
        # ── 0. Resolve date split (prevent look-ahead bias) ──────────
        opt_end = end_date
        if train_end is not None:
            opt_end = train_end
            bt_start = backtest_start or train_end
            bt_end = backtest_end or end_date
        elif backtest_start is not None and backtest_end is not None:
            bt_start = backtest_start
            bt_end = backtest_end
        else:
            raise ValueError(
                "Must provide either 'train_end' or both "
                "'backtest_start'/'backtest_end' to avoid look-ahead bias. "
                "The optimisation and backtest windows must not overlap."
            )

        if bt_start is None or bt_end is None:
            raise ValueError(
                "Backtest date range could not be determined. "
                "Provide 'train_end' or both 'backtest_start'/'backtest_end'."
            )

        # Warn about potential overlap
        effective_opt_end = opt_end or end_date
        if effective_opt_end and bt_start and bt_start < effective_opt_end:
            logger.warning(
                "Backtest start (%s) is before optimisation end (%s) — "
                "potential look-ahead bias!",
                bt_start, effective_opt_end,
            )

        # ── 1. Optimisation ──────────────────────────────────────────
        mu_dict: Optional[Dict[str, float]] = None

        if method == "evo":
            result = self.run_optimization(
                tickers=tickers,
                start_date=start_date,
                end_date=opt_end,
                frequency=frequency,
                pop_size=pop_size,
                n_generations=n_generations,
                max_cardinality=max_cardinality,
                risk_free_rate=risk_free_rate,
                seed=seed,
            )

        elif method == "lstm":
            if model_path is None:
                raise ValueError("model_path is required for method='lstm'")
            result, mu_dict = self.run_lstm_optimization(
                model_path=model_path,
                tickers=tickers,
                start_date=start_date,
                end_date=opt_end,
                seq_length=seq_length,
                alpha=alpha,
                pop_size=pop_size,
                n_generations=n_generations,
                max_cardinality=max_cardinality,
                risk_free_rate=risk_free_rate,
                seed=seed,
            )

        elif method == "plugin":
            if plugin_name is None:
                raise ValueError("plugin_name is required for method='plugin'")
            weights = self.run_plugin_optimization(
                plugin_name=plugin_name,
                tickers=tickers,
                start_date=start_date,
                end_date=opt_end,
                config=plugin_config,
            )
            # Wrap plugin output into an OptimizationResult for uniformity
            result = OptimizationResult(
                weights=weights,
                selected_assets=list(weights.keys()),
                sharpe_ratio=0.0,
                expected_return=0.0,
                portfolio_risk=0.0,
                n_generations=0,
                history=[],
            )

        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose from: 'evo', 'lstm', 'plugin'."
            )

        # ── 2. Backtest ──────────────────────────────────────────────
        spec = PortfolioSpec(
            name=f"{method.upper()} Strategy",
            weights=result.weights,
        )
        report = self.run_backtest(
            portfolios=[spec],
            start_date=bt_start,
            end_date=bt_end,
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate,
            rebalance_every=rebalance_every,
        )

        # ── 3. Persist (optional) ────────────────────────────────────
        if save:
            exp_name = experiment_name or (
                f"{method.upper()}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
            )
            # sharpe_ratio_exante = theoretical (μ,Σ-based) Sharpe from optimizer
            # backtest.sharpe_ratio = ex-post (realized returns) Sharpe
            metrics_to_save: Dict[str, Any] = {
                "sharpe_ratio_exante": result.sharpe_ratio,
                "expected_return": result.expected_return,
                "portfolio_risk": result.portfolio_risk,
                "selected_assets": result.selected_assets,
            }
            if report.results:
                bt_m = report.results[0].metrics
                metrics_to_save["backtest"] = {
                    "total_return": bt_m.total_return,
                    "cagr": bt_m.cagr,
                    "max_drawdown": bt_m.max_drawdown,
                    "sharpe_ratio": bt_m.sharpe_ratio,
                    "sortino_ratio": bt_m.sortino_ratio,
                }
            if mu_dict is not None:
                metrics_to_save["mu_dict"] = mu_dict

            params_to_save: Dict[str, Any] = {
                "method": method,
                "tickers": tickers,
                "start_date": start_date,
                "train_end": opt_end,
                "end_date": end_date,
                "pop_size": pop_size,
                "n_generations": n_generations,
                "max_cardinality": max_cardinality,
                "risk_free_rate": risk_free_rate,
                "seed": seed,
                "backtest_start": bt_start,
                "backtest_end": bt_end,
                "initial_capital": initial_capital,
                "rebalance_every": rebalance_every,
            }
            self.save_experiment(
                name=exp_name,
                algorithm=method,
                parameters=params_to_save,
                metrics=metrics_to_save,
            )

        logger.info(
            "run_and_backtest complete — method=%s  Sharpe(exante)=%.4f",
            method, result.sharpe_ratio,
        )
        return result, report
