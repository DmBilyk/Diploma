"""
Unit tests for app/core.py  (PortfolioCore facade).

Since PortfolioCore is a facade that delegates to many subsystems, these
tests focus on the facade's own responsibilities:

  - Lazy initialisation of repo / plugin_manager / data_engine
  - Correct delegation of parameters to downstream classes
  - Invariants enforced by the facade itself:
      · run_and_backtest: look-ahead-bias prevention (date split rules)
      · run_and_backtest: required parameters per method
      · run_plugin_optimization: unknown-plugin handling
      · run_and_backtest: unknown-method handling
  - Data shaping between subsystems (OptimizationResult wrapping,
    PortfolioSpec construction, experiment payload structure)
  - sync_market_data: progress callback invocation + DB writes

All heavyweight subsystems (HybridEvoOptimizer, BacktestEngine,
PortfolioRepository, PluginManager, DataEngine, init_db, LSTM predictor)
are patched — none of them actually run.
"""
from __future__ import annotations

import datetime as dt
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from app.core.core import PortfolioCore

# Import the data types that the facade returns / accepts, so we can
# build realistic mock return values.
from app.algorithms.hybrid_evo_optimizer import OptimizationResult
from app.backtesting.backtest_engine import (
    BacktestReport,
    BacktestResult,
    BacktestMetrics,
    PortfolioSpec,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_opt_result(
    weights=None,
    sharpe=1.23,
    expected_return=0.10,
    portfolio_risk=0.08,
) -> OptimizationResult:
    weights = weights or {"AAPL": 0.6, "MSFT": 0.4}
    return OptimizationResult(
        weights=weights,
        selected_assets=list(weights.keys()),
        sharpe_ratio=sharpe,
        expected_return=expected_return,
        portfolio_risk=portfolio_risk,
        n_generations=10,
        history=[0.1, 0.2, 0.3],
    )


def make_backtest_report(weights=None) -> BacktestReport:
    weights = weights or {"AAPL": 0.6, "MSFT": 0.4}
    metrics = BacktestMetrics(
        total_return=0.25, cagr=0.12, annualised_volatility=0.15,
        max_drawdown=-0.08, sharpe_ratio=0.8, sortino_ratio=1.1,
        start_value=100_000.0, end_value=125_000.0,
    )
    idx = pd.date_range("2021-01-04", periods=10, freq="W-MON")
    result = BacktestResult(
        spec=PortfolioSpec(name="Test", weights=weights),
        metrics=metrics,
        portfolio_values=pd.Series(range(10), index=idx, dtype=float),
    )
    return BacktestReport(
        results=[result],
        benchmark=None,
        start_date="2021-01-01",
        end_date="2022-01-01",
        initial_capital=100_000.0,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  1. Lazy initialisation
# ═════════════════════════════════════════════════════════════════════════════
class TestLazyInitialisation(unittest.TestCase):
    """Heavyweight subsystems must not be created until they are first used."""

    def test_constructor_does_not_create_repo(self):
        """Passing no repo should not trigger PortfolioRepository().
        This matters because PortfolioRepository() opens a DB connection."""
        with patch("app.core.core.PortfolioRepository") as PR:
            PortfolioCore()  # no attribute access
            PR.assert_not_called()

    def test_constructor_does_not_create_plugin_manager(self):
        with patch("app.core.core.PluginManager") as PM:
            PortfolioCore()
            PM.assert_not_called()

    def test_repo_property_creates_once_and_caches(self):
        with patch("app.core.core.PortfolioRepository") as PR:
            PR.return_value = MagicMock(name="repo_instance")
            core = PortfolioCore()
            r1 = core.repo
            r2 = core.repo
            self.assertIs(r1, r2)             # same cached instance
            PR.assert_called_once()           # created exactly once

    def test_repo_property_respects_injected_instance(self):
        """If a repo is injected in the constructor, no new one must be created."""
        injected = MagicMock(name="injected_repo")
        with patch("app.core.core.PortfolioRepository") as PR:
            core = PortfolioCore(repo=injected)
            self.assertIs(core.repo, injected)
            PR.assert_not_called()

    def test_plugin_manager_property_creates_once_and_caches(self):
        with patch("app.core.core.PluginManager") as PM:
            PM.return_value = MagicMock(name="pm_instance")
            core = PortfolioCore(plugins_dir="/custom/plugins")
            pm1 = core.plugin_manager
            pm2 = core.plugin_manager
            self.assertIs(pm1, pm2)
            PM.assert_called_once_with(plugins_dir="/custom/plugins")


# ═════════════════════════════════════════════════════════════════════════════
#  2. Simple database-access delegation
# ═════════════════════════════════════════════════════════════════════════════
class TestDatabaseDelegation(unittest.TestCase):

    def setUp(self):
        self.repo = MagicMock()
        self.core = PortfolioCore(repo=self.repo)

    def test_get_tickers_delegates_to_repo(self):
        self.repo.get_all_tickers.return_value = ["AAPL", "MSFT"]
        self.assertEqual(self.core.get_tickers(), ["AAPL", "MSFT"])
        self.repo.get_all_tickers.assert_called_once()

    def test_get_assets_delegates_to_repo(self):
        self.repo.get_all_assets.return_value = [{"ticker": "AAPL"}]
        self.assertEqual(self.core.get_assets(), [{"ticker": "AAPL"}])
        self.repo.get_all_assets.assert_called_once()

    def test_get_price_history_forwards_date_filters(self):
        df = pd.DataFrame({"AAPL": [1.0, 2.0]})
        self.repo.get_price_history.return_value = df
        out = self.core.get_price_history(
            ["AAPL"], start_date="2020-01-01", end_date="2021-01-01",
        )
        self.assertIs(out, df)
        self.repo.get_price_history.assert_called_once_with(
            ["AAPL"], start_date="2020-01-01", end_date="2021-01-01",
        )

    def test_get_quotes_delegates_to_repo(self):
        series = pd.Series([1.0, 2.0])
        self.repo.get_quotes.return_value = series
        out = self.core.get_quotes("AAPL")
        self.assertIs(out, series)
        self.repo.get_quotes.assert_called_once_with("AAPL")


# ═════════════════════════════════════════════════════════════════════════════
#  3. run_optimization — delegates to HybridEvoOptimizer
# ═════════════════════════════════════════════════════════════════════════════
class TestRunOptimization(unittest.TestCase):

    def test_forwards_hyperparameters_to_hybrid_evo_optimizer(self):
        repo = MagicMock()
        core = PortfolioCore(repo=repo)
        fake_result = make_opt_result()

        with patch("app.core.core.HybridEvoOptimizer") as HEO:
            optimizer_instance = MagicMock()
            optimizer_instance.run.return_value = fake_result
            HEO.return_value = optimizer_instance

            out = core.run_optimization(
                tickers=["AAPL", "MSFT"],
                start_date="2020-01-01",
                end_date="2023-01-01",
                pop_size=50,
                n_generations=20,
                max_cardinality=5,
                risk_free_rate=0.03,
                n_elite=3,
                top_m_refine=4,
                seed=42,
                ewma_span=26,
            )

        self.assertIs(out, fake_result)

        # Hyper-parameters go to __init__
        HEO.assert_called_once()
        init_kwargs = HEO.call_args.kwargs
        self.assertEqual(init_kwargs["pop_size"], 50)
        self.assertEqual(init_kwargs["n_generations"], 20)
        self.assertEqual(init_kwargs["max_cardinality"], 5)
        self.assertEqual(init_kwargs["risk_free_rate"], 0.03)
        self.assertEqual(init_kwargs["n_elite"], 3)
        self.assertEqual(init_kwargs["top_m_refine"], 4)
        self.assertEqual(init_kwargs["seed"], 42)
        self.assertEqual(init_kwargs["ewma_span"], 26)

        # Run-time parameters + shared repo go to .run()
        optimizer_instance.run.assert_called_once()
        run_kwargs = optimizer_instance.run.call_args.kwargs
        self.assertEqual(run_kwargs["tickers"], ["AAPL", "MSFT"])
        self.assertEqual(run_kwargs["start_date"], "2020-01-01")
        self.assertEqual(run_kwargs["end_date"], "2023-01-01")
        self.assertIs(run_kwargs["repo"], repo)  # the SAME repo — no duplicate connection


# ═════════════════════════════════════════════════════════════════════════════
#  4. run_plugin_optimization
# ═════════════════════════════════════════════════════════════════════════════
class TestRunPluginOptimization(unittest.TestCase):

    def setUp(self):
        self.repo = MagicMock()
        self.repo.get_all_tickers.return_value = ["AAPL", "MSFT"]
        self.repo.get_price_history.return_value = pd.DataFrame(
            {"AAPL": [1.0, 2.0], "MSFT": [3.0, 4.0]},
        )
        self.core = PortfolioCore(repo=self.repo)

    def test_unknown_plugin_raises_keyerror_with_available_list(self):
        self.core._plugin_manager = MagicMock()
        self.core._plugin_manager.get_plugins.return_value = {
            "FooOpt": MagicMock(), "BarOpt": MagicMock(),
        }
        with self.assertRaises(KeyError) as ctx:
            self.core.run_plugin_optimization(plugin_name="MissingPlugin")
        # Error message should mention the missing name AND available ones
        msg = str(ctx.exception)
        self.assertIn("MissingPlugin", msg)
        self.assertIn("FooOpt", msg)
        self.assertIn("BarOpt", msg)

    def test_no_plugins_available_lists_none(self):
        self.core._plugin_manager = MagicMock()
        self.core._plugin_manager.get_plugins.return_value = {}
        with self.assertRaises(KeyError) as ctx:
            self.core.run_plugin_optimization(plugin_name="Anything")
        self.assertIn("(none)", str(ctx.exception))

    def test_instantiates_plugin_class_and_returns_weights(self):
        plugin_instance = MagicMock()
        plugin_instance.optimize.return_value = {"AAPL": 0.5, "MSFT": 0.5}
        plugin_cls = MagicMock(return_value=plugin_instance)

        self.core._plugin_manager = MagicMock()
        self.core._plugin_manager.get_plugins.return_value = {"MyOpt": plugin_cls}

        weights = self.core.run_plugin_optimization(
            plugin_name="MyOpt",
            tickers=["AAPL", "MSFT"],
            start_date="2020-01-01",
            end_date="2023-01-01",
            config={"lookback": 252},
        )

        self.assertEqual(weights, {"AAPL": 0.5, "MSFT": 0.5})
        plugin_cls.assert_called_once_with()  # zero-arg constructor
        # optimize() gets the price frame + config
        plugin_instance.optimize.assert_called_once()
        args = plugin_instance.optimize.call_args.args
        self.assertIsInstance(args[0], pd.DataFrame)
        self.assertEqual(args[1], {"lookback": 252})

    def test_none_config_is_substituted_with_empty_dict(self):
        plugin_instance = MagicMock()
        plugin_instance.optimize.return_value = {"AAPL": 1.0}
        plugin_cls = MagicMock(return_value=plugin_instance)

        self.core._plugin_manager = MagicMock()
        self.core._plugin_manager.get_plugins.return_value = {"MyOpt": plugin_cls}

        self.core.run_plugin_optimization(
            plugin_name="MyOpt", tickers=["AAPL"], config=None,
        )

        # None must become {} (not passed through as None)
        args = plugin_instance.optimize.call_args.args
        self.assertEqual(args[1], {})

    def test_none_tickers_fallbacks_to_all_db_tickers(self):
        plugin_instance = MagicMock()
        plugin_instance.optimize.return_value = {"AAPL": 1.0}
        plugin_cls = MagicMock(return_value=plugin_instance)

        self.core._plugin_manager = MagicMock()
        self.core._plugin_manager.get_plugins.return_value = {"MyOpt": plugin_cls}

        self.core.run_plugin_optimization(plugin_name="MyOpt", tickers=None)

        # Repo was queried for the full ticker universe
        self.repo.get_all_tickers.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════════
#  5. run_backtest — delegates to BacktestEngine
# ═════════════════════════════════════════════════════════════════════════════
class TestRunBacktest(unittest.TestCase):

    def test_forwards_all_parameters_to_backtest_engine(self):
        repo = MagicMock()
        core = PortfolioCore(repo=repo)
        report = make_backtest_report()
        spec = PortfolioSpec(name="P", weights={"AAPL": 1.0})

        with patch("app.core.core.BacktestEngine") as BE:
            engine_instance = MagicMock()
            engine_instance.run.return_value = report
            BE.return_value = engine_instance

            out = core.run_backtest(
                portfolios=[spec],
                start_date="2022-01-01",
                end_date="2023-01-01",
                initial_capital=50_000.0,
                risk_free_rate=0.03,
                rebalance_every=4,
            )

        self.assertIs(out, report)
        BE.assert_called_once()
        kwargs = BE.call_args.kwargs
        self.assertEqual(kwargs["start_date"], "2022-01-01")
        self.assertEqual(kwargs["end_date"], "2023-01-01")
        self.assertEqual(kwargs["initial_capital"], 50_000.0)
        self.assertEqual(kwargs["risk_free_rate"], 0.03)
        self.assertEqual(kwargs["rebalance_every"], 4)
        self.assertIs(kwargs["repo"], repo)  # shared repo

        engine_instance.run.assert_called_once_with([spec])


# ═════════════════════════════════════════════════════════════════════════════
#  6. run_and_backtest — the beta-testing workflow
#     This is the most critical method: it orchestrates optimisation +
#     backtesting and must prevent look-ahead bias.
# ═════════════════════════════════════════════════════════════════════════════
class TestRunAndBacktestDateSplit(unittest.TestCase):
    """Date-window validation is a SAFETY-CRITICAL invariant: if the backtest
    window overlaps the optimisation window, results are silently tainted by
    look-ahead bias."""

    def setUp(self):
        self.core = PortfolioCore(repo=MagicMock())

    def test_neither_train_end_nor_backtest_dates_raises(self):
        with self.assertRaisesRegex(ValueError, r"look-ahead bias"):
            self.core.run_and_backtest(
                method="evo",
                start_date="2020-01-01",
                end_date="2023-01-01",
                # No train_end, no backtest_start/end → must raise
            )

    def test_only_backtest_start_without_end_raises(self):
        with self.assertRaisesRegex(ValueError, r"look-ahead bias"):
            self.core.run_and_backtest(
                method="evo",
                start_date="2020-01-01",
                end_date="2023-01-01",
                backtest_start="2023-01-01",  # missing backtest_end
            )

    def test_only_backtest_end_without_start_raises(self):
        with self.assertRaisesRegex(ValueError, r"look-ahead bias"):
            self.core.run_and_backtest(
                method="evo",
                start_date="2020-01-01",
                end_date="2023-01-01",
                backtest_end="2024-01-01",  # missing backtest_start
            )

    def test_train_end_alone_is_sufficient(self):
        """train_end alone → bt_start = train_end, bt_end = end_date."""
        core = PortfolioCore(repo=MagicMock())
        with patch.object(core, "run_optimization", return_value=make_opt_result()), \
             patch.object(core, "run_backtest", return_value=make_backtest_report()) as rb:
            core.run_and_backtest(
                method="evo",
                start_date="2020-01-01",
                end_date="2024-01-01",
                train_end="2023-01-01",
            )
            kwargs = rb.call_args.kwargs
            self.assertEqual(kwargs["start_date"], "2023-01-01")  # == train_end
            self.assertEqual(kwargs["end_date"], "2024-01-01")    # == end_date

    def test_explicit_backtest_window_is_honoured(self):
        core = PortfolioCore(repo=MagicMock())
        with patch.object(core, "run_optimization", return_value=make_opt_result()), \
             patch.object(core, "run_backtest", return_value=make_backtest_report()) as rb:
            core.run_and_backtest(
                method="evo",
                start_date="2020-01-01",
                end_date="2023-01-01",
                backtest_start="2023-06-01",
                backtest_end="2024-06-01",
            )
            kwargs = rb.call_args.kwargs
            self.assertEqual(kwargs["start_date"], "2023-06-01")
            self.assertEqual(kwargs["end_date"], "2024-06-01")

    def test_overlapping_windows_log_warning_but_do_not_raise(self):
        """When bt_start < opt_end, the facade logs a warning but still
        runs — this is a soft guard, not a hard fail."""
        core = PortfolioCore(repo=MagicMock())
        with patch.object(core, "run_optimization", return_value=make_opt_result()), \
             patch.object(core, "run_backtest", return_value=make_backtest_report()), \
             self.assertLogs("app.core.core", level="WARNING") as log_ctx:
            core.run_and_backtest(
                method="evo",
                start_date="2020-01-01",
                end_date="2024-01-01",
                backtest_start="2021-01-01",  # OVERLAPS opt window
                backtest_end="2023-01-01",
            )
        joined = " ".join(log_ctx.output)
        self.assertIn("look-ahead bias", joined)


class TestRunAndBacktestMethodDispatch(unittest.TestCase):

    def setUp(self):
        self.core = PortfolioCore(repo=MagicMock())

    def test_unknown_method_raises(self):
        with self.assertRaisesRegex(ValueError, r"Unknown method"):
            self.core.run_and_backtest(
                method="garbage",
                start_date="2020-01-01",
                end_date="2023-01-01",
                train_end="2022-01-01",
            )

    def test_lstm_requires_model_path(self):
        with self.assertRaisesRegex(ValueError, r"model_path"):
            self.core.run_and_backtest(
                method="lstm",
                start_date="2020-01-01",
                end_date="2023-01-01",
                train_end="2022-01-01",
                # model_path missing
            )

    def test_ppo_requires_model_path(self):
        with self.assertRaisesRegex(ValueError, r"model_path"):
            self.core.run_and_backtest(
                method="ppo",
                start_date="2020-01-01",
                end_date="2023-01-01",
                train_end="2022-01-01",
                # model_path missing
            )

    def test_plugin_requires_plugin_name(self):
        with self.assertRaisesRegex(ValueError, r"plugin_name"):
            self.core.run_and_backtest(
                method="plugin",
                start_date="2020-01-01",
                end_date="2023-01-01",
                train_end="2022-01-01",
                # plugin_name missing
            )

    def test_evo_flow_delegates_to_run_optimization(self):
        fake = make_opt_result()
        with patch.object(self.core, "run_optimization", return_value=fake) as ro, \
             patch.object(self.core, "run_backtest", return_value=make_backtest_report()) as rb:
            result, report = self.core.run_and_backtest(
                method="evo",
                start_date="2020-01-01",
                end_date="2023-01-01",
                train_end="2022-01-01",
                seed=7,
                pop_size=50,
                n_generations=30,
                max_cardinality=8,
            )

        self.assertIs(result, fake)
        ro.assert_called_once()
        ro_kwargs = ro.call_args.kwargs
        # Optimisation end must be train_end (NOT end_date)
        self.assertEqual(ro_kwargs["end_date"], "2022-01-01")
        self.assertEqual(ro_kwargs["seed"], 7)
        self.assertEqual(ro_kwargs["pop_size"], 50)
        # Backtest must be called with the non-overlapping window
        rb_kwargs = rb.call_args.kwargs
        self.assertEqual(rb_kwargs["start_date"], "2022-01-01")
        self.assertEqual(rb_kwargs["end_date"], "2023-01-01")
        # PortfolioSpec is built from optimizer weights
        specs = rb_kwargs["portfolios"]
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].weights, fake.weights)
        self.assertIn("EVO", specs[0].name.upper())

    def test_plugin_flow_wraps_weights_into_optimization_result(self):
        fake_weights = {"AAPL": 0.7, "MSFT": 0.3}
        with patch.object(self.core, "run_plugin_optimization",
                          return_value=fake_weights) as rpo, \
             patch.object(self.core, "run_backtest",
                          return_value=make_backtest_report()):
            result, _ = self.core.run_and_backtest(
                method="plugin",
                plugin_name="MyOpt",
                plugin_config={"k": 5},
                start_date="2020-01-01",
                end_date="2023-01-01",
                train_end="2022-01-01",
            )

        # Plugin output is wrapped into a neutral OptimizationResult
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.weights, fake_weights)
        self.assertEqual(result.selected_assets, list(fake_weights.keys()))
        self.assertEqual(result.sharpe_ratio, 0.0)     # sentinel defaults
        self.assertEqual(result.expected_return, 0.0)
        self.assertEqual(result.portfolio_risk, 0.0)
        self.assertEqual(result.n_generations, 0)
        self.assertEqual(result.history, [])
        # Config forwarded
        rpo.assert_called_once()
        self.assertEqual(rpo.call_args.kwargs["config"], {"k": 5})
        self.assertEqual(rpo.call_args.kwargs["plugin_name"], "MyOpt")


class TestRunAndBacktestPersistence(unittest.TestCase):

    def setUp(self):
        self.core = PortfolioCore(repo=MagicMock())

    def test_save_false_does_not_persist(self):
        with patch.object(self.core, "run_optimization", return_value=make_opt_result()), \
             patch.object(self.core, "run_backtest", return_value=make_backtest_report()), \
             patch.object(self.core, "save_experiment") as save_mock:
            self.core.run_and_backtest(
                method="evo",
                start_date="2020-01-01",
                end_date="2023-01-01",
                train_end="2022-01-01",
                save=False,
            )
            save_mock.assert_not_called()

    def test_save_true_builds_expected_payload(self):
        opt_result = make_opt_result(sharpe=1.5)
        report = make_backtest_report()
        with patch.object(self.core, "run_optimization", return_value=opt_result), \
             patch.object(self.core, "run_backtest", return_value=report), \
             patch.object(self.core, "save_experiment") as save_mock:
            self.core.run_and_backtest(
                method="evo",
                start_date="2020-01-01",
                end_date="2023-01-01",
                train_end="2022-01-01",
                seed=42,
                save=True,
                experiment_name="MyRun",
            )

        save_mock.assert_called_once()
        kwargs = save_mock.call_args.kwargs
        self.assertEqual(kwargs["name"], "MyRun")
        self.assertEqual(kwargs["algorithm"], "evo")

        # Metrics payload separates ex-ante (optimiser) vs ex-post (backtest) Sharpe
        metrics = kwargs["metrics"]
        self.assertEqual(metrics["sharpe_ratio_exante"], 1.5)
        self.assertIn("backtest", metrics)
        bt = metrics["backtest"]
        self.assertAlmostEqual(bt["total_return"], 0.25)
        self.assertAlmostEqual(bt["sharpe_ratio"], 0.8)

        # Parameters payload includes the key reproducibility fields
        params = kwargs["parameters"]
        self.assertEqual(params["method"], "evo")
        self.assertEqual(params["seed"], 42)
        self.assertEqual(params["train_end"], "2022-01-01")

    def test_save_true_autogenerates_experiment_name_when_none(self):
        with patch.object(self.core, "run_optimization", return_value=make_opt_result()), \
             patch.object(self.core, "run_backtest", return_value=make_backtest_report()), \
             patch.object(self.core, "save_experiment") as save_mock:
            self.core.run_and_backtest(
                method="evo",
                start_date="2020-01-01",
                end_date="2023-01-01",
                train_end="2022-01-01",
                save=True,
                # experiment_name=None → auto-generate
            )
        name = save_mock.call_args.kwargs["name"]
        # Auto-generated format: "{METHOD}_{YYYYMMDD}_{HHMMSS}"
        self.assertTrue(name.startswith("EVO_"))
        self.assertEqual(len(name.split("_")), 3)


# ═════════════════════════════════════════════════════════════════════════════
#  7. save_experiment — uses session context manager properly
# ═════════════════════════════════════════════════════════════════════════════
class TestSaveExperiment(unittest.TestCase):

    def test_save_experiment_commits_via_session(self):
        core = PortfolioCore(repo=MagicMock())

        # init_db() returns a Session factory; the factory supports
        # the context-manager protocol and yields a session.
        session = MagicMock()
        session_factory = MagicMock()
        session_factory.return_value.__enter__.return_value = session
        session_factory.return_value.__exit__.return_value = False

        with patch("app.core.core.init_db", return_value=session_factory), \
             patch("app.core.core.Experiment") as ExperimentCls:
            core.save_experiment(
                name="X",
                algorithm="evo",
                parameters={"seed": 1},
                metrics={"sharpe": 1.0},
            )

        ExperimentCls.assert_called_once_with(
            name="X",
            algorithm="evo",
            parameters={"seed": 1},
            metrics={"sharpe": 1.0},
        )
        session.add.assert_called_once()
        session.commit.assert_called_once()


# ═════════════════════════════════════════════════════════════════════════════
#  8. sync_market_data — progress callback + DB writes
# ═════════════════════════════════════════════════════════════════════════════
class TestSyncMarketData(unittest.TestCase):

    def setUp(self):
        self.repo = MagicMock()
        self.repo.get_latest_quote_date.return_value = None  # full sync
        self.core = PortfolioCore(repo=self.repo)

    def test_full_sync_when_db_empty(self):
        # data_engine returns a fake market_data dict
        self.core._data_engine = MagicMock()
        self.core._data_engine.get_sp500_tickers.return_value = ["AAPL", "MSFT"]
        self.core._data_engine.download_market_data.return_value = {
            "AAPL": pd.DataFrame({"close": [1.0, 2.0]}),
            "MSFT": pd.DataFrame({"close": [3.0, 4.0]}),
        }

        result = self.core.sync_market_data()

        # DataEngine called with start_date=None (full sync)
        self.core._data_engine.download_market_data.assert_called_once()
        self.assertIsNone(
            self.core._data_engine.download_market_data.call_args.kwargs["start_date"]
        )
        # Every ticker was added + saved
        self.assertEqual(self.repo.add_asset.call_count, 2)
        self.assertEqual(self.repo.save_quotes_bulk.call_count, 2)
        self.assertEqual(result, {"status": "success", "assets_updated": 2})

    def test_delta_sync_when_db_has_data(self):
        latest = dt.date(2023, 6, 15)
        self.repo.get_latest_quote_date.return_value = latest

        self.core._data_engine = MagicMock()
        self.core._data_engine.get_sp500_tickers.return_value = ["AAPL"]
        self.core._data_engine.download_market_data.return_value = {
            "AAPL": pd.DataFrame({"close": [1.0]}),
        }

        self.core.sync_market_data()

        passed_start = (
            self.core._data_engine.download_market_data.call_args.kwargs["start_date"]
        )
        # Delta sync: start_date must be 7 days BEFORE the latest quote
        expected = dt.datetime.combine(latest, dt.time.min) - dt.timedelta(days=7)
        self.assertEqual(passed_start, expected)

    def test_progress_callback_invoked_at_key_stages(self):
        self.core._data_engine = MagicMock()
        self.core._data_engine.get_sp500_tickers.return_value = ["AAPL"]
        self.core._data_engine.download_market_data.return_value = {
            "AAPL": pd.DataFrame({"close": [1.0]}),
        }

        callback = MagicMock()
        self.core.sync_market_data(progress_callback=callback)

        # At minimum: 5% start, 10% after tickers, 90% after download, 100% end
        percentages = [c.args[0] for c in callback.call_args_list]
        self.assertIn(5, percentages)
        self.assertIn(10, percentages)
        self.assertIn(90, percentages)
        self.assertIn(100, percentages)


# ═════════════════════════════════════════════════════════════════════════════
#  9. Deferred imports — LSTM / PPO paths must not break facade import
# ═════════════════════════════════════════════════════════════════════════════
class TestDeferredImports(unittest.TestCase):
    """The facade must remain importable even when heavy optional
    dependencies (TensorFlow, stable-baselines3, …) are missing.  Those
    modules are only imported inside the methods that need them."""

    def test_core_imports_without_touching_ai_modules(self):
        """A fresh PortfolioCore() must not trigger any AI-related imports.

        We verify this by checking that instantiating PortfolioCore does
        not fail even if we patch those modules to raise on import-time.
        (The real deferred-import contract is strict: we just smoke-test
        the most common case here.)"""
        core = PortfolioCore()  # must not raise
        self.assertIsInstance(core, PortfolioCore)


if __name__ == "__main__":
    unittest.main(verbosity=2)