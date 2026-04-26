# tests/plugins/test_plugin_system.py
"""
Unit tests for the plugin system: BaseOptimizer contract, concrete plugins,
and PluginManager discovery mechanics.
"""

from __future__ import annotations

import importlib.util
import os
import textwrap
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

N_ASSETS = 4
N_DAYS = 60


def _make_prices(n_days: int = N_DAYS, n_assets: int = N_ASSETS) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = 100 + np.cumsum(rng.normal(0, 1, (n_days, n_assets)), axis=0)
    tickers = [f"T{i}" for i in range(n_assets)]
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def prices_df() -> pd.DataFrame:
    return _make_prices()


# ─────────────────────────────────────────────────────────────────────────────
#  BaseOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseOptimizer:
    """Verifies the abstract interface contract."""

    def test_cannot_instantiate_base_optimizer_directly(self):
        # ABC must prevent direct instantiation without implement optimize()
        from app.plugins.base_optimizer import BaseOptimizer
        with pytest.raises(TypeError):
            BaseOptimizer()

    def test_concrete_subclass_without_optimize_is_still_abstract(self):
        # A subclass that skips optimize() must also be uninstantiable
        from app.plugins.base_optimizer import BaseOptimizer

        class Incomplete(BaseOptimizer):
            pass  # missing optimize()

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_with_optimize_can_be_instantiated(self):
        from app.plugins.base_optimizer import BaseOptimizer

        class Valid(BaseOptimizer):
            def optimize(self, prices_df, config_dict):
                return {}

        instance = Valid()
        assert isinstance(instance, BaseOptimizer)

    def test_optimize_signature_accepts_dataframe_and_dict(self, prices_df):
        from app.plugins.base_optimizer import BaseOptimizer

        class Minimal(BaseOptimizer):
            def optimize(self, prices_df: pd.DataFrame, config_dict: dict) -> Dict[str, float]:
                return {col: 1.0 / len(prices_df.columns) for col in prices_df.columns}

        result = Minimal().optimize(prices_df, {})
        assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
#  EqualWeightOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class TestEqualWeightOptimizer:
    """Tests for the equal-weight (dummy) plugin."""

    @pytest.fixture
    def optimizer(self):
        from app.plugins.dummy_optimizer import EqualWeightOptimizer
        return EqualWeightOptimizer()

    def test_weights_sum_to_one(self, optimizer, prices_df):
        weights = optimizer.optimize(prices_df, {})
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_all_weights_are_equal(self, optimizer, prices_df):
        weights = optimizer.optimize(prices_df, {})
        values = list(weights.values())
        assert all(v == pytest.approx(values[0]) for v in values)

    def test_each_weight_equals_one_over_n(self, optimizer, prices_df):
        weights = optimizer.optimize(prices_df, {})
        expected = 1.0 / N_ASSETS
        assert all(w == pytest.approx(expected) for w in weights.values())

    def test_keys_match_dataframe_columns(self, optimizer, prices_df):
        weights = optimizer.optimize(prices_df, {})
        assert set(weights.keys()) == set(prices_df.columns)

    def test_empty_dataframe_returns_empty_dict(self, optimizer):
        empty = pd.DataFrame()
        result = optimizer.optimize(empty, {})
        assert result == {}

    def test_single_asset_receives_full_weight(self, optimizer):
        single = pd.DataFrame({"ONLY": [100.0, 101.0, 102.0]})
        weights = optimizer.optimize(single, {})
        assert weights == {"ONLY": pytest.approx(1.0)}

    def test_config_dict_has_no_effect_on_equal_weight(self, optimizer, prices_df):
        # EqualWeight ignores config entirely — both calls must return the same result
        w1 = optimizer.optimize(prices_df, {})
        w2 = optimizer.optimize(prices_df, {"max_cardinality": 2, "anything": True})
        assert w1 == w2


# ─────────────────────────────────────────────────────────────────────────────
#  InverseVolatilityOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class TestInverseVolatilityOptimizer:
    """Tests for the inverse-volatility plugin."""

    @pytest.fixture
    def optimizer(self):
        from app.plugins.inverse_volatility import InverseVolatilityOptimizer
        return InverseVolatilityOptimizer()

    def test_weights_sum_to_one(self, optimizer, prices_df):
        weights = optimizer.optimize(prices_df, {})
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_all_weights_are_positive(self, optimizer, prices_df):
        weights = optimizer.optimize(prices_df, {})
        assert all(w > 0 for w in weights.values())

    def test_lower_volatility_asset_receives_higher_weight(self, optimizer):
        # T0 has near-zero variance; T1 has large variance → T0 must dominate
        rng = np.random.default_rng(1)
        data = {
            "T0": 100 + np.cumsum(rng.normal(0, 0.01, N_DAYS)),  # very stable
            "T1": 100 + np.cumsum(rng.normal(0, 5.0, N_DAYS)),   # very volatile
        }
        prices = pd.DataFrame(data)
        weights = optimizer.optimize(prices, {})
        assert weights["T0"] > weights["T1"]

    def test_max_cardinality_limits_number_of_assets(self, optimizer):
        prices = _make_prices(n_assets=10)
        weights = optimizer.optimize(prices, {"max_cardinality": 3})
        assert len(weights) <= 3

    def test_max_cardinality_defaults_to_fifteen(self, optimizer):
        # With 10 assets and default config, all 10 should be included
        prices = _make_prices(n_assets=10)
        weights = optimizer.optimize(prices, {})
        assert len(weights) == 10

    def test_zero_volatility_assets_are_dropped(self, optimizer):
        # A constant-price asset has std=0 → must be excluded via dropna
        data = {
            "FLAT": [100.0] * N_DAYS,           # zero variance
            "MOVING": list(range(100, 100 + N_DAYS)),  # non-zero variance
        }
        prices = pd.DataFrame(data)
        weights = optimizer.optimize(prices, {})
        assert "FLAT" not in weights
        assert "MOVING" in weights

    def test_returned_keys_are_subset_of_columns(self, optimizer, prices_df):
        weights = optimizer.optimize(prices_df, {})
        assert set(weights.keys()).issubset(set(prices_df.columns))

    def test_weights_sum_to_one_after_cardinality_pruning(self, optimizer):
        prices = _make_prices(n_assets=8)
        weights = optimizer.optimize(prices, {"max_cardinality": 3})
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
#  PluginManager
# ─────────────────────────────────────────────────────────────────────────────

class TestPluginManager:
    """Tests for dynamic plugin discovery."""

    def test_get_plugins_returns_dict(self):
        from app.plugins.plugin_manager import PluginManager
        manager = PluginManager()
        result = manager.get_plugins()
        assert isinstance(result, dict)

    def test_discovers_concrete_optimizer_subclasses(self):
        # Both built-in plugins must be found in the real plugins directory
        from app.plugins.plugin_manager import PluginManager
        plugins = PluginManager().get_plugins()
        assert len(plugins) >= 1

    def test_base_optimizer_itself_not_included(self):
        # BaseOptimizer is abstract — it must never appear as a discovered plugin
        from app.plugins.plugin_manager import PluginManager
        plugins = PluginManager().get_plugins()
        assert "BaseOptimizer" not in plugins

    def test_plugin_manager_itself_not_included(self):
        # plugin_manager.py is skipped by the loader — PluginManager must not appear
        from app.plugins.plugin_manager import PluginManager
        plugins = PluginManager().get_plugins()
        assert "PluginManager" not in plugins

    def test_discovered_classes_are_baseoptimizer_subclasses(self):
        from app.plugins.plugin_manager import PluginManager
        from app.plugins.base_optimizer import BaseOptimizer
        plugins = PluginManager().get_plugins()
        for name, cls in plugins.items():
            assert issubclass(cls, BaseOptimizer), f"{name} is not a BaseOptimizer subclass"

    def test_discovered_classes_can_be_instantiated(self):
        from app.plugins.plugin_manager import PluginManager
        plugins = PluginManager().get_plugins()
        for name, cls in plugins.items():
            instance = cls()
            assert instance is not None

    def test_nonexistent_directory_returns_empty_dict(self, tmp_path):
        from app.plugins.plugin_manager import PluginManager
        manager = PluginManager(plugins_dir=str(tmp_path / "no_such_dir"))
        assert manager.get_plugins() == {}

    def test_empty_directory_returns_empty_dict(self, tmp_path):
        from app.plugins.plugin_manager import PluginManager
        manager = PluginManager(plugins_dir=str(tmp_path))
        assert manager.get_plugins() == {}

    def test_non_plugin_py_files_are_ignored(self, tmp_path):
        # A .py file that defines no BaseOptimizer subclass must not appear
        (tmp_path / "helper.py").write_text("def util(): pass\n")
        from app.plugins.plugin_manager import PluginManager
        plugins = PluginManager(plugins_dir=str(tmp_path)).get_plugins()
        assert plugins == {}

    def test_dunder_files_are_skipped(self, tmp_path):
        # __init__.py and similar files must be ignored entirely
        (tmp_path / "__init__.py").write_text("")
        (tmp_path / "__version__.py").write_text("VERSION = '1.0'")
        from app.plugins.plugin_manager import PluginManager
        plugins = PluginManager(plugins_dir=str(tmp_path)).get_plugins()
        assert plugins == {}

    def test_broken_plugin_does_not_crash_manager(self, tmp_path):
        # A plugin with a syntax/import error must be logged and skipped,
        # not propagate an exception to the caller
        (tmp_path / "broken_plugin.py").write_text("raise RuntimeError('bad import')\n")
        from app.plugins.plugin_manager import PluginManager
        plugins = PluginManager(plugins_dir=str(tmp_path)).get_plugins()
        assert "broken_plugin" not in plugins

    def test_dynamically_written_plugin_is_discovered(self, tmp_path):
        # Write a valid plugin at runtime and verify PluginManager picks it up
        plugin_src = textwrap.dedent("""\
            import pandas as pd
            from typing import Dict
            from app.plugins.base_optimizer import BaseOptimizer

            class RuntimeOptimizer(BaseOptimizer):
                def optimize(self, prices_df: pd.DataFrame, config_dict: dict) -> Dict[str, float]:
                    tickers = list(prices_df.columns)
                    return {t: 1.0 / len(tickers) for t in tickers}
        """)
        (tmp_path / "runtime_plugin.py").write_text(plugin_src)

        from app.plugins.plugin_manager import PluginManager
        plugins = PluginManager(plugins_dir=str(tmp_path)).get_plugins()

        assert "RuntimeOptimizer" in plugins

    def test_dynamically_discovered_plugin_produces_valid_weights(self, tmp_path, prices_df):
        # End-to-end: discovered plugin must produce weights that sum to 1
        plugin_src = textwrap.dedent("""\
            import pandas as pd
            from typing import Dict
            from app.plugins.base_optimizer import BaseOptimizer

            class SumOneOptimizer(BaseOptimizer):
                def optimize(self, prices_df: pd.DataFrame, config_dict: dict) -> Dict[str, float]:
                    tickers = list(prices_df.columns)
                    return {t: 1.0 / len(tickers) for t in tickers}
        """)
        (tmp_path / "sum_one_plugin.py").write_text(plugin_src)

        from app.plugins.plugin_manager import PluginManager
        plugins = PluginManager(plugins_dir=str(tmp_path)).get_plugins()
        instance = plugins["SumOneOptimizer"]()
        weights = instance.optimize(prices_df, {})

        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_default_dir_points_to_plugins_package(self):
        from app.plugins.plugin_manager import PluginManager
        import app.plugins as pkg
        manager = PluginManager()
        assert os.path.abspath(manager.plugins_dir) == os.path.dirname(
            os.path.abspath(pkg.__file__)
        )