"""
Unit tests for hybrid_evo_optimizer.py

Covers the most important scenarios:
  - Numba kernels: Sharpe, penalised fitness, gradient, refiner
  - Individual dataclass (copy semantics)
  - FitnessEvaluator (Sharpe, evaluate, gradient, penalties)
  - EvoOperators (init, tournament, crossover, mutation, repair,
                  cardinality constraint, thermostat heat/cool, elitism)
  - LocalRefiner (refine improves or preserves fitness; simplex projection)
  - HybridEvoOptimizer.run() end-to-end with a mocked repository
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from app.algorithms.hybrid_evo_optimizer import (
    _sharpe_numba,
    _evaluate_numba,
    _gradient_numba,
    _refine_numba,
    Individual,
    OptimizationResult,
    FitnessEvaluator,
    EvoOperators,
    LocalRefiner,
    HybridEvoOptimizer,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_simple_market(n: int = 5, seed: int = 0):
    """Build a small well-conditioned (mu, cov) for testing."""
    rng = np.random.default_rng(seed)
    mu = np.linspace(0.05, 0.25, n)  # increasing expected returns
    A = rng.standard_normal((n, n))
    cov = A @ A.T / n + np.eye(n) * 0.04  # positive-definite
    return mu, cov


def make_feasible_individual(n: int, active_idx, rng=None) -> Individual:
    """Create an individual with weights summing to 1 on active_idx."""
    rng = rng or np.random.default_rng(0)
    binary = np.zeros(n, dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    binary[active_idx] = 1.0
    raw = rng.dirichlet(np.ones(len(active_idx)))
    weights[active_idx] = raw
    return Individual(weights=weights, binary=binary)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Numba kernels
# ═════════════════════════════════════════════════════════════════════════════
class TestNumbaKernels(unittest.TestCase):

    def test_sharpe_numba_matches_manual_formula(self):
        mu, cov = make_simple_market(4)
        w = np.array([0.25, 0.25, 0.25, 0.25])
        rf = 0.02
        port_ret = w @ mu
        port_std = np.sqrt(w @ cov @ w)
        expected = (port_ret - rf) / port_std
        self.assertAlmostEqual(_sharpe_numba(w, mu, cov, rf), expected, places=10)

    def test_sharpe_numba_returns_large_negative_for_zero_variance(self):
        # Zero-weight vector => port_var == 0 => must return -1e6 sentinel
        mu = np.array([0.1, 0.2])
        cov = np.array([[0.04, 0.0], [0.0, 0.09]])
        w = np.zeros(2)
        self.assertEqual(_sharpe_numba(w, mu, cov, 0.02), -1e6)

    def test_evaluate_numba_normalises_weights_and_computes_fitness(self):
        mu, cov = make_simple_market(4)
        # Pre-normalisation weights — evaluator should normalise them.
        weights = np.array([2.0, 0.0, 1.0, 1.0])
        binary = np.array([1.0, 0.0, 1.0, 1.0])
        # lam_conc=0.0 preserves the legacy fitness exactly.
        fitness, w_out = _evaluate_numba(
            weights, binary, mu, cov, 0.02,
            K=3, lam_k=50.0, lam_neg=200.0, lam_conc=0.0,
        )
        # Normalised weights must sum to 1 on active positions
        self.assertAlmostEqual(w_out.sum(), 1.0, places=10)
        self.assertEqual(w_out[1], 0.0)  # binary zero => weight zero
        self.assertGreater(fitness, -1e5)  # real fitness, not sentinel

    def test_evaluate_numba_penalises_cardinality_violation(self):
        mu, cov = make_simple_market(6)
        # All six assets selected, but K=3 → heavy cardinality penalty
        weights = np.ones(6) / 6
        binary = np.ones(6)
        fit_over, _ = _evaluate_numba(
            weights, binary, mu, cov, 0.02,
            K=3, lam_k=50.0, lam_neg=200.0, lam_conc=0.0,
        )
        # Same weights/binary but K=6 (no violation) should score higher
        fit_ok, _ = _evaluate_numba(
            weights, binary, mu, cov, 0.02,
            K=6, lam_k=50.0, lam_neg=200.0, lam_conc=0.0,
        )
        self.assertLess(fit_over, fit_ok)

    def test_evaluate_numba_returns_sentinel_on_empty_selection(self):
        mu, cov = make_simple_market(3)
        weights = np.zeros(3)
        binary = np.zeros(3)
        fitness, _ = _evaluate_numba(
            weights, binary, mu, cov, 0.02,
            K=3, lam_k=50.0, lam_neg=200.0, lam_conc=0.0,
        )
        self.assertEqual(fitness, -1e6)

    def test_gradient_numba_shape_and_finiteness(self):
        mu, cov = make_simple_market(4)
        w = np.array([0.25, 0.25, 0.25, 0.25])
        g = _gradient_numba(w, mu, cov, 0.02)
        self.assertEqual(g.shape, (4,))
        self.assertTrue(np.all(np.isfinite(g)))

    def test_gradient_numba_returns_zeros_for_degenerate_variance(self):
        mu = np.array([0.1, 0.2])
        cov = np.zeros((2, 2))
        w = np.array([0.5, 0.5])
        g = _gradient_numba(w, mu, cov, 0.02)
        np.testing.assert_array_equal(g, np.zeros(2))

    def test_refine_numba_does_not_worsen_sharpe(self):
        """Refiner uses hill-climbing with acceptance check: the final
        Sharpe must be >= starting Sharpe (never worse)."""
        mu, cov = make_simple_market(5, seed=1)
        rf = 0.02
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        binary = np.ones(5)

        sharpe_before = _sharpe_numba(weights, mu, cov, rf)
        w_new, _ = _refine_numba(
            weights, binary, mu, cov, rf,
            max_iter=30, lr_init=0.05, lr_decay=0.95,
        )
        sharpe_after = _sharpe_numba(w_new, mu, cov, rf)
        self.assertGreaterEqual(sharpe_after + 1e-10, sharpe_before)
        self.assertAlmostEqual(w_new.sum(), 1.0, places=8)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Individual dataclass
# ═════════════════════════════════════════════════════════════════════════════
class TestIndividual(unittest.TestCase):

    def test_copy_is_deep_for_arrays(self):
        w = np.array([0.5, 0.5])
        b = np.array([1.0, 1.0])
        ind = Individual(weights=w, binary=b, fitness=0.7)
        dup = ind.copy()

        dup.weights[0] = 999.0
        dup.binary[0] = 0.0
        dup.fitness = -1.0

        self.assertEqual(ind.weights[0], 0.5)
        self.assertEqual(ind.binary[0], 1.0)
        self.assertEqual(ind.fitness, 0.7)

    def test_default_fitness_is_neg_inf(self):
        ind = Individual(weights=np.zeros(3), binary=np.zeros(3))
        self.assertEqual(ind.fitness, -np.inf)


# ═════════════════════════════════════════════════════════════════════════════
# 3. FitnessEvaluator
# ═════════════════════════════════════════════════════════════════════════════
class TestFitnessEvaluator(unittest.TestCase):

    def setUp(self):
        self.mu, self.cov = make_simple_market(5)
        self.evaluator = FitnessEvaluator(
            mu=self.mu, cov=self.cov, risk_free_rate=0.02,
            max_cardinality=3,
        )

    def test_sharpe_method_matches_numba_kernel(self):
        w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        self.assertAlmostEqual(
            self.evaluator.sharpe(w),
            _sharpe_numba(w, self.mu, self.cov, 0.02),
            places=10,
        )

    def test_evaluate_updates_individual_in_place(self):
        ind = make_feasible_individual(5, [0, 1, 2])
        self.assertEqual(ind.fitness, -np.inf)
        fit = self.evaluator.evaluate(ind)
        self.assertEqual(fit, ind.fitness)
        self.assertTrue(np.isfinite(ind.fitness))
        self.assertAlmostEqual(ind.weights.sum(), 1.0, places=8)

    def test_evaluate_prefers_feasible_over_empty_selection(self):
        good = make_feasible_individual(5, [0, 1, 2])
        empty = Individual(weights=np.zeros(5), binary=np.zeros(5))
        self.evaluator.evaluate(good)
        self.evaluator.evaluate(empty)
        self.assertGreater(good.fitness, empty.fitness)

    def test_gradient_shape(self):
        w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        g = self.evaluator.gradient(w)
        self.assertEqual(g.shape, (5,))


# ═════════════════════════════════════════════════════════════════════════════
# 4. EvoOperators
# ═════════════════════════════════════════════════════════════════════════════
class TestEvoOperators(unittest.TestCase):

    def setUp(self):
        self.n = 10
        self.K = 4
        self.mu, self.cov = make_simple_market(self.n)
        self.rng = np.random.default_rng(42)
        self.ops = EvoOperators(
            n_assets=self.n,
            max_cardinality=self.K,
            tournament_size=3,
            mutation_sigma=0.05,
            mutation_prob=0.15,
            bit_flip_prob=0.10,
            rng=self.rng,
            cov=self.cov,
        )

    # ----- initialisation ---------------------------------------------------
    def test_random_individual_respects_cardinality(self):
        for _ in range(50):
            ind = self.ops.random_individual()
            self.assertLessEqual(int(ind.binary.sum()), self.K)
            self.assertGreaterEqual(int(ind.binary.sum()), 1)
            # Weights are zero wherever binary is zero
            self.assertTrue(np.all(ind.weights[ind.binary < 0.5] == 0.0))

    def test_init_population_size_and_seeding(self):
        pop_size = 30
        pop = self.ops.init_population(pop_size, mu=self.mu)
        self.assertEqual(len(pop), pop_size)
        for ind in pop:
            self.assertEqual(ind.weights.shape, (self.n,))
            self.assertEqual(ind.binary.shape, (self.n,))
            self.assertLessEqual(int(ind.binary.sum()), max(self.K, 20))

    # ----- tournament selection --------------------------------------------
    def test_tournament_select_picks_from_population(self):
        pop = self.ops.init_population(20, mu=self.mu)
        # Assign increasing fitnesses so we can identify who wins.
        for i, ind in enumerate(pop):
            ind.fitness = float(i)
        # Run many tournaments — the mean fitness of winners must exceed
        # the mean fitness of the population (selection pressure).
        winners = [self.ops.tournament_select(pop).fitness for _ in range(200)]
        self.assertGreater(np.mean(winners), np.mean([i.fitness for i in pop]))

    def test_tournament_select_returns_copy(self):
        pop = self.ops.init_population(10, mu=self.mu)
        for i, ind in enumerate(pop):
            ind.fitness = float(i)
        winner = self.ops.tournament_select(pop)
        winner.weights[0] = 12345.0  # mutate the copy
        # Population is untouched
        for ind in pop:
            self.assertNotEqual(ind.weights[0], 12345.0)

    # ----- crossover --------------------------------------------------------
    def test_crossover_produces_feasible_children(self):
        p1 = self.ops.random_individual()
        p2 = self.ops.random_individual()
        c1, c2 = self.ops.crossover(p1, p2)
        for child in (c1, c2):
            self.assertTrue(np.all(child.weights >= 0.0))
            self.assertLessEqual(int(child.binary.sum()), self.K)
            self.assertGreaterEqual(int(child.binary.sum()), 1)
            self.assertAlmostEqual(child.weights.sum(), 1.0, places=8)

    # ----- mutation ---------------------------------------------------------
    def test_mutate_preserves_feasibility(self):
        for _ in range(30):
            ind = self.ops.random_individual()
            mutated = self.ops.mutate(ind)
            self.assertTrue(np.all(mutated.weights >= 0.0))
            self.assertAlmostEqual(mutated.weights.sum(), 1.0, places=8)
            self.assertLessEqual(int(mutated.binary.sum()), self.K)
            self.assertGreaterEqual(int(mutated.binary.sum()), 1)

    # ----- repair -----------------------------------------------------------
    def test_repair_trims_excess_selection_to_K(self):
        ind = Individual(
            weights=np.ones(self.n) / self.n,
            binary=np.ones(self.n),  # all 10 selected — over K=4
        )
        self.ops._repair(ind)
        self.assertEqual(int(ind.binary.sum()), self.K)
        self.assertAlmostEqual(ind.weights.sum(), 1.0, places=8)

    def test_repair_recovers_from_empty_selection(self):
        ind = Individual(
            weights=np.zeros(self.n),
            binary=np.zeros(self.n),
        )
        self.ops._repair(ind)
        self.assertEqual(int(ind.binary.sum()), 1)
        self.assertAlmostEqual(ind.weights.sum(), 1.0, places=8)

    def test_repair_zeros_weights_for_deselected_assets(self):
        ind = Individual(
            weights=np.array([0.3, 0.3, 0.2, 0.2] + [0.0] * (self.n - 4)),
            binary=np.array([1.0, 1.0, 0.0, 0.0] + [0.0] * (self.n - 4)),
        )
        self.ops._repair(ind)
        # Positions with binary 0 must have weight 0
        for i in range(self.n):
            if ind.binary[i] < 0.5:
                self.assertEqual(ind.weights[i], 0.0)

    # ----- variance thermostat ---------------------------------------------
    def test_heat_up_increases_mutation_up_to_caps(self):
        # From baseline
        self.ops.heat_up(factor=1.25)
        self.assertGreater(self.ops.mut_sigma, 0.05)
        self.assertGreater(self.ops.mut_prob, 0.15)

        # Caps: even after many heat-ups, values stay bounded.
        for _ in range(100):
            self.ops.heat_up(factor=1.25, sigma_cap=0.30, prob_cap=0.60)
        self.assertLessEqual(self.ops.mut_sigma, 0.30 + 1e-12)
        self.assertLessEqual(self.ops.mut_prob, 0.60 + 1e-12)

    def test_cool_down_returns_toward_baseline(self):
        # Heat up first, then cool — must not fall below baseline.
        for _ in range(50):
            self.ops.heat_up(factor=1.25)
        for _ in range(200):
            self.ops.cool_down(decay=0.95)
        self.assertAlmostEqual(self.ops.mut_sigma, 0.05, places=8)
        self.assertAlmostEqual(self.ops.mut_prob, 0.15, places=8)

    # ----- elitism ----------------------------------------------------------
    def test_elitism_returns_top_n_copies(self):
        pop = self.ops.init_population(10, mu=self.mu)
        for i, ind in enumerate(pop):
            ind.fitness = float(i)
        elites = EvoOperators.elitism(pop, n_elite=3)
        self.assertEqual(len(elites), 3)
        fitnesses = sorted([e.fitness for e in elites], reverse=True)
        self.assertEqual(fitnesses, [9.0, 8.0, 7.0])
        # Elites are copies, not references
        elites[0].weights[0] = -9999.0
        self.assertNotEqual(pop[9].weights[0], -9999.0)


# ═════════════════════════════════════════════════════════════════════════════
# 5. LocalRefiner
# ═════════════════════════════════════════════════════════════════════════════
class TestLocalRefiner(unittest.TestCase):

    def setUp(self):
        self.mu, self.cov = make_simple_market(6, seed=7)
        self.evaluator = FitnessEvaluator(
            mu=self.mu, cov=self.cov, risk_free_rate=0.02,
            max_cardinality=4,
        )
        self.refiner = LocalRefiner(
            evaluator=self.evaluator, max_iter=30, lr_init=0.05
        )

    def test_refine_preserves_or_improves_fitness(self):
        ind = make_feasible_individual(6, [0, 1, 2, 3])
        self.evaluator.evaluate(ind)
        original_fitness = ind.fitness

        refined = self.refiner.refine(ind)
        # Refiner must at least match the original (hill-climbing guarantee).
        self.assertGreaterEqual(refined.fitness + 1e-9, original_fitness)
        self.assertAlmostEqual(refined.weights.sum(), 1.0, places=6)

    def test_refine_does_not_mutate_input(self):
        ind = make_feasible_individual(6, [0, 1, 2, 3])
        self.evaluator.evaluate(ind)
        w_snapshot = ind.weights.copy()
        b_snapshot = ind.binary.copy()
        _ = self.refiner.refine(ind)
        np.testing.assert_array_equal(ind.weights, w_snapshot)
        np.testing.assert_array_equal(ind.binary, b_snapshot)

    def test_project_simplex_sums_to_one_and_is_nonnegative(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            w = rng.standard_normal(8)  # can be negative, unbounded
            projected = LocalRefiner._project_simplex(w)
            self.assertTrue(np.all(projected >= -1e-12))
            self.assertAlmostEqual(projected.sum(), 1.0, places=8)

    def test_project_simplex_is_identity_on_simplex_point(self):
        w = np.array([0.25, 0.25, 0.25, 0.25])
        projected = LocalRefiner._project_simplex(w)
        np.testing.assert_allclose(projected, w, atol=1e-10)


# ═════════════════════════════════════════════════════════════════════════════
# 6. HybridEvoOptimizer end-to-end (with mocked repository)
# ═════════════════════════════════════════════════════════════════════════════
class TestHybridEvoOptimizerRun(unittest.TestCase):

    def _build_fake_prices(self, n_tickers=6, n_obs=120, seed=0):
        """Generate synthetic weekly price history."""
        rng = np.random.default_rng(seed)
        tickers = [f"T{i}" for i in range(n_tickers)]
        # Random-walk-ish log-returns → cumulative prices.
        rets = rng.normal(loc=0.002, scale=0.03, size=(n_obs, n_tickers))
        prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
        idx = pd.date_range("2020-01-01", periods=n_obs, freq="W")
        return tickers, pd.DataFrame(prices, index=idx, columns=tickers)

    def test_run_returns_valid_result_with_mocked_repo(self):
        tickers, prices_df = self._build_fake_prices(n_tickers=6, n_obs=150)

        fake_repo = MagicMock()
        fake_repo.get_all_tickers.return_value = tickers
        fake_repo.get_price_history.return_value = prices_df

        # Small pop/gen so the test runs quickly.
        optimizer = HybridEvoOptimizer(
            pop_size=20,
            n_generations=5,
            max_cardinality=3,
            n_elite=2,
            top_m_refine=3,
            local_iter=5,
            seed=123,
        )
        result = optimizer.run(repo=fake_repo)

        # Structural checks
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(len(result.history), 5)
        self.assertEqual(result.n_generations, 5)

        # Portfolio feasibility
        self.assertGreater(len(result.selected_assets), 0)
        self.assertLessEqual(len(result.selected_assets), 3)  # cardinality K
        total_weight = sum(result.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=4)
        for w in result.weights.values():
            self.assertGreaterEqual(w, 0.0)

        # All selected tickers are from the input universe
        for t in result.selected_assets:
            self.assertIn(t, tickers)

        # Finite numerical outputs
        self.assertTrue(np.isfinite(result.sharpe_ratio))
        self.assertTrue(np.isfinite(result.expected_return))
        self.assertGreaterEqual(result.portfolio_risk, 0.0)

    def test_run_is_reproducible_with_same_seed(self):
        tickers, prices_df = self._build_fake_prices(n_tickers=5, n_obs=100)

        def new_repo():
            r = MagicMock()
            r.get_all_tickers.return_value = tickers
            r.get_price_history.return_value = prices_df.copy()
            return r

        def run_once():
            opt = HybridEvoOptimizer(
                pop_size=15, n_generations=4, max_cardinality=3,
                n_elite=2, top_m_refine=3, local_iter=5, seed=777,
            )
            return opt.run(repo=new_repo())

        r1 = run_once()
        r2 = run_once()
        # Same seed → identical history and Sharpe.
        self.assertEqual(r1.history, r2.history)
        self.assertAlmostEqual(r1.sharpe_ratio, r2.sharpe_ratio, places=10)
        self.assertEqual(r1.selected_assets, r2.selected_assets)

    def test_run_clamps_K_to_number_of_available_assets(self):
        # Only 3 tickers available but K=10 is requested.
        tickers, prices_df = self._build_fake_prices(n_tickers=3, n_obs=80)
        fake_repo = MagicMock()
        fake_repo.get_all_tickers.return_value = tickers
        fake_repo.get_price_history.return_value = prices_df

        optimizer = HybridEvoOptimizer(
            pop_size=10, n_generations=3, max_cardinality=10,
            n_elite=1, top_m_refine=2, local_iter=3, seed=5,
        )
        result = optimizer.run(repo=fake_repo)
        # No more than 3 assets can be selected.
        self.assertLessEqual(len(result.selected_assets), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)