"""
hybrid_evo_optimizer.py
=======================


Architecture
------------
1. DataLoader         – loads prices and builds expected-return/covariance inputs
2. FitnessEvaluator   – scores portfolios with a constrained Sharpe objective
3. EvoOperators       – creates and evolves candidate portfolios
4. LocalRefiner       – improves the best candidates with gradient search
5. HybridEvoOptimizer – coordinates the complete optimisation workflow
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.data.repository import PortfolioRepository
from pypfopt import risk_models, expected_returns

logger = logging.getLogger(__name__)
# Library modules should not configure global logging; the application entry
# point owns logging setup.

# Fitness and covariance calculations may produce harmless numerical warnings
# for invalid candidates. Keep the filter narrow so unrelated issues still
# surface elsewhere.
warnings.filterwarnings(
    "ignore", category=RuntimeWarning,
    message=r"overflow encountered|invalid value encountered|divide by zero",
)

from numba import njit


# ═══════════════════════════════════════════════════════════════════════════
#  NUMBA JIT KERNELS
# ═══════════════════════════════════════════════════════════════════════════
@njit(cache=True)
def _sharpe_numba(w, mu, cov, rf):
    port_return = np.dot(w, mu)
    port_var = np.dot(w, np.dot(cov, w))
    if port_var <= 0:
        return -1e6
    port_std = np.sqrt(port_var)
    return (port_return - rf) / port_std


@njit(cache=True)
def _evaluate_numba(weights, binary, mu, cov, rf, K, lam_k, lam_neg, lam_conc):
    w = weights * binary
    w_sum = np.sum(w)

    if w_sum > 1e-8:
        w = w / w_sum
    else:
        return -1e6, w

    sr = _sharpe_numba(w, mu, cov, rf)

    p_card = lam_k * max(0.0, np.sum(binary) - float(K))
    p_neg = lam_neg * np.sum(np.maximum(0.0, -w))

    safe_max_weight = max(0.15, (1.0 / K) + 0.02)
    p_max_w = 100.0 * np.sum(np.maximum(0.0, w - safe_max_weight))

    # Penalise concentrated allocations through the Herfindahl index. With
    # lam_conc=0, this term has no effect.
    p_conc = lam_conc * np.sum(w * w)

    fitness = sr - p_card - p_neg - p_max_w - p_conc
    return fitness, w


@njit(cache=True)
def _gradient_numba(w, mu, cov, rf):
    port_ret = np.dot(w, mu)
    cov_w = np.dot(cov, w)
    port_var = np.dot(w, cov_w)
    if port_var <= 0:
        return np.zeros_like(w)
    port_std = np.sqrt(port_var)
    sr = (port_ret - rf) / port_std
    return (mu - rf) / port_std - sr * cov_w / port_var


@njit(cache=True)
def _refine_numba(weights, binary, mu, cov, rf, max_iter, lr_init, lr_decay):
    w = weights.copy()
    b = binary.copy()
    lr = lr_init
    n = len(w)

    # Phase 1: improve weights while keeping the selected assets fixed.
    for _ in range(max_iter):
        grad = _gradient_numba(w, mu, cov, rf)
        grad_masked = grad * b

        w_new = w + lr * grad_masked
        w_new = np.maximum(w_new, 0.0)
        w_new = w_new * b
        w_sum = np.sum(w_new)
        if w_sum < 1e-12:
            break
        w_new = w_new / w_sum

        if _sharpe_numba(w_new, mu, cov, rf) > _sharpe_numba(w, mu, cov, rf):
            w = w_new
            lr *= lr_decay
        else:
            lr *= 0.5
            if lr < 1e-8:
                break

    # Phase 2: replace weak selected assets with stronger inactive assets.
    failed_swaps = np.zeros(n, dtype=np.bool_)

    for _ in range(max_iter // 2):
        grad = _gradient_numba(w, mu, cov, rf)


        best_inactive = -1
        best_inactive_val = -1e9
        worst_active = -1
        worst_active_val = 1e9

        for i in range(n):
            if b[i] < 0.5 and not failed_swaps[i]:
                if grad[i] > best_inactive_val:
                    best_inactive_val = grad[i]
                    best_inactive = i
            elif b[i] > 0.5:
                if w[i] < worst_active_val:
                    worst_active_val = w[i]
                    worst_active = i

        if best_inactive == -1 or worst_active == -1:
            break

        if best_inactive_val <= grad[worst_active]:
            failed_swaps[best_inactive] = True
            continue

        new_b = b.copy()
        new_b[worst_active] = 0.0
        new_b[best_inactive] = 1.0

        new_w = w.copy()
        saved_weight = w[worst_active]
        new_w[worst_active] = 0.0
        new_w[best_inactive] = saved_weight
        new_w = new_w * new_b

        s = np.sum(new_w)
        if s < 1e-12:
            failed_swaps[best_inactive] = True
            continue
        new_w = new_w / s

        lr2 = lr_init
        for _ in range(15):
            g = _gradient_numba(new_w, mu, cov, rf) * new_b
            cand = np.maximum(new_w + lr2 * g, 0.0) * new_b
            cs = np.sum(cand)
            if cs < 1e-12:
                break
            cand = cand / cs
            if _sharpe_numba(cand, mu, cov, rf) > _sharpe_numba(new_w, mu, cov, rf):
                new_w = cand
                lr2 *= lr_decay
            else:
                lr2 *= 0.5
                if lr2 < 1e-8:
                    break

        if _sharpe_numba(new_w, mu, cov, rf) > _sharpe_numba(w, mu, cov, rf):
            w = new_w
            b = new_b
            failed_swaps[:] = False
        else:
            failed_swaps[best_inactive] = True

    return w, b


# ═══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass
class Individual:
    """Candidate portfolio inside the evolutionary population.

    Attributes
    ----------
    weights : np.ndarray
        Portfolio weight vector with one value per asset.
    binary : np.ndarray
        Asset-selection mask with one value per asset.
    fitness : float
        Evaluated objective value. Higher is better.
    """
    weights: np.ndarray
    binary: np.ndarray
    fitness: float = -np.inf

    def copy(self) -> "Individual":
        return Individual(
            weights=self.weights.copy(),
            binary=self.binary.copy(),
            fitness=self.fitness,
        )


@dataclasses.dataclass
class OptimizationResult:
    """Final portfolio and metrics produced by the hybrid optimizer."""
    weights: Dict[str, float]
    selected_assets: List[str]
    sharpe_ratio: float
    expected_return: float
    portfolio_risk: float
    n_generations: int
    history: List[float]  # Best fitness value from each generation.


# ═══════════════════════════════════════════════════════════════════════════
#  1. DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════
def _james_stein_shrink_mu(
    mu: np.ndarray,
    cov: np.ndarray,
    n_obs: int,
) -> np.ndarray:
    """Shrink expected returns toward their cross-sectional mean.

    The positive-part James-Stein estimator returns:

        μ_JS = ν + max(0, 1 − (n−2)·σ̄² / ||μ − ν||²) · (μ − ν)

    Here ν is the cross-sectional mean of μ, and σ̄² is the average sampling
    variance of the per-asset estimates.

    In portfolio optimisation this reduces the effect of noisy return
    estimates, which otherwise tend to create overconfident allocations.
    Returns ``mu`` unchanged when shrinkage is not defined.
    """
    n = len(mu)
    if n < 3 or n_obs < 2:
        return mu
    nu = float(np.mean(mu))
    sigma2 = float(np.mean(np.diag(cov))) / float(max(n_obs, 1))
    diff = mu - nu
    norm_sq = float(diff @ diff)
    if norm_sq <= 1e-12 or sigma2 <= 0.0:
        return mu
    factor = max(0.0, 1.0 - (n - 2) * sigma2 / norm_sq)
    return nu + factor * diff


class DataLoader:
    """Load prices and estimate the optimiser input matrices."""

    def __init__(
        self,
        repo: Optional[PortfolioRepository] = None,
        ewma_span: int = 52,
        mu_shrinkage: bool = False,
    ):
        self.repo = repo or PortfolioRepository()
        # Span reserved for EWMA-style return estimators.
        self.ewma_span = ewma_span
        # Optional shrinkage stabilises expected returns when enabled.
        self.mu_shrinkage = mu_shrinkage

    def load(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_history_frac: float = 0.80,
        frequency: int = 52,
    ) -> Tuple[List[str], np.ndarray, np.ndarray, pd.DataFrame]:
        """Load prices and compute expected returns and covariance.

        Parameters
        ----------
        tickers : list[str] | None
            Tickers to include.  ``None`` uses all database tickers.
        start_date, end_date : str | None
            Optional date filters (ISO-8601 strings).
        min_history_frac : float
            Minimum fraction of non-NaN observations a ticker must have.
        frequency : int
            Annualisation factor (52 = weekly, 252 = daily).

        Returns
        -------
        tickers : list[str]
        mu      : np.ndarray  (n,)
        cov     : np.ndarray  (n, n)
        prices  : pd.DataFrame
        """
        if tickers is None:
            tickers = self.repo.get_all_tickers()

        prices: pd.DataFrame = self.repo.get_price_history(
            tickers, start_date=start_date, end_date=end_date
        )
        if prices.empty:
            raise ValueError("No price data returned from the database.")

        prices = prices.sort_index()

        # --- keep only tickers with enough usable history ------------------
        coverage = prices.notna().sum()
        threshold = int(len(prices) * min_history_frac)
        valid = coverage[coverage >= threshold].index.tolist()
        if not valid:
            raise ValueError(
                f"No ticker has ≥{min_history_frac:.0%} non-NaN history "
                f"(need ≥{threshold}/{len(prices)} rows)."
            )
        prices = prices[valid].dropna()
        tickers = list(prices.columns)

        # --- estimate μ and Σ with PyPortfolioOpt -------------------------
        # CAGR keeps the return estimate less sensitive to the most recent
        # part of the sample than a short-horizon EWMA.
        mu = expected_returns.mean_historical_return(
            prices, frequency=frequency, compounding=True
        ).values
        cov = risk_models.CovarianceShrinkage(prices, frequency=frequency).ledoit_wolf().values

        # --- optionally shrink noisy expected returns ----------------------
        if self.mu_shrinkage:
            mu_pre = mu
            mu = _james_stein_shrink_mu(mu, cov, n_obs=len(prices))
            shift = float(np.linalg.norm(mu - mu_pre))
            logger.info(
                "DataLoader: James–Stein shrinkage applied to μ "
                "(‖Δμ‖=%.4g, |Δmean|=%.4g)",
                shift, float(abs(mu.mean() - mu_pre.mean())),
            )

        logger.info(
            "DataLoader: %d tickers, %d observations, annualisation=%d",
            len(tickers), len(prices), frequency,
        )
        return tickers, mu, cov, prices


# ═══════════════════════════════════════════════════════════════════════════
#  2. FITNESS EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════
class FitnessEvaluator:
    """Evaluate portfolios with a penalised Sharpe objective.

    fitness = sharpe_ratio
              - λ₁·|Σw − 1|
              - λ₂·max(0, Σb − K)
              - λ₃·Σmax(0, −w)
              - λ₄·Σw²            (Herfindahl concentration penalty, optional)

    The concentration penalty is optional. When it is zero, the objective keeps
    the original behaviour; positive values encourage broader diversification.
    """

    def __init__(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        risk_free_rate: float = 0.02,
        max_cardinality: int = 15,
        penalty_weights: float = 100.0,
        penalty_cardinality: float = 50.0,
        penalty_negativity: float = 200.0,
        penalty_concentration: float = 0.0,
    ):
        self.mu = mu
        self.cov = cov
        self.rf = risk_free_rate
        self.K = max_cardinality
        self.lam_w = penalty_weights
        self.lam_k = penalty_cardinality
        self.lam_neg = penalty_negativity
        self.lam_conc = penalty_concentration
        self.n = len(mu)

    # ----- core scoring ---------------------------------------------------
    def sharpe(self, w: np.ndarray) -> float:
        return _sharpe_numba(w, self.mu, self.cov, self.rf)

    def evaluate(self, ind: Individual) -> float:
        """Evaluate one individual through the compiled scoring kernel."""
        fitness, w = _evaluate_numba(
            ind.weights, ind.binary,
            self.mu, self.cov, self.rf,
            self.K, self.lam_k, self.lam_neg, self.lam_conc,
        )
        ind.fitness = fitness
        ind.weights = w
        return ind.fitness

    # ----- analytical gradient --------------------------------------------
    def gradient(self, w: np.ndarray) -> np.ndarray:
        return _gradient_numba(w, self.mu, self.cov, self.rf)


# ═══════════════════════════════════════════════════════════════════════════
#  3. EVOLUTIONARY OPERATORS
# ═══════════════════════════════════════════════════════════════════════════
class EvoOperators:
    """Create and transform individuals during evolutionary search."""

    def __init__(
        self,
        n_assets: int,
        max_cardinality: int = 15,
        tournament_size: int = 3,
        crossover_alpha: float = 0.5,
        mutation_sigma: float = 0.05,
        mutation_prob: float = 0.15,
        bit_flip_prob: float = 0.10,
        rng: Optional[np.random.Generator] = None,
        cov: Optional[np.ndarray] = None,
    ):
        self.n = n_assets
        self.K = max_cardinality
        self._cov = cov  # Used for risk-adjusted population seeding.
        self.tourn = tournament_size
        self.cx_alpha = crossover_alpha
        self.mut_sigma = mutation_sigma
        self.mut_prob = mutation_prob
        self.bit_flip = bit_flip_prob
        self.rng = rng or np.random.default_rng()

        # Mutation baselines used when the thermostat cools exploration.
        self._base_mut_sigma = mutation_sigma
        self._base_mut_prob = mutation_prob

    # ----- initialisation -------------------------------------------------
    def random_individual(self) -> Individual:
        """Create a random feasible individual."""
        # Select a random number of assets from the allowed cardinality range.
        k = self.rng.integers(1, self.K + 1)
        chosen = self.rng.choice(self.n, size=k, replace=False)
        binary = np.zeros(self.n, dtype=np.float64)
        binary[chosen] = 1.0

        # Dirichlet weights are non-negative and sum to one by construction.
        raw = self.rng.dirichlet(np.ones(k))
        weights = np.zeros(self.n, dtype=np.float64)
        weights[chosen] = raw

        return Individual(weights=weights, binary=binary)

    def init_population(self, pop_size: int, mu: Optional[np.ndarray] = None) -> List[Individual]:
        """Initialise the population with random and return-aware candidates."""
        population = []

        if mu is not None:
            # Seed about 20% of the population with high-scoring assets.
            n_seeded = max(1, pop_size // 5)
            # Rank by a Sharpe-like proxy when covariance is available;
            # otherwise rank by expected return alone.
            if self._cov is not None:
                per_asset_std = np.sqrt(np.maximum(np.diag(self._cov), 1e-12))
                sharpe_proxy = mu / per_asset_std
            else:
                sharpe_proxy = mu
            top_idx = np.argsort(sharpe_proxy)[::-1]

            for i in range(n_seeded):
                # Use a random subset of the best-ranked assets for diversity.
                k = self.rng.integers(2, min(self.K, 20) + 1)
                # Limit the pool so seeds remain biased toward strong assets.
                n_top = min(50, self.n)
                chosen = self.rng.choice(top_idx[:n_top], size=k, replace=False)
                binary = np.zeros(self.n, dtype=np.float64)
                binary[chosen] = 1.0
                raw = self.rng.dirichlet(np.ones(k))
                weights = np.zeros(self.n, dtype=np.float64)
                weights[chosen] = raw
                population.append(Individual(weights=weights, binary=binary))

        # Fill the remaining slots with fully random feasible candidates.
        while len(population) < pop_size:
            population.append(self.random_individual())

        return population

    # ----- selection ------------------------------------------------------
    def tournament_select(self, population: List[Individual]) -> Individual:
        """Select one parent with tournament selection."""
        idxs = self.rng.choice(len(population), size=self.tourn, replace=False)
        best = max(idxs, key=lambda i: population[i].fitness)
        return population[best].copy()

    # ----- crossover (blend / BLX-α) -------------------------------------
    def crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """Blend parent weights and mix parent selection masks."""
        alpha = self.cx_alpha
        # BLX-alpha explores around both parents' weight values.
        lo = np.minimum(p1.weights, p2.weights) - alpha * np.abs(p1.weights - p2.weights)
        hi = np.maximum(p1.weights, p2.weights) + alpha * np.abs(p1.weights - p2.weights)
        lo = np.maximum(lo, 0.0)
        c1_w = self.rng.uniform(lo, hi)
        c2_w = self.rng.uniform(lo, hi)

        # Uniform crossover chooses each selection bit from either parent.
        mask = self.rng.random(self.n) < 0.5
        c1_b = np.where(mask, p1.binary, p2.binary)
        c2_b = np.where(mask, p2.binary, p1.binary)

        c1 = Individual(weights=c1_w, binary=c1_b.astype(np.float64))
        c2 = Individual(weights=c2_w, binary=c2_b.astype(np.float64))
        self._repair(c1)
        self._repair(c2)
        return c1, c2

    # ----- mutation -------------------------------------------------------
    def mutate(self, ind: Individual) -> Individual:
        """Apply Gaussian weight noise and bit-flip selection changes."""
        # Add small random changes to selected weight genes.
        mask = self.rng.random(self.n) < self.mut_prob
        noise = self.rng.normal(0, self.mut_sigma, size=self.n) * mask
        ind.weights = np.maximum(ind.weights + noise, 0.0)

        # Flip selection bits to explore different asset sets.
        flips = self.rng.random(self.n) < self.bit_flip
        ind.binary = np.where(flips, 1.0 - ind.binary, ind.binary)

        self._repair(ind)
        return ind

    # ----- repair ---------------------------------------------------------
    def _repair(self, ind: Individual) -> None:
        """Restore non-negative weights, full investment, and cardinality."""
        ind.weights = np.maximum(ind.weights, 0.0)

        # Keep no more than K active assets.
        if ind.binary.sum() > self.K:
            active = np.where(ind.binary > 0.5)[0]
            keep = self.rng.choice(active, size=self.K, replace=False)
            ind.binary[:] = 0.0
            ind.binary[keep] = 1.0

        # Guarantee that the portfolio is never empty.
        if ind.binary.sum() == 0:
            idx = self.rng.integers(0, self.n)
            ind.binary[idx] = 1.0
            ind.weights[idx] = 1.0

        # Remove weight from inactive assets.
        ind.weights *= ind.binary

        # Re-normalise the active weights.
        w_sum = ind.weights.sum()
        if w_sum > 1e-12:
            ind.weights /= w_sum
        else:
            active = np.where(ind.binary > 0.5)[0]
            ind.weights[active] = 1.0 / len(active)

    # ----- variance thermostat --------------------------------------------
    def heat_up(
        self, factor: float = 1.25, sigma_cap: float = 0.30, prob_cap: float = 0.60,
    ) -> None:
        """Increase mutation rates when the population has stagnated."""
        self.mut_sigma = min(self.mut_sigma * factor, sigma_cap)
        self.mut_prob = min(self.mut_prob * factor, prob_cap)

    def cool_down(self, decay: float = 0.95) -> None:
        """Move mutation rates back toward their configured baselines."""
        self.mut_sigma = max(self.mut_sigma * decay, self._base_mut_sigma)
        self.mut_prob = max(self.mut_prob * decay, self._base_mut_prob)

    # ----- elitism --------------------------------------------------------
    @staticmethod
    def elitism(
        population: List[Individual],
        n_elite: int,
    ) -> List[Individual]:
        """Return copies of the strongest individuals."""
        ranked = sorted(population, key=lambda x: x.fitness, reverse=True)
        return [ind.copy() for ind in ranked[:n_elite]]


# ═══════════════════════════════════════════════════════════════════════════
#  4. LOCAL REFINER
# ═══════════════════════════════════════════════════════════════════════════
class LocalRefiner:
    """Improve strong candidates with constrained gradient ascent.

    For each candidate the refiner:
      1. Computes the Sharpe-ratio gradient.
      2. Adjusts weights while preserving long-only full investment.
      3. Tries asset swaps when inactive assets have stronger gradient signals.
    """

    def __init__(
        self,
        evaluator: FitnessEvaluator,
        max_iter: int = 30,
        lr_init: float = 0.05,
        lr_decay: float = 0.95,
    ):
        self.evaluator = evaluator
        self.max_iter = max_iter
        self.lr_init = lr_init
        self.lr_decay = lr_decay

    def refine(self, ind: Individual) -> Individual:
        """Run the compiled local-search routine for one individual."""
        best = ind.copy()

        new_w, new_b = _refine_numba(
            best.weights, best.binary,
            self.evaluator.mu, self.evaluator.cov, self.evaluator.rf,
            self.max_iter, self.lr_init, self.lr_decay
        )

        best.weights = new_w
        best.binary = new_b
        self.evaluator.evaluate(best)
        return best

    # ----- simplex projection ---------------------------------------------
    @staticmethod
    def _project_simplex(w: np.ndarray) -> np.ndarray:
        """Project ``w`` onto the non-negative simplex.

        Uses the efficient O(n log n) algorithm by Duchi et al. (2008).
        """
        n = len(w)
        u = np.sort(w)[::-1]
        cssv = np.cumsum(u) - 1.0
        rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
        theta = cssv[rho] / (rho + 1.0)
        return np.maximum(w - theta, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  5. HYBRID EVOLUTIONARY OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════
class HybridEvoOptimizer:
    """Hybrid evolutionary optimizer for constrained portfolios.

    Pipeline
    --------
    1. Load market data and compute μ, Σ.
    2. Initialise a random feasible population.
    3. For each generation:
       a. Evaluate penalised Sharpe fitness.
       b. Preserve elites.
       c. Build the next generation with selection, crossover, and mutation.
    4. Refine top candidates with local gradient search and SLSQP polish.
    5. Return the best feasible portfolio.

    Parameters
    ----------
    pop_size         : int   – population size                       (default 200)
    n_generations    : int   – number of evolutionary generations   (default 200)
    max_cardinality  : int   – maximum number of assets (K)         (default 15)
    risk_free_rate   : float – annualised risk-free rate             (default 0.02)
    n_elite          : int   – elites carried forward each gen      (default 5)
    top_m_refine     : int   – top individuals sent to local refiner(default 10)
    tournament_size  : int   – tournament selection pool             (default 3)
    crossover_alpha  : float – BLX-α crossover parameter            (default 0.5)
    mutation_sigma   : float – std-dev for Gaussian weight mutation  (default 0.05)
    mutation_prob    : float – per-gene probability of mutation      (default 0.15)
    bit_flip_prob    : float – per-bit flip probability              (default 0.10)
    local_lr         : float – initial learning rate for local search(default 0.05)
    local_iter       : int   – max local-search iterations          (default 30)
    seed             : int | None – random seed for reproducibility
    """

    def __init__(
        self,
        pop_size: int = 200,
        n_generations: int = 200,
        max_cardinality: int = 15,
        risk_free_rate: float = 0.02,
        n_elite: int = 5,
        top_m_refine: int = 10,
        tournament_size: int = 3,
        crossover_alpha: float = 0.5,
        mutation_sigma: float = 0.05,
        mutation_prob: float = 0.15,
        bit_flip_prob: float = 0.10,
        local_lr: float = 0.05,
        local_iter: int = 30,
        seed: Optional[int] = None,
        # Penalty strengths for constraints inside the fitness function.
        penalty_weights: float = 100.0,
        penalty_cardinality: float = 50.0,
        penalty_negativity: float = 200.0,
        # Optional Herfindahl concentration penalty. Zero disables it.
        penalty_concentration: float = 0.0,
        # Expected-return smoothing parameter passed to the data loader.
        ewma_span: int = 52,
        # Optional James-Stein shrinkage for noisy expected-return estimates.
        mu_shrinkage: bool = False,
        # Fitness variance below this threshold increases exploration.
        variance_threshold: float = 1e-5,
    ):
        if pop_size < 2:
            raise ValueError("pop_size must be ≥ 2")
        if n_generations < 1:
            raise ValueError("n_generations must be ≥ 1")
        if max_cardinality < 1:
            raise ValueError("max_cardinality must be ≥ 1")
        if not (0.0 <= risk_free_rate < 1.0):
            raise ValueError("risk_free_rate must be in [0, 1)")

        self.pop_size = pop_size
        self.n_gen = n_generations
        self.K = max_cardinality
        self.rf = risk_free_rate
        self.n_elite = n_elite
        self.top_m = top_m_refine
        self.tournament_size = tournament_size
        self.cx_alpha = crossover_alpha
        self.mut_sigma = mutation_sigma
        self.mut_prob = mutation_prob
        self.bit_flip = bit_flip_prob
        self.local_lr = local_lr
        self.local_iter = local_iter
        self.seed = seed
        self.penalty_weights = penalty_weights
        self.penalty_cardinality = penalty_cardinality
        self.penalty_negativity = penalty_negativity
        self.penalty_concentration = penalty_concentration
        self.ewma_span = ewma_span
        self.mu_shrinkage = mu_shrinkage
        self.variance_threshold = variance_threshold

    # =====================================================================
    #  PUBLIC API
    # =====================================================================
    def run(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: int = 52,
        repo: Optional[PortfolioRepository] = None,
    ) -> OptimizationResult:
        """Run the full hybrid optimisation pipeline.

        Parameters
        ----------
        tickers    : list[str] | None
            Subset of tickers. ``None`` uses all database tickers.
        start_date : str | None
            Optional ISO-8601 start filter.
        end_date   : str | None
            Optional ISO-8601 end filter.
        frequency  : int
            Annualisation factor. Use 52 for weekly data.
        repo       : PortfolioRepository | None
            Optional repository instance for tests or custom data access.

        Returns
        -------
        OptimizationResult
        """
        rng = np.random.default_rng(self.seed)

        # ── 1. Load optimisation inputs ──────────────────────────────────
        loader = DataLoader(
            repo=repo,
            ewma_span=self.ewma_span,
            mu_shrinkage=self.mu_shrinkage,
        )
        tickers, mu, cov, prices = loader.load(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
        )
        n_assets = len(tickers)
        logger.info("Assets after filtering: %d", n_assets)

        # Cardinality cannot exceed the number of available assets.
        K = min(self.K, n_assets)

        # ── 2. Build evaluator, operators, and local refiner ─────────────
        evaluator = FitnessEvaluator(
            mu=mu,
            cov=cov,
            risk_free_rate=self.rf,
            max_cardinality=K,
            penalty_weights=self.penalty_weights,
            penalty_cardinality=self.penalty_cardinality,
            penalty_negativity=self.penalty_negativity,
            penalty_concentration=self.penalty_concentration,
        )

        operators = EvoOperators(
            n_assets=n_assets,
            max_cardinality=K,
            tournament_size=self.tournament_size,
            crossover_alpha=self.cx_alpha,
            mutation_sigma=self.mut_sigma,
            mutation_prob=self.mut_prob,
            bit_flip_prob=self.bit_flip,
            rng=rng,
            cov=cov,
        )

        refiner = LocalRefiner(
            evaluator=evaluator,
            max_iter=self.local_iter,
            lr_init=self.local_lr,
        )

        # ── 3. Evolve the population with adaptive mutation ──────────────
        population = operators.init_population(self.pop_size, mu=mu)
        history: List[float] = []

        for gen in range(self.n_gen):
            # Score the current population before selection.
            for ind in population:
                evaluator.evaluate(ind)

            best_fit = max(ind.fitness for ind in population)
            history.append(best_fit)

            # Increase exploration when fitness variance collapses; reduce it
            # once the population is diverse enough again.
            pop_variance = np.var([ind.fitness for ind in population])

            if pop_variance < self.variance_threshold:
                # Low variance suggests stagnation, so explore more.
                operators.heat_up()
            else:
                # Healthy variance allows mutation to return toward baseline.
                operators.cool_down()

            if gen % 25 == 0 or gen == self.n_gen - 1:
                avg_fit = np.mean([ind.fitness for ind in population])
                logger.info(
                    "Gen %4d / %d  |  best=%.4f  avg=%.4f  var=%.2e  "
                    "σ_mut=%.4f  p_mut=%.3f",
                    gen, self.n_gen, best_fit, avg_fit, pop_variance,
                    operators.mut_sigma, operators.mut_prob,
                )

            # Preserve the strongest individuals unchanged.
            elites = operators.elitism(population, self.n_elite)

            # Fill the next generation from selected and mutated parents.
            next_gen: List[Individual] = list(elites)

            while len(next_gen) < self.pop_size:
                p1 = operators.tournament_select(population)
                p2 = operators.tournament_select(population)
                c1, c2 = operators.crossover(p1, p2)
                c1 = operators.mutate(c1)
                c2 = operators.mutate(c2)
                next_gen.append(c1)
                if len(next_gen) < self.pop_size:
                    next_gen.append(c2)

            population = next_gen

        # Score the final population before post-processing.
        for ind in population:
            evaluator.evaluate(ind)

        # ── 4. Refine the strongest candidates locally ───────────────────
        top_inds = sorted(population, key=lambda x: x.fitness, reverse=True)[
            : self.top_m
        ]
        logger.info(
            "Refining top %d individuals with local gradient search …", len(top_inds)
        )

        refined: List[Individual] = []
        for ind in top_inds:
            refined.append(refiner.refine(ind))

        # ── 4b. Polish the champion with SLSQP when SciPy is available ───
        try:
            from scipy.optimize import minimize

            champion_pre = max(refined, key=lambda x: x.fitness)
            active = np.where(champion_pre.binary > 0.5)[0]
            w0 = champion_pre.weights[active]
            w0 = np.maximum(w0, 0)
            w0 /= w0.sum()

            def neg_sharpe(w_sub):
                w_full = np.zeros(n_assets)
                w_full[active] = w_sub
                return -evaluator.sharpe(w_full)

            constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]


            safe_max_weight = max(0.15, (1.0 / len(active)) + 0.02)
            bounds = [(0.0, safe_max_weight)] * len(active)

            res = minimize(neg_sharpe, w0, method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={"maxiter": 500, "ftol": 1e-10})
            if res.success:
                w_full = np.zeros(n_assets)
                w_full[active] = np.maximum(res.x, 0)
                w_sum = w_full.sum()
                if w_sum > 1e-12:          # Guard against all-zero SciPy output.
                    w_full /= w_sum
                    # Evaluate the polished weights with the same penalised
                    # objective used by the evolutionary loop.
                    from copy import deepcopy
                    candidate = deepcopy(champion_pre)
                    candidate.weights = w_full
                    evaluator.evaluate(candidate)
                    if candidate.fitness > champion_pre.fitness:
                        champion_pre.weights = w_full
                        champion_pre.fitness = candidate.fitness
                        logger.info("Scipy SLSQP improved Sharpe to %.4f", champion_pre.fitness)
            refined.append(champion_pre)
        except ImportError:
            pass  # SLSQP polish is optional when SciPy is unavailable.
        except Exception as exc:           # Numerical failures should not abort GA output.
            logger.warning("Scipy polish failed and was skipped: %s", exc)

        # Pick the best candidate after refinement and optional polish.
        champion = max(refined, key=lambda x: x.fitness)

        # ── 5. Assemble user-facing result ───────────────────────────────
        w = champion.weights
        selected_mask = champion.binary > 0.5
        selected_tickers = [t for t, s in zip(tickers, selected_mask) if s]
        selected_weights = w[selected_mask]

        port_return = float(w @ mu)
        port_var = float(w @ cov @ w)
        port_risk = float(np.sqrt(max(port_var, 0.0)))
        sharpe = (port_return - self.rf) / port_risk if port_risk > 1e-12 else 0.0

        # Drop negligible positions and normalise the reported weights.
        raw_dict = {
            t: float(wt) for t, wt in zip(selected_tickers, selected_weights) if wt > 1e-6
        }
        _total = sum(raw_dict.values())
        weight_dict = (
            {t: wt / _total for t, wt in raw_dict.items()}
            if _total > 1e-12 else raw_dict
        )

        result = OptimizationResult(
            weights=weight_dict,
            selected_assets=list(weight_dict.keys()),
            sharpe_ratio=sharpe,
            expected_return=port_return,
            portfolio_risk=port_risk,
            n_generations=self.n_gen,
            history=history,
        )

        self._log_result(result)
        return result

    # ----- result logging -------------------------------------------------
    @staticmethod
    def _log_result(result: OptimizationResult) -> None:
        logger.info("=" * 60)
        logger.info("  HYBRID EVO OPTIMIZER – RESULT")
        logger.info("=" * 60)
        logger.info("  Sharpe Ratio      : %.4f", result.sharpe_ratio)
        logger.info("  Expected Return   : %.4f (%.2f%%)", result.expected_return, result.expected_return * 100)
        logger.info("  Portfolio Risk     : %.4f (%.2f%%)", result.portfolio_risk, result.portfolio_risk * 100)
        logger.info("  Selected Assets   : %d", len(result.selected_assets))
        logger.info("-" * 60)
        for ticker, w in sorted(result.weights.items(), key=lambda x: x[1], reverse=True):
            logger.info("    %-8s  %.2f%%", ticker, w * 100)
        logger.info("=" * 60)





# ═══════════════════════════════════════════════════════════════════════════
#  CONVENIENCE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def main() -> OptimizationResult:
    """Run the optimizer with demonstration defaults."""
    optimizer = HybridEvoOptimizer(
        pop_size=200,
        n_generations=200,
        max_cardinality=15,
        risk_free_rate=0.02,
        n_elite=10,          # Preserve more high-quality candidates.
        top_m_refine=20,     # Refine a wider set of finalists.
        tournament_size=5,   # Apply stronger selection pressure.
        mutation_prob=0.30,
        bit_flip_prob=0.15,
        crossover_alpha=0.7,
        local_lr=0.1,        # Start local refinement with a larger step.
        local_iter=50,       # Give local refinement more iterations.
        seed=42,
    )
    return optimizer.run()


if __name__ == "__main__":
    result = main()
    print(f"\nBest Sharpe : {result.sharpe_ratio:.4f}")
    print(f"Return      : {result.expected_return:.2%}")
    print(f"Risk        : {result.portfolio_risk:.2%}")
    print(f"Assets      : {result.selected_assets}")
