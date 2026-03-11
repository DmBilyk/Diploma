"""
predictor.py
============
Generates an "Expected Returns" vector from a trained LSTM and
feeds it into the HybridEvoOptimizer to produce an optimal portfolio.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pypfopt import risk_models

from app.ai.create_dataset import prepare_lstm_data
from app.ai.lstm_model import PortfolioLSTMModel
from app.data.repository import PortfolioRepository
from app.algorithms.hybrid_evo_optimizer import (
    FitnessEvaluator,
    EvoOperators,
    LocalRefiner,
    OptimizationResult,
)

logger = logging.getLogger(__name__)

# Annualisation factor for weekly data
_WEEKLY_FREQ = 52


def predict_expected_returns(
    model: PortfolioLSTMModel,
    repo: PortfolioRepository,
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    seq_length: int = 52,
) -> np.ndarray:
    """Generate an annualised expected-return vector using LSTM predictions.

    Steps
    -----
    1. Build the most recent input window from the database.
    2. Feed it through the trained LSTM → predicted weekly log-return.
    3. Annualise: μ_annual = predicted_weekly_return × 52.

    Returns
    -------
    mu : np.ndarray of shape (num_assets,)
    """
    dataset, used_tickers = prepare_lstm_data(
        repo=repo,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        seq_length=seq_length,
    )

    # Take the LAST window as the most recent context
    X_last = dataset.X[-1:].numpy()                      # (1, seq, assets)
    predicted_weekly = model.predict(X_last).flatten()    # (assets,)

    # Annualise the predicted weekly log-return
    mu_annual = predicted_weekly * _WEEKLY_FREQ

    logger.info(
        "LSTM μ: min=%.4f, max=%.4f, mean=%.4f",
        mu_annual.min(), mu_annual.max(), mu_annual.mean(),
    )
    return mu_annual


# ==================================================================
#  Integration: LSTM → Optimizer
# ==================================================================
def run_lstm_optimization(
    model: PortfolioLSTMModel,
    repo: PortfolioRepository,
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    seq_length: int = 52,
    alpha: float = 0.5,
    # Optimizer hyper-parameters
    pop_size: int = 200,
    n_generations: int = 200,
    max_cardinality: int = 15,
    risk_free_rate: float = 0.02,
    n_elite: int = 5,
    top_m_refine: int = 10,
    seed: Optional[int] = None,
) -> Tuple[OptimizationResult, Dict[str, float]]:
    """End-to-end pipeline: LSTM predictions → evolutionary optimisation.

    Parameters
    ----------
    model : PortfolioLSTMModel
        Already-trained LSTM model.
    alpha : float
        Blending weight.  ``mu = α · mu_lstm + (1-α) · mu_historical``.
        1.0 = pure LSTM,  0.0 = pure historical.

    Returns
    -------
    result : OptimizationResult
    mu_dict : dict[str, float]
        Per-ticker expected return used by the optimiser.
    """
    if tickers is None:
        tickers = repo.get_all_tickers()

    # ── 1.  Prices + covariance (same logic as optimizer's DataLoader) ──
    prices = repo.get_price_history(tickers, start_date=start_date, end_date=end_date)
    if prices.empty:
        raise ValueError("No price data returned from the database.")

    prices = prices.sort_index()
    threshold = int(len(prices) * 0.80)
    valid = prices.notna().sum()
    valid = valid[valid >= threshold].index.tolist()
    prices = prices[valid].dropna()
    tickers = list(prices.columns)
    n_assets = len(tickers)

    cov = risk_models.CovarianceShrinkage(
        prices, frequency=_WEEKLY_FREQ,
    ).ledoit_wolf().values

    # Historical μ (mean log-return, annualised)
    log_ret = np.log(prices / prices.shift(1)).dropna()
    mu_hist = log_ret.mean().values * _WEEKLY_FREQ

    # ── 2.  LSTM μ ──────────────────────────────────────────────────────
    mu_lstm = predict_expected_returns(
        model, repo, tickers,
        start_date=start_date, end_date=end_date,
        seq_length=seq_length,
    )

    # Blend
    mu = alpha * mu_lstm + (1.0 - alpha) * mu_hist
    logger.info("Blended μ (α=%.2f): mean=%.4f", alpha, mu.mean())

    # ── 3.  Run evolutionary optimisation with custom μ ─────────────────
    rng = np.random.default_rng(seed)
    K = min(max_cardinality, n_assets)

    evaluator = FitnessEvaluator(
        mu=mu, cov=cov, risk_free_rate=risk_free_rate, max_cardinality=K,
    )
    operators = EvoOperators(n_assets=n_assets, max_cardinality=K, rng=rng)
    refiner = LocalRefiner(evaluator=evaluator)

    population = operators.init_population(pop_size, mu=mu)
    history: List[float] = []
    best_ever = -np.inf
    stagnation = 0

    for gen in range(n_generations):
        for ind in population:
            evaluator.evaluate(ind)

        best_fit = max(ind.fitness for ind in population)
        history.append(best_fit)

        if best_fit > best_ever + 1e-4:
            best_ever = best_fit
            stagnation = 0
        else:
            stagnation += 1

        if stagnation >= 30:
            stagnation = 0
            ranked = sorted(population, key=lambda x: x.fitness, reverse=True)
            n_keep = pop_size // 2
            population = ranked[:n_keep] + operators.init_population(
                pop_size - n_keep, mu=mu,
            )
            continue

        elites = operators.elitism(population, n_elite)
        next_gen = list(elites)
        while len(next_gen) < pop_size:
            p1 = operators.tournament_select(population)
            p2 = operators.tournament_select(population)
            c1, c2 = operators.crossover(p1, p2)
            next_gen.append(operators.mutate(c1))
            if len(next_gen) < pop_size:
                next_gen.append(operators.mutate(c2))
        population = next_gen

    # Final evaluation
    for ind in population:
        evaluator.evaluate(ind)

    # ── 4.  Local refinement ────────────────────────────────────────────
    top_inds = sorted(population, key=lambda x: x.fitness, reverse=True)[
        :top_m_refine
    ]
    refined = [refiner.refine(ind) for ind in top_inds]
    champion = max(refined, key=lambda x: x.fitness)

    # ── 5.  Assemble result ─────────────────────────────────────────────
    w = champion.weights
    selected_mask = champion.binary > 0.5
    selected_tickers = [t for t, s in zip(tickers, selected_mask) if s]
    selected_weights = w[selected_mask]

    port_ret = float(w @ mu)
    port_var = float(w @ cov @ w)
    port_risk = float(np.sqrt(max(port_var, 0.0)))
    sharpe = (port_ret - risk_free_rate) / port_risk if port_risk > 1e-12 else 0.0

    weight_dict = {
        t: float(wt)
        for t, wt in zip(selected_tickers, selected_weights)
        if wt > 1e-6
    }

    result = OptimizationResult(
        weights=weight_dict,
        selected_assets=list(weight_dict.keys()),
        sharpe_ratio=sharpe,
        expected_return=port_ret,
        portfolio_risk=port_risk,
        n_generations=n_generations,
        history=history,
    )

    mu_dict = {t: float(m) for t, m in zip(tickers, mu)}

    logger.info(
        "LSTM Optimization done  |  Sharpe=%.4f  Return=%.2f%%  Risk=%.2f%%  Assets=%d",
        sharpe, port_ret * 100, port_risk * 100, len(weight_dict),
    )
    return result, mu_dict
