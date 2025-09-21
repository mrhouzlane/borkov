"""
borkov_agent_core.py

Core logic for Borkov – a protocol design and ROI optimization agent using Markov-process simulations.
Provides functions for simulating borrower journeys, computing state matrices, and running
parameter optimization sweeps for lending protocols.

No I/O or PDF generation; outputs are pure analytics/statistics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


def simulate_borkov_journey(
    n_loans: int,
    p_repay: float,
    d: float = 0.15,
    reset_on_default: bool = True
) -> Dict[str, Any]:
    """
    Simulate a single borrower's reward multiplier journey as a Markov process.

    Args:
        n_loans: Number of loan cycles (int)
        p_repay: Probability of successful repayment (float, 0–1)
        d: Multiplier increment per successful repayment
        reset_on_default: If True, reset multiplier to 1.0 on default

    Returns:
        Dict of journey states, default count, streaks, final multiplier, etc.
    """
    x_vals = [1.0]
    defaults = 0
    streak = 0
    max_streak = 0

    for _ in range(n_loans):
        if np.random.rand() < p_repay:
            x_vals.append(x_vals[-1] + d)
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            defaults += 1
            streak = 0
            if reset_on_default:
                x_vals.append(1.0)
            else:
                x_vals.append(max(1.0, x_vals[-1] - 0.5))
    return {
        "journey": x_vals,
        "defaults": defaults,
        "max_streak": max_streak,
        "final_multiplier": x_vals[-1]
    }


def mass_borkov_simulation(
    n_loans: int,
    archetypes: Dict[str, Dict[str, Any]] = None
) -> (pd.DataFrame, List[List[float]]):
    """
    Simulate a cohort of borrowers (multiple archetypes) for protocol analysis.

    Args:
        n_loans: Loan cycles per borrower
        archetypes: Dict {
            Name: {'p_repay': float, 'count': int, 'color': str},
            ...
        }

    Returns:
        (DataFrame: results, List: list-of-journey-paths)
    """
    if archetypes is None:
        archetypes = {
            'Diamond Hands': {'p_repay': 0.9, 'count': 300, 'color': 'green'},
            'Paper Hands': {'p_repay': 0.92, 'count': 500, 'color': 'orange'},
            'Degen Traders': {'p_repay': 0.95, 'count': 200, 'color': 'red'}
        }

    results = []
    all_journeys = []
    for label, params in archetypes.items():
        for _ in range(params['count']):
            sim = simulate_borkov_journey(n_loans, params['p_repay'])
            results.append({
                'archetype': label,
                'p_repay': params['p_repay'],
                'final_multiplier': sim['final_multiplier'],
                'defaults': sim['defaults'],
                'max_streak': sim['max_streak'],
                'journey': sim['journey'],
            })
            all_journeys.append(sim['journey'])
    df = pd.DataFrame(results)
    return df, all_journeys


def create_borkov_matrix(p_repay: float):
    """
    Create the state transition matrix for the Borkov Markov process.

    States: 0 = Reset, 1 = Building, 2 = Established

    Returns:
        numpy.ndarray shape (3, 3)
    """
    p_default = 1 - p_repay
    return np.array([
        [p_default, p_repay * 0.8, p_repay * 0.2],  # From Reset
        [p_default, p_repay * 0.3, p_repay * 0.7],  # From Building
        [p_default, 0.0, p_repay],                  # From Established
    ])


def steady_state_analysis(borkov_matrix: np.ndarray):
    """
    Compute the steady-state probabilities for the given transition matrix.

    Returns:
        np.ndarray: vector of length-3 state probabilities
    """
    eigenvals, eigenvects = np.linalg.eig(borkov_matrix.T)
    idx = np.argmax(eigenvals.real)
    steady = eigenvects[:, idx].real
    return steady / steady.sum()


def parameter_grid_search(
    interest_rate_range: (float, float, float),
    p_repay_range: (float, float, float),
    n_loans: int = 50,
    penalty: float = 0.5
) -> Dict[str, Any]:
    """
    Sweep grid of protocol parameters and return optimal design suggestion.

    Args:
        interest_rate_range: (start, stop, step)
        p_repay_range: (start, stop, step)
        n_loans: simulation cycles
        penalty: default penalty (currently fixed, can parameterize more if needed)

    Returns:
        Dict: with best param set, projected ROI, steady state, and table of top results
    """
    results = []
    best_roi = -np.inf
    best_params = {}
    best_steady = None
    for ir in np.arange(*interest_rate_range):
        for p_repay in np.arange(*p_repay_range):
            sim = simulate_borkov_journey(n_loans, p_repay, d=ir, reset_on_default=True)
            matrix = create_borkov_matrix(p_repay)
            steady = steady_state_analysis(matrix)
            avg_mult = sim["final_multiplier"]
            roi = avg_mult * ir
            results.append({
                "interest_rate": round(ir, 3),
                "p_repay": round(p_repay, 3),
                "projected_roi": round(roi, 4),
                "steady_state_established": round(float(steady[2]), 4)
            })
            if roi > best_roi:
                best_roi = roi
                best_params = {
                    "interest_rate": round(ir, 3),
                    "p_repay": round(p_repay, 3),
                    "penalty": penalty
                }
                best_steady = [round(float(x), 4) for x in steady]
    top_rows = sorted(results, key=lambda x: -x["projected_roi"])[:10]
    return {
        "best_params": best_params,
        "best_projected_roi": round(best_roi, 4),
        "best_steady_state": best_steady,
        "top_results": top_rows
    }
