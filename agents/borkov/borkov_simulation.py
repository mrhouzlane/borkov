"""
borkov_analysis.py

Core Markov-process analytics for the Borkov agent.
Simulates borrower journeys, evaluates protocol ROI, and provides state transition analysis
for optimizing lending protocol economics based on repayment behaviors.
"""

import numpy as np
import pandas as pd

def simulate_borkov_journey(n_loans, p_repay, d=0.15, reset_on_default=True):
    """
    Simulate the journey of a single borrower's multiplier via a Markov process.

    Args:
        n_loans (int): Number of loan events/cycles to simulate.
        p_repay (float): Probability [0,1] of successful repayment.
        d (float): Increment to multiplier for each successful repayment.
        reset_on_default (bool): If True, reset multiplier to 1.0 on default; else apply partial penalty.

    Returns:
        dict: {
            'journey': [list of multipliers per loan event],
            'defaults': int,            # number of defaults
            'max_streak': int,          # longest consecutive success streak
            'final_multiplier': float   # last multiplier
        }
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
        'journey': x_vals,
        'defaults': defaults,
        'max_streak': max_streak,
        'final_multiplier': x_vals[-1]
    }

def mass_borkov_simulation(n_borrowers=1000, n_loans=50, archetypes=None):
    """
    Simulate a cohort of borrowers following archetype definitions.

    Args:
        n_borrowers (int): (Not used, see 'archetypes' instead). Provided for interface consistency.
        n_loans (int): Number of loan cycles per borrower.
        archetypes (dict): {
            label: {'p_repay': float, 'count': int, 'color': str}, ...
        }

    Returns:
        (pd.DataFrame, list): DataFrame of results and list of all journeys.
    """
    if archetypes is None:
        archetypes = {
            'Diamond Hands': {'p_repay': 0.9, 'count': 300, 'color': 'green'},
            'Paper Hands': {'p_repay': 0.92, 'count': 500, 'color': 'orange'},
            'Degen Traders': {'p_repay': 0.95, 'count': 200, 'color': 'red'}
        }

    results = []
    all_journeys = []

    for arch, params in archetypes.items():
        for _ in range(params['count']):
            out = simulate_borkov_journey(n_loans, params['p_repay'])
            results.append({
                'archetype': arch,
                'p_repay': params['p_repay'],
                'final_multiplier': out['final_multiplier'],
                'defaults': out['defaults'],
                'max_streak': out['max_streak'],
                'journey': out['journey'],
            })
            all_journeys.append(out['journey'])

    df = pd.DataFrame(results)
    return df, all_journeys

def create_borkov_matrix(p_repay):
    """
    Create the Borkov Markov transition matrix for lending protocol state changes.

    Borkov States:
    0: Reset/New (multiplier = 1.0)
    1: Building (1.0 < multiplier < 3.0)
    2: Established (multiplier >= 3.0)

    Args:
        p_repay (float): Repayment probability.

    Returns:
        np.ndarray: 3x3 state transition matrix.
    """
    p_default = 1 - p_repay
    return np.array([
        [p_default, p_repay * 0.8, p_repay * 0.2],  # From Reset
        [p_default, p_repay * 0.3, p_repay * 0.7],  # From Building
        [p_default, 0.0, p_repay],                  # From Established
    ])

def steady_state_analysis(borkov_matrix):
    """
    Compute the steady-state probabilities for a Markov matrix.

    Args:
        borkov_matrix (np.ndarray): State transition matrix.

    Returns:
        np.ndarray: Steady-state probability vector (sum to 1).
    """
    eigenvals, eigenvects = np.linalg.eig(borkov_matrix.T)
    idx = np.argmax(eigenvals.real)
    steady = eigenvects[:, idx].real
    return steady / steady.sum()
