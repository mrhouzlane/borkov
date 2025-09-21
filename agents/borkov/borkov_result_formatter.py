"""
borkov_result_formatter.py

Utilities for formatting and packaging Borkov agent simulation/optimization results.
This module provides functions to summarize simulation stats, optimization sweeps,
and to structure actionable protocol design recommendations for downstream consumption.

Designed for output as JSON to UIs, dashboards, notebooks, or other agents.
"""

from typing import List, Dict, Any, Optional

def format_single_journey_result(journey_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the result of a single borrower's journey simulation.

    Args:
        journey_stats: Dict returned from simulate_borkov_journey.

    Returns:
        JSON-ready dict with highlights & summary.
    """
    return {
        "success": True,
        "final_multiplier": journey_stats['final_multiplier'],
        "total_defaults": journey_stats['defaults'],
        "longest_streak": journey_stats['max_streak'],
        "full_journey": journey_stats['journey'],
        "bullets": [
            f"Final multiplier: **{journey_stats['final_multiplier']:.2f}**",
            f"Number of defaults: **{journey_stats['defaults']}**",
            f"Longest success streak: **{journey_stats['max_streak']}**",
        ]
    }

def format_cohort_summary(df) -> Dict[str, Any]:
    """
    Format cohort groupby DataFrame and present summary stats.

    Args:
        df: pandas DataFrame as returned from mass_borkov_simulation.

    Returns:
        JSON-ready dict with archetype group summary table and highlights.
    """
    summary = df.groupby('archetype')[['final_multiplier', 'defaults', 'max_streak']].agg(['mean', 'std']).round(3)
    bullets = [
        f"{arch}: Final multiplier μ={summary.loc[arch, ('final_multiplier', 'mean')]:.2f}, σ={summary.loc[arch, ('final_multiplier', 'std')]:.2f}; "
        f"Defaults μ={summary.loc[arch, ('defaults', 'mean')]:.2f}, Max streak μ={summary.loc[arch, ('max_streak', 'mean')]:.2f}"
        for arch in summary.index
    ]
    return {
        "success": True,
        "archetype_stats": summary.to_dict(),
        "bullets": bullets
    }

def format_transition_matrix(matrix, p_repay: float) -> Dict[str, Any]:
    """
    Format Markov transition matrix for display.

    Args:
        matrix: np.ndarray (3x3)
        p_repay: float

    Returns:
        JSON dict with readable structure and markdown.
    """
    state_names = ["Reset", "Building", "Established"]
    matrix_list = [[float(f"{x:.4f}") for x in row] for row in matrix.tolist()]
    bullets = [
        f"Transition probability from Building to Established: **{matrix[1,2]:.2f}**",
        f"Probability of resetting from any state: **{1-p_repay:.2f}**"
    ]
    return {
        "success": True,
        "matrix": {
            "states": state_names,
            "rows": matrix_list
        },
        "bullets": bullets
    }

def format_steady_state(steady_probs, p_repay: float) -> Dict[str, Any]:
    """
    Format steady-state probabilities for protocol states.

    Args:
        steady_probs: np.ndarray of 3 floats
        p_repay: float

    Returns:
        Dict[str, Any]
    """
    state_names = ["Reset", "Building", "Established"]
    readable = {
        state: float(f"{steady_probs[i]:.4f}")
        for i, state in enumerate(state_names)
    }
    sorted_states = sorted(readable.items(), key=lambda x: -x[1])
    bullets = [f"Steady-state: {state} = {prob:.2%}" for state, prob in sorted_states]
    return {
        "success": True,
        "steady_state_probs": readable,
        "bullets": bullets
    }

def format_optimization_results(
    best_params: Dict[str, Any],
    best_projected_roi: float,
    best_steady_state: List[float],
    top_rows: List[Dict[str, Any]],
    rationale: Optional[str] = None
) -> Dict[str, Any]:
    """
    Package parameter optimization results and recos.

    Args:
        best_params: Best parameter set found in sweep/grid
        best_projected_roi: Corresponding ROI
        best_steady_state: Steady-state [Reset, Building, Established]
        top_rows: List of top parameter/ROI dicts
        rationale: Optional string giving reasoning

    Returns:
        Dict with recommendations, highlights, and top param table.
    """
    recos = [
        f"Set interest_rate to **{best_params['interest_rate']:.3f}**.",
        f"Set p_repay to **{best_params['p_repay']:.3f}**.",
        f"Projected ROI at optimum: **{best_projected_roi:.4f}**.",
        f"Steady-state distribution: Reset={best_steady_state[0]:.2%}, Building={best_steady_state[1]:.2%}, Established={best_steady_state[2]:.2%}."
    ]
    if rationale:
        recos.append(rationale)
    return {
        "success": True,
        "best_param_set": best_params,
        "best_projected_roi": best_projected_roi,
        "best_steady_state": {
            "Reset": best_steady_state[0],
            "Building": best_steady_state[1],
            "Established": best_steady_state[2]
        },
        "top10_results": top_rows,
        "recommendations": recos
    }
