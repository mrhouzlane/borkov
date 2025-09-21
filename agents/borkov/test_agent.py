#!/usr/bin/env python3
"""
Test script for the Borkov Protocol Optimization Agent

This script runs smoke tests for the Borkov agent's core analytics tools:
- Single borrower journey simulation
- Cohort simulation
- Markov transition matrix
- Steady-state calculation
- Parameter optimization sweep

No Coral/MCP required; runs functions directly.
"""

import os
from dotenv import load_dotenv

from main import (
    borkov_single_journey,
    borkov_cohort_analysis,
    borkov_transition_matrix,
    borkov_steady_state,
    borkov_optimize_roi
)

def test_single_journey():
    print("Testing borkov_single_journey...")
    result = borkov_single_journey.invoke({
        "n_loans": 40,
        "p_repay": 0.91,
        "d": 0.14,
        "reset_on_default": True
    })
    print("Result:", result)
    return result.get("success", False)

def test_cohort():
    print("\nTesting borkov_cohort_analysis...")
    archetypes = {
        'Cautious': {'p_repay': 0.99, 'count': 10, 'color': 'blue'},
        'Average': {'p_repay': 0.92, 'count': 15, 'color': 'yellow'}
    }
    result = borkov_cohort_analysis.invoke({
        "n_loans": 30,
        "archetypes": archetypes
    })
    print("Result:", result)
    return result.get("success", False)

def test_transition_matrix():
    print("\nTesting borkov_transition_matrix...")
    result = borkov_transition_matrix.invoke({"p_repay": 0.9})
    print("Matrix:", result)
    return result.get("success", False)

def test_steady_state():
    print("\nTesting borkov_steady_state...")
    result = borkov_steady_state.invoke({"p_repay": 0.9})
    print("Steady state:", result)
    return result.get("success", False)

def test_optimize_roi():
    print("\nTesting borkov_optimize_roi sweep...")
    result = borkov_optimize_roi.invoke({
        "interest_rate_start": 0.08,
        "interest_rate_end": 0.12,
        "interest_rate_step": 0.02,
        "p_repay_start": 0.88,
        "p_repay_end": 0.94,
        "p_repay_step": 0.03,
        "n_loans": 20,
        "penalty": 0.5
    })
    print("Best params:", result.get("best_params"))
    print("Top sweep rows:")
    for row in result.get("top10_results", []):
        print(row)
    return result.get("success", False)

def main():
    load_dotenv()
    print("Borkov Protocol Agent Test Suite")
    print("=" * 40)
    tests = [
        ("Single Borrower Journey", test_single_journey),
        ("Cohort Simulation", test_cohort),
        ("Transition Matrix", test_transition_matrix),
        ("Steady-State Calculation", test_steady_state),
        ("Optimization Sweep", test_optimize_roi),
    ]
    results = []
    for name, func in tests:
        try:
            ok = func()
            results.append((name, ok))
            print(f"✅ {name}: {'PASSED' if ok else 'FAILED'}")
        except Exception as e:
            results.append((name, False))
            print(f"❌ {name}: ERROR - {e}")
    print("\nTest Summary:")
    for name, ok in results:
        print(f"• {name}: {'PASSED' if ok else 'FAILED'}")
    print("=" * 40)

if __name__ == "__main__":
    main()
