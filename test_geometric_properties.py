#!/usr/bin/env python3
"""
Test automatic extraction of geometric properties from invariants.

This demonstrates how treating space/time/light consistently allows
automatic computation of surface area, volume, and curvature from
the lightlike invariant I = (A - B)².
"""

import numpy as np

from src.eigen_invariants import compute_geometric_properties


def test_problem_scales():
    """Test how geometric properties scale with problem size."""
    print("=" * 70)
    print("Automatic Geometric Property Extraction")
    print("=" * 70)
    print()

    test_cases = [
        ("Tiny problem", 0.1, 0.0),
        ("Small problem", 1.0, 0.0),
        ("Medium problem", 5.0, 0.0),
        ("Large problem", 10.0, 0.0),
        ("Equilibrium", 0.0, 0.0),
        ("Opposing forces balanced", 5.0, 5.0),
        ("Moderate imbalance", 7.0, 3.0),
        ("Large imbalance", 10.0, 1.0),
    ]

    for name, A, B in test_cases:
        props = compute_geometric_properties(A, B)

        print(f"{name}:")
        print(f"  A = {A:.2f}, B = {B:.2f}")
        print(f"  I = (A - B)² = {(A - B)**2:.4f}")
        print(f"  Status: {props['status']}")

        if props["status"] == "active":
            print(f"  Radius (r):       {props['radius']:.4f}")
            print(f"  Surface (4πr²):   {props['surface_area']:.4f}")
            print(f"  Volume (4/3πr³):  {props['volume']:.4f}")
            print(f"  Curvature (1/r):  {props['curvature']:.4f}")

            # Interpretation
            if props["curvature"] > 1.0:
                print(f"  → Well-conditioned problem (tight geometry)")
            elif props["curvature"] > 0.1:
                print(f"  → Moderate conditioning")
            else:
                print(f"  → Spread out problem (flat geometry)")
        else:
            print(f"  → Perfect equilibrium (I = 0)")

        print()


def test_opposing_pairs():
    """Test with different types of opposing pairs from the framework."""
    print("=" * 70)
    print("Different Opposing Pairs - Same Geometry")
    print("=" * 70)
    print()

    scenarios = [
        ("Position vs Velocity", 5.0, 2.0, "Spacelike-dominant"),
        ("Individual vs Collective", 3.0, 7.0, "Timelike-dominant"),
        ("Signal vs Noise", 8.0, 8.0, "Lightlike equilibrium"),
        ("Measurement vs Prediction", 6.0, 4.0, "Spacelike-dominant"),
    ]

    for name, A, B, regime in scenarios:
        props = compute_geometric_properties(A, B)
        delta = A - B

        print(f"{name}:")
        print(f"  A = {A:.2f}, B = {B:.2f}, Δ = {delta:.2f}")
        print(f"  Regime: {regime}")
        print(f"  Sphere radius: r = {props['radius']:.4f}")
        print(f"  Problem complexity: {props['surface_area']:.4f}")
        print(f"  Curvature: {props['curvature']:.4f}")
        print()


def test_convergence_prediction():
    """Show how geometric properties predict convergence behavior."""
    print("=" * 70)
    print("Convergence Prediction from Geometry")
    print("=" * 70)
    print()

    print("As system converges to equilibrium:")
    print()

    # Simulate convergence
    steps = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.01, 0.001]

    print(f"{'Step':>6} {'r':>8} {'Surface':>12} {'Volume':>12} {'Curvature':>12}")
    print("-" * 62)

    for i, A in enumerate(steps):
        B = 0.0  # Converging toward target
        props = compute_geometric_properties(A, B)

        print(
            f"{i:>6} {props['radius']:>8.4f} {props['surface_area']:>12.4f} "
            f"{props['volume']:>12.4f} {props['curvature']:>12.4f}"
        )

    print()
    print("Notice:")
    print("  - Surface area (complexity) decreases as r²")
    print("  - Volume (search space) decreases as r³")
    print("  - Curvature (conditioning) increases as 1/r")
    print("  → Problem becomes easier as it approaches equilibrium")
    print()


def test_universality():
    """Demonstrate that same r gives same geometry regardless of origin."""
    print("=" * 70)
    print("Universality: Same r = Same Geometry")
    print("=" * 70)
    print()

    print("Different problems with same radius:")
    print()

    # All have r = 3.0, but from different (A, B) pairs
    pairs = [
        ("Control scenario 1", 3.0, 0.0),
        ("Control scenario 2", 5.0, 2.0),
        ("Control scenario 3", 10.0, 7.0),
        ("Control scenario 4", -1.0, -4.0),
    ]

    for name, A, B in pairs:
        props = compute_geometric_properties(A, B)
        print(f"{name}: A={A:.1f}, B={B:.1f}")
        print(f"  r = {props['radius']:.4f}")
        print(f"  Surface = {props['surface_area']:.4f}")
        print(f"  Volume = {props['volume']:.4f}")
        print()

    print("→ All have identical geometry (same r)")
    print("→ Only the scale matters, not the specific values")
    print()


if __name__ == "__main__":
    test_problem_scales()
    test_opposing_pairs()
    test_convergence_prediction()
    test_universality()

    print("=" * 70)
    print("Key Insight:")
    print("=" * 70)
    print()
    print("The invariant I = (A - B)² contains complete geometric information:")
    print("  • Radius r = √I measures problem scale")
    print("  • Surface 4πr² measures complexity")
    print("  • Volume (4/3)πr³ measures search space")
    print("  • Curvature 1/r measures conditioning")
    print()
    print("All problems are spheres. Only r varies.")
    print("The geometry is universal. The scale is problem-specific.")
    print()
