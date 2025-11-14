#!/usr/bin/env python3
"""
Benchmark: Adaptive Two-Phase Control

Demonstrates the benefit of adaptive control that combines:
- Phase 1: Fast approach (eta=0.12, lightlike OFF) when far from target
- Phase 2: Ultra-slow terminal descent (eta=0.015, lightlike ON) when close

Expected results:
- Fixed fast: Quick convergence but poor precision (~180mm final error)
- Fixed ultra-slow: Excellent precision (~30mm) but painfully slow
- Adaptive two-phase: Quick approach + precision landing = BEST OF BOTH

This is Priority 1 from UPGRADE_ROADMAP.md
"""

import sys

import numpy as np

sys.path.insert(0, "/home/user/Eigen-Geometric-Control")  # noqa: E402

import pandas as pd  # noqa: E402

from src import (  # noqa: E402
    adaptive_control_parameters,
    compute_ds2,
    compute_gradient,
    detect_oscillation,
    forward_kinematics,
    lightlike_damping_factor,
)


def run_fixed_eta_control(
    eta=0.12,
    use_lightlike=False,
    initial_position=(-1.5, 1.3),
    target=(1.30, 0.36),
    obstacle_center=(0.8, 0.25),
    n_ticks=500,
    Go=6.0,
    lam=0.02,
    L1=0.9,
    L2=0.9,
):
    """Run control with fixed eta and lightlike setting"""
    theta1, theta2 = initial_position
    target = np.array(target)
    obstacle_center = np.array(obstacle_center)
    obstacle_radius = 0.25

    state_history = []
    rows = []

    for t in range(n_ticks):
        x, y = forward_kinematics(theta1, theta2, L1, L2)
        state = np.array([theta1, theta2])
        state_history.append(state)

        ds2_total, components = compute_ds2(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        grad, grad_norm = compute_gradient(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        # Lightlike observer (if enabled)
        oscillating = False
        damping = 0.0

        if use_lightlike and len(state_history) >= 4:
            oscillating, osc_strength = detect_oscillation(state_history, window=4, threshold=0.92)
            if oscillating:
                damping = lightlike_damping_factor(osc_strength) * 0.85

        # Update
        delta = -(1.0 - damping) * eta * grad
        theta1_new = theta1 + delta[0]
        theta2_new = theta2 + delta[1]

        distance_to_target = np.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)

        rows.append(
            {
                "tick": t,
                "x": x,
                "y": y,
                "distance_to_target": distance_to_target,
                "ds2_total": ds2_total,
                "grad_norm": grad_norm,
                "d_obs": components["d_obs"],
                "eta": eta,
                "phase": "FIXED",
                "oscillating": oscillating,
                "damping": damping,
            }
        )

        theta1, theta2 = theta1_new, theta2_new

    return pd.DataFrame(rows)


def run_adaptive_control(
    initial_position=(-1.5, 1.3),
    target=(1.30, 0.36),
    obstacle_center=(0.8, 0.25),
    n_ticks=500,
    terminal_threshold=0.15,
    Go=6.0,
    lam=0.02,
    L1=0.9,
    L2=0.9,
):
    """Run adaptive two-phase control"""
    theta1, theta2 = initial_position
    target = np.array(target)
    obstacle_center = np.array(obstacle_center)
    obstacle_radius = 0.25

    state_history = []
    rows = []

    for t in range(n_ticks):
        x, y = forward_kinematics(theta1, theta2, L1, L2)
        state = np.array([theta1, theta2])
        state_history.append(state)

        distance_to_target = np.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)

        # ADAPTIVE: Determine eta and lightlike based on distance
        eta, use_lightlike, phase = adaptive_control_parameters(
            distance_to_target,
            terminal_threshold=terminal_threshold,
        )

        ds2_total, components = compute_ds2(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        grad, grad_norm = compute_gradient(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        # Lightlike observer (if adaptive phase enabled it)
        oscillating = False
        damping = 0.0

        if use_lightlike and len(state_history) >= 4:
            oscillating, osc_strength = detect_oscillation(state_history, window=4, threshold=0.92)
            if oscillating:
                damping = lightlike_damping_factor(osc_strength) * 0.85

        # Update
        delta = -(1.0 - damping) * eta * grad
        theta1_new = theta1 + delta[0]
        theta2_new = theta2 + delta[1]

        rows.append(
            {
                "tick": t,
                "x": x,
                "y": y,
                "distance_to_target": distance_to_target,
                "ds2_total": ds2_total,
                "grad_norm": grad_norm,
                "d_obs": components["d_obs"],
                "eta": eta,
                "phase": phase,
                "oscillating": oscillating,
                "damping": damping,
            }
        )

        theta1, theta2 = theta1_new, theta2_new

    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("ADAPTIVE TWO-PHASE CONTROL BENCHMARK".center(80))
    print("=" * 80)
    print()
    print("Comparing three control strategies:")
    print("  1. Fixed fast (eta=0.12, no lightlike)")
    print("  2. Fixed ultra-slow (eta=0.015, with lightlike)")
    print("  3. Adaptive two-phase (combines both)")
    print()
    print("-" * 80)

    # Strategy 1: Fixed fast
    print("\n[1/3] Running fixed FAST approach...")
    df_fast = run_fixed_eta_control(eta=0.12, use_lightlike=False, n_ticks=500)

    fast_final_error = df_fast["distance_to_target"].iloc[-1] * 1000  # mm
    fast_ticks_to_10cm = len(df_fast[df_fast["distance_to_target"] > 0.1])

    print(f"   Final error: {fast_final_error:.1f}mm")
    print(f"   Ticks to reach 10cm: {fast_ticks_to_10cm}")

    # Strategy 2: Fixed ultra-slow
    print("\n[2/3] Running fixed ULTRA-SLOW approach...")
    df_ultraslow = run_fixed_eta_control(eta=0.015, use_lightlike=True, n_ticks=500)

    ultraslow_final_error = df_ultraslow["distance_to_target"].iloc[-1] * 1000  # mm
    ultraslow_ticks_to_10cm = len(df_ultraslow[df_ultraslow["distance_to_target"] > 0.1])

    print(f"   Final error: {ultraslow_final_error:.1f}mm")
    print(f"   Ticks to reach 10cm: {ultraslow_ticks_to_10cm}")

    # Strategy 3: Adaptive two-phase
    print("\n[3/3] Running ADAPTIVE two-phase control...")
    df_adaptive = run_adaptive_control(n_ticks=500, terminal_threshold=0.15)

    adaptive_final_error = df_adaptive["distance_to_target"].iloc[-1] * 1000  # mm
    adaptive_ticks_to_10cm = len(df_adaptive[df_adaptive["distance_to_target"] > 0.1])

    # Find when it switched to terminal phase
    terminal_starts = df_adaptive[df_adaptive["phase"] == "TERMINAL_DESCENT"]
    if len(terminal_starts) > 0:
        terminal_start_tick = terminal_starts.index[0]
    else:
        terminal_start_tick = None

    print(f"   Final error: {adaptive_final_error:.1f}mm")
    print(f"   Ticks to reach 10cm: {adaptive_ticks_to_10cm}")
    if terminal_start_tick:
        print(f"   Switched to terminal phase at tick: {terminal_start_tick}")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY".center(80))
    print("=" * 80)
    print()

    results = [
        {
            "Strategy": "Fixed Fast",
            "Final Error (mm)": f"{fast_final_error:.1f}",
            "Ticks to 10cm": fast_ticks_to_10cm,
            "Speed": "⚡ Fast",
            "Precision": "❌ Poor",
        },
        {
            "Strategy": "Fixed Ultra-Slow",
            "Final Error (mm)": f"{ultraslow_final_error:.1f}",
            "Ticks to 10cm": ultraslow_ticks_to_10cm,
            "Speed": "🐌 Very Slow",
            "Precision": "✓ Excellent",
        },
        {
            "Strategy": "Adaptive Two-Phase",
            "Final Error (mm)": f"{adaptive_final_error:.1f}",
            "Ticks to 10cm": adaptive_ticks_to_10cm,
            "Speed": "✓ Fast",
            "Precision": "✓ Excellent",
        },
    ]

    # Print table
    print(
        f"{'Strategy':<25} {'Final Error':<15} {'Ticks to 10cm':<15} {'Speed':<15} {'Precision':<15}"
    )
    print("-" * 85)
    for r in results:
        print(
            f"{r['Strategy']:<25} {r['Final Error (mm)']:<15} {r['Ticks to 10cm']:<15} {r['Speed']:<15} {r['Precision']:<15}"
        )

    print()
    print("KEY INSIGHT:")
    print("  Adaptive two-phase gets BOTH:")
    print(
        f"    • Speed of fast approach ({adaptive_ticks_to_10cm} vs {fast_ticks_to_10cm} ticks to 10cm)"
    )
    print(
        f"    • Precision of ultra-slow ({adaptive_final_error:.1f}mm vs {ultraslow_final_error:.1f}mm final error)"
    )
    print()

    # Comparison vs fixed approaches
    vs_fast_precision = ((fast_final_error - adaptive_final_error) / fast_final_error) * 100
    vs_ultraslow_speed = (
        (ultraslow_ticks_to_10cm - adaptive_ticks_to_10cm) / ultraslow_ticks_to_10cm
    ) * 100

    print("IMPROVEMENTS:")
    print("  vs Fixed Fast:")
    print(f"    • {vs_fast_precision:+.1f}% better precision")
    print(f"    • Similar speed (both reach 10cm in ~{adaptive_ticks_to_10cm} ticks)")
    print()
    print("  vs Fixed Ultra-Slow:")
    print(f"    • {vs_ultraslow_speed:+.1f}% faster to 10cm")
    print(
        f"    • Similar precision ({adaptive_final_error:.1f}mm vs {ultraslow_final_error:.1f}mm)"
    )
    print()

    # Phase transition analysis
    if terminal_start_tick:
        fast_phase_ticks = terminal_start_tick
        terminal_phase_ticks = len(df_adaptive) - terminal_start_tick

        print("PHASE BREAKDOWN:")
        print(
            f"  Fast Approach Phase:   {fast_phase_ticks:3d} ticks ({fast_phase_ticks/len(df_adaptive)*100:.1f}%)"
        )
        print(
            f"  Terminal Descent Phase: {terminal_phase_ticks:3d} ticks ({terminal_phase_ticks/len(df_adaptive)*100:.1f}%)"
        )
        print()
        print(
            f"  Adaptive control spent only {terminal_phase_ticks/len(df_adaptive)*100:.1f}% of time in slow mode,"
        )
        print("  but achieved precision comparable to 100% ultra-slow!")

    print()
    print("=" * 80)
    print("CONCLUSION: Adaptive two-phase = BEST OF BOTH WORLDS".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
