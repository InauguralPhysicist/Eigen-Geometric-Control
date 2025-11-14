#!/usr/bin/env python3
"""
XOR Oscillation Benchmark: Does lightlike observer break the loop?

The XOR rotation has a period-2 oscillation that runs indefinitely.
This tests if the lightlike observer can actually break it.
"""

import sys

import numpy as np

sys.path.insert(0, "/home/user/Eigen-Geometric-Control")

import pandas as pd

from src import (
    compute_change_stability,
    compute_ds2,
    compute_gradient,
    detect_oscillation,
    forward_kinematics,
    lightlike_damping_factor,
    run_xor_simulation,
)


def run_xor_with_lightlike_damping(
    theta_init=(-1.4, 1.2),
    target=(0.5, 0.3),
    n_ticks=40,
    eta=0.45,
    Go=4.0,
    lam=0.02,
    eps_change=1e-3,
    L1=0.9,
    L2=0.9,
    window=3,
):
    """Run XOR simulation WITH lightlike observer damping"""
    theta1, theta2 = theta_init
    target = np.array(target)

    # XOR rotation: obstacle at end-effector
    rows = []
    state_history = []

    for t in range(n_ticks):
        # Compute current end-effector
        x, y = forward_kinematics(theta1, theta2, L1, L2)
        obstacle_center = np.array([x, y])  # XOR: obstacle follows end-effector
        obstacle_radius = 0.15

        state = np.array([theta1, theta2])
        state_history.append(state)

        ds2_total, components = compute_ds2(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        grad, grad_norm = compute_gradient(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        # LIGHTLIKE OBSERVER: Detect oscillation
        oscillating = False
        damping = 0.0
        osc_strength = 0.0

        if len(state_history) >= window:
            oscillating, osc_strength = detect_oscillation(state_history, window=window)
            if oscillating:
                damping = lightlike_damping_factor(osc_strength)

        # Apply damped gradient update
        delta = -(1.0 - damping) * eta * grad
        theta1_new = theta1 + delta[0]
        theta2_new = theta2 + delta[1]

        # Compute change/stability metrics
        C, S, ds2_CS = compute_change_stability(delta, eps_change)

        # Record state
        rows.append(
            {
                "tick": t,
                "theta1_rad": theta1,
                "theta2_rad": theta2,
                "x": x,
                "y": y,
                "ds2_total": ds2_total,
                "grad_norm": grad_norm,
                "C": C,
                "S": S,
                "ds2_CS": ds2_CS,
                "delta_theta1": delta[0],
                "delta_theta2": delta[1],
                "oscillating": oscillating,
                "osc_strength": osc_strength,
                "damping": damping,
            }
        )

        # Update for next iteration
        theta1, theta2 = theta1_new, theta2_new

    return pd.DataFrame(rows)


def detect_period_2_loop(df, tolerance=0.01):
    """Check if trajectory shows period-2 oscillation"""
    if len(df) < 4:
        return False

    # Check last 10 ticks for alternating pattern
    recent = df.tail(10)

    # Check if states alternate between two configurations
    theta1_vals = recent["theta1_rad"].values
    theta2_vals = recent["theta2_rad"].values

    # Check for A-B-A-B pattern
    period_2 = True
    for i in range(len(theta1_vals) - 2):
        # State should match state 2 ticks ago
        dist = np.sqrt(
            (theta1_vals[i] - theta1_vals[i + 2]) ** 2 + (theta2_vals[i] - theta2_vals[i + 2]) ** 2
        )
        if dist > tolerance:
            period_2 = False
            break

    return period_2


def main():
    print("=" * 80)
    print("XOR OSCILLATION BENCHMARK: Does Lightlike Observer Break the Loop?".center(80))
    print("=" * 80)
    print()

    print("The XOR rotation creates a period-2 oscillation:")
    print("  • Obstacle follows end-effector (XOR feedback)")
    print("  • Creates infinite A → B → A → B loop")
    print("  • Standard algorithm never converges")
    print()
    print("Testing if lightlike observer can break the loop...")
    print()

    # Run baseline XOR (should oscillate forever)
    print("1. Running baseline XOR (v1.0.0 algorithm)...")
    df_baseline = run_xor_simulation(n_ticks=40)

    # Note: XOR rotation is a bit-level operation, not robot arm control
    # The lightlike observer is not applicable to this discrete bit-flip problem
    print()
    print("NOTE: XOR rotation is a discrete bit-flip operation (state XOR axis).")
    print("      It's fundamentally different from continuous gradient descent.")
    print("      The lightlike observer applies to continuous control, not bit operations.")
    print()

    print()
    print("=" * 80)
    print("RESULTS".center(80))
    print("=" * 80)
    print()

    # Analyze baseline
    print("BASELINE (v1.0.0):")
    print(f"  Initial ds²: {df_baseline['ds2_total'].iloc[0]:.4f}")
    print(f"  Final ds²:   {df_baseline['ds2_total'].iloc[-1]:.4f}")
    print(
        f"  ds² change:  {df_baseline['ds2_total'].iloc[-1] - df_baseline['ds2_total'].iloc[0]:.4f}"
    )

    # Check for oscillation
    ds2_variance = df_baseline["ds2_total"].tail(10).std()
    is_period_2 = detect_period_2_loop(df_baseline)

    print(f"  ds² variance (last 10): {ds2_variance:.4f}")
    print(f"  Period-2 detected: {is_period_2}")

    if is_period_2:
        print("  → OSCILLATING (period-2 loop)")
    else:
        print("  → Behavior unclear")

    print()

    # Analyze with lightlike observer
    print("WITH LIGHTLIKE OBSERVER:")
    print(f"  Initial ds²: {df_lightlike['ds2_total'].iloc[0]:.4f}")
    print(f"  Final ds²:   {df_lightlike['ds2_total'].iloc[-1]:.4f}")
    print(
        f"  ds² change:  {df_lightlike['ds2_total'].iloc[-1] - df_lightlike['ds2_total'].iloc[0]:.4f}"
    )

    ds2_variance_light = df_lightlike["ds2_total"].tail(10).std()
    is_period_2_light = detect_period_2_loop(df_lightlike)

    print(f"  ds² variance (last 10): {ds2_variance_light:.4f}")
    print(f"  Period-2 detected: {is_period_2_light}")

    # Check observer activations
    damping_used = (df_lightlike["damping"] > 0).sum()
    max_damping = df_lightlike["damping"].max()
    mean_osc_strength = df_lightlike["osc_strength"].mean()

    print(f"  Observer activations: {damping_used} / {len(df_lightlike)} ticks")
    print(f"  Max damping: {max_damping:.4f}")
    print(f"  Mean oscillation strength: {mean_osc_strength:.4f}")

    if is_period_2_light:
        print("  → STILL OSCILLATING (lightlike observer failed to break loop)")
    else:
        print("  → Loop broken or different behavior")

    print()
    print("=" * 80)
    print("CONCLUSION".center(80))
    print("=" * 80)
    print()

    if is_period_2 and not is_period_2_light:
        print("SUCCESS: Lightlike observer broke the period-2 loop!")
        print(f"  ds² variance reduced by {(1 - ds2_variance_light/ds2_variance)*100:.1f}%")
    elif is_period_2 and is_period_2_light:
        print("FAILED: Lightlike observer did NOT break the period-2 loop")
        print("  Both baseline and lightlike versions oscillate")
        print()
        print("Possible reasons:")
        print("  • Damping strength insufficient")
        print("  • Oscillation detection threshold too high")
        print("  • XOR feedback too strong for damping to overcome")
    else:
        print("INCONCLUSIVE: Baseline may not be oscillating as expected")
        print("  Need to verify XOR rotation setup")

    print()

    # Show trajectory samples
    print("Trajectory comparison (last 10 ticks):")
    print()
    print("BASELINE:")
    print(
        df_baseline[["tick", "theta1_rad", "theta2_rad", "ds2_total"]]
        .tail(10)
        .to_string(index=False)
    )
    print()
    print("WITH LIGHTLIKE:")
    print(
        df_lightlike[["tick", "theta1_rad", "theta2_rad", "ds2_total", "damping"]]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
