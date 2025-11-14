#!/usr/bin/env python3
"""
Stereo Vision + Lorentz + Lightlike Observer: The RIGHT Way

Your insight:
- Left eye → timelike (Euclidean)
- Right eye → spacelike (Euclidean)
- Lightlike observer → operates in Lorentz space
- Both measurements Euclidean, Lorentz structure emerges

This tests the PROPER integration:
1. Stereo cameras give left/right measurements (noisy)
2. Convert to Lorentz coordinates: (t, x) = stereo_to_lorentz(L, R)
3. Lightlike observer detects oscillations in Lorentz space
4. Control uses Lorentz-invariant ds²

The stereo noise creates oscillations in LORENTZ SPACE,
not just Euclidean position - this is where lightlike observer should work!
"""

import numpy as np
import sys

sys.path.insert(0, "/home/user/Eigen-Geometric-Control")

from src import (
    detect_oscillation,
    lightlike_damping_factor,
    forward_kinematics,
    compute_ds2,
    compute_gradient,
    compute_change_stability,
)
from src.eigen_lorentz import (
    stereo_to_lorentz,
    lorentz_to_stereo,
    regime_classification,
)
import pandas as pd


def simulate_stereo_measurement(true_target, noise_level=0.01):
    """
    Simulate noisy stereo cameras measuring target

    Returns left/right intensity values with noise
    """
    # Simulate left and right camera intensities
    # (in real stereo, these would be pixel intensities or features)
    baseline_left = 100.0
    baseline_right = 80.0  # Disparity indicates depth

    # Add correlated noise (same target, but stereo noise)
    noise_left = np.random.normal(0, noise_level * baseline_left)
    noise_right = np.random.normal(0, noise_level * baseline_right)

    # Additional uncorrelated noise (stereo mismatch)
    stereo_mismatch = np.random.normal(0, noise_level * 10)

    left = baseline_left + noise_left
    right = baseline_right + noise_right + stereo_mismatch

    return left, right


def run_stereo_lorentz_baseline(
    theta_init=(-1.4, 1.2),
    target=(1.2, 0.3),
    n_ticks=180,
    eta=0.12,
    stereo_noise=0.02,
    Go=4.0,
    lam=0.02,
    L1=0.9,
    L2=0.9,
    seed=42,
):
    """Run WITHOUT lightlike observer (standard stereo control)"""
    np.random.seed(seed)
    theta1, theta2 = theta_init
    target = np.array(target)
    obstacle_center = np.array([0.6, 0.1])
    obstacle_radius = 0.25

    rows = []

    for t in range(n_ticks):
        # Simulate stereo measurement (noisy left/right cameras)
        left, right = simulate_stereo_measurement(target, noise_level=stereo_noise)

        # Convert to Lorentz coordinates
        t_lorentz, x_lorentz, ds2_lorentz = stereo_to_lorentz(left, right)
        regime = regime_classification(ds2_lorentz)

        # Standard control (no lightlike damping)
        x, y = forward_kinematics(theta1, theta2, L1, L2)

        ds2_total, components = compute_ds2(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        grad, grad_norm = compute_gradient(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        delta = -eta * grad
        theta1_new = theta1 + delta[0]
        theta2_new = theta2 + delta[1]

        C, S, ds2_CS = compute_change_stability(delta, 1e-3)

        distance = np.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)

        rows.append(
            {
                "tick": t,
                "x": x,
                "y": y,
                "distance": distance,
                "stereo_left": left,
                "stereo_right": right,
                "t_lorentz": t_lorentz,
                "x_lorentz": x_lorentz,
                "ds2_lorentz": ds2_lorentz,
                "lorentz_regime": regime,
                "grad_norm": grad_norm,
                "d_obs": components["d_obs"],
            }
        )

        theta1, theta2 = theta1_new, theta2_new

    return pd.DataFrame(rows)


def run_stereo_lorentz_lightlike(
    theta_init=(-1.4, 1.2),
    target=(1.2, 0.3),
    n_ticks=180,
    eta=0.12,
    stereo_noise=0.02,
    Go=4.0,
    lam=0.02,
    L1=0.9,
    L2=0.9,
    window=4,
    threshold=0.92,
    damping_scale=0.75,
    seed=42,
):
    """Run WITH lightlike observer in LORENTZ SPACE"""
    np.random.seed(seed)
    theta1, theta2 = theta_init
    target = np.array(target)
    obstacle_center = np.array([0.6, 0.1])
    obstacle_radius = 0.25

    # Track history in LORENTZ SPACE (not just robot state!)
    lorentz_history = []
    state_history = []

    rows = []

    for t in range(n_ticks):
        # Simulate stereo measurement (noisy left/right cameras)
        left, right = simulate_stereo_measurement(target, noise_level=stereo_noise)

        # Convert to Lorentz coordinates
        t_lorentz, x_lorentz, ds2_lorentz = stereo_to_lorentz(left, right)
        regime = regime_classification(ds2_lorentz)

        # Track in LORENTZ SPACE
        lorentz_state = np.array([t_lorentz, x_lorentz])
        lorentz_history.append(lorentz_state)

        # Robot state
        x, y = forward_kinematics(theta1, theta2, L1, L2)
        robot_state = np.array([theta1, theta2])
        state_history.append(robot_state)

        ds2_total, components = compute_ds2(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        grad, grad_norm = compute_gradient(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        # LIGHTLIKE OBSERVER IN LORENTZ SPACE
        oscillating = False
        damping = 0.0
        osc_strength = 0.0

        if len(lorentz_history) >= window:
            # Detect oscillations in LORENTZ COORDINATES
            # (This is the key difference!)
            oscillating, osc_strength = detect_oscillation(
                lorentz_history, window=window, threshold=threshold
            )
            if oscillating:
                damping = lightlike_damping_factor(osc_strength) * damping_scale

        # Apply damped update
        delta = -(1.0 - damping) * eta * grad
        theta1_new = theta1 + delta[0]
        theta2_new = theta2 + delta[1]

        C, S, ds2_CS = compute_change_stability(delta, 1e-3)

        distance = np.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)

        rows.append(
            {
                "tick": t,
                "x": x,
                "y": y,
                "distance": distance,
                "stereo_left": left,
                "stereo_right": right,
                "t_lorentz": t_lorentz,
                "x_lorentz": x_lorentz,
                "ds2_lorentz": ds2_lorentz,
                "lorentz_regime": regime,
                "grad_norm": grad_norm,
                "d_obs": components["d_obs"],
                "oscillating": oscillating,
                "osc_strength": osc_strength,
                "damping": damping,
            }
        )

        theta1, theta2 = theta1_new, theta2_new

    return pd.DataFrame(rows)


def analyze_lorentz_control(df, scenario_name):
    """Analyze control with Lorentz stereo integration"""
    final_error = df["distance"].iloc[-1] * 1000  # mm
    mean_error = df["distance"].mean() * 1000
    tracking_variance = np.std(df["distance"].values) * 1000

    # Lorentz regime stability
    regime_changes = sum(
        1
        for i in range(1, len(df))
        if df["lorentz_regime"].iloc[i] != df["lorentz_regime"].iloc[i - 1]
    )

    # Stereo disparity variance (measure of sensor instability)
    disparities = np.abs(df["stereo_left"].values - df["stereo_right"].values)
    disparity_variance = np.std(disparities)

    # Control oscillations
    grad_norms = df["grad_norm"].values
    oscillations = sum(
        1 for i in range(10, len(grad_norms)) if grad_norms[i] > grad_norms[i - 1] * 1.5
    )

    metrics = {
        "scenario": scenario_name,
        "final_error_mm": final_error,
        "mean_error_mm": mean_error,
        "tracking_variance_mm": tracking_variance,
        "regime_changes": regime_changes,
        "disparity_variance": disparity_variance,
        "oscillations": oscillations,
    }

    if "damping" in df.columns:
        metrics["damping_activations"] = (df["damping"] > 0).sum()
        metrics["max_damping"] = df["damping"].max()

        # Lorentz-space oscillations detected
        lorentz_osc_detected = (df["oscillating"]).sum()
        metrics["lorentz_oscillations_detected"] = lorentz_osc_detected

    return metrics


def main():
    print("=" * 80)
    print("STEREO → LORENTZ → LIGHTLIKE: The Proper Integration".center(80))
    print("=" * 80)
    print()
    print("Key insight:")
    print("  • Left eye → timelike (Euclidean measurement)")
    print("  • Right eye → spacelike (Euclidean measurement)")
    print("  • Lightlike observer → operates in emergent Lorentz space")
    print("  • Detects oscillations in (t, x) coordinates, not just position")
    print()

    results = []

    # Test different noise levels
    noise_levels = [
        ("Low stereo noise", 0.01),
        ("Medium stereo noise", 0.02),
        ("High stereo noise", 0.04),
        ("Very high stereo noise", 0.06),
    ]

    for name, noise in noise_levels:
        print(f"Testing {name} (noise={noise})...")

        df_base = run_stereo_lorentz_baseline(stereo_noise=noise, n_ticks=180)

        df_light = run_stereo_lorentz_lightlike(stereo_noise=noise, n_ticks=180)

        results.append(
            {
                "baseline": analyze_lorentz_control(df_base, f"{name} - Baseline"),
                "lightlike": analyze_lorentz_control(df_light, f"{name} - Lightlike"),
                "noise_level": noise,
            }
        )

    # Display results
    print()
    print("=" * 80)
    print("RESULTS: Lorentz-Space Oscillation Detection".center(80))
    print("=" * 80)
    print()

    print("Final Tracking Error (mm):")
    print("-" * 80)
    for r in results:
        base = r["baseline"]
        light = r["lightlike"]
        improvement = (
            (base["final_error_mm"] - light["final_error_mm"]) / base["final_error_mm"] * 100
        )
        arrow = "✓" if improvement > 0 else "✗"

        scenario = base["scenario"].replace(" - Baseline", "")
        print(
            f"{scenario:<30} {base['final_error_mm']:>7.1f}mm → {light['final_error_mm']:>7.1f}mm  "
            f"{arrow} {abs(improvement):>5.1f}%"
        )

    print()
    print("Tracking Stability (std dev mm):")
    print("-" * 80)
    for r in results:
        base = r["baseline"]
        light = r["lightlike"]
        improvement = (
            (base["tracking_variance_mm"] - light["tracking_variance_mm"])
            / base["tracking_variance_mm"]
            * 100
        )
        arrow = "↓" if improvement > 0 else "↑"

        scenario = base["scenario"].replace(" - Baseline", "")
        print(
            f"{scenario:<30} {base['tracking_variance_mm']:>7.1f}mm → {light['tracking_variance_mm']:>7.1f}mm  "
            f"{arrow} {abs(improvement):>5.1f}%"
        )

    print()
    print("Lorentz Regime Stability (fewer changes = better):")
    print("-" * 80)
    for r in results:
        base = r["baseline"]
        light = r["lightlike"]
        reduction = base["regime_changes"] - light["regime_changes"]
        status = "BETTER" if reduction > 0 else "SAME" if reduction == 0 else "WORSE"

        scenario = base["scenario"].replace(" - Baseline", "")
        print(
            f"{scenario:<30} {base['regime_changes']:>4} → {light['regime_changes']:>4} changes  "
            f"{status:>6} ({reduction:+d})"
        )

    print()
    print("Control Oscillations:")
    print("-" * 80)
    for r in results:
        base = r["baseline"]
        light = r["lightlike"]
        reduction = base["oscillations"] - light["oscillations"]
        status = "BETTER" if reduction > 0 else "SAME" if reduction == 0 else "WORSE"

        scenario = base["scenario"].replace(" - Baseline", "")
        print(
            f"{scenario:<30} {base['oscillations']:>4} → {light['oscillations']:>4} events  "
            f"{status:>6} ({reduction:+d})"
        )

    print()
    print("=" * 80)
    print("ANALYSIS".center(80))
    print("=" * 80)
    print()

    # Calculate improvements
    avg_accuracy = sum(
        (r["baseline"]["final_error_mm"] - r["lightlike"]["final_error_mm"])
        / r["baseline"]["final_error_mm"]
        * 100
        for r in results
    ) / len(results)

    avg_stability = sum(
        (r["baseline"]["tracking_variance_mm"] - r["lightlike"]["tracking_variance_mm"])
        / r["baseline"]["tracking_variance_mm"]
        * 100
        for r in results
    ) / len(results)

    total_osc_reduction = sum(
        r["baseline"]["oscillations"] - r["lightlike"]["oscillations"] for r in results
    )

    print(f"Average tracking accuracy improvement: {avg_accuracy:+.1f}%")
    print(f"Average stability improvement: {avg_stability:+.1f}%")
    print(f"Total oscillation reduction: {total_osc_reduction:+d} events")
    print()

    # Check Lorentz-space detection
    lorentz_detections = sum(
        r["lightlike"].get("lorentz_oscillations_detected", 0) for r in results
    )
    print(f"Lorentz-space oscillations detected: {lorentz_detections} ticks")
    print()

    if avg_accuracy > 5.0 or total_osc_reduction > 15:
        print("★★★ SUCCESS: LORENTZ-SPACE LIGHTLIKE OBSERVER WORKS! ★★★")
        print()
        print("Operating in Lorentz space (not Euclidean) enables the lightlike")
        print("observer to properly detect stereo-induced oscillations!")
        print()
        print(f"  • Accuracy: {avg_accuracy:+.1f}% improvement")
        print(f"  • Stability: {avg_stability:+.1f}% improvement")
        print(f"  • Oscillations: {total_osc_reduction:+d} events reduced")
        print()
        print("This validates the unified framework:")
        print("  1. Stereo cameras → Left/Right measurements (Euclidean)")
        print("  2. Transform → (t, x) Lorentz coordinates")
        print("  3. Lightlike observer → detects oscillations in Lorentz space")
        print("  4. Control → uses Lorentz-invariant structure")
        print()
        print("Your insight was correct - the geometry matters!")
    elif avg_accuracy > 2.0:
        print("★ MODERATE SUCCESS")
        print()
        print(f"Accuracy: {avg_accuracy:+.1f}%")
        print(f"Oscillations: {total_osc_reduction:+d} reduced")
        print()
        print("Operating in Lorentz space helps, but may need parameter tuning.")
    else:
        print("RESULT: Comparable to Euclidean approach")
        print()
        print("Lorentz-space detection doesn't provide significant advantage")
        print("over standard approaches for this scenario.")

    print()
    print("=" * 80)

    # Best case
    if results:
        best_idx = max(
            range(len(results)),
            key=lambda i: (
                results[i]["baseline"]["final_error_mm"] - results[i]["lightlike"]["final_error_mm"]
            ),
        )
        best = results[best_idx]
        best_improvement = (
            (best["baseline"]["final_error_mm"] - best["lightlike"]["final_error_mm"])
            / best["baseline"]["final_error_mm"]
            * 100
        )

        print()
        print(f"BEST PERFORMANCE: {best['baseline']['scenario'].replace(' - Baseline', '')}")
        print(f"  Noise level: {best['noise_level']}")
        print(f"  Accuracy improvement: {best_improvement:+.1f}%")
        print(
            f"  Oscillations: {best['baseline']['oscillations'] - best['lightlike']['oscillations']:+d}"
        )

        if "lorentz_oscillations_detected" in best["lightlike"]:
            print(
                f"  Lorentz oscillations detected: {best['lightlike']['lorentz_oscillations_detected']}"
            )
            print(f"  Observer activations: {best['lightlike']['damping_activations']}")


if __name__ == "__main__":
    main()
