#!/usr/bin/env python3
"""
Visual Servoing with Lightlike Observer: Stereo-Guided Robot Control

Tests lightlike observer for visual servoing where:
- Stereo cameras provide noisy depth estimates
- Disparity oscillations affect control
- Temporal tracking requires stability
- Occlusions cause sudden changes

Visual servoing challenges perfect for lightlike observer:
1. Sensor noise → oscillating disparity estimates
2. Occlusion changes → sudden jumps in perceived position
3. Moving target tracking → temporal stability critical
4. Depth ambiguity → precision vs stability tradeoff

This tests whether lightlike observer can stabilize vision-based
control better than standard approaches.
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
import pandas as pd


def add_stereo_noise(target_pos, noise_level=0.01, occlusion_prob=0.0, tick=0):
    """Simulate noisy stereo vision measurements"""
    # Gaussian noise
    noise = np.random.normal(0, noise_level, 2)
    noisy_target = target_pos + noise

    # Random occlusion (sudden jump)
    if np.random.random() < occlusion_prob:
        # Occlusion causes large error
        occlusion_noise = np.random.normal(0, noise_level * 10, 2)
        noisy_target = target_pos + occlusion_noise

    return noisy_target


def run_visual_servoing_baseline(
    theta_init=(-1.4, 1.2),
    target_trajectory=None,
    n_ticks=200,
    eta=0.12,
    stereo_noise=0.015,
    occlusion_prob=0.02,
    Go=4.0,
    lam=0.02,
    L1=0.9,
    L2=0.9,
    seed=42,
):
    """Run visual servoing WITHOUT lightlike observer"""
    np.random.seed(seed)
    theta1, theta2 = theta_init
    obstacle_center = np.array([0.6, 0.1])
    obstacle_radius = 0.25

    rows = []

    for t in range(n_ticks):
        # Get true target position
        true_target = np.array(target_trajectory(t))

        # Simulate stereo vision measurement (noisy)
        measured_target = add_stereo_noise(
            true_target, noise_level=stereo_noise, occlusion_prob=occlusion_prob, tick=t
        )

        # Robot uses measured position for control
        x, y = forward_kinematics(theta1, theta2, L1, L2)

        ds2_total, components = compute_ds2(
            theta1, theta2, measured_target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        grad, grad_norm = compute_gradient(
            theta1, theta2, measured_target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        # Standard update
        delta = -eta * grad
        theta1_new = theta1 + delta[0]
        theta2_new = theta2 + delta[1]

        C, S, ds2_CS = compute_change_stability(delta, 1e-3)

        # Track both true and measured error
        true_distance = np.sqrt((x - true_target[0]) ** 2 + (y - true_target[1]) ** 2)
        measured_distance = np.sqrt((x - measured_target[0]) ** 2 + (y - measured_target[1]) ** 2)

        rows.append(
            {
                "tick": t,
                "x": x,
                "y": y,
                "true_target_x": true_target[0],
                "true_target_y": true_target[1],
                "measured_target_x": measured_target[0],
                "measured_target_y": measured_target[1],
                "true_distance": true_distance,
                "measured_distance": measured_distance,
                "vision_error": np.sqrt(
                    (true_target[0] - measured_target[0]) ** 2
                    + (true_target[1] - measured_target[1]) ** 2
                ),
                "grad_norm": grad_norm,
                "d_obs": components["d_obs"],
            }
        )

        theta1, theta2 = theta1_new, theta2_new

    return pd.DataFrame(rows)


def run_visual_servoing_lightlike(
    theta_init=(-1.4, 1.2),
    target_trajectory=None,
    n_ticks=200,
    eta=0.12,
    stereo_noise=0.015,
    occlusion_prob=0.02,
    Go=4.0,
    lam=0.02,
    L1=0.9,
    L2=0.9,
    window=4,
    threshold=0.90,
    damping_scale=0.75,
    seed=42,
):
    """Run visual servoing WITH lightlike observer"""
    np.random.seed(seed)
    theta1, theta2 = theta_init
    obstacle_center = np.array([0.6, 0.1])
    obstacle_radius = 0.25
    state_history = []

    rows = []

    for t in range(n_ticks):
        # Get true target position
        true_target = np.array(target_trajectory(t))

        # Simulate stereo vision measurement (noisy)
        measured_target = add_stereo_noise(
            true_target, noise_level=stereo_noise, occlusion_prob=occlusion_prob, tick=t
        )

        # Robot state
        x, y = forward_kinematics(theta1, theta2, L1, L2)
        state = np.array([theta1, theta2])
        state_history.append(state)

        ds2_total, components = compute_ds2(
            theta1, theta2, measured_target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        grad, grad_norm = compute_gradient(
            theta1, theta2, measured_target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        # LIGHTLIKE OBSERVER (stabilizes noisy vision)
        oscillating = False
        damping = 0.0
        osc_strength = 0.0

        if len(state_history) >= window:
            oscillating, osc_strength = detect_oscillation(
                state_history, window=window, threshold=threshold
            )
            if oscillating:
                damping = lightlike_damping_factor(osc_strength) * damping_scale

        # Damped update
        delta = -(1.0 - damping) * eta * grad
        theta1_new = theta1 + delta[0]
        theta2_new = theta2 + delta[1]

        C, S, ds2_CS = compute_change_stability(delta, 1e-3)

        # Track both true and measured error
        true_distance = np.sqrt((x - true_target[0]) ** 2 + (y - true_target[1]) ** 2)
        measured_distance = np.sqrt((x - measured_target[0]) ** 2 + (y - measured_target[1]) ** 2)

        rows.append(
            {
                "tick": t,
                "x": x,
                "y": y,
                "true_target_x": true_target[0],
                "true_target_y": true_target[1],
                "measured_target_x": measured_target[0],
                "measured_target_y": measured_target[1],
                "true_distance": true_distance,
                "measured_distance": measured_distance,
                "vision_error": np.sqrt(
                    (true_target[0] - measured_target[0]) ** 2
                    + (true_target[1] - measured_target[1]) ** 2
                ),
                "grad_norm": grad_norm,
                "d_obs": components["d_obs"],
                "oscillating": oscillating,
                "osc_strength": osc_strength,
                "damping": damping,
            }
        )

        theta1, theta2 = theta1_new, theta2_new

    return pd.DataFrame(rows)


def analyze_visual_servoing(df, scenario_name):
    """Analyze visual servoing performance"""
    # Final true accuracy (what matters)
    final_true_error = df["true_distance"].iloc[-1] * 1000  # mm
    mean_true_error = df["true_distance"].mean() * 1000

    # Vision noise impact
    mean_vision_error = df["vision_error"].mean() * 1000
    max_vision_error = df["vision_error"].max() * 1000

    # Stability (tracking smoothness)
    tracking_variance = np.std(df["true_distance"].values) * 1000

    # Oscillations
    grad_norms = df["grad_norm"].values
    oscillations = sum(
        1 for i in range(10, len(grad_norms)) if grad_norms[i] > grad_norms[i - 1] * 1.5
    )

    metrics = {
        "scenario": scenario_name,
        "final_true_error_mm": final_true_error,
        "mean_true_error_mm": mean_true_error,
        "tracking_variance_mm": tracking_variance,
        "mean_vision_error_mm": mean_vision_error,
        "max_vision_error_mm": max_vision_error,
        "oscillations": oscillations,
    }

    if "damping" in df.columns:
        metrics["damping_activations"] = (df["damping"] > 0).sum()
        metrics["max_damping"] = df["damping"].max()

    return metrics


def main():
    print("=" * 80)
    print("VISUAL SERVOING: Stereo Vision + Lightlike Observer".center(80))
    print("=" * 80)
    print()
    print("Testing lightlike observer for stereo-guided robot control with:")
    print("  • Noisy depth estimates from stereo vision")
    print("  • Occlusions causing measurement jumps")
    print("  • Moving target tracking")
    print()

    results = []

    # Scenario 1: Static target with stereo noise
    print("Scenario 1: Static target with stereo noise...")

    def static_target(t):
        return [1.2, 0.3]

    df_base_1 = run_visual_servoing_baseline(
        target_trajectory=static_target, n_ticks=180, stereo_noise=0.015, occlusion_prob=0.0
    )
    df_light_1 = run_visual_servoing_lightlike(
        target_trajectory=static_target, n_ticks=180, stereo_noise=0.015, occlusion_prob=0.0
    )

    results.append(
        {
            "baseline": analyze_visual_servoing(df_base_1, "Stereo Noise - Baseline"),
            "lightlike": analyze_visual_servoing(df_light_1, "Stereo Noise - Lightlike"),
        }
    )

    # Scenario 2: Static target with occlusions
    print("Scenario 2: Static target with occlusions...")
    df_base_2 = run_visual_servoing_baseline(
        target_trajectory=static_target,
        n_ticks=180,
        stereo_noise=0.015,
        occlusion_prob=0.05,  # 5% occlusion rate
    )
    df_light_2 = run_visual_servoing_lightlike(
        target_trajectory=static_target, n_ticks=180, stereo_noise=0.015, occlusion_prob=0.05
    )

    results.append(
        {
            "baseline": analyze_visual_servoing(df_base_2, "With Occlusions - Baseline"),
            "lightlike": analyze_visual_servoing(df_light_2, "With Occlusions - Lightlike"),
        }
    )

    # Scenario 3: Moving target with stereo noise
    print("Scenario 3: Moving target tracking with stereo noise...")

    def moving_target(t):
        return [1.1 + 0.002 * t, 0.3 + 0.001 * t]

    df_base_3 = run_visual_servoing_baseline(
        target_trajectory=moving_target,
        n_ticks=180,
        stereo_noise=0.020,  # Higher noise
        occlusion_prob=0.0,
    )
    df_light_3 = run_visual_servoing_lightlike(
        target_trajectory=moving_target, n_ticks=180, stereo_noise=0.020, occlusion_prob=0.0
    )

    results.append(
        {
            "baseline": analyze_visual_servoing(df_base_3, "Moving Target - Baseline"),
            "lightlike": analyze_visual_servoing(df_light_3, "Moving Target - Lightlike"),
        }
    )

    # Scenario 4: Oscillating target (periodic motion)
    print("Scenario 4: Oscillating target with high noise...")

    def oscillating_target(t):
        return [1.2 + 0.08 * np.sin(t * 0.2), 0.35 + 0.06 * np.cos(t * 0.15)]

    df_base_4 = run_visual_servoing_baseline(
        target_trajectory=oscillating_target,
        n_ticks=180,
        stereo_noise=0.025,  # Very noisy
        occlusion_prob=0.03,
    )
    df_light_4 = run_visual_servoing_lightlike(
        target_trajectory=oscillating_target, n_ticks=180, stereo_noise=0.025, occlusion_prob=0.03
    )

    results.append(
        {
            "baseline": analyze_visual_servoing(df_base_4, "Oscillating + Noise - Baseline"),
            "lightlike": analyze_visual_servoing(df_light_4, "Oscillating + Noise - Lightlike"),
        }
    )

    # Display results
    print()
    print("=" * 80)
    print("RESULTS: Vision-Based Control Performance".center(80))
    print("=" * 80)
    print()

    print("Final True Tracking Error (mm) - ACTUAL PERFORMANCE:")
    print("-" * 80)
    for r in results:
        base = r["baseline"]
        light = r["lightlike"]
        improvement = (
            (base["final_true_error_mm"] - light["final_true_error_mm"])
            / base["final_true_error_mm"]
            * 100
        )
        arrow = "✓" if improvement > 0 else "✗"

        scenario = base["scenario"].replace(" - Baseline", "")
        print(
            f"{scenario:<35} {base['final_true_error_mm']:>7.1f}mm → {light['final_true_error_mm']:>7.1f}mm  "
            f"{arrow} {abs(improvement):>5.1f}%"
        )

    print()
    print("Tracking Stability (std dev in mm):")
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
            f"{scenario:<35} {base['tracking_variance_mm']:>7.1f}mm → {light['tracking_variance_mm']:>7.1f}mm  "
            f"{arrow} {abs(improvement):>5.1f}%"
        )

    print()
    print("Control Oscillations (induced by vision noise):")
    print("-" * 80)
    for r in results:
        base = r["baseline"]
        light = r["lightlike"]
        reduction = base["oscillations"] - light["oscillations"]
        status = "BETTER" if reduction > 0 else "SAME" if reduction == 0 else "WORSE"

        scenario = base["scenario"].replace(" - Baseline", "")
        print(
            f"{scenario:<35} {base['oscillations']:>4} → {light['oscillations']:>4} events  "
            f"{status:>6} ({reduction:+d})"
        )

    print()
    print("=" * 80)
    print("ANALYSIS".center(80))
    print("=" * 80)
    print()

    # Calculate improvements
    avg_accuracy_improvement = sum(
        (r["baseline"]["final_true_error_mm"] - r["lightlike"]["final_true_error_mm"])
        / r["baseline"]["final_true_error_mm"]
        * 100
        for r in results
    ) / len(results)

    avg_stability_improvement = sum(
        (r["baseline"]["tracking_variance_mm"] - r["lightlike"]["tracking_variance_mm"])
        / r["baseline"]["tracking_variance_mm"]
        * 100
        for r in results
    ) / len(results)

    total_oscillation_reduction = sum(
        r["baseline"]["oscillations"] - r["lightlike"]["oscillations"] for r in results
    )

    print(f"Average tracking accuracy improvement: {avg_accuracy_improvement:+.1f}%")
    print(f"Average stability improvement: {avg_stability_improvement:+.1f}%")
    print(f"Total oscillation reduction: {total_oscillation_reduction:+d} events")
    print()

    if avg_accuracy_improvement > 5.0 or total_oscillation_reduction > 10:
        print("★★★ SIGNIFICANT BENEFIT FOR VISUAL SERVOING ★★★")
        print()
        print("The lightlike observer EXCELS at stabilizing vision-guided control:")
        print(f"  • Tracking accuracy: {avg_accuracy_improvement:+.1f}% better")
        print(f"  • Motion stability: {avg_stability_improvement:+.1f}% better")
        print(f"  • Oscillations reduced by {total_oscillation_reduction} events")
        print()
        print("Stereo vision noise naturally creates oscillations in control.")
        print("The lightlike observer dampens these, leading to:")
        print("  • Smoother trajectories despite noisy sensors")
        print("  • Better handling of occlusions")
        print("  • More robust tracking of moving targets")
        print()
        print("RECOMMENDATION: Deploy for all stereo vision-based robot control,")
        print("                especially in unstructured environments.")
    elif avg_accuracy_improvement > 2.0:
        print("★ MODERATE BENEFIT FOR VISUAL SERVOING")
        print()
        print(f"Accuracy improvement: {avg_accuracy_improvement:+.1f}%")
        print(f"Oscillation reduction: {total_oscillation_reduction:+d} events")
        print()
        print("Benefits are noticeable, especially for noisy/occluded scenarios.")
    else:
        print("MINIMAL IMPROVEMENT")
        print()
        print("Vision noise may not be the dominant factor in these scenarios.")

    print()
    print("=" * 80)

    # Best scenario
    if results:
        best_idx = max(
            range(len(results)),
            key=lambda i: (
                results[i]["baseline"]["final_true_error_mm"]
                - results[i]["lightlike"]["final_true_error_mm"]
            ),
        )
        best = results[best_idx]
        best_improvement = (
            (best["baseline"]["final_true_error_mm"] - best["lightlike"]["final_true_error_mm"])
            / best["baseline"]["final_true_error_mm"]
            * 100
        )

        print()
        print(f"BEST PERFORMANCE: {best['baseline']['scenario'].replace(' - Baseline', '')}")
        print(f"  Accuracy improvement: {best_improvement:+.1f}%")
        print(
            f"  Oscillations reduced: {best['baseline']['oscillations'] - best['lightlike']['oscillations']} events"
        )

        if "damping_activations" in best["lightlike"]:
            print(f"  Observer activations: {best['lightlike']['damping_activations']}")
            print(f"  Max damping: {best['lightlike']['max_damping']:.3f}")


if __name__ == "__main__":
    main()
