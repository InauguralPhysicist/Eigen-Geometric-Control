#!/usr/bin/env python3
"""
Mars Landing Scenarios: Precision Planetary Landing

Tests lightlike observer for Mars/planetary landing missions where:
- Atmospheric turbulence causes disturbances
- Sensor noise from dust/communication delays
- Terrain hazard avoidance requires mid-descent adjustments
- Precision landing in safe zones is critical
- No retry - oscillations during approach = mission failure

Mars EDL (Entry, Descent, Landing) phases:
1. Powered descent - initial approach
2. Hazard avoidance - terrain adjustment
3. Final descent - precision landing
4. Skycrane hover - touchdown (for Curiosity/Perseverance style)

Key differences from Earth landing:
- Thinner atmosphere → less predictable turbulence
- Communication delays → must be autonomous
- Dust → sensor noise
- Rocky terrain → precision critical
- One shot → stability > speed
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


def run_mars_baseline(
    initial_position=(-2.0, 1.8),  # High altitude start
    landing_trajectory=None,
    obstacle_trajectory=None,  # Rocks, hazards
    n_ticks=250,
    eta=0.12,
    Go=6.0,  # Strong hazard avoidance
    lam=0.02,
    L1=0.9,
    L2=0.9,
):
    """Run Mars landing baseline (standard gradient descent)"""
    theta1, theta2 = initial_position

    rows = []

    for t in range(n_ticks):
        target = np.array(landing_trajectory(t))
        obstacle_center = np.array(obstacle_trajectory(t))
        obstacle_radius = 0.25

        x, y = forward_kinematics(theta1, theta2, L1, L2)

        ds2_total, components = compute_ds2(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        grad, grad_norm = compute_gradient(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        # Standard gradient update
        delta = -eta * grad
        theta1_new = theta1 + delta[0]
        theta2_new = theta2 + delta[1]

        C, S, ds2_CS = compute_change_stability(delta, 1e-3)

        rows.append(
            {
                "tick": t,
                "x": x,
                "y": y,
                "target_x": target[0],
                "target_y": target[1],
                "distance_to_target": np.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2),
                "ds2_total": ds2_total,
                "grad_norm": grad_norm,
                "d_obs": components["d_obs"],
            }
        )

        theta1, theta2 = theta1_new, theta2_new

    return pd.DataFrame(rows)


def run_mars_lightlike(
    initial_position=(-2.0, 1.8),
    landing_trajectory=None,
    obstacle_trajectory=None,
    n_ticks=250,
    eta=0.12,
    Go=6.0,
    lam=0.02,
    L1=0.9,
    L2=0.9,
    window=4,
    threshold=0.93,  # More sensitive for Mars (can't tolerate oscillations)
    damping_scale=0.9,  # High damping for safety
):
    """Run Mars landing WITH lightlike observer (adaptive stability)"""
    theta1, theta2 = initial_position
    state_history = []

    rows = []

    for t in range(n_ticks):
        target = np.array(landing_trajectory(t))
        obstacle_center = np.array(obstacle_trajectory(t))
        obstacle_radius = 0.25

        x, y = forward_kinematics(theta1, theta2, L1, L2)
        state = np.array([theta1, theta2])
        state_history.append(state)

        ds2_total, components = compute_ds2(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        grad, grad_norm = compute_gradient(
            theta1, theta2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        # LIGHTLIKE OBSERVER (Mars-tuned)
        oscillating = False
        damping = 0.0
        osc_strength = 0.0

        if len(state_history) >= window:
            oscillating, osc_strength = detect_oscillation(
                state_history, window=window, threshold=threshold
            )
            if oscillating:
                damping = lightlike_damping_factor(osc_strength) * damping_scale

        delta = -(1.0 - damping) * eta * grad
        theta1_new = theta1 + delta[0]
        theta2_new = theta2 + delta[1]

        C, S, ds2_CS = compute_change_stability(delta, 1e-3)

        rows.append(
            {
                "tick": t,
                "x": x,
                "y": y,
                "target_x": target[0],
                "target_y": target[1],
                "distance_to_target": np.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2),
                "ds2_total": ds2_total,
                "grad_norm": grad_norm,
                "d_obs": components["d_obs"],
                "oscillating": oscillating,
                "osc_strength": osc_strength,
                "damping": damping,
            }
        )

        theta1, theta2 = theta1_new, theta2_new

    return pd.DataFrame(rows)


def analyze_mars_landing(df, scenario_name, safe_zone_radius=0.015):
    """Analyze Mars landing with mission-critical metrics"""
    distances = df["distance_to_target"].values

    # Final landing precision (mission critical)
    final_error = distances[-1] * 1000  # mm

    # Mission success (within safe zone?)
    mission_success = final_error < (safe_zone_radius * 1000)

    # Stability during descent (last 30%)
    final_descent_start = int(len(df) * 0.7)
    final_descent_distances = distances[final_descent_start:]
    descent_stability = np.std(final_descent_distances) * 1000

    # Oscillation count during final descent (CRITICAL for Mars)
    grad_norms = df["grad_norm"].values[final_descent_start:]
    descent_oscillations = 0
    for i in range(1, len(grad_norms)):
        if grad_norms[i] > grad_norms[i - 1] * 1.5:  # Significant gradient reversal
            descent_oscillations += 1

    # Hazard clearance (minimum distance to obstacles)
    min_clearance = df["d_obs"].min()
    hazard_violations = (df["d_obs"] < 0.30).sum()

    # Time to safe zone
    within_safe_zone = distances < safe_zone_radius
    time_to_safe = np.argmax(within_safe_zone) if within_safe_zone.any() else len(df)

    # Overall trajectory smoothness
    mean_distance = distances.mean() * 1000

    metrics = {
        "scenario": scenario_name,
        "final_landing_error_mm": final_error,
        "mission_success": mission_success,
        "safe_zone_radius_mm": safe_zone_radius * 1000,
        "descent_stability_mm": descent_stability,
        "descent_oscillations": descent_oscillations,
        "min_hazard_clearance_m": min_clearance,
        "hazard_violations": hazard_violations,
        "time_to_safe_zone": time_to_safe,
        "mean_tracking_error_mm": mean_distance,
    }

    if "damping" in df.columns:
        final_damping = df["damping"].iloc[final_descent_start:]
        metrics["descent_damping_activations"] = (final_damping > 0).sum()
        metrics["max_damping"] = df["damping"].max()

    return metrics


def main():
    print("=" * 80)
    print("MARS LANDING: Mission-Critical Precision Landing".center(80))
    print("=" * 80)
    print()
    print("Simulating Mars EDL (Entry, Descent, Landing) scenarios where:")
    print("  • Atmospheric turbulence causes disturbances")
    print("  • Sensor noise from dust/communication")
    print("  • Terrain hazards require avoidance")
    print("  • Precision landing is mission-critical")
    print("  • No retry - oscillations = mission failure")
    print()

    results = []

    # Scenario 1: Nominal Mars landing (like Perseverance)
    print("Scenario 1: Nominal Mars landing (clear weather)...")

    def nominal_landing(t):
        # Slow descent to landing zone
        return [1.35, 0.42 - 0.0005 * t]

    def boulder_field(t):
        return [0.75, 0.25]  # Large rock formation

    df_base_1 = run_mars_baseline(
        landing_trajectory=nominal_landing, obstacle_trajectory=boulder_field, n_ticks=200
    )
    df_light_1 = run_mars_lightlike(
        landing_trajectory=nominal_landing, obstacle_trajectory=boulder_field, n_ticks=200
    )

    results.append(
        {
            "baseline": analyze_mars_landing(df_base_1, "Nominal - Baseline"),
            "lightlike": analyze_mars_landing(df_light_1, "Nominal - Lightlike"),
        }
    )

    # Scenario 2: Atmospheric turbulence (dust storm effects)
    print("Scenario 2: Turbulent descent (dust storm effects)...")
    np.random.seed(42)
    turb_x = np.random.normal(0, 0.012, 200)
    turb_y = np.random.normal(0, 0.010, 200)

    def turbulent_descent(t):
        base_x, base_y = 1.32, 0.38 - 0.0005 * t
        return [base_x + turb_x[min(t, 199)], base_y + turb_y[min(t, 199)]]

    df_base_2 = run_mars_baseline(
        landing_trajectory=turbulent_descent, obstacle_trajectory=boulder_field, n_ticks=200
    )
    df_light_2 = run_mars_lightlike(
        landing_trajectory=turbulent_descent, obstacle_trajectory=boulder_field, n_ticks=200
    )

    results.append(
        {
            "baseline": analyze_mars_landing(df_base_2, "Turbulent - Baseline"),
            "lightlike": analyze_mars_landing(df_light_2, "Turbulent - Lightlike"),
        }
    )

    # Scenario 3: Mid-descent hazard avoidance (detected obstacle)
    print("Scenario 3: Hazard avoidance maneuver...")

    def hazard_avoidance(t):
        # Detect hazard at t=80, shift landing site
        if t < 80:
            return [1.35, 0.42 - 0.0005 * t]
        else:
            # Shift 10cm to avoid detected boulder
            return [1.45, 0.42 - 0.0005 * t]

    def moving_hazard(t):
        # Hazard in original landing zone
        return [1.35, 0.25]

    df_base_3 = run_mars_baseline(
        landing_trajectory=hazard_avoidance,
        obstacle_trajectory=moving_hazard,
        n_ticks=200,
        Go=8.0,  # Strong avoidance
    )
    df_light_3 = run_mars_lightlike(
        landing_trajectory=hazard_avoidance, obstacle_trajectory=moving_hazard, n_ticks=200, Go=8.0
    )

    results.append(
        {
            "baseline": analyze_mars_landing(df_base_3, "Hazard Avoidance - Baseline"),
            "lightlike": analyze_mars_landing(df_light_3, "Hazard Avoidance - Lightlike"),
        }
    )

    # Scenario 4: Precision landing near hazard (tight constraints)
    print("Scenario 4: Precision landing near hazard...")

    def precision_site(t):
        return [1.30, 0.38]  # Fixed, small safe zone

    def nearby_hazard(t):
        return [1.15, 0.35]  # Close to landing site

    df_base_4 = run_mars_baseline(
        landing_trajectory=precision_site,
        obstacle_trajectory=nearby_hazard,
        n_ticks=200,
        eta=0.10,  # Careful approach
        Go=7.0,
    )
    df_light_4 = run_mars_lightlike(
        landing_trajectory=precision_site,
        obstacle_trajectory=nearby_hazard,
        n_ticks=200,
        eta=0.10,
        Go=7.0,
    )

    results.append(
        {
            "baseline": analyze_mars_landing(
                df_base_4, "Precision Near Hazard - Baseline", safe_zone_radius=0.010
            ),
            "lightlike": analyze_mars_landing(
                df_light_4, "Precision Near Hazard - Lightlike", safe_zone_radius=0.010
            ),
        }
    )

    # Scenario 5: Sample return landing (must land on exact spot)
    print("Scenario 5: Sample return rendezvous (ultra-precision)...")

    def sample_return_site(t):
        # Oscillating slightly (Mars Ascent Vehicle preparing for launch)
        return [1.28 + 0.015 * np.sin(t * 0.1), 0.36 + 0.010 * np.cos(t * 0.15)]

    df_base_5 = run_mars_baseline(
        landing_trajectory=sample_return_site,
        obstacle_trajectory=boulder_field,
        n_ticks=200,
        eta=0.08,  # Very careful
    )
    df_light_5 = run_mars_lightlike(
        landing_trajectory=sample_return_site,
        obstacle_trajectory=boulder_field,
        n_ticks=200,
        eta=0.08,
    )

    results.append(
        {
            "baseline": analyze_mars_landing(
                df_base_5, "Sample Return - Baseline", safe_zone_radius=0.008
            ),
            "lightlike": analyze_mars_landing(
                df_light_5, "Sample Return - Lightlike", safe_zone_radius=0.008
            ),
        }
    )

    # Display results
    print()
    print("=" * 80)
    print("MISSION RESULTS: Mars Landing Performance".center(80))
    print("=" * 80)
    print()

    print("Final Landing Precision (mm) - MISSION CRITICAL:")
    print("-" * 80)
    for r in results:
        base = r["baseline"]
        light = r["lightlike"]
        improvement = (
            (base["final_landing_error_mm"] - light["final_landing_error_mm"])
            / base["final_landing_error_mm"]
            * 100
        )

        base_success = "✓ SUCCESS" if base["mission_success"] else "✗ FAIL"
        light_success = "✓ SUCCESS" if light["mission_success"] else "✗ FAIL"

        scenario = base["scenario"].replace(" - Baseline", "")
        safe_zone = base["safe_zone_radius_mm"]

        print(f"{scenario:<30}")
        print(
            f"  Baseline:  {base['final_landing_error_mm']:>6.1f}mm  {base_success}  (safe zone: <{safe_zone:.0f}mm)"
        )
        print(
            f"  Lightlike: {light['final_landing_error_mm']:>6.1f}mm  {light_success}  ({improvement:+.1f}%)"
        )

    print()
    print("Descent Stability (oscillations during final 30%):")
    print("-" * 80)
    for r in results:
        base = r["baseline"]
        light = r["lightlike"]
        reduction = base["descent_oscillations"] - light["descent_oscillations"]
        status = "SAFER" if reduction > 0 else "SAME" if reduction == 0 else "MORE RISK"

        scenario = base["scenario"].replace(" - Baseline", "")
        print(
            f"{scenario:<30} {base['descent_oscillations']:>3} → {light['descent_oscillations']:>3} oscillations  "
            f"{status:>10} ({reduction:+d})"
        )

    print()
    print("Descent Smoothness (std dev in mm):")
    print("-" * 80)
    for r in results:
        base = r["baseline"]
        light = r["lightlike"]
        improvement = (
            (base["descent_stability_mm"] - light["descent_stability_mm"])
            / base["descent_stability_mm"]
            * 100
        )
        arrow = "↓" if improvement > 0 else "↑"

        scenario = base["scenario"].replace(" - Baseline", "")
        print(
            f"{scenario:<30} {base['descent_stability_mm']:>7.1f}mm → {light['descent_stability_mm']:>7.1f}mm  "
            f"{arrow} {abs(improvement):>5.1f}%"
        )

    print()
    print("=" * 80)
    print("MISSION ANALYSIS".center(80))
    print("=" * 80)
    print()

    # Mission success rates
    baseline_successes = sum(1 for r in results if r["baseline"]["mission_success"])
    lightlike_successes = sum(1 for r in results if r["lightlike"]["mission_success"])

    print(f"MISSION SUCCESS RATE:")
    print(
        f"  Baseline:  {baseline_successes}/{len(results)} missions successful ({baseline_successes/len(results)*100:.0f}%)"
    )
    print(
        f"  Lightlike: {lightlike_successes}/{len(results)} missions successful ({lightlike_successes/len(results)*100:.0f}%)"
    )
    print()

    # Average metrics
    avg_precision_improvement = sum(
        (r["baseline"]["final_landing_error_mm"] - r["lightlike"]["final_landing_error_mm"])
        / r["baseline"]["final_landing_error_mm"]
        * 100
        for r in results
    ) / len(results)

    total_oscillation_reduction = sum(
        r["baseline"]["descent_oscillations"] - r["lightlike"]["descent_oscillations"]
        for r in results
    )

    avg_stability_improvement = sum(
        (r["baseline"]["descent_stability_mm"] - r["lightlike"]["descent_stability_mm"])
        / r["baseline"]["descent_stability_mm"]
        * 100
        for r in results
    ) / len(results)

    print(f"Average landing precision improvement: {avg_precision_improvement:+.1f}%")
    print(f"Total oscillation reduction: {total_oscillation_reduction:+d} events")
    print(f"Average descent stability improvement: {avg_stability_improvement:+.1f}%")
    print()

    if lightlike_successes > baseline_successes or total_oscillation_reduction > 5:
        print("★★★ MISSION-CRITICAL IMPROVEMENT ★★★")
        print()
        print(f"The lightlike observer SAVES MISSIONS:")
        print(f"  • {lightlike_successes - baseline_successes} additional successful landing(s)")
        print(f"  • {total_oscillation_reduction} fewer descent oscillations")
        print(f"  • {avg_precision_improvement:.1f}% better landing precision")
        print()
        print("For Mars/planetary missions where:")
        print("  • Atmospheric turbulence causes disturbances")
        print("  • Mid-descent hazard avoidance creates instability")
        print("  • Precision landing is mission-critical")
        print("  • No retry - one shot to succeed")
        print()
        print("The lightlike observer provides crucial stability and precision")
        print("that can make the difference between MISSION SUCCESS and FAILURE.")
        print()
        print("RECOMMENDATION: Deploy for all Mars precision landing missions.")
    elif avg_precision_improvement > 5.0:
        print("★ SIGNIFICANT BENEFIT FOR MARS LANDING")
        print()
        print(f"Precision improvement: {avg_precision_improvement:+.1f}%")
        print(f"Oscillation reduction: {total_oscillation_reduction:+d} events")
        print()
        print("Benefits are substantial for precision-critical scenarios.")
    else:
        print("MODERATE IMPROVEMENT")
        print()
        print("Benefits present but not dramatic for nominal scenarios.")
        print("Most useful for turbulent or hazard avoidance situations.")

    print()
    print("=" * 80)

    # Best scenario analysis
    if len(results) > 0:
        best_idx = max(
            range(len(results)),
            key=lambda i: (
                results[i]["baseline"]["final_landing_error_mm"]
                - results[i]["lightlike"]["final_landing_error_mm"]
            ),
        )
        best = results[best_idx]
        best_improvement = (
            (
                best["baseline"]["final_landing_error_mm"]
                - best["lightlike"]["final_landing_error_mm"]
            )
            / best["baseline"]["final_landing_error_mm"]
            * 100
        )

        print()
        print(f"BEST PERFORMANCE: {best['baseline']['scenario'].replace(' - Baseline', '')}")
        print(f"  Precision improvement: {best_improvement:+.1f}%")
        print(
            f"  Oscillations reduced: {best['baseline']['descent_oscillations'] - best['lightlike']['descent_oscillations']} events"
        )

        if "descent_damping_activations" in best["lightlike"]:
            print(
                f"  Damping activations during descent: {best['lightlike']['descent_damping_activations']}"
            )
            print(f"  Max damping: {best['lightlike']['max_damping']:.3f}")


if __name__ == "__main__":
    main()
