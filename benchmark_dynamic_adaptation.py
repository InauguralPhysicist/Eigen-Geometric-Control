"""
Benchmark: Dynamic Invariant Detection for Auto-Adaptation

This benchmark demonstrates that auto-detected parameters enable robust
control across different robot types and scales without manual tuning.

Comparison:
- Adaptive: Auto-detects c, threshold, tau, eta from trajectory
- Fixed: Uses one-size-fits-all default parameters

Test scenarios:
- Slow, heavy robot (low velocities, large mass)
- Fast, agile robot (high velocities, quick response)
- Medium robot (baseline)
"""

import numpy as np
import matplotlib.pyplot as plt

from src.eigen_adaptive import adaptive_control_step


def simulate_robot_control(
    robot_type: str,
    initial_state: np.ndarray,
    target: np.ndarray,
    n_steps: int = 100,
    use_adaptive: bool = True,
) -> tuple:
    """
    Simulate robot control with or without auto-adaptation.

    Args:
        robot_type: "slow", "medium", or "fast"
        initial_state: Starting position
        target: Target position
        n_steps: Number of control steps
        use_adaptive: Whether to use auto-adaptive parameters

    Returns:
        - trajectory: State trajectory
        - detected_params: Auto-detected parameters (if adaptive)
        - convergence_time: Steps to reach target
        - final_error: Final distance to target
    """
    # Robot-specific dynamics
    if robot_type == "slow":
        # Heavy robot: slow movements, needs small eta
        velocity_scale = 0.1
        noise_level = 0.05
        optimal_eta = 0.05
    elif robot_type == "fast":
        # Agile robot: fast movements, can handle large eta
        velocity_scale = 2.0
        noise_level = 0.2
        optimal_eta = 0.3
    else:  # medium
        # Baseline robot
        velocity_scale = 0.5
        noise_level = 0.1
        optimal_eta = 0.1

    state = initial_state.copy()
    trajectory = [state.copy()]
    detected_params_history = []

    for step in range(n_steps):
        # Move toward target with robot-specific dynamics
        direction = target - state
        distance = np.linalg.norm(direction)

        if distance < 0.01:
            # Already at target
            trajectory.append(state.copy())
            continue

        # Simulate robot dynamics (velocity-dependent movement)
        if use_adaptive:
            # Use adaptive control with auto-tuning
            new_state, info = adaptive_control_step(
                state,
                target,
                trajectory,
                auto_tune=True,
                min_samples_for_tuning=10,
            )
            detected_params_history.append(
                {
                    "c": info["c"],
                    "eta_recommended": info.get("eta_recommended", 0.1),
                    "confidence": info["confidence"],
                }
            )
        else:
            # Use fixed default parameters (no adaptation)
            new_state, info = adaptive_control_step(
                state,
                target,
                trajectory,
                auto_tune=False,
                eta=0.1,  # Fixed default
            )

        # Apply robot-specific velocity scaling
        movement = new_state - state
        scaled_movement = movement * velocity_scale

        # Add small noise
        noise = np.random.randn(*state.shape) * noise_level
        state = state + scaled_movement + noise

        trajectory.append(state.copy())

    # Compute metrics
    trajectory = np.array(trajectory)

    # Convergence time (steps to get within 5% of initial distance)
    initial_distance = np.linalg.norm(initial_state - target)
    threshold = initial_distance * 0.05
    convergence_time = n_steps
    for i, pos in enumerate(trajectory):
        if np.linalg.norm(pos - target) < threshold:
            convergence_time = i
            break

    # Final error
    final_error = np.linalg.norm(trajectory[-1] - target)

    # Average detected parameters (if adaptive)
    avg_params = None
    if use_adaptive and detected_params_history:
        # Use parameters after initial learning period
        learning_period = min(15, len(detected_params_history) // 2)
        stable_params = detected_params_history[learning_period:]
        if stable_params:
            avg_params = {
                "c": np.mean([p["c"] for p in stable_params]),
                "eta_recommended": np.mean(
                    [p["eta_recommended"] for p in stable_params]
                ),
                "confidence": np.mean([p["confidence"] for p in stable_params]),
            }

    return trajectory, avg_params, convergence_time, final_error


def run_benchmark():
    """
    Run comprehensive benchmark across robot types.
    """
    print("=" * 70)
    print("DYNAMIC INVARIANT DETECTION BENCHMARK")
    print("=" * 70)
    print()
    print("Testing auto-adaptation across different robot types")
    print()

    robot_types = ["slow", "medium", "fast"]
    n_trials = 20
    n_steps = 150
    initial_state = np.array([10.0, 10.0])
    target = np.array([0.0, 0.0])

    all_results = {}

    for robot_type in robot_types:
        print(f"Testing: {robot_type.upper()} Robot")
        print("-" * 40)

        # Run trials
        conv_times_adaptive = []
        conv_times_fixed = []
        final_errors_adaptive = []
        final_errors_fixed = []
        detected_params_list = []

        for trial in range(n_trials):
            # Same random seed for fair comparison
            np.random.seed(trial)

            # Adaptive control
            traj_adapt, params, conv_adapt, err_adapt = simulate_robot_control(
                robot_type,
                initial_state,
                target,
                n_steps=n_steps,
                use_adaptive=True,
            )
            conv_times_adaptive.append(conv_adapt)
            final_errors_adaptive.append(err_adapt)
            if params:
                detected_params_list.append(params)

            # Reset random seed
            np.random.seed(trial)

            # Fixed parameters
            traj_fixed, _, conv_fixed, err_fixed = simulate_robot_control(
                robot_type,
                initial_state,
                target,
                n_steps=n_steps,
                use_adaptive=False,
            )
            conv_times_fixed.append(conv_fixed)
            final_errors_fixed.append(err_fixed)

        # Average metrics
        avg_conv_adaptive = np.mean(conv_times_adaptive)
        avg_conv_fixed = np.mean(conv_times_fixed)
        avg_err_adaptive = np.mean(final_errors_adaptive)
        avg_err_fixed = np.mean(final_errors_fixed)

        # Compute improvements
        conv_improvement = (
            (avg_conv_fixed - avg_conv_adaptive) / avg_conv_fixed * 100
        )
        error_improvement = (avg_err_fixed - avg_err_adaptive) / avg_err_fixed * 100

        # Average detected parameters
        if detected_params_list:
            avg_c = np.mean([p["c"] for p in detected_params_list])
            avg_eta = np.mean([p["eta_recommended"] for p in detected_params_list])
            avg_confidence = np.mean([p["confidence"] for p in detected_params_list])
        else:
            avg_c = 1.0
            avg_eta = 0.1
            avg_confidence = 0.0

        results = {
            "robot_type": robot_type,
            "conv_time_adaptive": avg_conv_adaptive,
            "conv_time_fixed": avg_conv_fixed,
            "conv_improvement": conv_improvement,
            "error_adaptive": avg_err_adaptive,
            "error_fixed": avg_err_fixed,
            "error_improvement": error_improvement,
            "avg_c_detected": avg_c,
            "avg_eta_detected": avg_eta,
            "avg_confidence": avg_confidence,
            "sample_traj_adaptive": traj_adapt,
            "sample_traj_fixed": traj_fixed,
        }

        all_results[robot_type] = results

        print(f"  Convergence Time:")
        print(f"    Adaptive:  {avg_conv_adaptive:.1f} steps")
        print(f"    Fixed:     {avg_conv_fixed:.1f} steps")
        print(f"    Improvement: {conv_improvement:+.1f}%")
        print()
        print(f"  Final Error:")
        print(f"    Adaptive:  {avg_err_adaptive:.4f}")
        print(f"    Fixed:     {avg_err_fixed:.4f}")
        print(f"    Improvement: {error_improvement:+.1f}%")
        print()
        print(f"  Auto-Detected Parameters:")
        print(f"    c (characteristic velocity): {avg_c:.3f}")
        print(f"    eta (recommended):           {avg_eta:.3f}")
        print(f"    confidence:                  {avg_confidence:.3f}")
        print()

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    avg_conv_improvement = np.mean(
        [r["conv_improvement"] for r in all_results.values()]
    )
    avg_error_improvement = np.mean(
        [r["error_improvement"] for r in all_results.values()]
    )

    print(f"Average Convergence Improvement: {avg_conv_improvement:+.1f}%")
    print(f"Average Error Improvement:       {avg_error_improvement:+.1f}%")
    print()

    print("Breakdown by Robot Type:")
    for robot_type, result in all_results.items():
        print(
            f"  {robot_type.capitalize()}: Conv {result['conv_improvement']:+.1f}%, "
            f"Error {result['error_improvement']:+.1f}%"
        )

    print()

    # Demonstrate parameter adaptation
    print("Parameter Adaptation:")
    for robot_type, result in all_results.items():
        print(
            f"  {robot_type.capitalize()}: c={result['avg_c_detected']:.3f}, "
            f"eta={result['avg_eta_detected']:.3f}"
        )

    print()

    if avg_conv_improvement > 20 and avg_error_improvement > 20:
        print("✓✓✓ AUTO-ADAPTATION WINS!")
        print("    Dynamic parameter detection >> fixed parameters")
    elif avg_conv_improvement > 10 or avg_error_improvement > 10:
        print("✓ AUTO-ADAPTATION WINS")
        print("  Automatic tuning provides measurable benefits")
    else:
        print("~ COMPARABLE PERFORMANCE")
        print("  Benefits vary by robot type")

    print()
    print("=" * 70)

    # Visualization
    plot_results(all_results, target)

    return all_results


def plot_results(results: dict, target: np.ndarray):
    """
    Visualize adaptation performance across robot types.
    """
    n_types = len(results)
    fig, axes = plt.subplots(2, n_types, figsize=(5 * n_types, 10))

    if n_types == 1:
        axes = axes.reshape(-1, 1)

    for idx, (robot_type, data) in enumerate(results.items()):
        traj_adapt = data["sample_traj_adaptive"]
        traj_fixed = data["sample_traj_fixed"]

        # Top row: Trajectories
        ax = axes[0, idx]
        ax.set_title(
            f"{robot_type.capitalize()} Robot\n(c={data['avg_c_detected']:.2f}, eta={data['avg_eta_detected']:.2f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.plot(
            traj_fixed[:, 0],
            traj_fixed[:, 1],
            "r-",
            label="Fixed Params",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.plot(
            traj_adapt[:, 0],
            traj_adapt[:, 1],
            "b-",
            label="Adaptive",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.scatter(
            traj_adapt[0, 0],
            traj_adapt[0, 1],
            s=150,
            marker="*",
            color="green",
            label="Start",
            zorder=5,
        )
        ax.scatter(
            target[0],
            target[1],
            s=150,
            marker="X",
            color="red",
            label="Target",
            zorder=5,
        )
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Bottom row: Performance comparison
        ax = axes[1, idx]
        ax.set_title(
            f"Conv: {data['conv_improvement']:+.1f}%, Err: {data['error_improvement']:+.1f}%",
            fontsize=10,
        )

        metrics = ["Convergence\nTime", "Final\nError"]
        adaptive_vals = [data["conv_time_adaptive"], data["error_adaptive"]]
        fixed_vals = [data["conv_time_fixed"], data["error_fixed"]]

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(
            x - width / 2, fixed_vals, width, label="Fixed", color="red", alpha=0.7
        )
        ax.bar(
            x + width / 2,
            adaptive_vals,
            width,
            label="Adaptive",
            color="blue",
            alpha=0.7,
        )

        ax.set_ylabel("Value")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels
        for i, (fv, av) in enumerate(zip(fixed_vals, adaptive_vals)):
            ax.text(
                i - width / 2, fv, f"{fv:.2f}", ha="center", va="bottom", fontsize=8
            )
            ax.text(
                i + width / 2, av, f"{av:.2f}", ha="center", va="bottom", fontsize=8
            )

    plt.tight_layout()
    plt.savefig("benchmark_dynamic_adaptation.png", dpi=150, bbox_inches="tight")
    print("Plot saved: benchmark_dynamic_adaptation.png")


if __name__ == "__main__":
    np.random.seed(42)
    results = run_benchmark()
