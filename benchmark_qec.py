"""
Benchmark: Quantum-Inspired Error Correction for Sensor Robustness

This benchmark demonstrates that redundant encoding with parity checks
enables robust control despite sensor failures and measurement errors.

Comparison:
- Naive: Use sensor average (no error detection/correction)
- QEC: Redundant sensors + parity checks + error correction

Test scenarios:
- Low sensor failure rate (10%)
- Medium sensor failure rate (30%)
- High sensor failure rate (50%)
"""

import numpy as np
import matplotlib.pyplot as plt

from src.eigen_qec import (
    simulate_sensor_array,
    correct_measurement_errors,
    qec_control_step,
)


def simulate_navigation_with_sensors(
    start: np.ndarray,
    goal: np.ndarray,
    n_steps: int = 100,
    use_qec: bool = True,
    n_sensors: int = 3,
    failure_rate: float = 0.2,
    eta: float = 0.15,
) -> tuple:
    """
    Simulate robot navigation with redundant sensors.

    Args:
        start: Starting position
        goal: Goal position
        n_steps: Number of control steps
        use_qec: Whether to use quantum error correction
        n_sensors: Number of redundant sensors
        failure_rate: Probability of sensor failure per step
        eta: Learning rate / step size

    Returns:
        - trajectory: Path taken
        - errors_detected: Number of errors detected
        - final_error: Final distance to goal
    """
    position = start.copy()
    trajectory = [position.copy()]
    errors_detected = 0

    for step in range(n_steps):
        # Simulate sensor measurements with potential failures
        measurements = simulate_sensor_array(
            position,
            n_sensors=n_sensors,
            noise_level=0.05,
            failure_rate=failure_rate,
            failure_magnitude=3.0,
        )

        # Control step
        if use_qec:
            # Quantum error correction
            new_pos, info = qec_control_step(
                position,
                goal,
                measurements,
                eta=eta,
                error_threshold=0.5,
            )

            if info["error_detected"]:
                errors_detected += 1
        else:
            # Naive: average all measurements (no error correction)
            avg_measurement = np.mean(measurements, axis=0)

            # Blend with current state
            blend = 0.5
            updated_state = blend * avg_measurement + (1 - blend) * position

            # Move toward goal
            direction = goal - updated_state
            new_pos = updated_state + eta * direction

        # Check if reached goal
        if np.linalg.norm(new_pos - goal) < 0.1:
            position = goal.copy()
            trajectory.append(position.copy())
            break

        position = new_pos
        trajectory.append(position.copy())

    # Compute final error
    final_error = np.linalg.norm(position - goal)

    return np.array(trajectory), errors_detected, final_error


def run_benchmark():
    """
    Run comprehensive benchmark across sensor failure scenarios.
    """
    print("=" * 70)
    print("QUANTUM-INSPIRED ERROR CORRECTION BENCHMARK")
    print("=" * 70)
    print()
    print("Testing QEC vs naive averaging with sensor failures")
    print()

    failure_rates = [0.1, 0.3, 0.5]  # Low, medium, high
    n_trials = 20
    n_steps = 100
    n_sensors = 3
    start = np.array([10.0, 10.0])
    goal = np.array([0.0, 0.0])

    all_results = {}

    for failure_rate in failure_rates:
        print(f"Sensor Failure Rate: {failure_rate * 100:.0f}%")
        print("-" * 40)

        # Run trials
        final_errors_naive = []
        final_errors_qec = []
        errors_detected_list = []

        for trial in range(n_trials):
            # Naive (no QEC)
            np.random.seed(trial)
            traj_naive, _, err_naive = simulate_navigation_with_sensors(
                start,
                goal,
                n_steps=n_steps,
                use_qec=False,
                n_sensors=n_sensors,
                failure_rate=failure_rate,
                eta=0.15,
            )
            final_errors_naive.append(err_naive)

            # QEC
            np.random.seed(trial)  # Same random failures
            traj_qec, errors_detected, err_qec = simulate_navigation_with_sensors(
                start,
                goal,
                n_steps=n_steps,
                use_qec=True,
                n_sensors=n_sensors,
                failure_rate=failure_rate,
                eta=0.15,
            )
            final_errors_qec.append(err_qec)
            errors_detected_list.append(errors_detected)

        # Compute metrics
        avg_error_naive = np.mean(final_errors_naive)
        avg_error_qec = np.mean(final_errors_qec)
        avg_errors_detected = np.mean(errors_detected_list)

        # Success rates (within 10% of initial distance)
        initial_dist = np.linalg.norm(start - goal)
        threshold = initial_dist * 0.1
        success_naive = np.mean([e < threshold for e in final_errors_naive]) * 100
        success_qec = np.mean([e < threshold for e in final_errors_qec]) * 100

        # Error improvement
        error_improvement = (avg_error_naive - avg_error_qec) / avg_error_naive * 100
        success_improvement = success_qec - success_naive

        results = {
            "failure_rate": failure_rate,
            "avg_error_naive": avg_error_naive,
            "avg_error_qec": avg_error_qec,
            "error_improvement": error_improvement,
            "success_naive": success_naive,
            "success_qec": success_qec,
            "success_improvement": success_improvement,
            "avg_errors_detected": avg_errors_detected,
            "sample_traj_naive": traj_naive,
            "sample_traj_qec": traj_qec,
        }

        all_results[failure_rate] = results

        print(f"  Final Error:")
        print(f"    Naive: {avg_error_naive:.3f}")
        print(f"    QEC:   {avg_error_qec:.3f}")
        print(f"    Improvement: {error_improvement:+.1f}%")
        print()
        print(f"  Success Rate (within {threshold:.1f}):")
        print(f"    Naive: {success_naive:.1f}%")
        print(f"    QEC:   {success_qec:.1f}%")
        print(f"    Improvement: {success_improvement:+.1f}%")
        print()
        print(f"  Errors Detected: {avg_errors_detected:.1f} / {n_steps}")
        print()

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    avg_error_improvement = np.mean([r["error_improvement"] for r in all_results.values()])
    avg_success_improvement = np.mean(
        [r["success_improvement"] for r in all_results.values()]
    )

    print(f"Average Error Improvement:   {avg_error_improvement:+.1f}%")
    print(f"Average Success Improvement: {avg_success_improvement:+.1f}%")
    print()

    print("Breakdown by Failure Rate:")
    for rate, result in all_results.items():
        print(
            f"  {rate * 100:.0f}% failures: "
            f"Error {result['error_improvement']:+.1f}%, "
            f"Success {result['success_improvement']:+.1f}%"
        )

    print()

    if avg_error_improvement > 30 and avg_success_improvement > 20:
        print("✓✓✓ QEC WINS!")
        print("    Error correction >> naive averaging for sensor robustness")
    elif avg_error_improvement > 15 or avg_success_improvement > 10:
        print("✓ QEC WINS")
        print("  Redundant encoding provides measurable benefits")
    else:
        print("~ COMPARABLE PERFORMANCE")
        print("  Benefits depend on failure rate")

    print()
    print("=" * 70)

    # Visualization
    plot_results(all_results, start, goal)

    return all_results


def plot_results(results: dict, start: np.ndarray, goal: np.ndarray):
    """
    Visualize QEC performance.
    """
    n_scenarios = len(results)
    fig, axes = plt.subplots(2, n_scenarios, figsize=(6 * n_scenarios, 12))

    if n_scenarios == 1:
        axes = axes.reshape(-1, 1)

    for idx, (failure_rate, data) in enumerate(results.items()):
        traj_naive = data["sample_traj_naive"]
        traj_qec = data["sample_traj_qec"]

        # Top row: Trajectories
        ax = axes[0, idx]
        ax.set_title(
            f"Sensor Failure Rate: {failure_rate * 100:.0f}%",
            fontsize=12,
            fontweight="bold",
        )

        # Plot trajectories
        ax.plot(
            traj_naive[:, 0],
            traj_naive[:, 1],
            "r-",
            label="Naive",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.plot(
            traj_qec[:, 0],
            traj_qec[:, 1],
            "b-",
            label="QEC",
            linewidth=1.5,
            alpha=0.7,
        )

        # Mark start and goal
        ax.scatter(
            start[0], start[1], s=150, marker="*", color="green", label="Start", zorder=5
        )
        ax.scatter(
            goal[0], goal[1], s=150, marker="X", color="red", label="Goal", zorder=5
        )

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Bottom row: Performance comparison
        ax = axes[1, idx]
        ax.set_title(
            f"Error: {data['error_improvement']:+.1f}%, Success: {data['success_improvement']:+.1f}%",
            fontsize=10,
        )

        metrics = ["Final\nError", "Success\nRate (%)"]
        naive_vals = [data["avg_error_naive"], data["success_naive"]]
        qec_vals = [data["avg_error_qec"], data["success_qec"]]

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width / 2, naive_vals, width, label="Naive", color="red", alpha=0.7)
        ax.bar(x + width / 2, qec_vals, width, label="QEC", color="blue", alpha=0.7)

        ax.set_ylabel("Value")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels
        for i, (nv, qv) in enumerate(zip(naive_vals, qec_vals)):
            ax.text(
                i - width / 2, nv, f"{nv:.1f}", ha="center", va="bottom", fontsize=8
            )
            ax.text(i + width / 2, qv, f"{qv:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("benchmark_qec.png", dpi=150, bbox_inches="tight")
    print("Plot saved: benchmark_qec.png")


if __name__ == "__main__":
    np.random.seed(42)
    results = run_benchmark()
