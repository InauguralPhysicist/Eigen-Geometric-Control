"""
Benchmark: Moving Target Tracking with vs without Lorentz Boost

This benchmark demonstrates that predictive tracking (using estimated velocity)
significantly outperforms naive tracking for moving targets.

Comparison:
- Naive: Just move toward current target position (no velocity estimation)
- Predictive (Lorentz): Estimate velocity and move toward predicted position
"""

import numpy as np
import matplotlib.pyplot as plt

from src.eigen_lorentz import (
    moving_target_control_step,
    estimate_target_velocity,
)


def simulate_moving_target_tracking(
    initial_robot: np.ndarray,
    target_trajectory: list,
    method: str = "predictive",
    eta: float = 0.3,
    lookahead_time: float = 1.0,
) -> tuple:
    """
    Simulate robot tracking a moving target.

    Args:
        initial_robot: Starting robot position
        target_trajectory: List of target positions over time
        method: "predictive" or "naive"
        eta: Control gain
        lookahead_time: How far ahead to predict (for predictive method)

    Returns:
        - robot_trajectory: Robot positions over time
        - tracking_errors: Distance to target at each step
    """
    robot = initial_robot.copy()
    robot_trajectory = [robot.copy()]
    tracking_errors = []

    for i, target in enumerate(target_trajectory):
        # Get target history for velocity estimation
        history_start = max(0, i - 5)
        history = target_trajectory[history_start : i + 1]

        if method == "predictive":
            # Use predictive tracking
            robot, velocity, beta = moving_target_control_step(
                robot,
                target,
                history,
                eta=eta,
                lookahead_time=lookahead_time,
            )
        elif method == "naive":
            # Naive approach: just move toward current target
            direction = target - robot
            robot = robot + eta * direction
        else:
            raise ValueError(f"Unknown method: {method}")

        robot_trajectory.append(robot.copy())

        # Compute tracking error
        error = np.linalg.norm(robot - target)
        tracking_errors.append(error)

    return np.array(robot_trajectory), np.array(tracking_errors)


def generate_moving_target_trajectory(
    n_steps: int,
    trajectory_type: str = "linear",
    initial_position: np.ndarray = np.array([0.0, 0.0]),
    velocity: float = 0.5,
) -> list:
    """
    Generate different types of moving target trajectories.

    Args:
        n_steps: Number of time steps
        trajectory_type: "linear", "circular", "sinusoidal", or "accelerating"
        initial_position: Starting position
        velocity: Speed parameter

    Returns:
        List of target positions
    """
    trajectory = []

    for t in range(n_steps):
        if trajectory_type == "linear":
            # Constant velocity, straight line
            pos = initial_position + np.array([velocity * t, velocity * t * 0.5])

        elif trajectory_type == "circular":
            # Circular motion
            angle = velocity * t * 0.2
            radius = 5.0
            pos = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        elif trajectory_type == "sinusoidal":
            # Sinusoidal path
            pos = np.array([velocity * t, 3.0 * np.sin(velocity * t * 0.3)])

        elif trajectory_type == "accelerating":
            # Accelerating target
            accel = 0.05
            pos = initial_position + np.array(
                [
                    velocity * t + 0.5 * accel * t**2,
                    velocity * t * 0.3,
                ]
            )

        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")

        trajectory.append(pos)

    return trajectory


def run_benchmark():
    """
    Run comprehensive benchmark comparing predictive vs naive tracking.
    """
    print("=" * 70)
    print("MOVING TARGET TRACKING BENCHMARK")
    print("=" * 70)
    print()
    print("Comparing predictive (Lorentz boost) vs naive (current position) tracking")
    print()

    trajectory_types = [
        ("linear", "Linear Motion (Constant Velocity)"),
        ("circular", "Circular Motion"),
        ("sinusoidal", "Sinusoidal Path"),
        ("accelerating", "Accelerating Target"),
    ]

    n_steps = 100
    initial_robot = np.array([10.0, 10.0])

    all_results = {}

    for traj_type, traj_name in trajectory_types:
        print(f"Testing: {traj_name}")
        print("-" * 40)

        # Generate target trajectory
        target_traj = generate_moving_target_trajectory(
            n_steps, trajectory_type=traj_type, velocity=0.5
        )

        # Naive tracking
        robot_naive, errors_naive = simulate_moving_target_tracking(
            initial_robot, target_traj, method="naive", eta=0.3
        )

        # Predictive tracking
        robot_pred, errors_pred = simulate_moving_target_tracking(
            initial_robot, target_traj, method="predictive", eta=0.3, lookahead_time=1.0
        )

        # Statistics
        mean_error_naive = np.mean(errors_naive)
        mean_error_pred = np.mean(errors_pred)
        final_error_naive = errors_naive[-1]
        final_error_pred = errors_pred[-1]

        improvement = (mean_error_naive - mean_error_pred) / mean_error_naive * 100

        all_results[traj_type] = {
            "name": traj_name,
            "target_traj": target_traj,
            "robot_naive": robot_naive,
            "robot_pred": robot_pred,
            "errors_naive": errors_naive,
            "errors_pred": errors_pred,
            "mean_error_naive": mean_error_naive,
            "mean_error_pred": mean_error_pred,
            "improvement": improvement,
        }

        print(f"  Naive Mean Error:      {mean_error_naive:.3f}")
        print(f"  Predictive Mean Error: {mean_error_pred:.3f}")
        print(f"  Improvement:           {improvement:+.1f}%")
        print(f"  Final Error (Naive):   {final_error_naive:.3f}")
        print(f"  Final Error (Pred):    {final_error_pred:.3f}")
        print()

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    avg_improvement = np.mean([r["improvement"] for r in all_results.values()])

    print(f"Average Improvement: {avg_improvement:+.1f}%")
    print()

    for traj_type, result in all_results.items():
        print(f"{result['name']:30s} {result['improvement']:+6.1f}%")

    print()

    if avg_improvement > 20:
        print("✓✓✓ PREDICTIVE TRACKING WINS!")
        print("    Velocity estimation + lookahead → significantly better tracking")
    elif avg_improvement > 10:
        print("✓ PREDICTIVE TRACKING WINS")
        print("  Velocity-based prediction provides measurable benefit")
    else:
        print("~ COMPARABLE PERFORMANCE")
        print("  Benefits depend on trajectory type")

    print()
    print("=" * 70)

    # Visualization
    plot_results(all_results)

    return all_results


def plot_results(results: dict):
    """
    Visualize tracking performance.
    """
    n_scenarios = len(results)
    fig, axes = plt.subplots(2, n_scenarios, figsize=(5 * n_scenarios, 10))

    if n_scenarios == 1:
        axes = axes.reshape(-1, 1)

    for idx, (traj_type, data) in enumerate(results.items()):
        target_traj = np.array(data["target_traj"])
        robot_naive = data["robot_naive"]
        robot_pred = data["robot_pred"]
        errors_naive = data["errors_naive"]
        errors_pred = data["errors_pred"]

        # Top row: Trajectories
        ax = axes[0, idx]
        ax.set_title(data["name"], fontsize=12, fontweight="bold")
        ax.plot(
            target_traj[:, 0],
            target_traj[:, 1],
            "k--",
            label="Target",
            linewidth=2,
            alpha=0.7,
        )
        ax.plot(
            robot_naive[:, 0],
            robot_naive[:, 1],
            "r-",
            label="Naive",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.plot(
            robot_pred[:, 0],
            robot_pred[:, 1],
            "b-",
            label="Predictive",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.scatter(
            target_traj[0, 0],
            target_traj[0, 1],
            s=150,
            marker="*",
            color="green",
            label="Start",
            zorder=5,
        )
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Bottom row: Tracking Error Over Time
        ax = axes[1, idx]
        ax.set_title(f"Tracking Error ({data['improvement']:+.1f}% improvement)", fontsize=11)
        ax.plot(errors_naive, "r-", label="Naive", linewidth=1.5, alpha=0.7)
        ax.plot(errors_pred, "b-", label="Predictive", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Tracking Error (distance)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Add mean error lines
        ax.axhline(
            data["mean_error_naive"],
            color="r",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
            label=f"Mean (Naive): {data['mean_error_naive']:.2f}",
        )
        ax.axhline(
            data["mean_error_pred"],
            color="b",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
            label=f"Mean (Pred): {data['mean_error_pred']:.2f}",
        )

    plt.tight_layout()
    plt.savefig("benchmark_moving_targets.png", dpi=150, bbox_inches="tight")
    print("Plot saved: benchmark_moving_targets.png")


if __name__ == "__main__":
    np.random.seed(42)
    results = run_benchmark()
