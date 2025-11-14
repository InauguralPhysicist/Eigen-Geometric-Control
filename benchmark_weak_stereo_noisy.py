"""
Benchmark: Weak vs Strong Measurement in High-Noise Stereo Vision

This benchmark uses REALISTIC NOISE levels to demonstrate where
weak measurements truly excel over strong measurements.

Key insight: Weak measurements preserve coherence even when
observations are very noisy, while strong measurements collapse
to the noisy observation and create jittery motion.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.eigen_weak_measurement import (
    coherence_metric,
    weak_stereo_control_step,
)


def simulate_noisy_stereo_control(
    initial_state: np.ndarray,
    target: np.ndarray,
    n_steps: int = 100,
    measurement_strength: float = 0.1,
    eta: float = 0.1,
    observation_noise: float = 0.3,  # HIGH NOISE
    disparity_noise: float = 0.1,  # Noisy depth estimates
) -> tuple:
    """
    Simulate stereo control with realistic sensor noise.

    In real robotics:
    - Camera observations are noisy (pixel jitter, lighting, motion blur)
    - Stereo disparity is especially noisy (matching errors)
    - Weak measurements should smooth this out

    Args:
        observation_noise: Standard deviation of observation noise
        disparity_noise: Standard deviation of disparity measurement noise
    """
    trajectory = [initial_state.copy()]
    current = initial_state.copy()
    measurements = []

    for step in range(n_steps):
        # Simulate noisy stereo observations
        # Left camera
        left_obs = current + np.random.randn(2) * observation_noise

        # Right camera (with noisy disparity)
        true_disparity = 0.05
        noisy_disparity = true_disparity + np.random.randn() * disparity_noise
        right_obs = current + np.random.randn(2) * observation_noise
        right_obs[0] -= noisy_disparity

        # Control step
        new_state, depth, measurements = weak_stereo_control_step(
            current,
            target,
            left_obs,
            right_obs,
            eta=eta,
            measurement_strength=measurement_strength,
            accumulated_measurements=measurements,
        )

        current = new_state
        trajectory.append(current.copy())

    # Evaluate
    coherence = coherence_metric(trajectory)
    final_error = np.linalg.norm(trajectory[-1] - target)

    # Measure smoothness (via velocity changes)
    velocities = [trajectory[i] - trajectory[i - 1] for i in range(1, len(trajectory))]
    accelerations = [velocities[i] - velocities[i - 1] for i in range(1, len(velocities))]
    jitter = np.mean([np.linalg.norm(a) for a in accelerations])

    return trajectory, coherence, final_error, jitter


def run_noisy_benchmark():
    """
    Benchmark with realistic sensor noise.
    """
    print("=" * 70)
    print("HIGH-NOISE STEREO VISION BENCHMARK")
    print("=" * 70)
    print()
    print("Scenario: Noisy camera observations + noisy depth estimates")
    print("Prediction: Weak measurements preserve coherence despite noise")
    print()

    # Setup
    initial = np.array([8.0, 8.0])
    target = np.array([0.0, 0.0])
    n_steps = 100
    n_trials = 20

    configs = [
        {"name": "Weak (0.05)", "strength": 0.05, "color": "blue"},
        {"name": "Weak (0.1)", "strength": 0.1, "color": "cyan"},
        {"name": "Medium (0.5)", "strength": 0.5, "color": "orange"},
        {"name": "Strong (0.9)", "strength": 0.9, "color": "red"},
    ]

    results = {}

    for config in configs:
        print(f"Testing: {config['name']}")
        print("-" * 40)

        coherences = []
        final_errors = []
        jitters = []
        trajectories = []

        for trial in range(n_trials):
            traj, coh, err, jit = simulate_noisy_stereo_control(
                initial, target, n_steps=n_steps, measurement_strength=config["strength"]
            )

            coherences.append(coh)
            final_errors.append(err)
            jitters.append(jit)

            if trial == 0:
                trajectories.append(traj)

        # Statistics
        results[config["name"]] = {
            "coherence_mean": np.mean(coherences),
            "coherence_std": np.std(coherences),
            "error_mean": np.mean(final_errors),
            "error_std": np.std(final_errors),
            "jitter_mean": np.mean(jitters),
            "jitter_std": np.std(jitters),
            "trajectory": trajectories[0],
            "color": config["color"],
        }

        print(
            f"  Coherence:    {results[config['name']]['coherence_mean']:.4f} ± {results[config['name']]['coherence_std']:.4f}"
        )
        print(
            f"  Final Error:  {results[config['name']]['error_mean']:.4f} ± {results[config['name']]['error_std']:.4f}"
        )
        print(
            f"  Jitter:       {results[config['name']]['jitter_mean']:.4f} ± {results[config['name']]['jitter_std']:.4f}"
        )
        print()

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    weakest = results["Weak (0.05)"]
    weak = results["Weak (0.1)"]
    strong = results["Strong (0.9)"]

    # Compare weakest vs strong
    coherence_improvement = (
        (weakest["coherence_mean"] - strong["coherence_mean"]) / strong["coherence_mean"] * 100
    )
    jitter_reduction = (
        (strong["jitter_mean"] - weakest["jitter_mean"]) / strong["jitter_mean"] * 100
    )
    error_improvement = (strong["error_mean"] - weakest["error_mean"]) / strong["error_mean"] * 100

    print(f"WEAK (0.05) vs STRONG (0.9):")
    print("-" * 40)
    print(f"Coherence:      {coherence_improvement:+.1f}% (higher is smoother)")
    print(f"  Weak:   {weakest['coherence_mean']:.4f}")
    print(f"  Strong: {strong['coherence_mean']:.4f}")
    print()
    print(f"Jitter:         {jitter_reduction:+.1f}% (reduction)")
    print(f"  Weak:   {weakest['jitter_mean']:.4f} (less jerky)")
    print(f"  Strong: {strong['jitter_mean']:.4f} (more jerky)")
    print()
    print(f"Final Error:    {error_improvement:+.1f}%")
    print(f"  Weak:   {weakest['error_mean']:.4f}")
    print(f"  Strong: {strong['error_mean']:.4f}")
    print()

    # Verdict
    if coherence_improvement > 5 and jitter_reduction > 10:
        print("✓✓✓ WEAK MEASUREMENT CLEAR WINNER!")
        print("    Quantum coherence preservation → dramatically smoother control")
    elif coherence_improvement > 2:
        print("✓ WEAK MEASUREMENT WINS")
        print("  Coherence preservation provides measurable benefit")
    else:
        print("~ COMPARABLE PERFORMANCE")
        print("  Noise level may not be high enough to show clear benefit")

    print()
    print("=" * 70)

    # Visualization
    plot_noisy_results(results, target)

    return results


def plot_noisy_results(results: dict, target: np.ndarray):
    """
    Visualize high-noise scenario results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Trajectories
    ax = axes[0, 0]
    ax.set_title("Noisy Stereo Trajectories", fontsize=14, fontweight="bold")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, alpha=0.3)

    for name, data in results.items():
        traj = data["trajectory"]
        traj_array = np.array(traj)
        ax.plot(
            traj_array[:, 0],
            traj_array[:, 1],
            marker="",
            linewidth=2,
            label=name,
            color=data["color"],
            alpha=0.7,
        )

    ax.scatter([8.0], [8.0], s=200, marker="*", color="green", label="Start", zorder=5)
    ax.scatter([target[0]], [target[1]], s=200, marker="X", color="red", label="Target", zorder=5)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_aspect("equal")

    # Plot 2: Coherence
    ax = axes[0, 1]
    ax.set_title("Coherence (Smoothness)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Coherence Score")

    names = list(results.keys())
    coherences = [results[name]["coherence_mean"] for name in names]
    errors = [results[name]["coherence_std"] for name in names]
    colors = [results[name]["color"] for name in names]

    bars = ax.bar(range(len(names)), coherences, yerr=errors, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, coherences)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + errors[i],
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 3: Jitter
    ax = axes[1, 0]
    ax.set_title("Jitter (Lower is Better)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Acceleration Magnitude")

    jitters = [results[name]["jitter_mean"] for name in names]
    errors = [results[name]["jitter_std"] for name in names]

    bars = ax.bar(range(len(names)), jitters, yerr=errors, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, jitters)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + errors[i],
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 4: Final Error
    ax = axes[1, 1]
    ax.set_title("Final Error (Lower is Better)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Distance to Target")

    final_errors = [results[name]["error_mean"] for name in names]
    errors = [results[name]["error_std"] for name in names]

    bars = ax.bar(range(len(names)), final_errors, yerr=errors, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, final_errors)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + errors[i],
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("benchmark_weak_stereo_noisy.png", dpi=150, bbox_inches="tight")
    print("Plot saved: benchmark_weak_stereo_noisy.png")


if __name__ == "__main__":
    np.random.seed(42)
    results = run_noisy_benchmark()
