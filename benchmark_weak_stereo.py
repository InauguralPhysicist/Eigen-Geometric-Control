"""
Benchmark: Weak Measurement vs Strong Measurement for Stereo Vision

This benchmark demonstrates that weak measurements provide smoother,
more coherent control than strong measurements.

Key comparison:
- Strong measurement: Fully collapses state → discrete observations → jittery control
- Weak measurement: Preserves coherence → continuous observations → smooth control
"""

import numpy as np
import matplotlib.pyplot as plt

from src.eigen_weak_measurement import (
    coherence_metric,
    weak_stereo_control_step,
)


def simulate_stereo_control(
    initial_state: np.ndarray,
    target: np.ndarray,
    n_steps: int = 50,
    measurement_strength: float = 0.1,
    eta: float = 0.15,
    noise_level: float = 0.05,
) -> tuple:
    """
    Simulate stereo-based control with specified measurement strength.

    Args:
        initial_state: Starting position
        target: Goal position
        n_steps: Number of control steps
        measurement_strength: 0.1 = weak, 0.9 = strong
        eta: Learning rate
        noise_level: Noise added to observations

    Returns:
        - trajectory: List of states
        - coherence: Coherence score of trajectory
        - final_error: Distance to target at end
    """
    trajectory = [initial_state.copy()]
    current = initial_state.copy()
    measurements = []

    for step in range(n_steps):
        # Simulate noisy stereo observations
        # Left eye sees current state + noise
        left_obs = current + np.random.randn(2) * noise_level

        # Right eye sees current state with disparity (parallax)
        right_obs = current + np.random.randn(2) * noise_level
        right_obs[0] -= 0.05  # Disparity in x-direction

        # Control step with weak/strong measurement
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

    # Evaluate performance
    coherence = coherence_metric(trajectory)
    final_error = np.linalg.norm(trajectory[-1] - target)

    return trajectory, coherence, final_error


def run_benchmark():
    """
    Run comprehensive benchmark comparing weak vs strong measurements.
    """
    print("=" * 70)
    print("WEAK MEASUREMENT STEREO VISION BENCHMARK")
    print("=" * 70)
    print()
    print("Comparing weak measurement (quantum-inspired, coherence-preserving)")
    print("vs strong measurement (classical, state-collapsing)")
    print()

    # Setup
    initial = np.array([5.0, 5.0])
    target = np.array([0.0, 0.0])
    n_steps = 50
    n_trials = 10

    # Test configurations
    configs = [
        {"name": "Weak Measurement (0.1)", "strength": 0.1, "color": "blue"},
        {"name": "Medium Measurement (0.5)", "strength": 0.5, "color": "orange"},
        {"name": "Strong Measurement (0.9)", "strength": 0.9, "color": "red"},
    ]

    results = {}

    for config in configs:
        print(f"Testing: {config['name']}")
        print("-" * 40)

        coherences = []
        final_errors = []
        trajectories = []

        for trial in range(n_trials):
            traj, coh, err = simulate_stereo_control(
                initial,
                target,
                n_steps=n_steps,
                measurement_strength=config["strength"],
            )

            coherences.append(coh)
            final_errors.append(err)
            if trial == 0:  # Save first trajectory for plotting
                trajectories.append(traj)

        # Statistics
        mean_coherence = np.mean(coherences)
        std_coherence = np.std(coherences)
        mean_error = np.mean(final_errors)
        std_error = np.std(final_errors)

        results[config["name"]] = {
            "coherence_mean": mean_coherence,
            "coherence_std": std_coherence,
            "error_mean": mean_error,
            "error_std": std_error,
            "trajectory": trajectories[0],
            "color": config["color"],
        }

        print(f"  Coherence:    {mean_coherence:.4f} ± {std_coherence:.4f}")
        print(f"  Final Error:  {mean_error:.4f} ± {std_error:.4f}")
        print()

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    weak = results["Weak Measurement (0.1)"]
    strong = results["Strong Measurement (0.9)"]

    coherence_improvement = (
        (weak["coherence_mean"] - strong["coherence_mean"]) / strong["coherence_mean"] * 100
    )
    error_improvement = (strong["error_mean"] - weak["error_mean"]) / strong["error_mean"] * 100

    print(f"Coherence Improvement: {coherence_improvement:+.1f}%")
    print(f"  Weak:   {weak['coherence_mean']:.4f} (smoother trajectory)")
    print(f"  Strong: {strong['coherence_mean']:.4f} (jittery trajectory)")
    print()
    print(f"Error Improvement: {error_improvement:+.1f}%")
    print(f"  Weak:   {weak['error_mean']:.4f} (better convergence)")
    print(f"  Strong: {strong['error_mean']:.4f} (worse convergence)")
    print()

    # Key insight
    if coherence_improvement > 10:
        print("✓ WEAK MEASUREMENT WINS!")
        print("  Preserving quantum coherence → smoother, more accurate control")
    elif coherence_improvement < -10:
        print("✗ STRONG MEASUREMENT WINS (unexpected)")
        print("  This suggests noise overwhelms coherence benefits")
    else:
        print("~ COMPARABLE PERFORMANCE")
        print("  Coherence preservation has minimal impact in this scenario")

    print()
    print("=" * 70)

    # Visualization
    plot_results(results, target)

    return results


def plot_results(results: dict, target: np.ndarray):
    """
    Visualize trajectories for weak vs strong measurement.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Trajectories
    ax = axes[0]
    ax.set_title("Stereo Control Trajectories", fontsize=14, fontweight="bold")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, alpha=0.3)

    for name, data in results.items():
        traj = data["trajectory"]
        traj_array = np.array(traj)
        ax.plot(
            traj_array[:, 0],
            traj_array[:, 1],
            marker="o",
            markersize=3,
            label=name,
            color=data["color"],
            alpha=0.7,
        )

    # Mark start and target
    ax.scatter([5.0], [5.0], s=200, marker="*", color="green", label="Start", zorder=5)
    ax.scatter([target[0]], [target[1]], s=200, marker="X", color="red", label="Target", zorder=5)

    ax.legend(loc="upper right")
    ax.set_aspect("equal")

    # Plot 2: Coherence Comparison
    ax = axes[1]
    ax.set_title("Coherence Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Coherence Score (higher = smoother)")
    ax.set_xlabel("Measurement Type")

    names = list(results.keys())
    coherences = [results[name]["coherence_mean"] for name in names]
    errors = [results[name]["coherence_std"] for name in names]
    colors = [results[name]["color"] for name in names]

    bars = ax.bar(range(len(names)), coherences, yerr=errors, color=colors, alpha=0.7)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, coherences)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + errors[i],
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("benchmark_weak_stereo.png", dpi=150, bbox_inches="tight")
    print("Plot saved: benchmark_weak_stereo.png")


if __name__ == "__main__":
    np.random.seed(42)  # Reproducibility
    results = run_benchmark()
