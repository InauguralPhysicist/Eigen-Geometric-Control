"""
Benchmark: Coherent/Incoherent Decomposition for Noise Filtering

This benchmark demonstrates that signal decomposition enables better
control in noisy environments by separating predictable signals from
random noise.

Comparison:
- Naive: Use observations directly (no noise filtering)
- Coherent Filtering: Decompose and suppress incoherent noise
"""

import numpy as np
import matplotlib.pyplot as plt

from src.eigen_decomposition import (
    coherent_control_step,
)


def simulate_noisy_trajectory(
    initial_state: np.ndarray,
    target: np.ndarray,
    n_steps: int = 100,
    use_filtering: bool = True,
    noise_level: float = 0.5,
    eta: float = 0.2,
    noise_suppression: float = 0.8,
) -> tuple:
    """
    Simulate control with or without coherent filtering.

    Args:
        initial_state: Starting position
        target: Target position
        n_steps: Number of steps
        use_filtering: Whether to use coherent/incoherent decomposition
        noise_level: Standard deviation of observation noise
        eta: Learning rate
        noise_suppression: How much to suppress incoherent component

    Returns:
        - trajectory: State trajectory
        - observations: Noisy observations
        - coherence_scores: Coherence at each step
    """
    state = initial_state.copy()
    trajectory = [state.copy()]
    observations = []
    coherence_scores = []

    for step in range(n_steps):
        # Generate noisy observation
        true_observation = state.copy()
        noise = np.random.randn(*state.shape) * noise_level
        noisy_obs = true_observation + noise
        observations.append(noisy_obs)

        # Build history for coherence detection
        history = trajectory[-min(10, len(trajectory)) :]

        if use_filtering:
            # Use coherent filtering
            new_state, info = coherent_control_step(
                state,
                target,
                noisy_obs,
                history,
                eta=eta,
                noise_suppression=noise_suppression,
            )
            coherence_scores.append(info["coherence_score"])
        else:
            # Naive: use noisy observation directly
            # Blend observation with current state
            blend = 0.5
            updated_state = blend * noisy_obs + (1 - blend) * state

            # Move toward target
            direction = target - updated_state
            new_state = updated_state + eta * direction

            coherence_scores.append(0.0)  # No coherence computed

        state = new_state
        trajectory.append(state.copy())

    return np.array(trajectory), np.array(observations), np.array(coherence_scores)


def compute_trajectory_metrics(
    trajectory: np.ndarray,
    target: np.ndarray,
) -> dict:
    """
    Compute performance metrics for a trajectory.

    Args:
        trajectory: State trajectory
        target: Target position

    Returns:
        Dictionary of metrics
    """
    # Tracking error
    errors = [np.linalg.norm(state - target) for state in trajectory]
    mean_error = np.mean(errors)
    final_error = errors[-1]

    # Path smoothness (lower is smoother)
    if len(trajectory) > 2:
        velocities = [
            trajectory[i] - trajectory[i - 1] for i in range(1, len(trajectory))
        ]
        accelerations = [
            velocities[i] - velocities[i - 1] for i in range(1, len(velocities))
        ]
        jitter = np.mean([np.linalg.norm(a) for a in accelerations])
    else:
        jitter = 0.0

    # Convergence time (steps to reach within 10% of target)
    target_dist = np.linalg.norm(trajectory[0] - target)
    threshold = target_dist * 0.1
    convergence_time = None
    for i, error in enumerate(errors):
        if error < threshold:
            convergence_time = i
            break
    if convergence_time is None:
        convergence_time = len(errors)

    return {
        "mean_error": mean_error,
        "final_error": final_error,
        "jitter": jitter,
        "convergence_time": convergence_time,
    }


def run_benchmark():
    """
    Run comprehensive benchmark comparing filtering vs no filtering.
    """
    print("=" * 70)
    print("COHERENT/INCOHERENT DECOMPOSITION BENCHMARK")
    print("=" * 70)
    print()
    print("Testing signal decomposition for noise filtering in control")
    print()

    # Test different noise levels
    noise_levels = [0.1, 0.3, 0.5, 1.0]
    n_trials = 20
    n_steps = 100
    initial_state = np.array([10.0, 10.0])
    target = np.array([0.0, 0.0])

    all_results = {}

    for noise_level in noise_levels:
        print(f"Testing: Noise Level = {noise_level:.1f}")
        print("-" * 40)

        # Run trials
        metrics_naive = []
        metrics_filtered = []

        for trial in range(n_trials):
            # Same random seed for both methods (fair comparison)
            np.random.seed(trial)

            # Naive (no filtering)
            traj_naive, obs_naive, _ = simulate_noisy_trajectory(
                initial_state,
                target,
                n_steps=n_steps,
                use_filtering=False,
                noise_level=noise_level,
                eta=0.2,
            )
            metrics_naive.append(compute_trajectory_metrics(traj_naive, target))

            # Reset random seed
            np.random.seed(trial)

            # Filtered (coherent decomposition)
            traj_filt, obs_filt, coh_scores = simulate_noisy_trajectory(
                initial_state,
                target,
                n_steps=n_steps,
                use_filtering=True,
                noise_level=noise_level,
                eta=0.2,
                noise_suppression=0.8,
            )
            metrics_filtered.append(compute_trajectory_metrics(traj_filt, target))

        # Average metrics
        def avg_metric(metrics, key):
            return np.mean([m[key] for m in metrics])

        results = {
            "noise_level": noise_level,
            "naive_mean_error": avg_metric(metrics_naive, "mean_error"),
            "filtered_mean_error": avg_metric(metrics_filtered, "mean_error"),
            "naive_jitter": avg_metric(metrics_naive, "jitter"),
            "filtered_jitter": avg_metric(metrics_filtered, "jitter"),
            "naive_conv_time": avg_metric(metrics_naive, "convergence_time"),
            "filtered_conv_time": avg_metric(metrics_filtered, "convergence_time"),
            "sample_traj_naive": traj_naive,  # Last trial
            "sample_traj_filtered": traj_filt,  # Last trial
        }

        # Compute improvements
        error_improvement = (
            (results["naive_mean_error"] - results["filtered_mean_error"])
            / results["naive_mean_error"]
            * 100
        )
        jitter_reduction = (
            (results["naive_jitter"] - results["filtered_jitter"])
            / results["naive_jitter"]
            * 100
        )

        results["error_improvement"] = error_improvement
        results["jitter_reduction"] = jitter_reduction

        all_results[noise_level] = results

        print(f"  Naive Mean Error:      {results['naive_mean_error']:.3f}")
        print(f"  Filtered Mean Error:   {results['filtered_mean_error']:.3f}")
        print(f"  Error Improvement:     {error_improvement:+.1f}%")
        print()
        print(f"  Naive Jitter:          {results['naive_jitter']:.4f}")
        print(f"  Filtered Jitter:       {results['filtered_jitter']:.4f}")
        print(f"  Jitter Reduction:      {jitter_reduction:+.1f}%")
        print()

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    avg_error_improvement = np.mean(
        [r["error_improvement"] for r in all_results.values()]
    )
    avg_jitter_reduction = np.mean(
        [r["jitter_reduction"] for r in all_results.values()]
    )

    print(f"Average Error Improvement: {avg_error_improvement:+.1f}%")
    print(f"Average Jitter Reduction:  {avg_jitter_reduction:+.1f}%")
    print()

    print("Breakdown by Noise Level:")
    for noise_level, result in all_results.items():
        print(
            f"  Noise {noise_level:.1f}: Error {result['error_improvement']:+.1f}%, "
            f"Jitter {result['jitter_reduction']:+.1f}%"
        )

    print()

    if avg_error_improvement > 15:
        print("✓✓✓ COHERENT FILTERING WINS!")
        print("    Signal decomposition → significantly better performance")
    elif avg_error_improvement > 10:
        print("✓ COHERENT FILTERING WINS")
        print("  Noise filtering provides measurable benefit")
    else:
        print("~ COMPARABLE PERFORMANCE")
        print("  Benefits depend on noise level")

    print()
    print("=" * 70)

    # Visualization
    plot_results(all_results, target)

    return all_results


def plot_results(results: dict, target: np.ndarray):
    """
    Visualize filtering performance.
    """
    n_noise_levels = len(results)
    fig, axes = plt.subplots(2, n_noise_levels, figsize=(5 * n_noise_levels, 10))

    if n_noise_levels == 1:
        axes = axes.reshape(-1, 1)

    for idx, (noise_level, data) in enumerate(results.items()):
        traj_naive = data["sample_traj_naive"]
        traj_filt = data["sample_traj_filtered"]

        # Top row: Trajectories
        ax = axes[0, idx]
        ax.set_title(f"Noise Level = {noise_level:.1f}", fontsize=12, fontweight="bold")
        ax.plot(
            traj_naive[:, 0],
            traj_naive[:, 1],
            "r-",
            label="Naive",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.plot(
            traj_filt[:, 0],
            traj_filt[:, 1],
            "b-",
            label="Filtered",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.scatter(
            traj_naive[0, 0],
            traj_naive[0, 1],
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
            f"Improvement: {data['error_improvement']:+.1f}%", fontsize=11
        )

        metrics = ["Mean Error", "Jitter"]
        naive_vals = [data["naive_mean_error"], data["naive_jitter"]]
        filt_vals = [data["filtered_mean_error"], data["filtered_jitter"]]

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width / 2, naive_vals, width, label="Naive", color="red", alpha=0.7)
        ax.bar(
            x + width / 2, filt_vals, width, label="Filtered", color="blue", alpha=0.7
        )

        ax.set_ylabel("Value")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels
        for i, (nv, fv) in enumerate(zip(naive_vals, filt_vals)):
            ax.text(i - width / 2, nv, f"{nv:.3f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + width / 2, fv, f"{fv:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("benchmark_coherent_decomposition.png", dpi=150, bbox_inches="tight")
    print("Plot saved: benchmark_coherent_decomposition.png")


if __name__ == "__main__":
    np.random.seed(42)
    results = run_benchmark()
