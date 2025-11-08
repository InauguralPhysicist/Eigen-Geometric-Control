#!/usr/bin/env python3
"""
Quickstart Example: Basic Eigen Usage

Demonstrates simplest possible use case:
1. Import framework
2. Run simulation
3. Analyze results
4. Optional: Visualize trajectory

This is the fastest way to see Eigen in action.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import src
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src import run_arm_simulation


def main():
    print("=" * 60)
    print("EIGEN GEOMETRIC CONTROL - QUICKSTART EXAMPLE")
    print("=" * 60)
    print()

    # Run simulation with default parameters
    print("Running simulation with default parameters...")
    print("  Initial configuration: θ₁=-1.4, θ₂=1.2 rad")
    print("  Target: (1.2, 0.3) m")
    print("  Obstacle: center=(0.6, 0.1), radius=0.25 m")
    print("  Learning rate: η=0.12")
    print("  Ticks: 100")
    print()

    results = run_arm_simulation(n_ticks=100)

    # Print summary
    print("RESULTS:")
    print("-" * 60)
    print(f"Initial state:")
    print(f"  ds² (error):      {results['ds2_total'].iloc[0]:.6f}")
    print(f"  Gradient norm:    {results['grad_norm'].iloc[0]:.6f}")
    print(f"  End-effector:     ({results['x'].iloc[0]:.3f}, {results['y'].iloc[0]:.3f}) m")
    print()

    print(f"Final state:")
    print(f"  ds² (error):      {results['ds2_total'].iloc[-1]:.6f}")
    print(f"  Gradient norm:    {results['grad_norm'].iloc[-1]:.2e}")
    print(f"  End-effector:     ({results['x'].iloc[-1]:.3f}, {results['y'].iloc[-1]:.3f}) m")
    print()

    # Compute metrics
    error_reduction = results["ds2_total"].iloc[0] / results["ds2_total"].iloc[-1]
    gradient_reduction = results["grad_norm"].iloc[0] / results["grad_norm"].iloc[-1]
    min_obstacle_dist = results["d_obs"].min()

    print(f"Performance metrics:")
    print(f"  Error reduction:  {error_reduction:.1f}×")
    print(f"  Gradient collapse: {gradient_reduction:.2e}×")
    print(
        f"  Min obstacle dist: {min_obstacle_dist:.3f} m (safe: >{results['d_obs'].iloc[0] - 0.25:.3f} m)"
    )
    print()

    # Check convergence
    if results["grad_norm"].iloc[-1] < 1e-4:
        print("✓ System converged to eigenstate (gradient < 10⁻⁴)")
    else:
        print("⚠ System still converging (run more ticks)")

    if min_obstacle_dist > 0.25:
        print("✓ Obstacle avoided (maintained clearance)")
    else:
        print("⚠ Obstacle boundary violated!")

    print()
    print("=" * 60)
    print("Simulation complete!")
    print()
    print("Next steps:")
    print("  - Try different parameters: python quickstart.py --help")
    print("  - Visualize results: see examples/visualize_trajectory.py")
    print("  - Extend framework: see examples/custom_objective.py")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = main()

    # Optional: simple matplotlib visualization if available
    try:
        import matplotlib.pyplot as plt

        print("\nGenerating visualization...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Trajectory
        ax1.plot(results["x"], results["y"], "b-", linewidth=2, label="Path")
        ax1.plot(results["x"].iloc[0], results["y"].iloc[0], "go", markersize=10, label="Start")
        ax1.plot(results["x"].iloc[-1], results["y"].iloc[-1], "ro", markersize=10, label="End")
        ax1.plot(1.2, 0.3, "r*", markersize=15, label="Target")

        # Add obstacle circle
        circle = plt.Circle(
            (0.6, 0.1), 0.25, fill=False, color="red", linestyle="--", linewidth=2, label="Obstacle"
        )
        ax1.add_patch(circle)

        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_title("Arm Trajectory")
        ax1.legend()
        ax1.axis("equal")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Convergence
        ax2.semilogy(results["tick"], results["grad_norm"], "b-", linewidth=2)
        ax2.set_xlabel("Tick")
        ax2.set_ylabel("Gradient Norm |∇ds²|")
        ax2.set_title("Convergence (log scale)")
        ax2.grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        plt.savefig("quickstart_result.png", dpi=150)
        print("✓ Visualization saved to: quickstart_result.png")

        # Optionally show plot
        # plt.show()

    except ImportError:
        print("\nMatplotlib not available. Skipping visualization.")
        print("Install with: pip install matplotlib")
