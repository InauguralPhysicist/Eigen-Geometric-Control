"""
Benchmark: Geodesic Path Planning for Obstacle Avoidance

This benchmark demonstrates that geodesics in curved configuration space
provide natural and optimal obstacle avoidance compared to naive control.

Comparison:
- Naive: Straight-line control toward target (ignores obstacles)
- Geodesic: Follows shortest path in curved space (natural avoidance)

Test scenarios:
- Single obstacle between start and goal
- Multiple scattered obstacles
- Corridor navigation (obstacles on sides)
"""

import numpy as np
import matplotlib.pyplot as plt

from src.eigen_geodesic import (
    Obstacle,
    geodesic_control_step,
)


def check_collision(position: np.ndarray, obstacles: list) -> bool:
    """Check if position collides with any obstacle."""
    for obs in obstacles:
        dist = np.linalg.norm(position - obs.position)
        if dist < obs.radius * 0.8:  # Collision if within 80% of radius
            return True
    return False


def simulate_navigation(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: list,
    n_steps: int = 200,
    use_geodesic: bool = True,
    eta: float = 0.15,
) -> tuple:
    """
    Simulate robot navigation with or without geodesic planning.

    Args:
        start: Starting position
        goal: Goal position
        obstacles: List of Obstacle objects
        n_steps: Maximum number of steps
        use_geodesic: Whether to use geodesic path planning
        eta: Step size / learning rate

    Returns:
        - trajectory: Path taken
        - collisions: Number of collisions
        - success: Whether goal was reached
        - path_length: Total path length
    """
    position = start.copy()
    trajectory = [position.copy()]
    collisions = 0

    for step in range(n_steps):
        # Check if reached goal
        dist_to_goal = np.linalg.norm(position - goal)
        if dist_to_goal < 0.2:
            # Success
            return (
                np.array(trajectory),
                collisions,
                True,
                compute_path_length(trajectory),
            )

        # Control step
        if use_geodesic:
            # Geodesic path planning
            new_pos, info = geodesic_control_step(
                position,
                goal,
                obstacles,
                eta=eta,
                use_geodesic_direction=True,
            )
        else:
            # Naive straight-line control
            direction = goal - position
            new_pos = position + eta * direction

        # Check collision
        if check_collision(new_pos, obstacles):
            collisions += 1
            # Don't move if collision (stay at current position)
            # This simulates hitting an obstacle
            new_pos = position.copy()

        position = new_pos
        trajectory.append(position.copy())

    # Did not reach goal
    return (
        np.array(trajectory),
        collisions,
        False,
        compute_path_length(trajectory),
    )


def compute_path_length(trajectory: list) -> float:
    """Compute total path length."""
    if len(trajectory) < 2:
        return 0.0

    total_length = 0.0
    for i in range(1, len(trajectory)):
        total_length += np.linalg.norm(trajectory[i] - trajectory[i - 1])

    return total_length


def run_benchmark():
    """
    Run comprehensive benchmark across obstacle scenarios.
    """
    print("=" * 70)
    print("GEODESIC PATH PLANNING BENCHMARK")
    print("=" * 70)
    print()
    print("Testing geodesic paths vs naive straight-line control")
    print()

    scenarios = {
        "single_obstacle": {
            "start": np.array([0.0, 0.0]),
            "goal": np.array([10.0, 0.0]),
            "obstacles": [Obstacle(np.array([5.0, 0.5]), radius=1.5, strength=60.0)],
        },
        "multiple_obstacles": {
            "start": np.array([0.0, 0.0]),
            "goal": np.array([10.0, 10.0]),
            "obstacles": [
                Obstacle(np.array([3.0, 3.0]), radius=1.2, strength=50.0),
                Obstacle(np.array([7.0, 5.0]), radius=1.5, strength=50.0),
                Obstacle(np.array([5.0, 8.0]), radius=1.0, strength=50.0),
            ],
        },
        "corridor": {
            "start": np.array([0.0, 5.0]),
            "goal": np.array([15.0, 5.0]),
            "obstacles": [
                # Obstacles on sides (corridor walls)
                Obstacle(np.array([5.0, 2.0]), radius=1.0, strength=40.0),
                Obstacle(np.array([5.0, 8.0]), radius=1.0, strength=40.0),
                Obstacle(np.array([10.0, 2.5]), radius=1.2, strength=40.0),
                Obstacle(np.array([10.0, 7.5]), radius=1.2, strength=40.0),
            ],
        },
    }

    n_trials = 10
    all_results = {}

    for scenario_name, scenario in scenarios.items():
        print(f"Scenario: {scenario_name.upper().replace('_', ' ')}")
        print("-" * 40)

        start = scenario["start"]
        goal = scenario["goal"]
        obstacles = scenario["obstacles"]

        # Run trials
        success_naive = []
        success_geodesic = []
        collisions_naive = []
        collisions_geodesic = []
        length_naive = []
        length_geodesic = []

        for trial in range(n_trials):
            # Naive control
            traj_naive, coll_naive, succ_naive, len_naive = simulate_navigation(
                start, goal, obstacles, n_steps=200, use_geodesic=False, eta=0.15
            )
            success_naive.append(succ_naive)
            collisions_naive.append(coll_naive)
            length_naive.append(len_naive)

            # Geodesic planning
            traj_geo, coll_geo, succ_geo, len_geo = simulate_navigation(
                start, goal, obstacles, n_steps=200, use_geodesic=True, eta=0.15
            )
            success_geodesic.append(succ_geo)
            collisions_geodesic.append(coll_geo)
            length_geodesic.append(len_geo)

        # Compute metrics
        naive_success_rate = np.mean(success_naive) * 100
        geo_success_rate = np.mean(success_geodesic) * 100
        naive_avg_collisions = np.mean(collisions_naive)
        geo_avg_collisions = np.mean(collisions_geodesic)
        naive_avg_length = np.mean(length_naive)
        geo_avg_length = np.mean(length_geodesic)

        # Improvements
        success_improvement = geo_success_rate - naive_success_rate
        collision_reduction = (
            (naive_avg_collisions - geo_avg_collisions) / (naive_avg_collisions + 1e-6)
        ) * 100

        results = {
            "scenario_name": scenario_name,
            "naive_success_rate": naive_success_rate,
            "geo_success_rate": geo_success_rate,
            "success_improvement": success_improvement,
            "naive_collisions": naive_avg_collisions,
            "geo_collisions": geo_avg_collisions,
            "collision_reduction": collision_reduction,
            "naive_path_length": naive_avg_length,
            "geo_path_length": geo_avg_length,
            "sample_traj_naive": traj_naive,
            "sample_traj_geodesic": traj_geo,
            "obstacles": obstacles,
            "start": start,
            "goal": goal,
        }

        all_results[scenario_name] = results

        print(f"  Success Rate:")
        print(f"    Naive:      {naive_success_rate:.1f}%")
        print(f"    Geodesic:   {geo_success_rate:.1f}%")
        print(f"    Improvement: {success_improvement:+.1f}%")
        print()
        print(f"  Collisions:")
        print(f"    Naive:      {naive_avg_collisions:.2f}")
        print(f"    Geodesic:   {geo_avg_collisions:.2f}")
        print(f"    Reduction:  {collision_reduction:+.1f}%")
        print()
        print(f"  Path Length:")
        print(f"    Naive:      {naive_avg_length:.2f}")
        print(f"    Geodesic:   {geo_avg_length:.2f}")
        print()

    # Summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    avg_success_improvement = np.mean([r["success_improvement"] for r in all_results.values()])
    avg_collision_reduction = np.mean([r["collision_reduction"] for r in all_results.values()])

    print(f"Average Success Improvement:  {avg_success_improvement:+.1f}%")
    print(f"Average Collision Reduction:  {avg_collision_reduction:+.1f}%")
    print()

    print("Breakdown by Scenario:")
    for name, result in all_results.items():
        print(
            f"  {name.replace('_', ' ').title()}: "
            f"Success {result['success_improvement']:+.1f}%, "
            f"Collisions {result['collision_reduction']:+.1f}%"
        )

    print()

    if avg_success_improvement > 30 and avg_collision_reduction > 50:
        print("✓✓✓ GEODESIC PLANNING WINS!")
        print("    Natural obstacle avoidance >> straight-line control")
    elif avg_success_improvement > 15 or avg_collision_reduction > 30:
        print("✓ GEODESIC PLANNING WINS")
        print("  Curved paths enable better navigation")
    else:
        print("~ COMPARABLE PERFORMANCE")
        print("  Benefits depend on obstacle configuration")

    print()
    print("=" * 70)

    # Visualization
    plot_results(all_results)

    return all_results


def plot_results(results: dict):
    """
    Visualize geodesic path planning performance.
    """
    n_scenarios = len(results)
    fig, axes = plt.subplots(2, n_scenarios, figsize=(6 * n_scenarios, 12))

    if n_scenarios == 1:
        axes = axes.reshape(-1, 1)

    for idx, (scenario_name, data) in enumerate(results.items()):
        traj_naive = data["sample_traj_naive"]
        traj_geo = data["sample_traj_geodesic"]
        obstacles = data["obstacles"]
        start = data["start"]
        goal = data["goal"]

        # Top row: Trajectories
        ax = axes[0, idx]
        ax.set_title(f"{scenario_name.replace('_', ' ').title()}", fontsize=12, fontweight="bold")

        # Plot obstacles
        for obs in obstacles:
            circle = plt.Circle(obs.position, obs.radius, color="red", alpha=0.3, label="Obstacle")
            ax.add_patch(circle)

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
            traj_geo[:, 0],
            traj_geo[:, 1],
            "b-",
            label="Geodesic",
            linewidth=1.5,
            alpha=0.7,
        )

        # Mark start and goal
        ax.scatter(start[0], start[1], s=150, marker="*", color="green", label="Start", zorder=5)
        ax.scatter(goal[0], goal[1], s=150, marker="X", color="darkgreen", label="Goal", zorder=5)

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        # Bottom row: Performance comparison
        ax = axes[1, idx]
        ax.set_title(
            f"Success: {data['success_improvement']:+.1f}%, "
            f"Collisions: {data['collision_reduction']:+.1f}%",
            fontsize=10,
        )

        metrics = ["Success\nRate (%)", "Collisions"]
        naive_vals = [data["naive_success_rate"], data["naive_collisions"]]
        geo_vals = [data["geo_success_rate"], data["geo_collisions"]]

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width / 2, naive_vals, width, label="Naive", color="red", alpha=0.7)
        ax.bar(x + width / 2, geo_vals, width, label="Geodesic", color="blue", alpha=0.7)

        ax.set_ylabel("Value")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=9)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        # Add value labels
        for i, (nv, gv) in enumerate(zip(naive_vals, geo_vals)):
            ax.text(i - width / 2, nv, f"{nv:.1f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + width / 2, gv, f"{gv:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("benchmark_geodesic_planning.png", dpi=150, bbox_inches="tight")
    print("Plot saved: benchmark_geodesic_planning.png")


if __name__ == "__main__":
    np.random.seed(42)
    results = run_benchmark()
