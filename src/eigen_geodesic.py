"""
Geodesic Path Planning in Curved Configuration Space

Uses differential geometry to find optimal paths around obstacles.
Key insight: Obstacles curve the configuration space, and geodesics
(shortest paths in curved space) naturally avoid them.

Mathematical Framework:
- Metric tensor g_μν: Defines distances, increases near obstacles
- Christoffel symbols Γ^μ_νρ: Connection coefficients (how metric changes)
- Geodesic equation: d²x^μ/dτ² + Γ^μ_νρ (dx^ν/dτ)(dx^ρ/dτ) = 0

This enables:
- Natural obstacle avoidance (no explicit collision detection)
- Optimal path curvature (minimal "energy" in curved space)
- Integration with lightlike damping framework
"""

from typing import List, Tuple

import numpy as np


class Obstacle:
    """
    Obstacle in configuration space.

    Attributes
    ----------
    position : ndarray
        Center position of obstacle
    radius : float
        Radius of influence
    strength : float
        How much it curves space (higher = stronger repulsion)
    """

    def __init__(
        self,
        position: np.ndarray,
        radius: float = 1.0,
        strength: float = 10.0,
    ):
        self.position = np.asarray(position)
        self.radius = radius
        self.strength = strength


def metric_tensor(
    position: np.ndarray,
    obstacles: List[Obstacle],
    base_metric: float = 1.0,
) -> np.ndarray:
    """
    Compute metric tensor at given position.

    The metric defines "distance" in configuration space. Near obstacles,
    the metric increases (making that region "expensive" to traverse).

    Parameters
    ----------
    position : ndarray, shape (d,)
        Current position in configuration space
    obstacles : list of Obstacle
        List of obstacles
    base_metric : float
        Base metric value (flat space)

    Returns
    -------
    g : ndarray, shape (d, d)
        Metric tensor (symmetric positive-definite matrix)

    Notes
    -----
    We use a simple diagonal metric that increases near obstacles:
        g_μν = δ_μν * (1 + Σ_i strength_i * exp(-dist_i² / radius_i²))

    This makes the space "curved" near obstacles, so geodesics naturally
    avoid them.

    Examples
    --------
    >>> pos = np.array([1.0, 1.0])
    >>> obs = [Obstacle(np.array([0.0, 0.0]), radius=1.0, strength=10.0)]
    >>> g = metric_tensor(pos, obs)
    >>> g.shape
    (2, 2)
    >>> g[0, 0] > 1.0  # Metric increased near obstacle
    True
    """
    d = len(position)

    # Start with flat metric
    metric_scale = base_metric

    # Add obstacle contributions
    for obs in obstacles:
        # Distance to obstacle
        diff = position - obs.position
        dist_sq = np.sum(diff**2)

        # Gaussian repulsion (exponential decay)
        # Increases metric near obstacle
        metric_scale += obs.strength * np.exp(-dist_sq / (obs.radius**2))

    # Diagonal metric (isotropic space)
    g = np.eye(d) * metric_scale

    return g


def metric_gradient(
    position: np.ndarray,
    obstacles: List[Obstacle],
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Compute gradient of metric tensor components.

    Uses finite differences to compute ∂g_μν/∂x^ρ.

    Parameters
    ----------
    position : ndarray, shape (d,)
        Current position
    obstacles : list of Obstacle
        Obstacles
    epsilon : float
        Step size for finite differences

    Returns
    -------
    dg : ndarray, shape (d, d, d)
        Gradient tensor: dg[μ, ν, ρ] = ∂g_μν/∂x^ρ
    """
    d = len(position)
    dg = np.zeros((d, d, d))

    # Central differences
    for rho in range(d):
        # Step in rho direction
        step = np.zeros(d)
        step[rho] = epsilon

        g_plus = metric_tensor(position + step, obstacles)
        g_minus = metric_tensor(position - step, obstacles)

        # ∂g_μν/∂x^ρ
        dg[:, :, rho] = (g_plus - g_minus) / (2 * epsilon)

    return dg


def christoffel_symbols(
    position: np.ndarray,
    obstacles: List[Obstacle],
) -> np.ndarray:
    """
    Compute Christoffel symbols of the second kind.

    The Christoffel symbols Γ^μ_νρ describe how the metric changes
    and are needed for the geodesic equation.

    Parameters
    ----------
    position : ndarray, shape (d,)
        Current position
    obstacles : list of Obstacle
        Obstacles

    Returns
    -------
    Gamma : ndarray, shape (d, d, d)
        Christoffel symbols: Gamma[μ, ν, ρ] = Γ^μ_νρ

    Notes
    -----
    Formula:
        Γ^μ_νρ = (1/2) g^μσ (∂g_σν/∂x^ρ + ∂g_σρ/∂x^ν - ∂g_νρ/∂x^σ)

    For a diagonal metric this simplifies considerably.
    """
    d = len(position)

    # Metric and its inverse
    g = metric_tensor(position, obstacles)
    g_inv = np.linalg.inv(g)

    # Metric gradient
    dg = metric_gradient(position, obstacles)

    # Compute Christoffel symbols
    Gamma = np.zeros((d, d, d))

    for mu in range(d):
        for nu in range(d):
            for rho in range(d):
                # Sum over sigma
                for sigma in range(d):
                    Gamma[mu, nu, rho] += (
                        0.5
                        * g_inv[mu, sigma]
                        * (dg[sigma, nu, rho] + dg[sigma, rho, nu] - dg[nu, rho, sigma])
                    )

    return Gamma


def geodesic_acceleration(
    position: np.ndarray,
    velocity: np.ndarray,
    obstacles: List[Obstacle],
) -> np.ndarray:
    """
    Compute geodesic acceleration.

    The geodesic equation gives the acceleration needed to follow
    a geodesic (shortest path in curved space):

        d²x^μ/dτ² = -Γ^μ_νρ (dx^ν/dτ)(dx^ρ/dτ)

    Parameters
    ----------
    position : ndarray, shape (d,)
        Current position
    velocity : ndarray, shape (d,)
        Current velocity dx^μ/dτ
    obstacles : list of Obstacle
        Obstacles

    Returns
    -------
    acceleration : ndarray, shape (d,)
        Geodesic acceleration d²x^μ/dτ²

    Notes
    -----
    This acceleration keeps you on a geodesic. It naturally steers
    around obstacles due to space curvature.
    """
    d = len(position)

    # Christoffel symbols at current position
    Gamma = christoffel_symbols(position, obstacles)

    # Compute acceleration: -Γ^μ_νρ v^ν v^ρ
    acceleration = np.zeros(d)

    for mu in range(d):
        for nu in range(d):
            for rho in range(d):
                acceleration[mu] -= Gamma[mu, nu, rho] * velocity[nu] * velocity[rho]

    return acceleration


def compute_geodesic_path(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: List[Obstacle],
    n_steps: int = 100,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Compute geodesic path from start to goal.

    Integrates the geodesic equation with initial velocity toward goal.

    Parameters
    ----------
    start : ndarray
        Starting position
    goal : ndarray
        Goal position
    obstacles : list of Obstacle
        Obstacles to avoid
    n_steps : int
        Number of integration steps
    dt : float
        Time step size

    Returns
    -------
    path : ndarray, shape (n_steps+1, d)
        Geodesic path from start to goal

    Algorithm
    ---------
    1. Initialize with velocity toward goal
    2. Integrate geodesic equation using RK4
    3. Return trajectory
    """
    # Initial conditions
    position = start.copy()
    velocity = (goal - start) / np.linalg.norm(goal - start)  # Unit velocity

    # Store path
    path = [position.copy()]

    # Integrate geodesic equation (RK4)
    for _ in range(n_steps):
        # RK4 integration
        # k1
        k1_v = geodesic_acceleration(position, velocity, obstacles)
        k1_x = velocity

        # k2
        k2_v = geodesic_acceleration(
            position + 0.5 * dt * k1_x,
            velocity + 0.5 * dt * k1_v,
            obstacles,
        )
        k2_x = velocity + 0.5 * dt * k1_v

        # k3
        k3_v = geodesic_acceleration(
            position + 0.5 * dt * k2_x,
            velocity + 0.5 * dt * k2_v,
            obstacles,
        )
        k3_x = velocity + 0.5 * dt * k2_v

        # k4
        k4_v = geodesic_acceleration(
            position + dt * k3_x,
            velocity + dt * k3_v,
            obstacles,
        )
        k4_x = velocity + dt * k3_v

        # Update
        position = position + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        velocity = velocity + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        path.append(position.copy())

    return np.array(path)


def geodesic_control_step(
    current_position: np.ndarray,
    target_position: np.ndarray,
    obstacles: List[Obstacle],
    eta: float = 0.1,
    use_geodesic_direction: bool = True,
    repulsion_strength: float = 2.0,
) -> Tuple[np.ndarray, dict]:
    """
    Control step following geodesic path with obstacle avoidance.

    Combines geodesic path planning with explicit repulsive forces
    for robust obstacle avoidance.

    Parameters
    ----------
    current_position : ndarray
        Current position
    target_position : ndarray
        Target position
    obstacles : list of Obstacle
        Obstacles to avoid
    eta : float
        Step size / learning rate
    use_geodesic_direction : bool
        If True, use geodesic acceleration to modify direction
    repulsion_strength : float
        Strength of repulsive forces from obstacles

    Returns
    -------
    new_position : ndarray
        Updated position
    info : dict
        Diagnostic information

    Algorithm
    ---------
    1. Compute attractive force toward target
    2. Compute geodesic acceleration (space curvature effects)
    3. Compute repulsive forces from nearby obstacles
    4. Blend forces: F_total = F_attract + F_geodesic + F_repel
    5. Take step in direction of total force
    """
    # Attractive force (toward target)
    attractive_force = target_position - current_position

    if not use_geodesic_direction or len(obstacles) == 0:
        # No geodesic correction
        new_position = current_position + eta * attractive_force

        info = {
            "geodesic_acceleration": np.zeros_like(current_position),
            "repulsive_force": np.zeros_like(current_position),
            "correction_magnitude": 0.0,
            "metric_at_position": 1.0,
        }

        return new_position, info

    # Compute forces
    dist_to_target = np.linalg.norm(attractive_force)
    if dist_to_target < 1e-12:
        # Already at target
        return current_position.copy(), {
            "geodesic_acceleration": np.zeros_like(current_position),
            "repulsive_force": np.zeros_like(current_position),
        }

    velocity = attractive_force / dist_to_target

    # Geodesic acceleration (space curvature)
    geo_accel = geodesic_acceleration(current_position, velocity, obstacles)

    # Repulsive forces from obstacles (artificial potential field)
    repulsive_force = np.zeros_like(current_position)
    for obs in obstacles:
        diff = current_position - obs.position
        dist = np.linalg.norm(diff)

        if dist < 1e-12:
            # At obstacle center - push in random direction
            repulsion = np.random.randn(len(current_position))
            repulsion = repulsion / (np.linalg.norm(repulsion) + 1e-12)
            repulsive_force += repulsion * repulsion_strength
        elif dist < obs.radius * 3.0:  # Repulsion range
            # Repulsion magnitude: stronger when closer
            # F_repel ~ 1/dist² (like Coulomb force)
            repulsion_magnitude = obs.strength * repulsion_strength / (dist**2 + 0.1)
            # Direction: away from obstacle
            repulsion_direction = diff / dist
            repulsive_force += repulsion_magnitude * repulsion_direction

    # Total force: attractive + geodesic + repulsive
    total_force = attractive_force + geo_accel * dist_to_target + repulsive_force

    # Take step
    new_position = current_position + eta * total_force

    # Diagnostics
    g = metric_tensor(current_position, obstacles)
    metric_value = np.mean(np.diag(g))

    info = {
        "geodesic_acceleration": geo_accel,
        "repulsive_force": repulsive_force,
        "correction_magnitude": float(np.linalg.norm(geo_accel + repulsive_force)),
        "metric_at_position": float(metric_value),
        "attractive_force": attractive_force,
        "total_force": total_force,
    }

    return new_position, info
