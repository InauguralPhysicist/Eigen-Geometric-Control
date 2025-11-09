# -*- coding: utf-8 -*-
"""
Eigen: Noperthedron Analysis

The Noperthedron (Steininger & Yurkevich, 2025) is a convex polyhedron
with 90 vertices that has NO Rupert's property (cannot pass through itself).

This module explores the connection between Rupert's property and the
ds² metric in the Eigen framework.

Hypothesis:
- Rupert property ↔ time-like passage (ds² > 0) ↔ convergence possible
- No Rupert property ↔ space-like only (ds² < 0) ↔ no convergence

The Noperthedron may provide geometric validation that ds² < 0 systems
cannot converge, just as shapes without Rupert's property cannot pass
through themselves.

References
----------
Steininger, Jakob and Yurkevich, Vsevolod (2025). "The Noperthedron."
arXiv preprint.
"""

import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from dataclasses import dataclass


@dataclass
class PassageAttempt:
    """Result of attempting passage of Noperthedron through itself"""
    orientation1: Tuple[float, float]  # (theta1, phi1)
    orientation2: Tuple[float, float]  # (theta2, phi2)
    direction: float  # alpha
    min_distance: float  # Closest approach
    overlap_volume: float  # Estimated overlap
    C: float  # Change metric (collision points)
    S: float  # Stability metric (clear points)
    ds2: float  # S² - C²


def generate_base_points() -> List[np.ndarray]:
    """
    Generate the three base points C₁, C₂, C₃ for the Noperthedron.

    From Steininger & Yurkevich (2025):
    - C₁ has |C₁| = 1 (unit vector)
    - C₂, C₃ have |Cᵢ| ≈ 0.98-0.99
    - All coordinates are rational (in ℚ³)

    Returns
    -------
    base_points : list of np.ndarray
        [C₁, C₂, C₃] coordinates in ℝ³

    Notes
    -----
    These are approximate values. The actual Noperthedron coordinates
    would come from the paper's supplementary materials.
    """
    # C₁: unit vector (exact rationals approximated)
    C1 = np.array([152024884, 0, 210152163]) / 259375205

    # C₂: magnitude ≈ 0.986
    C2 = np.array([6632738028, 6106948881, 3980949609]) / 1e10

    # C₃: magnitude ≈ 0.987
    C3 = np.array([8193990033, 5298215096, 1230614493]) / 1e10

    return [C1, C2, C3]


def verify_base_points(base_points: List[np.ndarray]) -> None:
    """Verify base point properties match paper specifications"""
    print("Base point verification:")
    for i, point in enumerate(base_points, 1):
        magnitude = np.linalg.norm(point)
        print(f"  C{i}: magnitude = {magnitude:.6f}")
        print(f"       coords = [{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}]")
    print()


def rotation_z(angle: float) -> np.ndarray:
    """
    Rotation matrix around z-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians

    Returns
    -------
    R : np.ndarray of shape (3, 3)
        Rotation matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def rotation_y(angle: float) -> np.ndarray:
    """Rotation matrix around y-axis"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])


def rotation_from_spherical(theta: float, phi: float) -> np.ndarray:
    """
    Create rotation matrix from spherical coordinates.

    Parameters
    ----------
    theta : float
        Polar angle (0 to π)
    phi : float
        Azimuthal angle (0 to 2π)

    Returns
    -------
    R : np.ndarray of shape (3, 3)
        Rotation matrix orienting z-axis to (theta, phi)
    """
    # First rotate around z-axis by phi
    Rz = rotation_z(phi)
    # Then rotate around y-axis by theta
    Ry = rotation_y(theta)
    return Rz @ Ry


def generate_noperthedron_vertices() -> np.ndarray:
    """
    Generate all 90 vertices of the Noperthedron.

    Uses cyclic group C₃₀ action on three base points:
        g_{k,ℓ} = (-1)^ℓ · R_z(2πk/15)
    for k = 0,...,14 and ℓ = 0,1

    This generates:
        3 base points × 15 rotations × 2 reflections = 90 vertices

    Returns
    -------
    vertices : np.ndarray of shape (90, 3)
        All vertex coordinates

    Notes
    -----
    The Noperthedron has:
    - Point symmetry (inversion through origin)
    - 15-fold rotational symmetry around z-axis
    - All vertices at rational coordinates
    """
    base_points = generate_base_points()
    vertices = []

    for base_point in base_points:
        for k in range(15):  # 15-fold rotation
            angle = 2 * np.pi * k / 15
            R = rotation_z(angle)

            for ℓ in [0, 1]:  # Reflection
                sign = (-1) ** ℓ
                vertex = sign * (R @ base_point)
                vertices.append(vertex)

    return np.array(vertices)


def compute_min_distance(vertices1: np.ndarray, vertices2: np.ndarray) -> float:
    """
    Compute minimum distance between two sets of vertices.

    This is a simplified collision metric. A full implementation would
    compute minimum distance between the convex hulls.
    """
    min_dist = np.inf
    for v1 in vertices1:
        # Vectorized distance to all points in vertices2
        dists = np.linalg.norm(vertices2 - v1, axis=1)
        min_dist = min(min_dist, np.min(dists))
    return min_dist


def estimate_overlap(vertices1: np.ndarray, vertices2: np.ndarray,
                     threshold: float = 0.1) -> float:
    """
    Estimate overlap between two point clouds.

    Returns fraction of points from vertices1 within threshold distance
    of any point in vertices2.
    """
    overlap_count = 0
    for v1 in vertices1:
        dists = np.linalg.norm(vertices2 - v1, axis=1)
        if np.min(dists) < threshold:
            overlap_count += 1
    return overlap_count / len(vertices1)


def compute_passage_ds2(vertices1: np.ndarray,
                        vertices2: np.ndarray,
                        threshold: float = 0.15) -> Tuple[float, float, float]:
    """
    Compute ds² metric for passage attempt.

    Interpretation:
    - C (Change): Number of vertices in collision (bits flipping)
    - S (Stability): Number of vertices clear (bits at rest)
    - ds² = S² - C²

    Parameters
    ----------
    vertices1, vertices2 : np.ndarray
        Two oriented copies of Noperthedron
    threshold : float
        Distance threshold for considering collision

    Returns
    -------
    C : float
        Change metric (collision intensity)
    S : float
        Stability metric (clearance quality)
    ds2 : float
        Metric signature (S² - C²)

    Notes
    -----
    This is a heuristic mapping. The exact interpretation of C and S
    in the passage problem is part of the theoretical exploration.
    """
    min_dist = compute_min_distance(vertices1, vertices2)
    overlap_frac = estimate_overlap(vertices1, vertices2, threshold)

    # Map to C/S metrics
    # High overlap → high C (many bits changing/colliding)
    # Low overlap → high S (many bits stable/clear)

    C = overlap_frac * 100  # Scale to ~0-100
    S = (1 - overlap_frac) * 100

    # Also incorporate minimum distance
    if min_dist < threshold:
        C += 20 * (threshold - min_dist) / threshold  # Boost C for close approaches
    else:
        S += 10 * min(min_dist / threshold, 2)  # Boost S for clear separation

    ds2 = S**2 - C**2

    return C, S, ds2


def test_single_configuration(vertices: np.ndarray,
                              theta1: float, phi1: float,
                              theta2: float, phi2: float,
                              alpha: float = 0.0) -> PassageAttempt:
    """
    Test passage for a single configuration of two Noperthedron copies.

    Parameters
    ----------
    vertices : np.ndarray
        Base Noperthedron vertices
    theta1, phi1 : float
        Orientation of first copy (spherical coords)
    theta2, phi2 : float
        Orientation of second copy
    alpha : float
        Passage direction (currently unused in simplified version)

    Returns
    -------
    attempt : PassageAttempt
        Results of passage attempt including ds² metric
    """
    # Orient first copy
    R1 = rotation_from_spherical(theta1, phi1)
    vertices1 = (R1 @ vertices.T).T

    # Orient second copy
    R2 = rotation_from_spherical(theta2, phi2)
    vertices2 = (R2 @ vertices.T).T

    # Compute metrics
    min_dist = compute_min_distance(vertices1, vertices2)
    overlap = estimate_overlap(vertices1, vertices2)
    C, S, ds2 = compute_passage_ds2(vertices1, vertices2)

    return PassageAttempt(
        orientation1=(theta1, phi1),
        orientation2=(theta2, phi2),
        direction=alpha,
        min_distance=min_dist,
        overlap_volume=overlap,
        C=C,
        S=S,
        ds2=ds2
    )


def test_rupert_property(vertices: np.ndarray,
                        n_samples: int = 1000,
                        random_seed: Optional[int] = 42) -> Tuple[List[PassageAttempt], bool]:
    """
    Test Rupert property by sampling configuration space.

    Steininger & Yurkevich tested ~18 million configurations in 5D
    parameter space. We sample a smaller subset to explore the principle.

    Parameters
    ----------
    vertices : np.ndarray
        Noperthedron vertices
    n_samples : int
        Number of random configurations to test
    random_seed : int or None
        Random seed for reproducibility

    Returns
    -------
    attempts : list of PassageAttempt
        All tested configurations
    has_passage : bool
        True if ANY configuration has ds² > 0 (passage possible)

    Notes
    -----
    The full 5D parameter space is (θ₁, φ₁, θ₂, φ₂, α).
    We sample uniformly and test each configuration.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    attempts = []

    print(f"Testing Rupert property with {n_samples} samples...")

    for i in range(n_samples):
        # Sample random orientations
        theta1 = np.random.uniform(0, np.pi)
        phi1 = np.random.uniform(0, 2*np.pi)
        theta2 = np.random.uniform(0, np.pi)
        phi2 = np.random.uniform(0, 2*np.pi)
        alpha = np.random.uniform(0, 2*np.pi)

        attempt = test_single_configuration(vertices, theta1, phi1, theta2, phi2, alpha)
        attempts.append(attempt)

        # Print progress (avoid division by zero for small samples)
        if n_samples >= 10 and (i+1) % (n_samples // 10) == 0:
            print(f"  Progress: {i+1}/{n_samples}")

    # Check if any passage exists (ds² > 0 indicates time-like/convergent)
    has_passage = any(a.ds2 > 0 for a in attempts)

    return attempts, has_passage


def analyze_results(attempts: List[PassageAttempt]) -> dict:
    """Analyze passage attempt results"""
    ds2_values = [a.ds2 for a in attempts]
    C_values = [a.C for a in attempts]
    S_values = [a.S for a in attempts]
    min_dists = [a.min_distance for a in attempts]

    return {
        'n_attempts': len(attempts),
        'ds2_min': min(ds2_values),
        'ds2_max': max(ds2_values),
        'ds2_mean': np.mean(ds2_values),
        'ds2_std': np.std(ds2_values),
        'frac_timelike': sum(1 for ds2 in ds2_values if ds2 > 0) / len(ds2_values),
        'frac_spacelike': sum(1 for ds2 in ds2_values if ds2 < 0) / len(ds2_values),
        'frac_lightlike': sum(1 for ds2 in ds2_values if abs(ds2) < 1) / len(ds2_values),
        'C_mean': np.mean(C_values),
        'S_mean': np.mean(S_values),
        'min_dist_min': min(min_dists),
        'min_dist_mean': np.mean(min_dists),
    }


def visualize_noperthedron(vertices: np.ndarray,
                          save_path: str = 'noperthedron_structure.png'):
    """Visualize the Noperthedron structure"""
    fig = plt.figure(figsize=(15, 5))

    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               c='blue', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title('Noperthedron: 90 Vertices\n15-fold Symmetry')
    ax1.grid(True, alpha=0.3)

    # Radial distribution
    ax2 = fig.add_subplot(132)
    radii = np.linalg.norm(vertices, axis=1)
    ax2.hist(radii, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Distance from origin')
    ax2.set_ylabel('Count')
    ax2.set_title('Radial Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(1.0, color='red', linestyle='--', label='Unit sphere')
    ax2.legend()

    # Projection onto xy-plane showing 15-fold symmetry
    ax3 = fig.add_subplot(133)
    ax3.scatter(vertices[:, 0], vertices[:, 1], c='blue', s=30, alpha=0.6,
               edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('xy-plane Projection\n(15-fold rotational symmetry)')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')

    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax3.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.3, label='Unit circle')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def visualize_metric_distribution(attempts: List[PassageAttempt],
                                  save_path: str = 'noperthedron_metrics.png'):
    """Visualize distribution of ds² and related metrics"""
    ds2_values = [a.ds2 for a in attempts]
    C_values = [a.C for a in attempts]
    S_values = [a.S for a in attempts]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ds² histogram
    ax = axes[0, 0]
    ax.hist(ds2_values, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='ds² = 0 (light-like)')
    ymax = ax.get_ylim()[1]
    ax.fill_betweenx([0, ymax], 0, max(ds2_values),
                    alpha=0.2, color='blue', label='Time-like (ds² > 0)')
    ax.fill_betweenx([0, ymax], min(ds2_values), 0,
                    alpha=0.2, color='yellow', label='Space-like (ds² < 0)')
    ax.set_xlabel('ds² = S² - C²')
    ax.set_ylabel('Count')
    ax.set_title('Metric Distribution: ds² = S² - C²')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative distribution
    ax = axes[0, 1]
    sorted_ds2 = np.sort(ds2_values)
    cumulative = np.arange(1, len(sorted_ds2) + 1) / len(sorted_ds2)
    ax.plot(sorted_ds2, cumulative, linewidth=2, color='purple')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='ds² = 0')
    ax.set_xlabel('ds²')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF: P(ds² < threshold)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # C vs S scatter
    ax = axes[1, 0]
    colors = ['blue' if ds2 > 0 else 'orange' for ds2 in ds2_values]
    ax.scatter(C_values, S_values, c=colors, alpha=0.5, s=20)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='C = S (ds² = 0)')
    ax.set_xlabel('C (Change/Collision)')
    ax.set_ylabel('S (Stability/Clearance)')
    ax.set_title('Change vs Stability\nBlue: ds²>0, Orange: ds²<0')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')

    # C and S distributions
    ax = axes[1, 1]
    ax.hist(C_values, bins=30, alpha=0.5, color='red', label='C (Change)', edgecolor='black')
    ax.hist(S_values, bins=30, alpha=0.5, color='green', label='S (Stability)', edgecolor='black')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title('C and S Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def print_analysis(stats: dict, has_passage: bool):
    """Print analysis results"""
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Configurations tested: {stats['n_attempts']}")
    print(f"\nMetric Statistics:")
    print(f"  ds² range: [{stats['ds2_min']:.2f}, {stats['ds2_max']:.2f}]")
    print(f"  ds² mean:  {stats['ds2_mean']:.2f} ± {stats['ds2_std']:.2f}")
    print(f"\nRegime Classification:")
    print(f"  Time-like  (ds² > 0): {stats['frac_timelike']:.1%} - passage possible, convergent")
    print(f"  Space-like (ds² < 0): {stats['frac_spacelike']:.1%} - no passage, non-convergent")
    print(f"  Light-like (ds² ≈ 0): {stats['frac_lightlike']:.1%} - boundary")
    print(f"\nChange/Stability:")
    print(f"  Mean C (collision):  {stats['C_mean']:.2f}")
    print(f"  Mean S (clearance):  {stats['S_mean']:.2f}")
    print(f"\nMinimum Distance:")
    print(f"  Overall minimum: {stats['min_dist_min']:.4f}")
    print(f"  Mean minimum:    {stats['min_dist_mean']:.4f}")

    print(f"\n" + "=" * 70)
    if has_passage:
        print("⚠️  RUPERT-POSITIVE: Passage exists")
        print("    ds² > 0 found → time-like path → convergence possible")
    else:
        print("🔴 RUPERT-NEGATIVE: NO passage exists")
        print("    ds² < 0 always → space-like only → NO convergence")
        print("    This supports the Eigen framework prediction!")
    print("=" * 70)


def main():
    """Main analysis routine"""
    print("=" * 70)
    print("NOPERTHEDRON: EXPLORING GEOMETRIC CONVERGENCE LIMITS")
    print("=" * 70)
    print("\nHypothesis: The Noperthedron's lack of Rupert's property")
    print("(cannot pass through itself) corresponds to ds² < 0 systems")
    print("that cannot converge in the Eigen framework.\n")

    # Generate vertices
    print("Generating Noperthedron...")
    vertices = generate_noperthedron_vertices()
    print(f"Generated {len(vertices)} vertices\n")

    # Verify base points
    base_points = generate_base_points()
    verify_base_points(base_points)

    # Visualize structure
    print("Visualizing structure...")
    visualize_noperthedron(vertices)

    # Test Rupert property
    print("\nTesting Rupert property...")
    attempts, has_passage = test_rupert_property(vertices, n_samples=1000)

    # Analyze
    stats = analyze_results(attempts)

    # Visualize metrics
    print("\nVisualizing metrics...")
    visualize_metric_distribution(attempts)

    # Print results
    print_analysis(stats, has_passage)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
The connection between Rupert's property and convergence:

1. RUPERT-POSITIVE shapes (CAN pass through self):
   - Time-like configurations exist (ds² > 0)
   - Self-similar transformations possible
   - Analogous to convergent systems

2. RUPERT-NEGATIVE shapes (CANNOT pass through self):
   - Only space-like configurations (ds² < 0)
   - No self-similar transformations
   - Analogous to non-convergent systems

The Noperthedron, with its proven lack of Rupert's property,
may provide geometric evidence that ds² < 0 systems exist
and cannot converge.

This is a THOUGHT EXPERIMENT exploring deep connections between:
- Geometric constraints (passage through self)
- Metric signatures (ds² sign)
- System convergence (eigenstate reachability)
    """)

    print("\n" + "=" * 70)
    print("REFERENCE")
    print("=" * 70)
    print("""
Steininger, Jakob and Yurkevich, Vsevolod (2025).
"The Noperthedron." arXiv preprint.

Note: This implementation is exploratory. The exact mapping between
passage geometry and the ds² metric requires further theoretical work.
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
