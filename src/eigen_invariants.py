# -*- coding: utf-8 -*-
"""
Eigen: Geometric Control Framework - Invariant Computations

Provides geometric property extraction from lightlike invariants.
This module treats space/time/light consistently to allow automatic
computation of surface area, volume, and curvature from opposing pairs.
"""

from typing import Dict

import numpy as np


def compute_geometric_properties(A: float, B: float) -> Dict[str, float]:
    """
    Extract all geometric properties from opposing pair (A, B).

    Given any two opposing quantities, this function computes the lightlike
    invariant I = (A - B)² and extracts all geometric properties that scale
    from this fundamental measure.

    The algorithm treats space/time/light consistently, allowing automatic
    computation of:
    - Problem scale (radius r = √I)
    - Problem complexity (surface area = 4πr²)
    - Search space size (volume = 4/3πr³)
    - Problem conditioning (curvature = 1/r)

    Parameters
    ----------
    A : float
        First component of the opposing pair
    B : float
        Second component of the opposing pair

    Returns
    -------
    properties : dict
        Dictionary containing:
        - radius : float
            Problem scale r = √I (1D measure)
        - surface_area : float
            Complexity measure 4πr² (2D measure)
        - volume : float
            Search space size 4/3πr³ (3D measure)
        - curvature : float
            Problem conditioning 1/r (inverse scale)
        - status : str
            Either "equilibrium" (I ≈ 0) or "active" (I > 0)

    Notes
    -----
    The invariant I = (A - B)² is the lightlike invariant, analogous to
    the spacetime interval in special relativity. All geometric properties
    derive from this single quantity.

    When I ≈ 0, the system is at equilibrium and curvature becomes infinite.
    For active problems (I > 0), all properties scale deterministically:
    - Linear scale: r ∝ √I
    - Area scale: A ∝ I
    - Volume scale: V ∝ I^(3/2)
    - Curvature: κ ∝ 1/√I

    Examples
    --------
    >>> props = compute_geometric_properties(5.0, 2.0)
    >>> print(f"Radius: {props['radius']:.2f}")
    Radius: 3.00
    >>> print(f"Volume: {props['volume']:.2f}")
    Volume: 113.10

    >>> # Equilibrium case
    >>> props = compute_geometric_properties(5.0, 5.0)
    >>> print(props['status'])
    equilibrium

    References
    ----------
    The geometric interpretation is based on treating all problems as
    spheres in an abstract geometric space, where only the radius varies.
    This universality principle simplifies convergence analysis.
    """
    # The invariant contains everything
    I = (A - B) ** 2

    if I < 1e-12:  # Near equilibrium
        return {
            "radius": 0.0,
            "surface_area": 0.0,
            "volume": 0.0,
            "curvature": np.inf,
            "status": "equilibrium",
        }

    # All properties derive from r = sqrt(I)
    r = np.sqrt(I)

    # Geometric properties (treating space/time/light consistently)
    surface_area = 4 * np.pi * I  # 4πr²
    volume = (4 / 3) * np.pi * I ** (3 / 2)  # (4/3)πr³
    curvature = 1 / r  # κ = 1/r

    return {
        "radius": r,
        "surface_area": surface_area,
        "volume": volume,
        "curvature": curvature,
        "status": "active",
    }
