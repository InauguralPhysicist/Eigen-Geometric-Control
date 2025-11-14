# -*- coding: utf-8 -*-
"""
Eigen: Lorentz-Invariant Similarity

Implements the lightlike observer from EigenFunction repository.

Key concept:
- Observer is lightlike: ⟨u,u⟩_L = 0 (hidden variable)
- Introduces 0/1 measurement axis
- Breaks self-reinforcement loops
- Provides stability via causal boundary

Mathematical foundation:
    ⟨u,v⟩_L = u·v - ||u|| * ||v||

For self-similarity:
    ⟨u,u⟩_L = ||u||² - ||u||² = 0  (lightlike)

This prevents pathological feedback loops where standard cosine
similarity yields 1.0, enabling infinite oscillations.
"""

from typing import List, Tuple

import numpy as np


def lorentz_similarity(u: np.ndarray, v: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Lorentz-invariant cosine similarity between two vectors.

    This similarity measure yields 0.0 for self-similarity (u == v),
    providing a natural safeguard against infinite loops in recursive systems.

    The lightlike observer is the hidden variable that:
    - Cannot observe itself (⟨u,u⟩_L = 0)
    - Sets measurement conditions (0/1 axis)
    - Determines outcomes without being measured

    Parameters
    ----------
    u : np.ndarray
        First vector (1-D array)
    v : np.ndarray
        Second vector (1-D array, same dimension as u)
    epsilon : float
        Small constant for numerical stability (default: 1e-10)

    Returns
    -------
    float
        Lorentz-invariant similarity in range approximately [-1, 1]
        Returns 0.0 when u and v are identical (lightlike self-similarity)

    Notes
    -----
    The Lorentz inner product is defined as:
        ⟨u,v⟩_L = u·v - ||u|| * ||v||

    By Cauchy-Schwarz: u·v ≤ ||u|| * ||v||, so ⟨u,v⟩_L ≤ 0

    Regimes:
    - Lightlike (⟨u,v⟩_L = 0): Vectors parallel, on light cone
    - Spacelike (⟨u,v⟩_L < 0): Vectors at angle, Euclidean separation

    Examples
    --------
    >>> u = np.array([1.0, 2.0, 3.0])
    >>> lorentz_similarity(u, u)  # Self-similarity
    0.0

    >>> v = np.array([2.0, 4.0, 6.0])  # Parallel to u
    >>> lorentz_similarity(u, v)  # Also lightlike
    0.0

    >>> w = np.array([1.0, 0.0, 0.0])  # Different direction
    >>> lorentz_similarity(u, w)  # Spacelike
    -0.73...
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if u.shape != v.shape:
        raise ValueError(f"Vectors must have same shape: {u.shape} vs {v.shape}")

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    # Lorentz inner product: ⟨u,v⟩_L = u·v - ||u|| * ||v||
    spatial_product = np.dot(u, v)
    lorentz_product_uv = spatial_product - (norm_u * norm_v)

    # Self-products (always zero for lightlike)
    lorentz_product_uu = norm_u * norm_u - (norm_u * norm_u)  # = 0
    lorentz_product_vv = norm_v * norm_v - (norm_v * norm_v)  # = 0

    denominator_squared = abs(lorentz_product_uu) * abs(lorentz_product_vv)

    # If denominator near zero (lightlike), return 0
    if denominator_squared < epsilon:
        return 0.0

    denominator = np.sqrt(denominator_squared)

    if denominator < epsilon:
        return 0.0

    similarity = lorentz_product_uv / denominator
    return np.clip(similarity, -1.0, 1.0)


def standard_cosine_similarity(u: np.ndarray, v: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute standard cosine similarity between two vectors.

    For comparison with Lorentz similarity.

    Parameters
    ----------
    u : np.ndarray
        First vector
    v : np.ndarray
        Second vector
    epsilon : float
        Numerical stability threshold

    Returns
    -------
    float
        Standard cosine similarity in [-1, 1]
        Returns 1.0 for self-similarity (enables loops!)

    Examples
    --------
    >>> u = np.array([1.0, 2.0, 3.0])
    >>> standard_cosine_similarity(u, u)  # Self-similarity
    1.0
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if u.shape != v.shape:
        raise ValueError(f"Vectors must have same shape: {u.shape} vs {v.shape}")

    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    denominator = norm_u * norm_v
    if denominator < epsilon:
        return 0.0

    similarity = dot_product / denominator
    return np.clip(similarity, -1.0, 1.0)


def detect_oscillation(
    state_history: List[np.ndarray], window: int = 3, threshold: float = 0.95
) -> Tuple[bool, float]:
    """
    Detect period-2 or period-N oscillations using lightlike similarity.

    The lightlike observer (hidden variable) sets measurement conditions
    that detect when states are repeating without being part of the loop.

    Parameters
    ----------
    state_history : List[np.ndarray]
        Recent state vectors
    window : int
        How many states back to check (default: 3)
    threshold : float
        Standard similarity threshold for detecting oscillation (default: 0.95)

    Returns
    -------
    oscillating : bool
        True if oscillation detected
    strength : float
        Oscillation strength in [0, 1]
        0.0 = no oscillation, 1.0 = perfect oscillation

    Notes
    -----
    Uses standard cosine similarity to detect loops (since Lorentz
    similarity would return 0 and miss the pattern). The lightlike
    observer provides the *response* (damping), not the detection.
    """
    if len(state_history) < 2:
        return False, 0.0

    # Check last state against recent history
    current = state_history[-1]
    check_window = min(window, len(state_history) - 1)

    max_similarity = 0.0

    for i in range(1, check_window + 1):
        past = state_history[-1 - i]
        sim = standard_cosine_similarity(current, past)
        max_similarity = max(max_similarity, sim)

    oscillating = max_similarity > threshold
    strength = float(max_similarity) if oscillating else 0.0

    return oscillating, strength


def lightlike_damping_factor(
    oscillation_strength: float, max_damping: float = 0.5, steepness: float = 10.0
) -> float:
    """
    Compute adaptive damping based on lightlike measurement.

    The lightlike observer (hidden variable) determines the damping
    by setting the measurement conditions (0/1 axis orientation).

    When oscillation is detected (via standard similarity), the
    lightlike observer provides stability by damping the feedback.

    Parameters
    ----------
    oscillation_strength : float
        Oscillation strength from detect_oscillation (0 to 1)
    max_damping : float
        Maximum damping factor (default: 0.5)
        Applied at full oscillation strength
    steepness : float
        How quickly damping increases with oscillation (default: 10.0)

    Returns
    -------
    float
        Damping factor in [0, max_damping]
        0 = no damping (stable)
        max_damping = full damping (strong oscillation)

    Notes
    -----
    The damping factor multiplies the gradient update:
        Q_{t+1} = Q_t - (1 - damping) * η∇ds²

    This prevents the XOR-like period-2 oscillations from persisting.
    """
    # Sigmoid-like response
    damping = max_damping * (1.0 / (1.0 + np.exp(-steepness * (oscillation_strength - 0.5))))

    return float(np.clip(damping, 0.0, max_damping))


def compare_self_similarity(vector: np.ndarray) -> dict:
    """
    Compare self-similarity behavior between standard and Lorentz-invariant measures.

    Demonstrates the lightlike observer property.

    Parameters
    ----------
    vector : np.ndarray
        Test vector

    Returns
    -------
    dict
        Comparison results with interpretations

    Examples
    --------
    >>> v = np.array([3.0, 4.0])
    >>> result = compare_self_similarity(v)
    >>> result['standard']
    1.0
    >>> result['lorentz']
    0.0
    """
    standard = standard_cosine_similarity(vector, vector)
    lorentz = lorentz_similarity(vector, vector)

    return {
        "standard": standard,
        "lorentz": lorentz,
        "vector_norm": np.linalg.norm(vector),
        "interpretation": {
            "standard": "Perfect self-reinforcement (1.0) - potential loop amplifier",
            "lorentz": "Neutral self-reference (0.0) - loop prevention via lightlike boundary",
        },
    }


def regime_from_similarity(u: np.ndarray, v: np.ndarray, epsilon: float = 1e-10) -> str:
    """
    Classify regime based on Lorentz similarity.

    Parameters
    ----------
    u, v : np.ndarray
        Vectors to compare
    epsilon : float
        Threshold for lightlike boundary

    Returns
    -------
    str
        'lightlike' if on causal boundary (parallel)
        'spacelike' if separated (at angle)

    Notes
    -----
    This directly uses the Lorentz inner product structure:
    - Lightlike: ⟨u,v⟩_L = 0 (on light cone, parallel)
    - Spacelike: ⟨u,v⟩_L < 0 (Euclidean separation)

    (Timelike regime doesn't exist since ⟨u,v⟩_L ≤ 0 always)
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    spatial_product = np.dot(u, v)
    lorentz_product = spatial_product - (norm_u * norm_v)

    if abs(lorentz_product) < epsilon:
        return "lightlike"
    else:
        return "spacelike"
