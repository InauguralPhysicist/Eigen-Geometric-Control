"""
Meta-Control via Self-Simulation

Demonstrates paradox-mediated computation: the system observes its own
control parameters and auto-tunes them using the lightlike observer.

Key insight:
- System observes itself: a = b (self-reference creates paradox)
- Must adjust itself: b ≠ a (observation requires difference)
- Lightlike observer mediates paradox at ds² = 0
- Enables self-simulation without infinite loops

This is the first concrete implementation of TC+ computation.
"""

from typing import List, Tuple

import numpy as np

from .eigen_similarity import detect_oscillation, standard_cosine_similarity


def observe_parameter_performance(
    control_params: dict,
    trajectory_history: List[np.ndarray],
    target: np.ndarray,
    min_history: int = 10,
) -> Tuple[bool, float, dict]:
    """
    System observes how its own parameters are performing.

    This is SELF-OBSERVATION: the control system looks at itself.
    - a = current parameter state
    - b = current parameter state (self-reference)
    - Paradox: How can a system observe itself without infinite regress?
    - Answer: Lightlike observer mediates via oscillation detection

    Args:
        control_params: Current control parameters {eta, damping, threshold}
        trajectory_history: Recent trajectory under these parameters
        target: Target state
        min_history: Minimum trajectory length to evaluate

    Returns:
        - needs_adjustment: bool - True if parameters causing issues
        - performance_score: float - Quality metric [0, 1]
        - diagnosis: dict - What's wrong and suggested corrections
    """
    if len(trajectory_history) < min_history:
        return False, 1.0, {"status": "insufficient_data"}

    recent_trajectory = trajectory_history[-min_history:]

    # Metric 1: Are we oscillating? (lightlike observer check)
    oscillating, similarity = detect_oscillation(
        recent_trajectory,
        window=min(5, len(recent_trajectory) - 1),
        threshold=control_params.get("oscillation_threshold", 0.92),
    )

    # Metric 2: Are we converging toward target?
    distances = [np.linalg.norm(state - target) for state in recent_trajectory]
    converging = distances[-1] < distances[0]  # Getting closer?
    convergence_rate = (distances[0] - distances[-1]) / (distances[0] + 1e-9)

    # Metric 3: Is motion smooth or erratic?
    if len(recent_trajectory) >= 3:
        velocities = [
            recent_trajectory[i] - recent_trajectory[i - 1]
            for i in range(1, len(recent_trajectory))
        ]
        velocity_changes = [
            np.linalg.norm(velocities[i] - velocities[i - 1])
            for i in range(1, len(velocities))
        ]
        smoothness = 1.0 / (1.0 + np.mean(velocity_changes))
    else:
        smoothness = 1.0

    # Compute overall performance score
    oscillation_penalty = 0.5 if oscillating else 0.0
    convergence_score = max(0.0, convergence_rate) if converging else 0.0

    performance_score = (
        (1.0 - oscillation_penalty) * 0.4
        + convergence_score * 0.4
        + smoothness * 0.2
    )

    # Diagnose what needs adjustment
    diagnosis = {
        "oscillating": oscillating,
        "oscillation_similarity": float(similarity),
        "converging": converging,
        "convergence_rate": float(convergence_rate),
        "smoothness": float(smoothness),
        "suggested_adjustments": {},
    }

    needs_adjustment = False

    if oscillating:
        # Too much energy, need more damping or lower eta
        needs_adjustment = True
        diagnosis["suggested_adjustments"]["increase_damping"] = True
        diagnosis["suggested_adjustments"]["decrease_eta"] = True
        diagnosis["issue"] = "oscillation_detected"

    if not converging and len(trajectory_history) > min_history:
        # Not making progress, might need higher eta or less damping
        needs_adjustment = True
        diagnosis["suggested_adjustments"]["increase_eta"] = True
        diagnosis["suggested_adjustments"]["decrease_damping"] = True
        diagnosis["issue"] = "stagnation_detected"

    if smoothness < 0.5:
        # Erratic motion, need smoother control
        needs_adjustment = True
        diagnosis["suggested_adjustments"]["decrease_eta"] = True
        diagnosis["issue"] = "erratic_motion"

    return needs_adjustment, float(performance_score), diagnosis


def meta_update_parameters(
    current_params: dict,
    diagnosis: dict,
    meta_eta: float = 0.1,
) -> dict:
    """
    Update control parameters based on self-observation.

    This is the META-LEVEL UPDATE: parameters adjust themselves.
    - Current params = a
    - Updated params = b
    - a ≠ b (change required for adaptation)
    - But updating self creates paradox (a observing a)
    - Lightlike observer has already mediated via diagnosis

    Args:
        current_params: Current {eta, max_damping, oscillation_threshold}
        diagnosis: Output from observe_parameter_performance()
        meta_eta: Learning rate for parameter updates (slow adaptation)

    Returns:
        Updated parameters dict
    """
    updated = current_params.copy()
    suggestions = diagnosis.get("suggested_adjustments", {})

    # Extract current values with defaults
    eta = updated.get("eta", 0.1)
    max_damping = updated.get("max_damping", 0.5)
    osc_threshold = updated.get("oscillation_threshold", 0.92)

    # Apply suggested adjustments with smooth updates
    if suggestions.get("decrease_eta"):
        eta = eta * (1.0 - meta_eta * 0.5)  # Decrease by up to 5%
        eta = max(0.001, eta)  # Floor

    if suggestions.get("increase_eta"):
        eta = eta * (1.0 + meta_eta * 0.5)  # Increase by up to 5%
        eta = min(0.5, eta)  # Ceiling

    if suggestions.get("increase_damping"):
        max_damping = max_damping + meta_eta * 0.1
        max_damping = min(0.95, max_damping)  # Ceiling

    if suggestions.get("decrease_damping"):
        max_damping = max_damping - meta_eta * 0.1
        max_damping = max(0.0, max_damping)  # Floor

    # Update the dict
    updated["eta"] = float(eta)
    updated["max_damping"] = float(max_damping)
    updated["oscillation_threshold"] = float(osc_threshold)

    return updated


def self_tuning_control_step(
    current_state: np.ndarray,
    target: np.ndarray,
    control_params: dict,
    trajectory_history: List[np.ndarray],
    enable_meta_control: bool = True,
    meta_eta: float = 0.1,
) -> Tuple[dict, dict]:
    """
    Single step of self-tuning control.

    This demonstrates SELF-SIMULATION WITHOUT INFINITE LOOPS:
    1. System observes itself (self-reference)
    2. Lightlike observer detects if self-observation reveals problems
    3. System adjusts itself based on observation
    4. No infinite loop because lightlike observer mediates the paradox

    Args:
        current_state: Current robot state
        target: Target state
        control_params: Current control parameters
        trajectory_history: Recent trajectory
        enable_meta_control: If True, allow self-tuning
        meta_eta: Learning rate for parameter updates

    Returns:
        - updated_params: Potentially adjusted control parameters
        - meta_info: Diagnostic information about self-observation
    """
    meta_info = {
        "meta_control_active": enable_meta_control,
        "adjustment_made": False,
        "performance_score": 1.0,
    }

    if not enable_meta_control:
        return control_params, meta_info

    # SELF-OBSERVATION: System looks at itself
    needs_adjustment, performance, diagnosis = observe_parameter_performance(
        control_params, trajectory_history, target
    )

    meta_info["performance_score"] = performance
    meta_info["diagnosis"] = diagnosis
    meta_info["needs_adjustment"] = needs_adjustment

    # If lightlike observer detected issues, adjust parameters
    if needs_adjustment:
        updated_params = meta_update_parameters(
            control_params, diagnosis, meta_eta=meta_eta
        )
        meta_info["adjustment_made"] = True
        meta_info["param_changes"] = {
            key: {
                "old": control_params.get(key),
                "new": updated_params.get(key),
                "delta": updated_params.get(key, 0) - control_params.get(key, 0),
            }
            for key in ["eta", "max_damping", "oscillation_threshold"]
            if key in control_params or key in updated_params
        }
        return updated_params, meta_info

    # No adjustment needed - system in equilibrium
    return control_params, meta_info
