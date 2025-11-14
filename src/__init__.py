# -*- coding: utf-8 -*-
"""
Eigen: Geometric Robot Control Framework

A minimal control framework where motion emerges from gradient descent
on a geometric objective function.

Core equation: Q_{t+1} = Q_t - η∇ds²(Q)

Extensions:
- Lorentz boost framework for multi-sensor fusion
- Stereo vision integration via relativistic transformations
"""

from .eigen_arm_control import run_arm_simulation
from .eigen_core import (
    compute_change_stability,
    compute_ds2,
    compute_gradient,
    forward_kinematics,
    jacobian,
)
from .eigen_lorentz import (
    LorentzState,
    apply_boost,
    boost_lorentz_state,
    change_stability_to_lorentz,
    create_lorentz_state,
    disparity_to_rapidity,
    estimate_target_velocity,
    inverse_boost,
    lorentz_boost_from_rapidity,
    lorentz_boost_matrix,
    lorentz_to_stereo,
    moving_target_control_step,
    proper_distance,
    proper_time,
    regime_classification,
    stereo_to_lorentz,
    verify_lorentz_invariance,
)
from .eigen_noperthedron import (
    analyze_results,
    check_rupert_property,
    compute_passage_ds2_discrete,
    generate_noperthedron_vertices,
    hamming_distance,
    run_single_passage_test,
    spatial_overlap,
    vertices_to_bitstring,
)
from .eigen_similarity import (
    adaptive_control_parameters,
    compare_self_similarity,
    detect_oscillation,
    detect_oscillation_frequency,
    frequency_selective_damping,
    lightlike_damping_factor,
    lorentz_similarity,
    regime_from_similarity,
    standard_cosine_similarity,
)
from .eigen_meta_control import (
    meta_update_parameters,
    observe_parameter_performance,
    self_tuning_control_step,
)
from .eigen_multi_agent import (
    check_convergence,
    collective_lightlike_observer,
    multi_agent_step,
    paradox_mediated_damping,
    simulate_multi_agent,
)
from .eigen_weak_measurement import (
    accumulate_weak_measurements,
    apply_weak_measurement,
    coherence_metric,
    stereo_weak_measurement,
    weak_measurement_operator,
    weak_stereo_control_step,
)
from .eigen_decomposition import (
    coherence_score,
    coherent_control_step,
    compute_autocorrelation,
    decompose_signal,
    filtered_observation,
)
from .eigen_adaptive import (
    adaptive_control_step,
    detect_system_parameters,
    estimate_characteristic_velocity,
    estimate_natural_timescale,
    estimate_oscillation_threshold,
    renormalize_lorentz_state,
)
from .eigen_xor_rotation import run_xor_simulation

__all__ = [
    # Core geometric functions
    "forward_kinematics",
    "jacobian",
    "compute_ds2",
    "compute_gradient",
    "compute_change_stability",
    # Simulation runners
    "run_arm_simulation",
    "run_xor_simulation",
    # Noperthedron analysis
    "generate_noperthedron_vertices",
    "check_rupert_property",
    "run_single_passage_test",
    "analyze_results",
    "vertices_to_bitstring",
    "spatial_overlap",
    "hamming_distance",
    "compute_passage_ds2_discrete",
    # Lorentz transformation framework
    "lorentz_boost_matrix",
    "lorentz_boost_from_rapidity",
    "apply_boost",
    "inverse_boost",
    "LorentzState",
    "create_lorentz_state",
    "boost_lorentz_state",
    "stereo_to_lorentz",
    "lorentz_to_stereo",
    "change_stability_to_lorentz",
    "disparity_to_rapidity",
    "regime_classification",
    "proper_time",
    "proper_distance",
    "verify_lorentz_invariance",
    # Moving target tracking
    "estimate_target_velocity",
    "moving_target_control_step",
    # Lorentz-invariant similarity (lightlike observer)
    "lorentz_similarity",
    "standard_cosine_similarity",
    "detect_oscillation",
    "detect_oscillation_frequency",
    "frequency_selective_damping",
    "lightlike_damping_factor",
    "compare_self_similarity",
    "regime_from_similarity",
    "adaptive_control_parameters",
    # Meta-control (self-simulation)
    "observe_parameter_performance",
    "meta_update_parameters",
    "self_tuning_control_step",
    # Multi-agent coordination (paradox-mediated)
    "collective_lightlike_observer",
    "paradox_mediated_damping",
    "multi_agent_step",
    "check_convergence",
    "simulate_multi_agent",
    # Weak measurement (stereo vision)
    "weak_measurement_operator",
    "apply_weak_measurement",
    "accumulate_weak_measurements",
    "stereo_weak_measurement",
    "weak_stereo_control_step",
    "coherence_metric",
    # Coherent/incoherent decomposition
    "compute_autocorrelation",
    "coherence_score",
    "decompose_signal",
    "filtered_observation",
    "coherent_control_step",
    # Adaptive/dynamic parameter detection
    "estimate_characteristic_velocity",
    "estimate_oscillation_threshold",
    "estimate_natural_timescale",
    "detect_system_parameters",
    "adaptive_control_step",
    "renormalize_lorentz_state",
]

__version__ = "1.0.0"
__author__ = "Jon McReynolds"
