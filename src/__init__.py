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
    lorentz_boost_from_rapidity,
    lorentz_boost_matrix,
    lorentz_to_stereo,
    proper_distance,
    proper_time,
    regime_classification,
    stereo_to_lorentz,
    verify_lorentz_invariance,
)
from .eigen_noperthedron import (
    analyze_results,
    compute_passage_ds2_discrete,
    generate_noperthedron_vertices,
    hamming_distance,
    spatial_overlap,
    test_rupert_property,
    vertices_to_bitstring,
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
    "test_rupert_property",
    "analyze_results",
    "vertices_to_bitstring",
    "spatial_overlap",
    "hamming_distance",
    "compute_passage_ds2_discrete",
    # Lorentz transformation framework
    "lorentz_boost_matrix",
    "lorentz_boost_from_rapidity",
    "apply_boost",
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
]

__version__ = "1.0.0"
__author__ = "Jon McReynolds"
