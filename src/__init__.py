# -*- coding: utf-8 -*-
"""
Eigen: Geometric Robot Control Framework

A minimal control framework where motion emerges from gradient descent
on a geometric objective function.

Core equation: Q_{t+1} = Q_t - η∇ds²(Q)
"""

from .eigen_core import (
    forward_kinematics,
    jacobian,
    compute_ds2,
    compute_gradient,
    compute_change_stability
)

from .eigen_arm_control import run_arm_simulation
from .eigen_xor_rotation import run_xor_simulation

__all__ = [
    'forward_kinematics',
    'jacobian',
    'compute_ds2',
    'compute_gradient',
    'compute_change_stability',
    'run_arm_simulation',
    'run_xor_simulation'
]

__version__ = '1.0.0'
__author__ = 'Jon McReynolds'
