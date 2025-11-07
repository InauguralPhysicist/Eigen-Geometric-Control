"""
Unit tests for Eigen core functions

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '.')

from src.eigen_core import (
    forward_kinematics,
    jacobian,
    compute_ds2,
    compute_gradient,
    compute_change_stability
)


class TestForwardKinematics:
    """Test forward kinematics computations"""
    
    def test_straight_configuration(self):
        """Test FK for straight-out configuration (0, 0)"""
        x, y = forward_kinematics(0, 0, L1=1.0, L2=1.0)
        assert np.isclose(x, 2.0), f"Expected x=2.0, got {x}"
        assert np.isclose(y, 0.0, atol=1e-10), f"Expected y=0.0, got {y}"
    
    def test_right_angle_configuration(self):
        """Test FK for 90-degree configuration"""
        x, y = forward_kinematics(np.pi/2, 0, L1=1.0, L2=1.0)
        assert np.isclose(x, 0.0, atol=1e-10), f"Expected x=0.0, got {x}"
        assert np.isclose(y, 2.0), f"Expected y=2.0, got {y}"
    
    def test_default_link_lengths(self):
        """Test FK with default link lengths"""
        x, y = forward_kinematics(0, 0)  # Uses L1=0.9, L2=0.9
        assert np.isclose(x, 1.8), f"Expected x=1.8, got {x}"
        assert np.isclose(y, 0.0, atol=1e-10), f"Expected y=0.0, got {y}"


class TestJacobian:
    """Test Jacobian matrix computation"""
    
    def test_jacobian_shape(self):
        """Test Jacobian has correct dimensions (2x2)"""
        J = jacobian(0.5, 0.3)
        assert J.shape == (2, 2), f"Expected shape (2,2), got {J.shape}"
    
    def test_jacobian_numerical_consistency(self):
        """Test Jacobian against numerical differentiation"""
        theta1, theta2 = 0.5, 0.3
        J = jacobian(theta1, theta2)
        
        # Small perturbation for numerical derivative
        eps = 1e-7
        x0, y0 = forward_kinematics(theta1, theta2)
        x1, y1 = forward_kinematics(theta1 + eps, theta2)
        x2, y2 = forward_kinematics(theta1, theta2 + eps)
        
        # Numerical derivatives
        dx_dtheta1 = (x1 - x0) / eps
        dy_dtheta1 = (y1 - y0) / eps
        dx_dtheta2 = (x2 - x0) / eps
        dy_dtheta2 = (y2 - y0) / eps
        
        # Compare with analytical Jacobian
        assert np.isclose(J[0, 0], dx_dtheta1, rtol=1e-5)
        assert np.isclose(J[1, 0], dy_dtheta1, rtol=1e-5)
        assert np.isclose(J[0, 1], dx_dtheta2, rtol=1e-5)
        assert np.isclose(J[1, 1], dy_dtheta2, rtol=1e-5)


class TestComputeDS2:
    """Test ds² objective function computation"""
    
    def test_ds2_at_target_no_obstacle(self):
        """Test ds² at target with no obstacle should be minimal"""
        theta1, theta2 = 0.5, 0.3
        x, y = forward_kinematics(theta1, theta2)
        
        ds2_total, components = compute_ds2(
            theta1, theta2,
            target=np.array([x, y]),  # At target
            obstacle_center=np.array([10.0, 10.0]),  # Far away
            obstacle_radius=0.1,
            Go=0.0,  # No obstacle repulsion
            lam=0.0  # No regularization
        )
        
        assert ds2_total < 1e-10, f"Expected ds²≈0 at target, got {ds2_total}"
        assert components['target_term'] < 1e-10
        assert components['obs_term'] == 0.0
        assert components['reg_term'] == 0.0
    
    def test_ds2_components_present(self):
        """Test that all ds² components are returned"""
        ds2_total, components = compute_ds2(
            0, 0,
            target=np.array([1.0, 0.0]),
            obstacle_center=np.array([0.5, 0.5]),
            obstacle_radius=0.2
        )
        
        assert 'target_term' in components
        assert 'obs_term' in components
        assert 'reg_term' in components
        assert 'd_obs' in components
    
    def test_ds2_non_negative(self):
        """Test that ds² is always non-negative"""
        ds2_total, _ = compute_ds2(
            0.5, 0.3,
            target=np.array([1.0, 0.5]),
            obstacle_center=np.array([0.8, 0.2]),
            obstacle_radius=0.1
        )
        assert ds2_total >= 0, f"ds² must be non-negative, got {ds2_total}"


class TestComputeGradient:
    """Test gradient computation"""
    
    def test_gradient_at_target_vanishes(self):
        """Test gradient is near zero at target (no obstacle, no reg)"""
        theta1, theta2 = 0.5, 0.3
        x, y = forward_kinematics(theta1, theta2)
        
        grad, grad_norm = compute_gradient(
            theta1, theta2,
            target=np.array([x, y]),
            obstacle_center=np.array([10, 10]),  # Far away
            obstacle_radius=0.1,
            Go=0.0,  # No obstacle
            lam=0.0  # No regularization
        )
        
        assert grad_norm < 1e-6, f"Gradient should vanish at target, got {grad_norm}"
    
    def test_gradient_shape(self):
        """Test gradient has correct shape"""
        grad, grad_norm = compute_gradient(
            0.5, 0.3,
            target=np.array([1.0, 0.5]),
            obstacle_center=np.array([0.8, 0.2]),
            obstacle_radius=0.1
        )
        assert grad.shape == (2,), f"Expected gradient shape (2,), got {grad.shape}"
        assert isinstance(grad_norm, float)


class TestChangeStability:
    """Test change and stability metrics"""
    
    def test_large_change(self):
        """Test detection of large changes"""
        delta = np.array([0.1, 0.2])
        C, S, ds2_CS = compute_change_stability(delta, eps_change=1e-4)
        
        assert C == 2, f"Both joints changing, expected C=2, got {C}"
        assert S == 0, f"No joints stable, expected S=0, got {S}"
        assert ds2_CS == -4, f"Expected ds²=-4, got {ds2_CS}"
    
    def test_no_change(self):
        """Test detection of stability (no change)"""
        delta = np.array([1e-5, 1e-6])
        C, S, ds2_CS = compute_change_stability(delta, eps_change=1e-4)
        
        assert C == 0, f"No joints changing, expected C=0, got {C}"
        assert S == 2, f"Both joints stable, expected S=2, got {S}"
        assert ds2_CS == 4, f"Expected ds²=4, got {ds2_CS}"
    
    def test_partial_change(self):
        """Test detection of partial change"""
        delta = np.array([0.1, 1e-6])
        C, S, ds2_CS = compute_change_stability(delta, eps_change=1e-4)
        
        assert C == 1, f"One joint changing, expected C=1, got {C}"
        assert S == 1, f"One joint stable, expected S=1, got {S}"
        assert ds2_CS == 0, f"Expected ds²=0, got {ds2_CS}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
