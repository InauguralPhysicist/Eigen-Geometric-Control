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


class TestErrorHandling:
    """Tests for error handling and input validation"""
    
    def test_forward_kinematics_negative_link(self):
        """Test FK rejects negative link lengths"""
        with pytest.raises(ValueError, match="Link lengths must be positive"):
            forward_kinematics(0, 0, L1=-1.0, L2=1.0)
        
        with pytest.raises(ValueError, match="Link lengths must be positive"):
            forward_kinematics(0, 0, L1=1.0, L2=-1.0)
    
    def test_forward_kinematics_non_numeric(self):
        """Test FK rejects non-numeric inputs"""
        with pytest.raises(TypeError, match="All inputs must be numeric"):
            forward_kinematics("0", 0)
        
        with pytest.raises(TypeError, match="All inputs must be numeric"):
            forward_kinematics(0, "0")
    
    def test_jacobian_validation(self):
        """Test Jacobian validates inputs"""
        with pytest.raises(ValueError, match="Link lengths must be positive"):
            jacobian(0, 0, L1=0, L2=1.0)
        
        with pytest.raises(TypeError, match="All inputs must be numeric"):
            jacobian(None, 0)
    
    def test_compute_ds2_invalid_shapes(self):
        """Test ds² validates array shapes"""
        with pytest.raises(ValueError, match="target must have shape"):
            compute_ds2(0, 0, target=[1.0], obstacle_center=[0.5, 0.5], obstacle_radius=0.2)
        
        with pytest.raises(ValueError, match="obstacle_center must have shape"):
            compute_ds2(0, 0, target=[1.0, 0.0], obstacle_center=[0.5], obstacle_radius=0.2)
    
    def test_compute_ds2_negative_params(self):
        """Test ds² rejects negative parameters"""
        with pytest.raises(ValueError, match="obstacle_radius must be non-negative"):
            compute_ds2(0, 0, target=[1.0, 0.0], obstacle_center=[0.5, 0.5], obstacle_radius=-0.1)
        
        with pytest.raises(ValueError, match="Go must be non-negative"):
            compute_ds2(0, 0, target=[1.0, 0.0], obstacle_center=[0.5, 0.5], 
                       obstacle_radius=0.2, Go=-1.0)
        
        with pytest.raises(ValueError, match="lam must be non-negative"):
            compute_ds2(0, 0, target=[1.0, 0.0], obstacle_center=[0.5, 0.5], 
                       obstacle_radius=0.2, lam=-0.1)
    
    def test_change_stability_empty_delta(self):
        """Test C/S computation rejects empty input"""
        with pytest.raises(ValueError, match="delta must be non-empty"):
            compute_change_stability([])
    
    def test_change_stability_negative_threshold(self):
        """Test C/S computation rejects negative threshold"""
        with pytest.raises(ValueError, match="eps_change must be non-negative"):
            compute_change_stability([0.1, 0.2], eps_change=-0.1)
    


class TestNumericalAccuracy:
    """Tests for numerical accuracy and edge cases"""
    
    def test_forward_kinematics_zero_angles(self):
        """Test FK at zero configuration"""
        x, y = forward_kinematics(0, 0, L1=1.0, L2=1.0)
        assert np.isclose(x, 2.0)
        assert np.isclose(y, 0.0, atol=1e-15)
    
    def test_jacobian_numerical_derivative(self):
        """Test Jacobian matches numerical differentiation"""
        theta1, theta2 = 0.7, 0.5
        J_analytical = jacobian(theta1, theta2)
        
        eps = 1e-8
        x0, y0 = forward_kinematics(theta1, theta2)
        x1, y1 = forward_kinematics(theta1 + eps, theta2)
        x2, y2 = forward_kinematics(theta1, theta2 + eps)
        
        # Numerical Jacobian
        J_numerical = np.array([
            [(x1 - x0) / eps, (x2 - x0) / eps],
            [(y1 - y0) / eps, (y2 - y0) / eps]
        ])
        
        assert np.allclose(J_analytical, J_numerical, rtol=1e-5)
    
    def test_gradient_descent_step_reduces_ds2(self):
        """Test single gradient descent step reduces objective"""
        theta1, theta2 = 0, 0
        target = np.array([1.5, 0.5])
        obstacle_center = np.array([10, 10])  # Far away
        obstacle_radius = 0.1
        
        ds2_before, _ = compute_ds2(theta1, theta2, target, obstacle_center, obstacle_radius)
        grad, _ = compute_gradient(theta1, theta2, target, obstacle_center, obstacle_radius)
        
        # Take gradient descent step
        eta = 0.01
        theta1_new = theta1 - eta * grad[0]
        theta2_new = theta2 - eta * grad[1]
        
        ds2_after, _ = compute_ds2(theta1_new, theta2_new, target, obstacle_center, obstacle_radius)
        
        # Should decrease (or stay same at minimum)
        assert ds2_after <= ds2_before
    
    def test_obstacle_repulsion_activates(self):
        """Test obstacle term only activates inside radius"""
        theta1, theta2 = 0, 0
        target = np.array([2.0, 0.0])
        
        # Obstacle far away
        ds2_far, comps_far = compute_ds2(
            theta1, theta2, target,
            obstacle_center=np.array([0, 5.0]),
            obstacle_radius=0.5
        )
        
        # Obstacle close
        ds2_near, comps_near = compute_ds2(
            theta1, theta2, target,
            obstacle_center=np.array([1.5, 0.0]),
            obstacle_radius=1.0
        )
        
        # Far obstacle shouldn't contribute
        assert comps_far['obs_term'] == 0.0
        
        # Near obstacle should contribute
        assert comps_near['obs_term'] > 0.0
        
        # Total should be higher with near obstacle
        assert ds2_near > ds2_far


class TestPhysicalConstraints:
    """Tests for physical realism and constraints"""
    
    def test_workspace_reachability(self):
        """Test end-effector stays within reachable workspace"""
        L1, L2 = 0.9, 0.9
        max_reach = L1 + L2  # 1.8m
        min_reach = abs(L1 - L2)  # 0.0m (can fold)
        
        # Test various configurations
        for theta1 in np.linspace(-np.pi, np.pi, 10):
            for theta2 in np.linspace(-np.pi, np.pi, 10):
                x, y = forward_kinematics(theta1, theta2, L1, L2)
                reach = np.sqrt(x**2 + y**2)
                
                # Should be within physical limits
                assert reach <= max_reach + 1e-10, \
                    f"Exceeds max reach: {reach} > {max_reach} at θ=({theta1}, {theta2})"
                assert reach >= min_reach - 1e-10, \
                    f"Below min reach: {reach} < {min_reach} at θ=({theta1}, {theta2})"
    
    def test_regularization_penalizes_large_angles(self):
        """Test regularization term increases with angle magnitude"""
        target = np.array([1.0, 0.5])
        obstacle_center = np.array([10, 10])
        obstacle_radius = 0.1
        
        # Small angles
        ds2_small, comps_small = compute_ds2(
            0.1, 0.1, target, obstacle_center, obstacle_radius,
            Go=0.0, lam=1.0  # Only regularization active
        )
        
        # Large angles
        ds2_large, comps_large = compute_ds2(
            1.0, 1.0, target, obstacle_center, obstacle_radius,
            Go=0.0, lam=1.0
        )
        
        # Regularization should be larger for larger angles
        assert comps_large['reg_term'] > comps_small['reg_term']
    
    def test_continuous_gradient(self):
        """Test gradient is continuous (no jumps)"""
        target = np.array([1.0, 0.5])
        obstacle_center = np.array([0.8, 0.2])
        obstacle_radius = 0.3
        
        theta1 = 0.5
        
        # Compute gradients at nearby points
        grad1, _ = compute_gradient(theta1, 0.3, target, obstacle_center, obstacle_radius)
        grad2, _ = compute_gradient(theta1, 0.3001, target, obstacle_center, obstacle_radius)
        
        # Should be very similar (continuous)
        assert np.allclose(grad1, grad2, rtol=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src/eigen_core", "--cov-report=term-missing"])
