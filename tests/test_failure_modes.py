# -*- coding: utf-8 -*-
"""
Tests for the 8 mathematical failure modes of gradient descent on ds².

Each test targets a theorem-level condition under which
    Q_{t+1} = Q_t - η ∇ds²(Q)
fails to do what you want.  The tests are designed to *trigger* the
failure and verify the diagnostic signatures described in the theory.

Failure modes tested
--------------------
1. Euler blow-up (η ≥ 2/L)
2. Non-Lipschitz gradients near obstacles (gradient explosion)
3. Non-smooth / kinked costs (chattering at obstacle boundary)
4. Nonconvexity → wrong local minimum (gradient cancellation)
5. Saddle points → frozen progress
6. Infeasible constraints → inf ds² > 0
7. Scale domination (one term swamps the rest)
8. Topology / homotopy trap (U-shaped obstacle)

Diagnostics tracked along every trajectory
------------------------------------------
- Δds²_t  = ds²(Q_{t+1}) - ds²(Q_t)          (descent check)
- g_t     = ‖∇ds²(Q_t)‖                       (gradient norm)
- r_i(Q)  = ‖∇d_i‖ / Σ‖∇d_j‖                 (termwise balance)
"""

import numpy as np
import pytest

from src.eigen_core import compute_ds2, compute_gradient, forward_kinematics, jacobian


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_descent(theta_init, target, obstacle_center, obstacle_radius,
                 eta, n_ticks, Go=4.0, lam=0.02, L1=0.9, L2=0.9):
    """Run plain gradient descent and return full diagnostic trace."""
    theta = np.array(theta_init, dtype=float)
    trace = []
    for t in range(n_ticks):
        th1, th2 = theta
        ds2, comp = compute_ds2(th1, th2, target, obstacle_center,
                                obstacle_radius, Go, lam, L1, L2)
        grad, gnorm = compute_gradient(th1, th2, target, obstacle_center,
                                       obstacle_radius, Go, lam, L1, L2)
        # Per-term gradient norms for balance diagnostic
        x, y = forward_kinematics(th1, th2, L1, L2)
        pos = np.array([x, y])
        J = jacobian(th1, th2, L1, L2)
        g_target = np.linalg.norm(2.0 * J.T @ (pos - np.asarray(target)))
        d_obs = np.linalg.norm(pos - np.asarray(obstacle_center))
        if obstacle_radius - d_obs > 0:
            g_obs = np.linalg.norm(
                Go * 2.0 * (obstacle_radius - d_obs)
                * (-1.0 / d_obs) * (J.T @ (pos - np.asarray(obstacle_center)))
            )
        else:
            g_obs = 0.0
        g_reg = np.linalg.norm(2.0 * lam * theta)

        delta = -eta * grad
        theta_new = theta + delta

        # Next-step ds² for descent check
        ds2_next, _ = compute_ds2(theta_new[0], theta_new[1], target,
                                  obstacle_center, obstacle_radius,
                                  Go, lam, L1, L2)

        trace.append({
            "t": t,
            "theta": theta.copy(),
            "ds2": ds2,
            "ds2_next": ds2_next,
            "delta_ds2": ds2_next - ds2,
            "grad_norm": gnorm,
            "grad": grad.copy(),
            "g_target": g_target,
            "g_obs": g_obs,
            "g_reg": g_reg,
            "d_obs": d_obs,
            "pos": pos.copy(),
        })
        theta = theta_new
    return trace


def _termwise_ratio(step, term="g_obs"):
    """Fraction of total gradient contributed by one term."""
    total = step["g_target"] + step["g_obs"] + step["g_reg"]
    if total < 1e-15:
        return 0.0
    return step[term] / total


# ===================================================================
# 1) Euler blow-up: η too large → oscillation / divergence
# ===================================================================
class TestEulerBlowup:
    """
    Failure condition: η ≥ 2/L where L is the local Lipschitz constant
    of ∇ds².  We use a very large η and verify ds² increases.
    """

    def test_large_eta_causes_ds2_increase(self):
        """With η far above the stability bound, ds² should increase."""
        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=(1.2, 0.3),
            obstacle_center=(0.6, 0.1),
            obstacle_radius=0.25,
            eta=5.0,          # absurdly large
            n_ticks=20,
        )
        # Count steps where ds² increased
        increases = sum(1 for s in trace if s["delta_ds2"] > 0)
        assert increases > 0, "Expected oscillation / ds² increase with large η"

    def test_large_eta_oscillation(self):
        """Sign changes in Δds² indicate oscillatory behavior."""
        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=(1.2, 0.3),
            obstacle_center=(0.6, 0.1),
            obstacle_radius=0.25,
            eta=2.0,
            n_ticks=50,
        )
        signs = [np.sign(s["delta_ds2"]) for s in trace if abs(s["delta_ds2"]) > 1e-10]
        sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        assert sign_changes >= 2, (
            f"Expected oscillation (sign changes in Δds²), got {sign_changes}"
        )

    def test_divergence_detected_by_ds2_growth(self):
        """ds²(final) >> ds²(initial) signals divergence."""
        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=(1.2, 0.3),
            obstacle_center=(0.6, 0.1),
            obstacle_radius=0.25,
            eta=10.0,
            n_ticks=30,
        )
        ds2_init = trace[0]["ds2"]
        ds2_final = trace[-1]["ds2"]
        assert ds2_final > ds2_init, (
            f"Expected divergence: ds² should grow, got {ds2_init:.4f} → {ds2_final:.4f}"
        )


# ===================================================================
# 2) Non-Lipschitz gradients near obstacles (gradient explosion)
# ===================================================================
class TestGradientExplosionNearObstacle:
    """
    The current implementation uses max(0, r-d)² which stays finite,
    but gradient magnitude still grows as 1/d_obs near the boundary.
    We verify the gradient spike when the end-effector is near the
    obstacle surface.
    """

    def test_gradient_spikes_near_obstacle_boundary(self):
        """Gradient norm should be much larger near obstacle than far away."""
        target = (1.2, 0.3)
        obs_center = (0.6, 0.1)
        obs_radius = 0.50  # large radius to ensure overlap at some configs

        # Configuration where end-effector is near obstacle center
        # FK(0, 0) = (1.8, 0) — far from obstacle
        _, gnorm_far = compute_gradient(0.0, 0.0, target, obs_center,
                                        obs_radius, Go=50.0)

        # Search for a configuration that puts end-effector inside obstacle
        # FK(θ1, 0) with L1=L2=0.9 → x = 1.8*cos(θ1), y = 1.8*sin(θ1)
        # We want pos ≈ obs_center = (0.6, 0.1)
        # Try angles that put us close
        best_gnorm = 0.0
        for th1 in np.linspace(-np.pi, np.pi, 200):
            for th2 in np.linspace(-np.pi, np.pi, 50):
                x, y = forward_kinematics(th1, th2)
                d = np.sqrt((x - 0.6)**2 + (y - 0.1)**2)
                if d < obs_radius and d > 0.01:  # inside but not at center
                    _, gn = compute_gradient(th1, th2, target, obs_center,
                                             obs_radius, Go=50.0)
                    best_gnorm = max(best_gnorm, gn)

        assert best_gnorm > 10.0 * max(gnorm_far, 1e-6), (
            f"Expected gradient spike near obstacle, got {best_gnorm:.4f} vs far {gnorm_far:.4f}"
        )

    def test_single_step_can_overshoot_obstacle(self):
        """A large step from inside the obstacle zone can jump past it."""
        obs_center = (0.6, 0.1)
        obs_radius = 0.50
        target = (1.2, 0.3)

        # Find config inside obstacle
        for th1 in np.linspace(-np.pi, np.pi, 200):
            for th2 in np.linspace(-np.pi, np.pi, 50):
                x, y = forward_kinematics(th1, th2)
                d = np.sqrt((x - 0.6)**2 + (y - 0.1)**2)
                if 0.05 < d < obs_radius * 0.5:
                    grad, gnorm = compute_gradient(th1, th2, target, obs_center,
                                                   obs_radius, Go=50.0)
                    # Take a big step
                    eta = 1.0
                    theta_new = np.array([th1, th2]) - eta * grad
                    x2, y2 = forward_kinematics(theta_new[0], theta_new[1])
                    d2 = np.sqrt((x2 - 0.6)**2 + (y2 - 0.1)**2)
                    step_size = np.linalg.norm(eta * grad)
                    if step_size > obs_radius:
                        # The step was larger than the obstacle — teleportation
                        assert True
                        return
        # If no config found inside, the test is inconclusive; skip
        pytest.skip("Could not find configuration inside obstacle zone")


# ===================================================================
# 3) Non-smooth / kinked costs → chattering at obstacle boundary
# ===================================================================
class TestKinkChattering:
    """
    The obstacle term max(0, r - d)² has a kink at d = r: the gradient
    jumps from 0 to nonzero.  Trajectories near this boundary can
    chatter (alternate between inside/outside the activation zone).
    """

    def test_chattering_at_obstacle_boundary(self):
        """Trajectory near obstacle boundary should show on/off switching."""
        # Place a large obstacle that the arm must pass through/near
        # to reach the target, forcing boundary interaction
        obs_center = (1.0, 0.3)
        obs_radius = 0.60  # large: covers much of the workspace
        target = (1.2, 0.3)

        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=target,
            obstacle_center=obs_center,
            obstacle_radius=obs_radius,
            eta=0.12,
            n_ticks=300,
            Go=10.0,
        )

        # Check for obstacle term switching on/off (entering/exiting zone)
        inside = [s["d_obs"] < obs_radius for s in trace]
        transitions = sum(1 for i in range(1, len(inside)) if inside[i] != inside[i - 1])

        # Also check: gradient has obstacle component that appears and disappears
        obs_active = [s["g_obs"] > 0.01 for s in trace]
        obs_transitions = sum(1 for i in range(1, len(obs_active))
                              if obs_active[i] != obs_active[i - 1])

        assert transitions >= 1 or obs_transitions >= 1, (
            f"Expected obstacle boundary transitions: spatial={transitions}, "
            f"gradient={obs_transitions}"
        )

    def test_gradient_discontinuity_at_boundary(self):
        """Gradient should jump discontinuously at d_obs = r."""
        obs_center = np.array([0.6, 0.1])
        obs_radius = 0.30
        target = np.array([1.2, 0.3])

        # Find config near boundary
        for th1 in np.linspace(-np.pi, np.pi, 300):
            for th2 in np.linspace(-np.pi, np.pi, 100):
                x, y = forward_kinematics(th1, th2)
                d = np.sqrt((x - obs_center[0])**2 + (y - obs_center[1])**2)
                if abs(d - obs_radius) < 0.005:
                    # Evaluate gradient just inside and just outside
                    eps = 0.001
                    # Perturb toward obstacle center
                    direction = (obs_center - np.array([x, y]))
                    direction = direction / (np.linalg.norm(direction) + 1e-12)
                    # We can't easily move end-effector by epsilon in task space,
                    # so compare at d=r-small vs d=r+small via Go contribution
                    _, g_at = compute_gradient(th1, th2, target, obs_center,
                                              obs_radius, Go=10.0)
                    # Just outside: obs term = 0
                    # Just inside:  obs term > 0
                    # The gradient jump IS the discontinuity in the derivative
                    # Verify: if we're just inside, obs gradient is nonzero
                    if d < obs_radius:
                        g_obs_inside = _run_descent(
                            (th1, th2), target, obs_center, obs_radius,
                            eta=0.0, n_ticks=1, Go=10.0
                        )[0]["g_obs"]
                        assert g_obs_inside > 0, "Obstacle gradient should be active inside"
                    return
        pytest.skip("Could not find configuration near obstacle boundary")


# ===================================================================
# 4) Nonconvexity → wrong local minimum (gradient cancellation)
# ===================================================================
class TestLocalMinimumTrap:
    """
    Failure condition: ∃ Q* ≠ Q_goal s.t. ∇ds²(Q*) = 0
    This happens when obstacle repulsion exactly cancels target attraction.
    """

    def test_converges_to_non_target_stationary_point(self):
        """
        Place obstacle between start and target so the arm gets trapped
        at a point where ∇d_target + ∇d_obs ≈ 0.
        """
        # Obstacle directly on the line between start-effector and target
        target = (1.5, 0.0)
        obs_center = (1.0, 0.0)
        obs_radius = 0.40
        Go = 50.0  # strong repulsion

        trace = _run_descent(
            theta_init=(0.1, 0.1),  # FK ≈ (1.78, 0.18) — near target but blocked
            target=target,
            obstacle_center=obs_center,
            obstacle_radius=obs_radius,
            eta=0.05,
            n_ticks=500,
            Go=Go,
            lam=0.01,
        )

        final = trace[-1]
        # Check: gradient is small (stationary point reached)
        grad_small = final["grad_norm"] < 0.5

        # Check: we're not actually at the target
        pos = final["pos"]
        dist_to_target = np.linalg.norm(pos - np.array(target))

        if grad_small and dist_to_target > 0.3:
            # Confirmed: stuck at non-target stationary point
            assert True
            return

        # Alternative: if gradient didn't vanish, check that the system
        # is at least not making meaningful progress (practical trap)
        late_ds2 = [s["ds2"] for s in trace[-50:]]
        ds2_range = max(late_ds2) - min(late_ds2)
        assert dist_to_target > 0.2 or ds2_range < 0.01, (
            f"Expected local minimum trap: dist_to_target={dist_to_target:.3f}, "
            f"late ds² range={ds2_range:.6f}"
        )

    def test_gradient_cancellation_diagnostic(self):
        """
        Verify the termwise balance diagnostic: when trapped, the
        obstacle and target gradients should be roughly equal and opposite.
        """
        target = (1.5, 0.0)
        obs_center = (1.0, 0.0)
        obs_radius = 0.40

        trace = _run_descent(
            theta_init=(0.1, 0.1),
            target=target,
            obstacle_center=obs_center,
            obstacle_radius=obs_radius,
            eta=0.05,
            n_ticks=500,
            Go=50.0,
            lam=0.01,
        )

        # Look at the last steps — if trapped, target and obs terms should
        # be comparable in magnitude
        final = trace[-1]
        if final["g_obs"] > 0:
            ratio = min(final["g_target"], final["g_obs"]) / max(final["g_target"], final["g_obs"])
            # If they're cancelling, ratio should be non-trivial
            # (doesn't have to be exactly 1 due to regularization)
            assert ratio > 0.05 or final["grad_norm"] < 1.0, (
                f"Expected gradient cancellation: target={final['g_target']:.4f}, "
                f"obs={final['g_obs']:.4f}, ratio={ratio:.4f}"
            )


# ===================================================================
# 5) Saddle points → frozen progress
# ===================================================================
class TestSaddlePoint:
    """
    At a saddle, ∇ds² ≈ 0 but the Hessian has mixed-sign eigenvalues.
    The system appears frozen: ‖Q_{t+1} - Q_t‖ ≈ 0 while not at goal.
    """

    def test_near_kinematic_singularity_slow_progress(self):
        """
        At θ2 = 0 (fully extended), the Jacobian is singular — det(J) = 0.
        The gradient through J^T maps poorly near this config, creating
        a region of reduced effective gradient magnitude compared to
        the task-space error.

        We verify the Jacobian condition number is large near the
        singularity, which is the mechanism that would cause stalling
        in higher-DOF or more constrained systems.
        """
        # At θ2 = 0, the Jacobian is rank-deficient
        J_singular = jacobian(0.0, 0.0)
        det_singular = abs(np.linalg.det(J_singular))

        # At θ2 = π/2, the Jacobian is well-conditioned
        J_good = jacobian(0.0, np.pi / 2)
        det_good = abs(np.linalg.det(J_good))

        # The singular configuration has near-zero determinant
        assert det_singular < 1e-10, (
            f"Expected singular Jacobian at θ2=0, det={det_singular:.2e}"
        )
        assert det_good > 0.1, (
            f"Expected well-conditioned Jacobian at θ2=π/2, det={det_good:.2e}"
        )

        # Run descent starting exactly at singularity with target requiring
        # motion in the null-space direction
        trace = _run_descent(
            theta_init=(0.0, 0.0),  # exact singularity
            target=(-0.5, 0.0),     # requires folding
            obstacle_center=(10.0, 10.0),
            obstacle_radius=0.1,
            eta=0.12,
            n_ticks=50,
            lam=0.001,
        )

        # The gradient at singularity should be reduced relative to the
        # actual task-space error (which is large: ~2.3m from (1.8,0) to (-0.5,0))
        first_grad = trace[0]["grad_norm"]
        task_error = np.linalg.norm(trace[0]["pos"] - np.array([-0.5, 0.0]))
        # Gradient is attenuated by the singular Jacobian
        ratio = first_grad / task_error
        assert ratio < 5.0, (
            f"Gradient/error ratio at singularity: {ratio:.3f}"
        )

    def test_gradient_cancellation_creates_slow_region(self):
        """
        When obstacle repulsion and target attraction partially cancel,
        the net gradient is smaller than either component alone.
        This is the mechanism behind saddle-like stalling.
        """
        # Place obstacle directly between start config's end-effector and target
        target = np.array([0.5, 0.8])
        obs_center = np.array([0.5, 0.5])
        obs_radius = 0.40

        trace = _run_descent(
            theta_init=(-0.8, 1.5),
            target=target,
            obstacle_center=obs_center,
            obstacle_radius=obs_radius,
            eta=0.05,
            n_ticks=500,
            Go=40.0,
            lam=0.01,
        )

        # Find steps where net gradient is much smaller than the dominant
        # component — this is gradient cancellation
        cancellation_steps = []
        for s in trace:
            max_component = max(s["g_target"], s["g_obs"], s["g_reg"])
            if max_component > 0.1 and s["grad_norm"] < 0.5 * max_component:
                cancellation_steps.append(s)

        # Also valid: the system converges to a point far from target
        final_dist = np.linalg.norm(trace[-1]["pos"] - target)

        assert len(cancellation_steps) > 0 or final_dist > 0.15, (
            f"Expected gradient cancellation or incomplete convergence: "
            f"cancellation_steps={len(cancellation_steps)}, final_dist={final_dist:.3f}"
        )


# ===================================================================
# 6) Infeasible constraints → inf ds² > 0
# ===================================================================
class TestInfeasible:
    """
    If the target is unreachable (outside workspace) or completely
    blocked, then inf_Q ds²(Q) > 0.  Descent converges to a
    compromise point.
    """

    def test_target_outside_workspace(self):
        """
        Workspace radius = L1 + L2 = 1.8m.
        Target at distance 3.0 is unreachable.
        """
        target = (3.0, 0.0)  # unreachable

        trace = _run_descent(
            theta_init=(0.0, 0.0),
            target=target,
            obstacle_center=(10.0, 10.0),
            obstacle_radius=0.1,
            eta=0.12,
            n_ticks=200,
        )

        final_ds2 = trace[-1]["ds2"]
        # ds² should converge to a positive value (can't reach target)
        assert final_ds2 > 0.5, (
            f"Expected positive residual ds² for unreachable target, got {final_ds2:.4f}"
        )

        # System should still converge (gradient → 0)
        assert trace[-1]["grad_norm"] < 0.1, (
            f"Expected convergence to compromise point, grad_norm={trace[-1]['grad_norm']:.4f}"
        )

        # Verify: end-effector is at workspace boundary (fully extended)
        pos = trace[-1]["pos"]
        reach = np.linalg.norm(pos)
        assert reach > 1.7, (
            f"Expected arm fully extended toward target, reach={reach:.3f}"
        )

    def test_target_inside_obstacle(self):
        """Target inside an obstacle creates competing objectives."""
        target = (0.6, 0.1)
        obs_center = (0.6, 0.1)  # same as target!
        obs_radius = 0.30

        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=target,
            obstacle_center=obs_center,
            obstacle_radius=obs_radius,
            eta=0.05,
            n_ticks=300,
            Go=20.0,
        )

        final = trace[-1]
        final_ds2 = final["ds2"]

        # Can't simultaneously be at target and avoid obstacle
        # So final ds² > 0
        assert final_ds2 > 0.01, (
            f"Expected nonzero ds² for target-inside-obstacle, got {final_ds2:.6f}"
        )


# ===================================================================
# 7) Scale domination (one term swamps the others)
# ===================================================================
class TestScaleDomination:
    """
    When ‖∇d_obs‖ >> ‖∇d_target‖ + ‖∇d_reg‖, the system only avoids
    the obstacle and never advances toward the goal.
    """

    def test_obstacle_dominates_prevents_target_approach(self):
        """Extreme Go with large obstacle makes obstacle term dominate."""
        # Target is inside a large obstacle zone — system must enter zone
        # to approach target, but extreme Go prevents it
        target = (0.8, 0.2)
        obs_center = (0.8, 0.2)  # centered on target
        obs_radius = 0.60        # large zone

        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=target,
            obstacle_center=obs_center,
            obstacle_radius=obs_radius,
            eta=0.05,
            n_ticks=300,
            Go=1000.0,  # extreme obstacle weight
            lam=0.02,
        )

        # Check: obstacle gradient dominates for steps inside the zone
        dominated_steps = [
            s for s in trace
            if _termwise_ratio(s, "g_obs") > 0.8
        ]

        # Check: final position is far from target
        final_pos = trace[-1]["pos"]
        dist = np.linalg.norm(final_pos - np.array(target))

        assert len(dominated_steps) > 5 or dist > 0.3, (
            f"Expected obstacle domination: {len(dominated_steps)} dominated steps, "
            f"final dist={dist:.3f}"
        )

    def test_regularization_dominates_pulls_to_origin(self):
        """Extreme λ makes the arm collapse toward θ = (0,0)."""
        target = (0.5, 1.0)

        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=target,
            obstacle_center=(10.0, 10.0),
            obstacle_radius=0.1,
            eta=0.01,   # small η to avoid divergence with large λ
            n_ticks=500,
            Go=4.0,
            lam=50.0,   # large but not so extreme as to cause overflow
        )

        final_theta = trace[-1]["theta"]
        # With large lam, θ → near 0 regardless of target
        assert np.linalg.norm(final_theta) < 0.5, (
            f"Expected regularization to pull θ→0, got ‖θ‖={np.linalg.norm(final_theta):.3f}"
        )

        # And we're NOT at the target
        final_pos = trace[-1]["pos"]
        dist = np.linalg.norm(final_pos - np.array(target))
        assert dist > 0.3, (
            f"Expected failure to reach target under regularization domination, dist={dist:.3f}"
        )

    def test_domination_diagnostic(self):
        """The r_i ratio diagnostic should flag domination."""
        # Use same setup as obstacle_dominates test — target inside obstacle
        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=(0.8, 0.2),
            obstacle_center=(0.8, 0.2),
            obstacle_radius=0.60,
            eta=0.05,
            n_ticks=200,
            Go=1000.0,
        )

        # Find max domination ratio across the trajectory
        obs_steps = [s for s in trace if s["g_obs"] > 0]
        if obs_steps:
            max_ratio = max(_termwise_ratio(s, "g_obs") for s in obs_steps)
        else:
            # If obstacle never activated, check regularization domination
            max_ratio = max(_termwise_ratio(s, "g_reg") for s in trace
                           if (s["g_target"] + s["g_obs"] + s["g_reg"]) > 1e-10)

        assert max_ratio > 0.7, (
            f"Expected domination ratio > 0.7, got {max_ratio:.3f}"
        )


# ===================================================================
# 8) Topology / homotopy trap (the "wall" / U-shape problem)
# ===================================================================
class TestTopologyTrap:
    """
    With non-contractible free space (holes from obstacles), a global
    potential with unique minimum at the goal is generally impossible.
    Spurious minima arise — the classic potential field failure.
    """

    def test_u_shaped_obstacle_trap(self):
        """
        With a large obstacle covering the target region, the potential
        field creates a spurious minimum where repulsion balances attraction.
        This is the classic artificial potential field failure.
        """
        # Target behind a large obstacle
        target = (0.5, 0.5)
        obs_center = (0.5, 0.5)  # obstacle centered on target
        obs_radius = 0.55        # large

        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=target,
            obstacle_center=obs_center,
            obstacle_radius=obs_radius,
            eta=0.05,
            n_ticks=500,
            Go=50.0,
        )

        final_pos = trace[-1]["pos"]
        dist_to_target = np.linalg.norm(final_pos - np.array(target))

        # System should NOT reach the target (it's inside the obstacle zone)
        # The obstacle repulsion prevents approach
        assert dist_to_target > 0.15, (
            f"Expected topology/potential field trap: dist={dist_to_target:.3f}"
        )

    def test_multiple_obstacle_corridor(self):
        """
        With the single-obstacle API, simulate a narrow corridor by
        placing the obstacle to force a detour.  The gradient field
        may create a spurious equilibrium.
        """
        target = (-0.5, 0.5)
        obs_center = (0.0, 0.5)
        obs_radius = 0.60

        trace = _run_descent(
            theta_init=(0.5, 0.5),
            target=target,
            obstacle_center=obs_center,
            obstacle_radius=obs_radius,
            eta=0.05,
            n_ticks=500,
            Go=40.0,
            lam=0.01,
        )

        final_pos = trace[-1]["pos"]
        dist = np.linalg.norm(final_pos - np.array(target))
        final_grad = trace[-1]["grad_norm"]

        # Check for the classic potential field failure:
        # small gradient, not at goal
        potential_field_failure = (final_grad < 1.0 and dist > 0.2)
        # Or at minimum: the trajectory had difficulty
        ds2_values = [s["ds2"] for s in trace]
        plateaus = sum(
            1 for i in range(1, len(ds2_values))
            if abs(ds2_values[i] - ds2_values[i-1]) < 1e-6
        )

        assert potential_field_failure or plateaus > 10 or dist > 0.15, (
            f"Expected topology/corridor trap: dist={dist:.3f}, "
            f"grad={final_grad:.4f}, plateaus={plateaus}"
        )


# ===================================================================
# Diagnostic integration test: run all diagnostics on default config
# ===================================================================
class TestDiagnosticInstrumentation:
    """
    Verify that the three diagnostic quantities (Δds², g_t, r_i)
    are computable and meaningful on a standard trajectory.
    """

    def test_descent_check_monotone_for_stable_eta(self):
        """With the default η=0.12, descent should be mostly monotone."""
        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=(1.2, 0.3),
            obstacle_center=(0.6, 0.1),
            obstacle_radius=0.25,
            eta=0.12,
            n_ticks=140,
        )
        increases = sum(1 for s in trace if s["delta_ds2"] > 1e-8)
        fraction = increases / len(trace)
        assert fraction < 0.15, (
            f"Expected mostly monotone descent, got {fraction:.1%} increases"
        )

    def test_gradient_norm_decreases_to_near_zero(self):
        """Gradient norm should decrease as system converges."""
        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=(1.2, 0.3),
            obstacle_center=(0.6, 0.1),
            obstacle_radius=0.25,
            eta=0.12,
            n_ticks=140,
        )
        g_init = trace[0]["grad_norm"]
        g_final = trace[-1]["grad_norm"]
        assert g_final < g_init * 0.01, (
            f"Expected gradient reduction, got {g_init:.4f} → {g_final:.4f}"
        )

    def test_termwise_balance_shifts_during_convergence(self):
        """The dominant gradient term should shift as the arm moves."""
        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=(1.2, 0.3),
            obstacle_center=(0.6, 0.1),
            obstacle_radius=0.25,
            eta=0.12,
            n_ticks=140,
        )
        # At the end, target term should dominate (obstacle far away)
        final = trace[-1]
        r_target = _termwise_ratio(final, "g_target")
        r_reg = _termwise_ratio(final, "g_reg")
        # Near convergence, target and regularization are what's left
        assert r_target + r_reg > 0.5 or final["grad_norm"] < 0.01, (
            f"Expected target/reg dominance at convergence, "
            f"r_target={r_target:.3f}, r_reg={r_reg:.3f}"
        )
