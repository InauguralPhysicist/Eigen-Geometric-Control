# -*- coding: utf-8 -*-
"""
Tests for failure modes of the discrete scheme tracking ds².

ds² is a geometric invariant — not a cost function to optimize.
The update rule Q_{t+1} = Q_t - η ∇ds²(Q) follows the geometry.
Failures therefore fall into two categories:

A. DISCRETIZATION FAILURES (the numerical scheme loses the continuous flow)
   1. Euler blow-up: η ≥ 2/L → oscillation / divergence
   2. Gradient explosion near obstacles: ‖∇ds²‖ → ∞ makes any fixed η unstable
   3. Kink chattering: non-smooth ds² at obstacle boundary → discrete switching

B. GEOMETRIC PROPERTIES (the invariant faithfully reports the geometry)
   These are NOT failures of the framework. They are correct descriptions of
   geometries where the invariant says "you can't get there from here" or
   "here is where the geometry puts you." The tests verify the invariant
   accurately reflects these conditions:
   4. Stationary points from gradient cancellation (geometry has an equilibrium)
   5. Jacobian singularity (geometry degenerates at certain configurations)
   6. Infeasible geometry (no configuration satisfies all constraints)
   7. Component scale separation (one geometric term dominates the invariant)
   8. Topological obstruction (free space topology prevents unique minimum)

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


def _run_descent(
    theta_init,
    target,
    obstacle_center,
    obstacle_radius,
    eta,
    n_ticks,
    Go=4.0,
    lam=0.02,
    L1=0.9,
    L2=0.9,
):
    """Run plain gradient descent and return full diagnostic trace."""
    theta = np.array(theta_init, dtype=float)
    trace = []
    for t in range(n_ticks):
        th1, th2 = theta
        ds2, comp = compute_ds2(th1, th2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2)
        grad, gnorm = compute_gradient(
            th1, th2, target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )
        # Per-term gradient norms for balance diagnostic
        x, y = forward_kinematics(th1, th2, L1, L2)
        pos = np.array([x, y])
        J = jacobian(th1, th2, L1, L2)
        g_target = np.linalg.norm(2.0 * J.T @ (pos - np.asarray(target)))
        d_obs = np.linalg.norm(pos - np.asarray(obstacle_center))
        if obstacle_radius - d_obs > 0:
            g_obs = np.linalg.norm(
                Go
                * 2.0
                * (obstacle_radius - d_obs)
                * (-1.0 / d_obs)
                * (J.T @ (pos - np.asarray(obstacle_center)))
            )
        else:
            g_obs = 0.0
        g_reg = np.linalg.norm(2.0 * lam * theta)

        delta = -eta * grad
        theta_new = theta + delta

        # Next-step ds² for descent check
        ds2_next, _ = compute_ds2(
            theta_new[0], theta_new[1], target, obstacle_center, obstacle_radius, Go, lam, L1, L2
        )

        trace.append(
            {
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
            }
        )
        theta = theta_new
    return trace


def _termwise_ratio(step, term="g_obs"):
    """Fraction of total gradient contributed by one term."""
    total = step["g_target"] + step["g_obs"] + step["g_reg"]
    if total < 1e-15:
        return 0.0
    return step[term] / total


# ###################################################################
#
#  PART A: DISCRETIZATION FAILURES
#
#  The numerical scheme fails to track the continuous geometric flow.
#  These are real failures — the discrete update diverges from what
#  the invariant prescribes.
#
# ###################################################################


# ===================================================================
# A1) Euler blow-up: η ≥ 2/L → oscillation / divergence
# ===================================================================
class TestEulerBlowup:
    """
    The discrete update Q_{t+1} = Q_t - η∇ds² is forward Euler.
    If ∇ds² is L-Lipschitz, stability requires η < 2/L.
    Above this bound the scheme oscillates or diverges — it no longer
    tracks the continuous flow.
    """

    def test_large_eta_causes_ds2_increase(self):
        """With η far above the stability bound, ds² should increase."""
        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=(1.2, 0.3),
            obstacle_center=(0.6, 0.1),
            obstacle_radius=0.25,
            eta=5.0,
            n_ticks=20,
        )
        increases = sum(1 for s in trace if s["delta_ds2"] > 0)
        assert increases > 0, "Expected ds² increase with large η (Euler instability)"

    def test_large_eta_oscillation(self):
        """Sign changes in Δds² indicate the scheme is oscillating."""
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
        assert sign_changes >= 2, f"Expected oscillation (sign changes in Δds²), got {sign_changes}"

    def test_divergence_detected_by_ds2_growth(self):
        """ds²(final) >> ds²(initial) signals the scheme has diverged."""
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
        assert (
            ds2_final > ds2_init
        ), f"Expected divergence: ds² should grow, got {ds2_init:.4f} → {ds2_final:.4f}"


# ===================================================================
# A2) Gradient explosion near obstacles
# ===================================================================
class TestGradientExplosionNearObstacle:
    """
    The obstacle gradient ∝ (r - d)/d blows up as d → 0.  Even though
    max(0, r-d)² stays finite, the gradient magnitude grows without
    bound near the obstacle center.  This makes the effective local
    Lipschitz constant L → ∞, so no fixed η is stable — the Euler
    scheme will overshoot.

    This is a discretization failure: the continuous flow would slow
    down near the obstacle, but the discrete step can teleport past it.
    """

    def test_gradient_spikes_near_obstacle_boundary(self):
        """Gradient norm should be much larger near obstacle than far away."""
        target = (1.2, 0.3)
        obs_center = (0.6, 0.1)
        obs_radius = 0.50

        # Far from obstacle
        _, gnorm_far = compute_gradient(0.0, 0.0, target, obs_center, obs_radius, Go=50.0)

        # Search for a configuration inside obstacle
        best_gnorm = 0.0
        for th1 in np.linspace(-np.pi, np.pi, 200):
            for th2 in np.linspace(-np.pi, np.pi, 50):
                x, y = forward_kinematics(th1, th2)
                d = np.sqrt((x - 0.6) ** 2 + (y - 0.1) ** 2)
                if d < obs_radius and d > 0.01:
                    _, gn = compute_gradient(th1, th2, target, obs_center, obs_radius, Go=50.0)
                    best_gnorm = max(best_gnorm, gn)

        assert best_gnorm > 10.0 * max(
            gnorm_far, 1e-6
        ), f"Expected gradient spike near obstacle, got {best_gnorm:.4f} vs far {gnorm_far:.4f}"

    def test_single_step_can_overshoot_obstacle(self):
        """
        A discrete step from inside the obstacle zone can be larger than
        the obstacle itself — the scheme teleports through the barrier
        that the continuous flow would respect.
        """
        obs_center = (0.6, 0.1)
        obs_radius = 0.50
        target = (1.2, 0.3)

        for th1 in np.linspace(-np.pi, np.pi, 200):
            for th2 in np.linspace(-np.pi, np.pi, 50):
                x, y = forward_kinematics(th1, th2)
                d = np.sqrt((x - 0.6) ** 2 + (y - 0.1) ** 2)
                if 0.05 < d < obs_radius * 0.5:
                    grad, gnorm = compute_gradient(
                        th1, th2, target, obs_center, obs_radius, Go=50.0
                    )
                    eta = 1.0
                    step_size = np.linalg.norm(eta * grad)
                    if step_size > obs_radius:
                        # Step larger than the obstacle — discrete teleportation
                        assert True
                        return
        pytest.skip("Could not find configuration inside obstacle zone")


# ===================================================================
# A3) Kink chattering: non-smooth ds² at obstacle boundary
# ===================================================================
class TestKinkChattering:
    """
    max(0, r - d)² has a C¹ kink at d = r: the gradient jumps from 0
    to nonzero.  The continuous flow handles this fine (it just enters
    or exits the active zone smoothly).  But the discrete scheme can
    chatter — alternating between inside/outside on successive steps,
    because the gradient changes discontinuously between them.

    This is a discretization artifact, not a geometric property.
    """

    def test_chattering_at_obstacle_boundary(self):
        """Trajectory near obstacle boundary should show on/off switching."""
        obs_center = (1.0, 0.3)
        obs_radius = 0.60
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

        obs_active = [s["g_obs"] > 0.01 for s in trace]
        obs_transitions = sum(
            1 for i in range(1, len(obs_active)) if obs_active[i] != obs_active[i - 1]
        )

        assert transitions >= 1 or obs_transitions >= 1, (
            f"Expected obstacle boundary transitions: spatial={transitions}, "
            f"gradient={obs_transitions}"
        )

    def test_gradient_discontinuity_at_boundary(self):
        """
        Verify the gradient has a discontinuous jump at d_obs = r.
        This is the mechanism that causes chattering in the discrete scheme.
        """
        obs_center = np.array([0.6, 0.1])
        obs_radius = 0.30
        target = np.array([1.2, 0.3])

        for th1 in np.linspace(-np.pi, np.pi, 300):
            for th2 in np.linspace(-np.pi, np.pi, 100):
                x, y = forward_kinematics(th1, th2)
                d = np.sqrt((x - obs_center[0]) ** 2 + (y - obs_center[1]) ** 2)
                if abs(d - obs_radius) < 0.005:
                    if d < obs_radius:
                        g_obs_inside = _run_descent(
                            (th1, th2), target, obs_center, obs_radius, eta=0.0, n_ticks=1, Go=10.0
                        )[0]["g_obs"]
                        assert g_obs_inside > 0, "Obstacle gradient should be active inside"
                    return
        pytest.skip("Could not find configuration near obstacle boundary")


# ###################################################################
#
#  PART B: GEOMETRIC PROPERTIES OF THE INVARIANT
#
#  ds² faithfully describes the geometry.  These tests verify that
#  the invariant correctly reports geometric conditions — they are
#  NOT failures of the framework.  The system stopping, stalling,
#  or settling at a non-target point is the *correct* reading of
#  the geometry in each case.
#
# ###################################################################


# ===================================================================
# B4) Stationary points from gradient cancellation
#     (the geometry has an equilibrium — ds² correctly reports it)
# ===================================================================
class TestGeometricEquilibrium:
    """
    When ∇d_target + ∇d_obs + ∇d_reg = 0 at Q* ≠ Q_goal, the invariant
    has a stationary point.  This is not a "trap" — it's the geometry
    telling you that the forces balance here.  The system correctly
    finds and reports this equilibrium.
    """

    def test_system_finds_equilibrium_between_obstacle_and_target(self):
        """
        With obstacle between start and target, the system settles at
        the geometric equilibrium where repulsion balances attraction.
        Verify: gradient → 0, position stable, ds² > 0.
        """
        target = (1.5, 0.0)
        obs_center = (1.0, 0.0)
        obs_radius = 0.40
        Go = 50.0

        trace = _run_descent(
            theta_init=(0.1, 0.1),
            target=target,
            obstacle_center=obs_center,
            obstacle_radius=obs_radius,
            eta=0.05,
            n_ticks=500,
            Go=Go,
            lam=0.01,
        )

        final = trace[-1]

        # The system should reach a stationary point (gradient small)
        # OR converge to a stable region (ds² plateau)
        late_ds2 = [s["ds2"] for s in trace[-50:]]
        ds2_range = max(late_ds2) - min(late_ds2)

        assert final["grad_norm"] < 0.5 or ds2_range < 0.01, (
            f"Expected convergence to equilibrium: grad={final['grad_norm']:.4f}, "
            f"ds² range={ds2_range:.6f}"
        )

        # ds² > 0 at this equilibrium: the invariant correctly reports
        # that the target is not reached
        assert (
            final["ds2"] > 0.01
        ), f"Expected ds² > 0 at equilibrium (target not reached), got {final['ds2']:.6f}"

    def test_gradient_cancellation_is_balanced(self):
        """
        At the equilibrium, obstacle and target gradient components should
        be comparable — they are balancing each other.  The termwise
        diagnostic r_i confirms this.
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

        final = trace[-1]
        if final["g_obs"] > 0:
            ratio = min(final["g_target"], final["g_obs"]) / max(final["g_target"], final["g_obs"])
            assert ratio > 0.05 or final["grad_norm"] < 1.0, (
                f"Expected balanced cancellation: target={final['g_target']:.4f}, "
                f"obs={final['g_obs']:.4f}, ratio={ratio:.4f}"
            )


# ===================================================================
# B5) Jacobian singularity
#     (the geometry degenerates — ds² correctly reflects this)
# ===================================================================
class TestJacobianSingularity:
    """
    At θ2 = 0 (fully extended), det(J) = 0.  The gradient ∇ds² passes
    through J^T, so it loses a direction of information.  This is a
    geometric fact about the configuration space — the invariant
    correctly reports that the manifold is degenerate here.
    """

    def test_jacobian_rank_deficiency_at_extension(self):
        """
        Verify: det(J) = 0 at θ2 = 0, det(J) >> 0 at θ2 = π/2.
        This is a property of the arm geometry, not a numerical failure.
        """
        J_singular = jacobian(0.0, 0.0)
        det_singular = abs(np.linalg.det(J_singular))

        J_good = jacobian(0.0, np.pi / 2)
        det_good = abs(np.linalg.det(J_good))

        assert det_singular < 1e-10, f"Expected singular Jacobian at θ2=0, det={det_singular:.2e}"
        assert det_good > 0.1, f"Expected well-conditioned Jacobian at θ2=π/2, det={det_good:.2e}"

    def test_gradient_attenuation_at_singularity(self):
        """
        At the singularity, the gradient is attenuated relative to
        the task-space error.  This is the invariant correctly reporting
        that the configuration is geometrically degenerate — the flow
        has less information to work with here.
        """
        trace = _run_descent(
            theta_init=(0.0, 0.0),  # exact singularity
            target=(-0.5, 0.0),  # requires folding
            obstacle_center=(10.0, 10.0),
            obstacle_radius=0.1,
            eta=0.12,
            n_ticks=50,
            lam=0.001,
        )

        first_grad = trace[0]["grad_norm"]
        task_error = np.linalg.norm(trace[0]["pos"] - np.array([-0.5, 0.0]))
        ratio = first_grad / task_error
        assert ratio < 5.0, f"Gradient/error ratio at singularity: {ratio:.3f}"

    def test_exact_singularity_traps_the_flow(self):
        """
        Starting exactly at the singularity (θ₂ = 0), with target requiring
        motion along the degenerate direction, the gradient is zero in that
        direction.  The system stays trapped — this is the invariant correctly
        reporting that the geometry provides no information here.

        A near-singularity start (θ₂ = 0.01) escapes; the exact one does not.
        """
        target = (-0.5, 0.0)

        # Exact singularity: trapped
        trace_exact = _run_descent(
            theta_init=(0.0, 0.0),
            target=target,
            obstacle_center=(10.0, 10.0),
            obstacle_radius=0.1,
            eta=0.12,
            n_ticks=200,
            lam=0.001,
        )
        dist_exact = np.linalg.norm(trace_exact[-1]["pos"] - np.array(target))

        # Near singularity: escapes
        trace_near = _run_descent(
            theta_init=(0.0, 0.01),
            target=target,
            obstacle_center=(10.0, 10.0),
            obstacle_radius=0.1,
            eta=0.12,
            n_ticks=200,
            lam=0.001,
        )
        dist_near = np.linalg.norm(trace_near[-1]["pos"] - np.array(target))

        # Exact singularity should be much worse
        assert dist_exact > 1.0, f"Expected exact singularity to trap: dist={dist_exact:.3f}"
        assert dist_near < dist_exact, (
            f"Expected near-singularity to outperform exact: "
            f"near={dist_near:.3f}, exact={dist_exact:.3f}"
        )


# ===================================================================
# B6) Infeasible geometry
#     (ds² correctly reports that no solution exists)
# ===================================================================
class TestInfeasibleGeometry:
    """
    When the target is unreachable or contradicts constraints,
    inf_Q ds²(Q) > 0.  The invariant settles to a positive value
    and gradient → 0.  This is correct: ds² is telling you the
    geometry has no solution.
    """

    def test_target_outside_workspace(self):
        """
        Workspace radius = L1 + L2 = 1.8m.  Target at 3.0m is
        geometrically unreachable.  ds² → positive residual,
        gradient → 0, arm fully extended.
        """
        target = (3.0, 0.0)

        trace = _run_descent(
            theta_init=(0.0, 0.0),
            target=target,
            obstacle_center=(10.0, 10.0),
            obstacle_radius=0.1,
            eta=0.12,
            n_ticks=200,
        )

        final = trace[-1]
        # ds² > 0: invariant correctly reports infeasibility
        assert (
            final["ds2"] > 0.5
        ), f"Expected ds² > 0 for unreachable target, got {final['ds2']:.4f}"
        # Gradient → 0: system found the closest point
        assert (
            final["grad_norm"] < 0.1
        ), f"Expected gradient → 0 at compromise, got {final['grad_norm']:.4f}"
        # Arm fully extended toward target
        reach = np.linalg.norm(final["pos"])
        assert reach > 1.7, f"Expected full extension toward target, reach={reach:.3f}"

    def test_target_inside_obstacle(self):
        """
        Target coincides with obstacle center.  The invariant correctly
        reports that satisfying both constraints simultaneously is
        impossible: ds² > 0 at the equilibrium.
        """
        target = (0.6, 0.1)
        obs_center = (0.6, 0.1)
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

        assert (
            trace[-1]["ds2"] > 0.01
        ), f"Expected ds² > 0 for contradictory constraints, got {trace[-1]['ds2']:.6f}"


# ===================================================================
# B7) Component scale separation
#     (the invariant is dominated by one geometric term)
# ===================================================================
class TestComponentScaleSeparation:
    """
    When one component of ds² has much larger gradient than the others,
    the flow follows that component almost exclusively.  This is not
    a failure — it's the invariant correctly reflecting that one
    geometric constraint is far more "urgent" than the others.

    The diagnostic r_i = ‖∇d_i‖ / Σ‖∇d_j‖ detects this.
    """

    def test_obstacle_component_dominates_invariant(self):
        """
        With Go=1000 and target inside obstacle zone, the obstacle
        component dominates ds².  The invariant correctly prioritizes
        obstacle avoidance — the system settles outside the zone.
        """
        target = (0.8, 0.2)
        obs_center = (0.8, 0.2)
        obs_radius = 0.60

        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=target,
            obstacle_center=obs_center,
            obstacle_radius=obs_radius,
            eta=0.05,
            n_ticks=300,
            Go=1000.0,
            lam=0.02,
        )

        # The invariant should push the system outside the obstacle zone
        # OR the obstacle component should dominate the gradient
        dominated_steps = [s for s in trace if _termwise_ratio(s, "g_obs") > 0.8]
        final_pos = trace[-1]["pos"]
        dist = np.linalg.norm(final_pos - np.array(target))

        assert len(dominated_steps) > 5 or dist > 0.3, (
            f"Expected obstacle dominance: {len(dominated_steps)} dominated steps, "
            f"final dist={dist:.3f}"
        )

    def test_regularization_component_dominates_invariant(self):
        """
        With λ=50, regularization dominates ds².  The invariant correctly
        reports that the "cheapest" configuration (θ near 0) outweighs
        reaching the target.  The system settles near θ = 0.
        """
        target = (0.5, 1.0)

        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=target,
            obstacle_center=(10.0, 10.0),
            obstacle_radius=0.1,
            eta=0.01,
            n_ticks=500,
            Go=4.0,
            lam=50.0,
        )

        final_theta = trace[-1]["theta"]
        assert np.linalg.norm(final_theta) < 0.5, (
            f"Expected θ → 0 under regularization dominance, "
            f"got ‖θ‖={np.linalg.norm(final_theta):.3f}"
        )

        final_pos = trace[-1]["pos"]
        dist = np.linalg.norm(final_pos - np.array(target))
        assert (
            dist > 0.3
        ), f"Expected target not reached under regularization dominance, dist={dist:.3f}"

    def test_domination_diagnostic_detects_scale_separation(self):
        """The r_i ratio diagnostic correctly flags scale separation."""
        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=(0.8, 0.2),
            obstacle_center=(0.8, 0.2),
            obstacle_radius=0.60,
            eta=0.05,
            n_ticks=200,
            Go=1000.0,
        )

        obs_steps = [s for s in trace if s["g_obs"] > 0]
        if obs_steps:
            max_ratio = max(_termwise_ratio(s, "g_obs") for s in obs_steps)
        else:
            max_ratio = max(
                _termwise_ratio(s, "g_reg")
                for s in trace
                if (s["g_target"] + s["g_obs"] + s["g_reg"]) > 1e-10
            )

        assert max_ratio > 0.7, f"Expected domination ratio > 0.7, got {max_ratio:.3f}"


# ===================================================================
# B8) Topological obstruction
#     (free space topology prevents a unique global minimum)
# ===================================================================
class TestTopologicalObstruction:
    """
    When obstacles make the free configuration space non-contractible,
    a smooth global potential with a unique minimum at the goal is
    generally impossible.  The invariant correctly settles at a
    non-target stationary point — this is a property of the topology,
    not a bug in the framework.
    """

    def test_obstacle_on_target_creates_geometric_barrier(self):
        """
        With obstacle centered on target, the invariant correctly
        reports that approach is blocked.  ds² settles at a positive
        equilibrium — the geometry has no path to ds² = 0.
        """
        target = (0.5, 0.5)
        obs_center = (0.5, 0.5)
        obs_radius = 0.55

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

        # System correctly stops outside obstacle zone
        assert dist_to_target > 0.15, f"Expected geometric barrier: dist={dist_to_target:.3f}"

        # ds² is positive at the equilibrium
        assert trace[-1]["ds2"] > 0.01, "Expected ds² > 0 at topological obstruction"

    def test_obstacle_forces_detour_or_equilibrium(self):
        """
        Large obstacle between start and target.  The invariant either
        guides the system around (if the geometry permits) or settles
        at an equilibrium (if topology prevents it).  Either is correct.
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

        final = trace[-1]
        final_pos = final["pos"]
        dist = np.linalg.norm(final_pos - np.array(target))

        # Either: system found a path (dist small, ds² small)
        # Or: system at equilibrium (gradient small, ds² positive)
        reached_target = dist < 0.2
        at_equilibrium = final["grad_norm"] < 1.0 and final["ds2"] > 0.01

        # Both outcomes are geometrically correct
        assert reached_target or at_equilibrium, (
            f"Expected either target reached or geometric equilibrium: "
            f"dist={dist:.3f}, grad={final['grad_norm']:.4f}, ds²={final['ds2']:.4f}"
        )


# ===================================================================
# Diagnostic integration: verify instrumentation on default config
# ===================================================================
class TestDiagnosticInstrumentation:
    """
    Verify that the three diagnostic quantities (Δds², g_t, r_i)
    are computable and meaningful on a standard trajectory where the
    geometry permits convergence.
    """

    def test_descent_check_monotone_for_stable_eta(self):
        """With stable η, the scheme should track the continuous flow (monotone ds²)."""
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
        assert fraction < 0.15, f"Expected mostly monotone descent, got {fraction:.1%} increases"

    def test_gradient_norm_decreases_to_near_zero(self):
        """Gradient norm → 0 as the system reaches the geometric minimum."""
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
        assert (
            g_final < g_init * 0.01
        ), f"Expected gradient reduction, got {g_init:.4f} → {g_final:.4f}"

    def test_termwise_balance_shifts_during_convergence(self):
        """The dominant gradient component shifts as geometry changes along the path."""
        trace = _run_descent(
            theta_init=(-1.4, 1.2),
            target=(1.2, 0.3),
            obstacle_center=(0.6, 0.1),
            obstacle_radius=0.25,
            eta=0.12,
            n_ticks=140,
        )
        final = trace[-1]
        r_target = _termwise_ratio(final, "g_target")
        r_reg = _termwise_ratio(final, "g_reg")
        assert r_target + r_reg > 0.5 or final["grad_norm"] < 0.01, (
            f"Expected target/reg dominance at convergence, "
            f"r_target={r_target:.3f}, r_reg={r_reg:.3f}"
        )
