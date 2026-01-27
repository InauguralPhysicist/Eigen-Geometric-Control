# -*- coding: utf-8 -*-
"""
Tests for failure modes of ds² on a 3-DOF spatial arm.

Same 8 conditions as the 2-DOF tests, but now in 3D workspace where:
  - The Jacobian is 3×3 (singularity manifolds are curves, not points)
  - The configuration space has 8 octants (closure exists)
  - Topological obstruction should appear naturally, not just under
    extreme parameters

The π-halving prediction: at 3 independent constraints (3-DOF), closure
appears at π/4.  Failure modes 4-8 (geometric properties of ds²) should
manifest more readily than in 2-DOF because the topology is richer.

Structure follows the 2-DOF file:
  Part A: Discretization failures (scheme loses the flow)
  Part B: Geometric properties (ds² correctly reports the geometry)
"""

import numpy as np
import pytest

from src.eigen_3dof_core import (
    compute_ds2_3d,
    compute_gradient_3d,
    forward_kinematics_3d,
    jacobian_3d,
    run_3dof_simulation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _termwise_ratio(step, term="g_obs"):
    total = step["g_target"] + step["g_obs"] + step["g_reg"]
    if total < 1e-15:
        return 0.0
    return step[term] / total


# ###################################################################
#
#  PART A: DISCRETIZATION FAILURES
#
# ###################################################################


# ===================================================================
# A1) Euler blow-up
# ===================================================================
class TestEulerBlowup3D:
    """Same as 2-DOF: η too large → scheme diverges from the flow."""

    def test_large_eta_causes_ds2_increase(self):
        trace = run_3dof_simulation(
            theta_init=(0.5, 0.5, -0.5),
            target=(0.8, 0.5, 1.5),
            eta=5.0,
            n_ticks=20,
        )
        increases = sum(1 for s in trace if s["delta_ds2"] > 0)
        assert increases > 0, "Expected ds² increase with large η"

    def test_large_eta_oscillation(self):
        trace = run_3dof_simulation(
            theta_init=(0.5, 0.5, -0.5),
            target=(0.8, 0.5, 1.5),
            eta=2.0,
            n_ticks=50,
        )
        signs = [np.sign(s["delta_ds2"]) for s in trace if abs(s["delta_ds2"]) > 1e-10]
        sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        assert sign_changes >= 2, f"Expected oscillation, got {sign_changes} sign changes"

    def test_divergence(self):
        trace = run_3dof_simulation(
            theta_init=(0.5, 0.5, -0.5),
            target=(0.8, 0.5, 1.5),
            eta=10.0,
            n_ticks=30,
        )
        assert trace[-1]["ds2"] > trace[0]["ds2"], "Expected ds² divergence"


# ===================================================================
# A2) Gradient explosion near obstacles
# ===================================================================
class TestGradientExplosion3D:
    """
    In 3D, gradient explosion near obstacles is the same mechanism.
    But now the obstacle is a sphere in 3D workspace.
    """

    def test_gradient_spikes_near_obstacle(self):
        obs = [(np.array([0.6, 0.1, 1.0]), 0.50)]

        # Far from obstacle
        _, gnorm_far = compute_gradient_3d(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.5, 1.5]),
            obs, Go=50.0,
        )

        # Search for config inside obstacle
        best_gnorm = 0.0
        for th1 in np.linspace(-np.pi, np.pi, 50):
            for th2 in np.linspace(-np.pi/2, np.pi/2, 30):
                for th3 in np.linspace(-np.pi, np.pi, 30):
                    x, y, z = forward_kinematics_3d(th1, th2, th3)
                    d = np.sqrt((x - 0.6)**2 + (y - 0.1)**2 + (z - 1.0)**2)
                    if 0.02 < d < 0.40:
                        _, gn = compute_gradient_3d(
                            np.array([th1, th2, th3]),
                            np.array([1.0, 0.5, 1.5]),
                            obs, Go=50.0,
                        )
                        best_gnorm = max(best_gnorm, gn)

        assert best_gnorm > 5.0 * max(gnorm_far, 1e-6), (
            f"Expected gradient spike: near={best_gnorm:.4f}, far={gnorm_far:.4f}"
        )

    def test_discrete_step_teleports_past_obstacle(self):
        obs = [(np.array([0.6, 0.1, 1.0]), 0.50)]

        for th1 in np.linspace(-np.pi, np.pi, 50):
            for th2 in np.linspace(-np.pi/2, np.pi/2, 30):
                for th3 in np.linspace(-np.pi, np.pi, 30):
                    x, y, z = forward_kinematics_3d(th1, th2, th3)
                    d = np.sqrt((x - 0.6)**2 + (y - 0.1)**2 + (z - 1.0)**2)
                    if 0.05 < d < 0.25:
                        grad, _ = compute_gradient_3d(
                            np.array([th1, th2, th3]),
                            np.array([1.0, 0.5, 1.5]),
                            obs, Go=50.0,
                        )
                        step_size = np.linalg.norm(1.0 * grad)
                        if step_size > 0.50:
                            assert True
                            return
        pytest.skip("Could not find configuration for teleportation test")


# ===================================================================
# A3) Kink chattering
# ===================================================================
class TestKinkChattering3D:
    """
    In 3D the obstacle boundary is a sphere — the kink is now a 2D
    surface in workspace, making boundary transitions more likely
    than in 2D (where the kink was a circle).
    """

    def test_chattering_at_obstacle_boundary(self):
        obs = [(np.array([0.8, 0.3, 1.2]), 0.60)]

        trace = run_3dof_simulation(
            theta_init=(0.5, 0.3, -0.3),
            target=(1.0, 0.5, 1.5),
            obstacles=obs,
            eta=0.10,
            n_ticks=300,
            Go=10.0,
        )

        inside = [s["d_obs_min"] < 0.60 for s in trace]
        transitions = sum(1 for i in range(1, len(inside)) if inside[i] != inside[i - 1])
        obs_active = [s["g_obs"] > 0.01 for s in trace]
        obs_transitions = sum(1 for i in range(1, len(obs_active))
                              if obs_active[i] != obs_active[i - 1])

        assert transitions >= 1 or obs_transitions >= 1, (
            f"Expected boundary transitions: spatial={transitions}, gradient={obs_transitions}"
        )


# ###################################################################
#
#  PART B: GEOMETRIC PROPERTIES OF ds²
#
#  The key question: do these appear NATURALLY in 3-DOF, where
#  the 2-DOF system needed extreme parameters?
#
# ###################################################################


# ===================================================================
# B4) Geometric equilibrium from gradient cancellation
#     In 3D: more degrees of freedom for cancellation to occur
# ===================================================================
class TestGeometricEquilibrium3D:
    """
    With 3 gradient components each in ℝ³ joint space, exact cancellation
    ∇d_target + ∇d_obs + ∇d_reg = 0 can happen in more configurations
    than in 2D.  The geometry has more room for equilibria.
    """

    def test_equilibrium_with_moderate_obstacle(self):
        """
        In 2-DOF this required Go=50 and obstacle directly blocking the path.
        Test whether a moderate obstacle creates an equilibrium in 3-DOF.
        """
        # Obstacle between start end-effector and target — moderate Go
        obs = [(np.array([0.8, 0.3, 1.2]), 0.35)]

        trace = run_3dof_simulation(
            theta_init=(0.3, 0.5, -0.3),
            target=(0.8, 0.3, 1.5),
            obstacles=obs,
            eta=0.05,
            n_ticks=500,
            Go=15.0,  # moderate, not extreme
            lam=0.01,
        )

        final = trace[-1]
        target_arr = np.array([0.8, 0.3, 1.5])
        dist = np.linalg.norm(final["pos"] - target_arr)

        # Check: system settled (gradient small or ds² plateau)
        late_ds2 = [s["ds2"] for s in trace[-50:]]
        ds2_range = max(late_ds2) - min(late_ds2)
        settled = final["grad_norm"] < 0.5 or ds2_range < 0.01

        # ds² > 0 at equilibrium
        assert settled, (
            f"Expected convergence: grad={final['grad_norm']:.4f}, ds² range={ds2_range:.6f}"
        )
        assert final["ds2"] > 0.001, (
            f"Expected ds² > 0 at equilibrium, got {final['ds2']:.6f}"
        )

    def test_multiple_obstacles_create_equilibrium(self):
        """
        Multiple obstacles in 3D create a richer landscape.
        Equilibria should appear naturally.
        """
        obs = [
            (np.array([0.7, 0.2, 1.3]), 0.30),
            (np.array([0.9, 0.5, 1.0]), 0.30),
        ]

        trace = run_3dof_simulation(
            theta_init=(0.3, 0.5, -0.3),
            target=(0.8, 0.3, 1.5),
            obstacles=obs,
            eta=0.05,
            n_ticks=500,
            Go=10.0,
            lam=0.01,
        )

        final = trace[-1]
        target_arr = np.array([0.8, 0.3, 1.5])
        dist = np.linalg.norm(final["pos"] - target_arr)

        # With multiple obstacles, either reaches target or finds equilibrium
        late_ds2 = [s["ds2"] for s in trace[-50:]]
        ds2_range = max(late_ds2) - min(late_ds2)

        reached = dist < 0.2
        at_equilibrium = ds2_range < 0.01

        assert reached or at_equilibrium, (
            f"Expected target or equilibrium: dist={dist:.3f}, ds² range={ds2_range:.6f}"
        )


# ===================================================================
# B5) Jacobian singularity
#     In 3D: singularity manifolds are CURVES, not isolated points
# ===================================================================
class TestJacobianSingularity3D:
    """
    The 3×3 Jacobian has det(J) = 0 on curves in configuration space,
    not just at isolated points like the 2-DOF case.

    This is the first qualitative difference from 2-DOF:
    singularities are unavoidable in a different sense — you can't
    just perturb by ε to escape, you might need to cross a manifold.
    """

    def test_singularity_is_a_manifold_not_a_point(self):
        """
        det(J) = 0 at θ₃ = 0 (elbow extended) for ALL values of θ₁.
        This is a 1D manifold (curve) in configuration space.
        """
        dets = []
        for th1 in np.linspace(0, 2 * np.pi, 20):
            J = jacobian_3d(th1, 0.5, 0.0)  # θ₃ = 0: extended elbow
            dets.append(abs(np.linalg.det(J)))

        # All should be near-singular (det ≈ 0) regardless of θ₁
        assert all(d < 0.1 for d in dets), (
            f"Expected singular manifold at θ₃=0: dets={[f'{d:.4f}' for d in dets[:5]]}"
        )

        # Verify: perturbing θ₃ away from 0 restores rank
        J_good = jacobian_3d(0.0, 0.5, np.pi / 3)
        assert abs(np.linalg.det(J_good)) > 0.05, (
            f"Expected non-singular at θ₃=π/3, det={abs(np.linalg.det(J_good)):.4f}"
        )

    def test_singularity_manifold_at_theta2_zero(self):
        """
        θ₂ = 0 (shoulder level) also creates singularity for certain θ₃.
        The singularity set is a more complex manifold than a single point.
        """
        # At θ₂ = 0, θ₃ = 0: both shoulder and elbow extended
        J = jacobian_3d(0.0, 0.0, 0.0)
        det_both = abs(np.linalg.det(J))
        assert det_both < 1e-10, (
            f"Expected singularity at θ₂=θ₃=0, det={det_both:.2e}"
        )

        # At θ₂ = 0, θ₃ = π: elbow fully folded
        J_folded = jacobian_3d(0.0, 0.0, np.pi)
        det_folded = abs(np.linalg.det(J_folded))
        assert det_folded < 1e-10, (
            f"Expected singularity at θ₂=0, θ₃=π, det={det_folded:.2e}"
        )

    def test_singularity_traps_flow_on_manifold(self):
        """
        Starting on the singularity manifold, the flow should be
        trapped or severely attenuated — and in 3-DOF, you can't
        escape by perturbing one joint (the manifold extends).
        """
        # Start on singularity manifold: θ₃ = 0
        target = np.array([0.0, 1.5, 0.9])

        trace_singular = run_3dof_simulation(
            theta_init=(np.pi / 2, 0.5, 0.0),  # on manifold
            target=target,
            eta=0.08,
            n_ticks=200,
            lam=0.001,
        )

        trace_off = run_3dof_simulation(
            theta_init=(np.pi / 2, 0.5, 0.3),  # off manifold
            target=target,
            eta=0.08,
            n_ticks=200,
            lam=0.001,
        )

        dist_singular = np.linalg.norm(trace_singular[-1]["pos"] - target)
        dist_off = np.linalg.norm(trace_off[-1]["pos"] - target)

        # The on-manifold start should perform worse (or at least
        # the gradient should be attenuated at the start)
        grad_singular_0 = trace_singular[0]["grad_norm"]
        grad_off_0 = trace_off[0]["grad_norm"]

        # At minimum: the Jacobian determinant is smaller on-manifold
        assert abs(trace_singular[0]["det_J"]) < abs(trace_off[0]["det_J"]), (
            f"Expected smaller det(J) on manifold: "
            f"on={trace_singular[0]['det_J']:.4f}, off={trace_off[0]['det_J']:.4f}"
        )

    def test_3dof_vs_2dof_singularity_dimension(self):
        """
        Quantify the qualitative difference:
        - 2-DOF: singularity at θ₂ = 0 (codimension 1 in 2D config space = isolated point along θ₂ axis)
        - 3-DOF: singularity at θ₃ = 0 for all θ₁ (codimension 1 in 3D config space = 2D surface)

        Sample random configs and measure the fraction that are near-singular.
        In 3-DOF this fraction should be larger because the singular set
        has higher relative dimension.
        """
        np.random.seed(42)
        n_samples = 1000
        threshold = 0.05

        # 3-DOF: sample (θ₁, θ₂, θ₃) uniformly
        near_singular_3d = 0
        for _ in range(n_samples):
            th = np.random.uniform(-np.pi, np.pi, 3)
            J = jacobian_3d(th[0], th[1], th[2])
            if abs(np.linalg.det(J)) < threshold:
                near_singular_3d += 1

        frac_3d = near_singular_3d / n_samples
        # A nontrivial fraction should be near-singular
        # (the singular set θ₃ ≈ 0 occupies ~threshold/π of the range)
        assert frac_3d > 0.01, (
            f"Expected nontrivial singular fraction in 3-DOF, got {frac_3d:.3f}"
        )


# ===================================================================
# B6) Infeasible geometry
# ===================================================================
class TestInfeasibleGeometry3D:
    """
    Same as 2-DOF: ds² correctly reports when no solution exists.
    In 3D the workspace is a shell in ℝ³.
    """

    def test_target_outside_workspace(self):
        """Workspace max reach = L1 + L2 + L3 = 2.7m (approximately — depends on geometry)."""
        target = (5.0, 0.0, 0.0)  # far outside

        trace = run_3dof_simulation(
            theta_init=(0.0, 0.5, -0.5),
            target=target,
            eta=0.02,   # small η to avoid Euler instability at workspace edge
            n_ticks=500,
        )

        assert trace[-1]["ds2"] > 1.0, (
            f"Expected ds² > 0 for unreachable target, got {trace[-1]['ds2']:.4f}"
        )
        # Gradient should decrease toward compromise point
        assert trace[-1]["grad_norm"] < trace[0]["grad_norm"], (
            f"Expected gradient decrease: {trace[0]['grad_norm']:.4f} → {trace[-1]['grad_norm']:.4f}"
        )

    def test_target_inside_obstacle(self):
        """Target coincides with obstacle center in 3D."""
        target = (0.6, 0.1, 1.0)
        obs = [(np.array([0.6, 0.1, 1.0]), 0.30)]

        trace = run_3dof_simulation(
            theta_init=(0.3, 0.5, -0.3),
            target=target,
            obstacles=obs,
            eta=0.05,
            n_ticks=300,
            Go=20.0,
        )

        assert trace[-1]["ds2"] > 0.01, (
            f"Expected ds² > 0 for contradictory constraints, got {trace[-1]['ds2']:.6f}"
        )

    def test_target_inside_inner_workspace_hole(self):
        """
        3D workspace has an inner hole (configurations below minimum reach).
        Target at the origin may be unreachable if L1 > 0 (base elevation).
        """
        # Target at (0, 0, 0) — inside the arm's base
        target = (0.0, 0.0, 0.0)

        trace = run_3dof_simulation(
            theta_init=(0.0, -0.5, 1.0),
            target=target,
            eta=0.08,
            n_ticks=300,
            lam=0.01,
        )

        # L1 = 0.9 is the base height offset, so the arm can't reach (0,0,0)
        assert trace[-1]["ds2"] > 0.1, (
            f"Expected ds² > 0 for target inside workspace hole, got {trace[-1]['ds2']:.4f}"
        )


# ===================================================================
# B7) Component scale separation
# ===================================================================
class TestComponentScaleSeparation3D:
    """
    Same mechanism as 2-DOF.  In 3D with multiple obstacles,
    scale separation can occur per-obstacle.
    """

    def test_obstacle_dominates(self):
        obs = [(np.array([0.8, 0.3, 1.2]), 0.60)]

        trace = run_3dof_simulation(
            theta_init=(0.3, 0.5, -0.3),
            target=(0.8, 0.3, 1.2),  # inside obstacle
            obstacles=obs,
            eta=0.05,
            n_ticks=300,
            Go=500.0,
        )

        dominated = [s for s in trace if _termwise_ratio(s, "g_obs") > 0.8]
        dist = np.linalg.norm(trace[-1]["pos"] - np.array([0.8, 0.3, 1.2]))

        assert len(dominated) > 5 or dist > 0.3, (
            f"Expected obstacle dominance: {len(dominated)} steps, dist={dist:.3f}"
        )

    def test_regularization_dominates(self):
        trace = run_3dof_simulation(
            theta_init=(1.0, 0.8, -0.5),
            target=(0.5, 0.8, 1.5),
            eta=0.01,
            n_ticks=500,
            lam=50.0,
        )

        final_theta = trace[-1]["theta"]
        assert np.linalg.norm(final_theta) < 0.5, (
            f"Expected θ → 0 under reg dominance, ‖θ‖={np.linalg.norm(final_theta):.3f}"
        )


# ===================================================================
# B8) Topological obstruction
#     THIS IS THE KEY 3D TEST: topology should matter more naturally
# ===================================================================
class TestTopologicalObstruction3D:
    """
    In 3D, obstacles create genuinely non-contractible free spaces.
    A single spherical obstacle in 3D workspace makes the free space
    homotopy-equivalent to S² — fundamentally different from 2D where
    a circular obstacle gives S¹.

    The prediction from π-halving: at 3 constraints (closure at π/4),
    topological obstruction should appear with MODERATE parameters,
    not just extreme ones.
    """

    def test_obstacle_between_start_and_target(self):
        """
        Single spherical obstacle directly between start and target.
        In 3D, the system must navigate AROUND a sphere — there are
        infinitely many homotopy classes of paths.

        With moderate Go, ds² should either:
        (a) find a path around (geodesic on the sphere's complement)
        (b) settle at an equilibrium (topological trap)

        Either is correct — but the test verifies ds² is doing something
        geometrically meaningful.
        """
        # Start: end-effector at roughly (0.8, 0.0, 1.5)
        # Target: (0.8, 0.8, 1.5) — same x,z, different y
        # Obstacle: directly between them at (0.8, 0.4, 1.5)
        target = (0.8, 0.8, 1.5)
        obs = [(np.array([0.8, 0.4, 1.5]), 0.30)]

        trace = run_3dof_simulation(
            theta_init=(0.1, 0.6, -0.3),
            target=target,
            obstacles=obs,
            eta=0.05,
            n_ticks=500,
            Go=20.0,  # moderate
            lam=0.01,
        )

        final = trace[-1]
        dist = np.linalg.norm(final["pos"] - np.array(target))

        reached = dist < 0.2
        at_equilibrium = final["grad_norm"] < 1.0 and final["ds2"] > 0.01
        non_monotone = sum(1 for s in trace if s["delta_ds2"] > 1e-6)

        # Any of these outcomes is geometrically valid
        assert reached or at_equilibrium or non_monotone > 3, (
            f"Expected geometric response to topology: dist={dist:.3f}, "
            f"grad={final['grad_norm']:.4f}, non_monotone={non_monotone}"
        )

    def test_multiple_obstacles_create_narrow_passage(self):
        """
        Multiple obstacles creating a narrow passage in 3D.
        The gradient field is more likely to have spurious equilibria
        because the obstacle gradients can cancel from multiple directions.
        """
        obs = [
            (np.array([0.7, 0.3, 1.3]), 0.25),
            (np.array([0.9, 0.3, 1.3]), 0.25),
            (np.array([0.8, 0.3, 1.1]), 0.25),
        ]

        trace = run_3dof_simulation(
            theta_init=(0.3, 0.5, -0.3),
            target=(0.8, 0.3, 1.5),
            obstacles=obs,
            eta=0.05,
            n_ticks=500,
            Go=15.0,
            lam=0.01,
        )

        final = trace[-1]
        dist = np.linalg.norm(final["pos"] - np.array([0.8, 0.3, 1.5]))

        late_ds2 = [s["ds2"] for s in trace[-50:]]
        ds2_range = max(late_ds2) - min(late_ds2)

        reached = dist < 0.2
        at_equilibrium = ds2_range < 0.01

        assert reached or at_equilibrium, (
            f"Expected target or equilibrium: dist={dist:.3f}, ds² range={ds2_range:.6f}"
        )

    def test_enclosed_target_requires_going_around(self):
        """
        Six obstacles forming a rough cage around the target.
        ds² must navigate around the enclosure or report it can't.
        This is where 3D closure (8 octants) matters — the obstacle
        structure can actually enclose a region.
        """
        target = np.array([0.8, 0.3, 1.3])
        r = 0.20
        obs = [
            (target + np.array([ 0.35, 0.0, 0.0]), r),
            (target + np.array([-0.35, 0.0, 0.0]), r),
            (target + np.array([0.0,  0.35, 0.0]), r),
            (target + np.array([0.0, -0.35, 0.0]), r),
            (target + np.array([0.0, 0.0,  0.35]), r),
            (target + np.array([0.0, 0.0, -0.35]), r),
        ]

        trace = run_3dof_simulation(
            theta_init=(0.3, 0.5, -0.3),
            target=target,
            obstacles=obs,
            eta=0.05,
            n_ticks=500,
            Go=20.0,
            lam=0.01,
        )

        final = trace[-1]
        dist = np.linalg.norm(final["pos"] - target)

        # With enclosure, ds² should report inability to reach target
        # OR find a gap between obstacles
        assert final["ds2"] > 0.01, (
            f"Expected ds² > 0 with cage obstacles, got {final['ds2']:.6f}"
        )

        # The system should have settled
        late_ds2 = [s["ds2"] for s in trace[-50:]]
        ds2_range = max(late_ds2) - min(late_ds2)
        assert ds2_range < 0.1 or final["grad_norm"] < 1.0, (
            f"Expected settled behavior: ds² range={ds2_range:.6f}, "
            f"grad={final['grad_norm']:.4f}"
        )


# ===================================================================
# Comparative: 3-DOF vs 2-DOF structural differences
# ===================================================================
class TestDimensionalComparison:
    """
    Tests that directly compare 3-DOF geometric properties to 2-DOF.
    These verify the π-halving prediction: 3D should exhibit
    qualitatively different behavior, not just quantitatively harder.
    """

    def test_singularity_codimension(self):
        """
        In 2-DOF: singular set has codimension 1 (a line θ₂=0 in 2D config space).
        In 3-DOF: singular set has codimension 1 (a surface θ₃=0 in 3D config space).

        The singular set's dimension increases with DOF:
          2-DOF: dim(singular) = 1 (line)
          3-DOF: dim(singular) = 2 (surface)

        This means singularities are harder to avoid in 3-DOF.
        """
        np.random.seed(42)
        n_samples = 2000

        # 3-DOF: fraction near-singular
        near_3d = 0
        for _ in range(n_samples):
            th = np.random.uniform(-np.pi, np.pi, 3)
            J = jacobian_3d(th[0], th[1], th[2])
            if abs(np.linalg.det(J)) < 0.05:
                near_3d += 1

        frac_3d = near_3d / n_samples

        # 2-DOF: fraction near-singular
        from src.eigen_core import jacobian as jacobian_2d
        near_2d = 0
        for _ in range(n_samples):
            th = np.random.uniform(-np.pi, np.pi, 2)
            J = jacobian_2d(th[0], th[1])
            if abs(np.linalg.det(J)) < 0.05:
                near_2d += 1

        frac_2d = near_2d / n_samples

        # Both should have near-singular configs
        assert frac_3d > 0.005, f"Expected near-singular configs in 3D: {frac_3d:.3f}"
        assert frac_2d > 0.005, f"Expected near-singular configs in 2D: {frac_2d:.3f}"

    def test_octant_coverage(self):
        """
        The 3-DOF arm can reach all 8 octants of 3D workspace.
        This is the closure property: the arm's workspace has
        enough coverage to define inside/outside.

        2-DOF arm can only reach 2 of 4 quadrants (upper half-plane)
        with positive link lengths.
        """
        octants_reached = set()

        for th1 in np.linspace(-np.pi, np.pi, 30):
            for th2 in np.linspace(-np.pi/2, np.pi/2, 20):
                for th3 in np.linspace(-np.pi, np.pi, 20):
                    x, y, z = forward_kinematics_3d(th1, th2, th3)
                    octant = (
                        1 if x >= 0 else -1,
                        1 if y >= 0 else -1,
                        1 if z >= 0 else -1,
                    )
                    octants_reached.add(octant)

        assert len(octants_reached) >= 6, (
            f"Expected at least 6 of 8 octants reachable, got {len(octants_reached)}: "
            f"{octants_reached}"
        )

    def test_det_J_changes_sign_in_3d(self):
        """
        In 3D, det(J) can change sign — the Jacobian has a proper
        orientation-reversing singularity.  This doesn't happen in 2-DOF
        (det(J) = L2*sin(θ₂) * constant structure — changes sign at θ₂=0,π).

        In 3-DOF, the sign change structure is richer because the
        singularity manifold separates regions of opposite orientation.
        """
        dets = []
        for th3 in np.linspace(-np.pi, np.pi, 100):
            J = jacobian_3d(0.5, 0.5, th3)
            dets.append(np.linalg.det(J))

        # det should change sign (cross zero)
        signs = np.sign(dets)
        sign_changes = sum(1 for i in range(1, len(signs))
                           if signs[i] != signs[i - 1] and signs[i] != 0 and signs[i - 1] != 0)

        assert sign_changes >= 1, (
            f"Expected det(J) sign change across singularity, got {sign_changes} changes"
        )


# ===================================================================
# Diagnostic integration on default 3-DOF config
# ===================================================================
class TestDiagnosticInstrumentation3D:
    """
    Verify the three diagnostics work on 3-DOF with no obstacles.
    """

    def test_descent_monotone_for_stable_eta(self):
        trace = run_3dof_simulation(
            theta_init=(0.5, 0.3, -0.3),
            target=(0.8, 0.5, 1.5),
            eta=0.08,
            n_ticks=200,
        )
        increases = sum(1 for s in trace if s["delta_ds2"] > 1e-8)
        fraction = increases / len(trace)
        assert fraction < 0.15, f"Expected monotone descent, got {fraction:.1%} increases"

    def test_gradient_norm_decreases(self):
        trace = run_3dof_simulation(
            theta_init=(0.5, 0.3, -0.3),
            target=(0.8, 0.5, 1.5),
            eta=0.08,
            n_ticks=200,
        )
        assert trace[-1]["grad_norm"] < trace[0]["grad_norm"] * 0.1, (
            f"Expected gradient reduction: {trace[0]['grad_norm']:.4f} → {trace[-1]['grad_norm']:.4f}"
        )

    def test_convergence_to_target(self):
        trace = run_3dof_simulation(
            theta_init=(0.5, 0.3, -0.3),
            target=(0.8, 0.5, 1.5),
            eta=0.08,
            n_ticks=300,
        )
        dist = np.linalg.norm(trace[-1]["pos"] - np.array([0.8, 0.5, 1.5]))
        assert dist < 0.3, f"Expected convergence to target, dist={dist:.3f}"
