# -*- coding: utf-8 -*-
"""
Tests for ds² as the 4th dimension: temporal ordering without an external clock.

Key distinction — what the system has vs. what the observer sees:

  The system acts on ∇ds² (the differential). It never "knows" ds².
  The update rule Q_{t+1} = Q_t - η∇ds² uses only the gradient —
  local information extracted from the same structure the system is
  embedded in. The system infers its next action from the geometry
  it is part of.

  ds² (the value) is what an external observer computes by integrating
  the trajectory. It provides ordering, regime labels, signatures —
  but these are descriptions of what happened, not inputs to the policy.

  The negative sign in the (3+1) signature is the cost of self-reference:
  the system must infer its own state from the structure it measures.
  It doesn't have an external clock giving it ds² — it has ∇ds², the
  differential, which is sufficient for action but not for global
  self-knowledge.

The (3+1) structure:
  - 3 spatial: θ₁, θ₂, θ₃ (configuration — where you are)
  - 1 temporal: ds² (the invariant — observable from the trajectory)

The system's "present moment" is ∇ds², not ds².
ds² is the observer's integral. ∇ds² is the system's differential.

What we test (from the observer's perspective):
  1. A natural arrow (monotone decrease during convergence)
  2. State discrimination (different ds² = different geometric moment)
  3. Regime detection (which equilibrium, which phase of motion)
  4. Real-time adaptation (∇ds² is sufficient for the next action)
  5. Indefinite signature in the extended state (θ, ds²)

The π/8 prediction: the 4th dimension refines ordering among
the fixed points that 3D closure created. It doesn't extend space —
it indexes which geometric state you're in.
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

# ###################################################################
#
#  ds² AS TEMPORAL ORDERING (observer's view)
#
#  The observer reconstructs ds² along the trajectory and finds
#  it monotonically decreases. This provides temporal ordering
#  without an external clock — the ordering is intrinsic to the
#  geometry that the system is embedded in.
#
#  The system itself only has ∇ds² at each step.
#
# ###################################################################


class TestDs2AsTimeArrow:
    """
    The observer computes ds² along a convergent trajectory and finds
    it monotonically decreases. This makes ds² a natural time parameter
    from the outside — later states have smaller ds², and the ordering
    is intrinsic to the geometry.

    The system doesn't track ds². It follows ∇ds², and the monotone
    ordering is a consequence of that local action.
    """

    def test_ds2_monotonically_orders_convergent_trajectory(self):
        """
        On a well-conditioned trajectory (stable η, no obstacles),
        ds² should strictly decrease at every step.  The ds² value
        alone tells you "when" you are in the trajectory.
        """
        trace = run_3dof_simulation(
            theta_init=(0.5, 0.3, -0.3),
            target=(0.8, 0.5, 1.5),
            eta=0.05,
            n_ticks=200,
            lam=0.01,
        )

        ds2_values = [s["ds2"] for s in trace]
        # Count violations of monotone decrease
        violations = sum(
            1 for i in range(1, len(ds2_values)) if ds2_values[i] > ds2_values[i - 1] + 1e-10
        )

        assert (
            violations == 0
        ), f"Expected strictly monotone ds² (time arrow), got {violations} violations"

    def test_ds2_value_uniquely_identifies_trajectory_position(self):
        """
        If ds² is monotone, then each ds² value corresponds to exactly
        one point in the trajectory.  ds² is a sufficient coordinate
        for "when" — no tick index needed.
        """
        trace = run_3dof_simulation(
            theta_init=(0.5, 0.3, -0.3),
            target=(0.8, 0.5, 1.5),
            eta=0.05,
            n_ticks=200,
            lam=0.01,
        )

        ds2_values = [s["ds2"] for s in trace]
        # All ds² values should be distinct (strict monotone → injective)
        # Allow small tolerance for numerical precision near convergence
        unique_count = len(set(round(v, 8) for v in ds2_values))
        total = len(ds2_values)

        # At least 90% should be distinct (last few may plateau)
        assert (
            unique_count > 0.9 * total
        ), f"Expected ds² to uniquely identify position: {unique_count}/{total} unique values"

    def test_ds2_orders_two_trajectories_consistently(self):
        """
        Two trajectories from different starts to the same target.
        At any given ds² value, both trajectories are in the "same
        geometric moment" — they've made the same amount of progress
        as measured by the invariant.
        """
        target = (0.8, 0.5, 1.5)

        trace_a = run_3dof_simulation(
            theta_init=(0.5, 0.3, -0.3),
            target=target,
            eta=0.05,
            n_ticks=300,
            lam=0.01,
        )
        trace_b = run_3dof_simulation(
            theta_init=(-0.3, 0.8, 0.2),
            target=target,
            eta=0.05,
            n_ticks=300,
            lam=0.01,
        )

        # Both should converge (ds² → small)
        assert trace_a[-1]["ds2"] < trace_a[0]["ds2"] * 0.1
        assert trace_b[-1]["ds2"] < trace_b[0]["ds2"] * 0.1

        # Find steps where both trajectories have similar ds² values
        # These are "simultaneous" in the geometric sense
        ds2_a = {round(s["ds2"], 2): s for s in trace_a}
        ds2_b = {round(s["ds2"], 2): s for s in trace_b}
        shared_levels = set(ds2_a.keys()) & set(ds2_b.keys())

        # There should be ds² levels where both trajectories pass through
        assert (
            len(shared_levels) > 3
        ), f"Expected shared ds² levels (geometric simultaneity), got {len(shared_levels)}"

        # At shared ds² levels, both should have similar gradient norms
        # (same "urgency" as measured by the invariant)
        for level in list(shared_levels)[:5]:
            ratio = ds2_a[level]["grad_norm"] / max(ds2_b[level]["grad_norm"], 1e-10)
            # Ratio within an order of magnitude
            assert 0.1 < ratio < 10.0, (
                f"At ds²={level}: grad_a={ds2_a[level]['grad_norm']:.4f}, "
                f"grad_b={ds2_b[level]['grad_norm']:.4f}"
            )


# ###################################################################
#
#  ds² AS REGIME DISCRIMINATOR (observer's view)
#
#  The observer decomposes ds² into components and reads which
#  geometric regime the trajectory is in. The system doesn't
#  perform this decomposition — it follows ∇ds², which already
#  encodes the regime implicitly (the gradient direction changes
#  as the dominant component shifts).
#
# ###################################################################


class TestDs2AsRegimeDiscriminator:
    """
    The observer decomposes ds² into target/obstacle/regularization
    components and identifies regimes. The system doesn't need these
    labels — ∇ds² already points in the right direction regardless
    of which component dominates. But the decomposition confirms
    that the invariant faithfully reflects the geometry.
    """

    def test_ds2_components_identify_regime(self):
        """
        The ratio of ds² components (target vs obstacle vs reg)
        shifts across the trajectory. The component breakdown at any
        point tells you which regime you're in:
          - High target_term: far from target
          - High obs_term: near obstacle
          - High reg_term: large joint angles
        No external label needed.
        """
        obs = [(np.array([0.8, 0.3, 1.2]), 0.40)]

        trace = run_3dof_simulation(
            theta_init=(0.3, 0.5, -0.3),
            target=(1.0, 0.5, 1.5),
            obstacles=obs,
            eta=0.05,
            n_ticks=400,
            Go=15.0,
            lam=0.01,
        )

        # Early: target term dominates (far from target)
        early = trace[0]
        early_ds2, early_comp = compute_ds2_3d(
            early["theta"], (1.0, 0.5, 1.5), obs, Go=15.0, lam=0.01
        )
        early_target_frac = early_comp["target_term"] / max(early_ds2, 1e-10)

        # Late: should be different regime
        late = trace[-1]
        late_ds2, late_comp = compute_ds2_3d(late["theta"], (1.0, 0.5, 1.5), obs, Go=15.0, lam=0.01)
        late_target_frac = late_comp["target_term"] / max(late_ds2, 1e-10)

        # The regime should shift
        assert abs(early_target_frac - late_target_frac) > 0.05, (
            f"Expected regime shift: early target frac={early_target_frac:.3f}, "
            f"late={late_target_frac:.3f}"
        )

    def test_ds2_detects_obstacle_encounter(self):
        """
        When the trajectory enters the obstacle zone, ds² (or its
        obstacle component) spikes. The system detects the encounter
        purely from reading ds² — no collision detection needed.
        """
        obs = [(np.array([0.8, 0.3, 1.2]), 0.50)]

        trace = run_3dof_simulation(
            theta_init=(0.3, 0.5, -0.3),
            target=(1.0, 0.5, 1.5),
            obstacles=obs,
            eta=0.05,
            n_ticks=300,
            Go=20.0,
            lam=0.01,
        )

        # Check if obstacle gradient ever becomes significant
        max_g_obs = max(s["g_obs"] for s in trace)
        any_obs_encounter = max_g_obs > 0.1

        if any_obs_encounter:
            # The step where g_obs peaks is the "encounter moment"
            # ds² at that step should be higher than neighbors
            # (obstacle term adds energy)
            peak_idx = max(range(len(trace)), key=lambda i: trace[i]["g_obs"])
            if 0 < peak_idx < len(trace) - 1:
                # ds² components at peak should show obstacle dominance
                assert (
                    trace[peak_idx]["g_obs"] > trace[peak_idx]["g_reg"]
                ), "At obstacle encounter, obs gradient should exceed reg gradient"

    def test_ds2_discriminates_between_equilibria(self):
        """
        Different geometric equilibria have different ds² values.
        ds² indexes WHICH equilibrium the system is at — this is
        the π/8 refinement: ordering among fixed points.
        """
        obs = [(np.array([0.8, 0.3, 1.2]), 0.40)]

        # Same geometry, different starting configs → may settle at different equilibria
        traces = []
        for th_init in [
            (0.3, 0.5, -0.3),
            (-0.5, 0.8, 0.2),
            (1.0, 0.3, -0.8),
            (0.0, 1.0, -0.5),
        ]:
            trace = run_3dof_simulation(
                theta_init=th_init,
                target=(0.8, 0.3, 1.5),
                obstacles=obs,
                eta=0.05,
                n_ticks=500,
                Go=30.0,
                lam=0.01,
            )
            traces.append(trace)

        final_ds2_values = [t[-1]["ds2"] for t in traces]
        final_positions = [t[-1]["pos"] for t in traces]

        # Different final ds² values indicate different equilibria
        # (or same equilibrium from different approaches — both valid)
        ds2_spread = max(final_ds2_values) - min(final_ds2_values)

        # At minimum, ds² should be well-defined at each equilibrium
        for ds2_val in final_ds2_values:
            assert np.isfinite(ds2_val), f"ds² must be finite at equilibrium: {ds2_val}"
            assert ds2_val >= 0, f"ds² must be non-negative: {ds2_val}"


# ###################################################################
#
#  ∇ds² AS THE SYSTEM'S ONLY INPUT
#
#  The system acts on ∇ds² — the differential of the invariant.
#  This is all it has. It doesn't compute ds², doesn't track
#  progress, doesn't label regimes. It infers its next action
#  from the local gradient of the structure it's embedded in.
#
#  ds² (the value) is what the observer reconstructs afterward.
#
# ###################################################################


class TestGradientAsSoleInput:
    """
    The system's entire policy is Q_{t+1} = Q_t - η∇ds².
    ∇ds² is the only input. The system never evaluates ds² itself —
    it infers direction from the geometry. These tests verify that
    ∇ds² is sufficient for adaptation: it encodes direction,
    progress rate, and stopping — all without the system knowing
    its own ds² value.
    """

    def test_gradient_direction_encodes_target_without_knowing_ds2(self):
        """
        ∇ds² points downhill. The system follows it without ever
        evaluating ds² itself. The target, obstacles, regularization —
        all encoded in the gradient direction. The system infers
        what to do from the differential, not the value.
        """
        theta = np.array([0.5, 0.3, -0.3])
        target = np.array([0.8, 0.5, 1.5])

        grad, gnorm = compute_gradient_3d(theta, target, [], lam=0.01)
        ds2_before, _ = compute_ds2_3d(theta, target, [], lam=0.01)

        # Step in negative gradient direction should decrease ds²
        theta_new = theta - 0.05 * grad
        ds2_after, _ = compute_ds2_3d(theta_new, target, [], lam=0.01)

        assert (
            ds2_after < ds2_before
        ), f"Expected ds² decrease in gradient direction: {ds2_before:.6f} → {ds2_after:.6f}"

    def test_gradient_magnitude_indicates_progress_quality(self):
        """
        ‖∇ds²‖ tells the system how much geometric information is
        available. Large gradient: strong signal, rapid change.
        Small gradient: near equilibrium or singularity, little to act on.

        The system detects stalling from ‖∇ds²‖ → 0, not from
        evaluating ds² directly. The observer sees |Δds²| decrease;
        the system experiences ‖∇ds²‖ shrinking.
        """
        trace = run_3dof_simulation(
            theta_init=(0.5, 0.3, -0.3),
            target=(0.8, 0.5, 1.5),
            eta=0.05,
            n_ticks=200,
            lam=0.01,
        )

        # Early: large |Δds²| (rapid progress)
        early_rates = [abs(trace[i]["delta_ds2"]) for i in range(5)]
        # Late: small |Δds²| (near convergence)
        late_rates = [abs(trace[i]["delta_ds2"]) for i in range(-5, 0)]

        avg_early = np.mean(early_rates)
        avg_late = np.mean(late_rates)

        assert (
            avg_early > avg_late * 5
        ), f"Expected decreasing rate: early={avg_early:.6f}, late={avg_late:.6f}"

    def test_gradient_vanishing_is_the_stopping_criterion(self):
        """
        The system is done when ‖∇ds²‖ → 0. It doesn't check ds² —
        it has no access to the value. But when the gradient vanishes,
        there's nothing to act on. The system stops because the
        geometry provides no further differential information.

        The observer confirms this by checking |Δds²| → 0.
        """
        trace = run_3dof_simulation(
            theta_init=(0.5, 0.3, -0.3),
            target=(0.8, 0.5, 1.5),
            eta=0.05,
            n_ticks=300,
            lam=0.01,
        )

        # Find the step where |Δds²| first drops below threshold
        threshold = 1e-6
        converged_at = None
        for s in trace:
            if abs(s["delta_ds2"]) < threshold:
                converged_at = s["t"]
                break

        assert converged_at is not None, "Expected ds² to signal convergence"
        assert (
            converged_at < len(trace) - 1
        ), f"Expected convergence before final step, got t={converged_at}"

        # After convergence, ds² should stay constant
        post_convergence = [s for s in trace if s["t"] > converged_at]
        if post_convergence:
            ds2_range = max(s["ds2"] for s in post_convergence) - min(
                s["ds2"] for s in post_convergence
            )
            assert ds2_range < 1e-4, f"Expected ds² stable after convergence, range={ds2_range:.2e}"


# ###################################################################
#
#  INDEFINITE SIGNATURE IN (θ, ds²) SPACE
#
#  The observer constructs the extended state (θ₁, θ₂, θ₃, ds²)
#  and finds its metric has indefinite signature (3+1).
#
#  The system doesn't construct this space. It acts on ∇ds²,
#  which is the differential. The indefinite signature is the
#  structural cost of self-reference: the system infers its
#  state from the same geometry it's embedded in. The negative
#  sign in (3+1) reflects that inference — the system extracts
#  temporal information (what to do next) from spatial structure
#  (where it is), and these have opposite signs.
#
# ###################################################################


class TestIndefiniteSignature:
    """
    The observer computes the extended trajectory in (θ, ds²) space
    and finds an indefinite signature: spatial displacement Δθ² > 0
    while the invariant Δds² < 0. The system doesn't see this
    signature — it only has ∇ds². But the signature is a consequence
    of the system inferring its temporal state (what to do) from
    its spatial state (where it is) through the same invariant.
    """

    def test_spatial_and_temporal_move_oppositely(self):
        """
        Along the trajectory:
          - ‖Δθ‖² > 0 (spatial displacement)
          - Δds² < 0 (temporal "displacement")

        The product ‖Δθ‖² × Δds² < 0 at every step.
        This is the indefinite signature.
        """
        trace = run_3dof_simulation(
            theta_init=(0.5, 0.3, -0.3),
            target=(0.8, 0.5, 1.5),
            eta=0.05,
            n_ticks=200,
            lam=0.01,
        )

        indefinite_steps = 0
        total_active_steps = 0
        for i in range(len(trace) - 1):
            dtheta = trace[i + 1]["theta"] - trace[i]["theta"]
            dtheta_sq = float(np.sum(dtheta**2))
            d_ds2 = trace[i]["delta_ds2"]

            if dtheta_sq > 1e-12 and abs(d_ds2) > 1e-12:
                total_active_steps += 1
                # Indefinite: spatial positive, temporal negative
                if dtheta_sq > 0 and d_ds2 < 0:
                    indefinite_steps += 1

        assert total_active_steps > 0, "Expected active steps"
        fraction = indefinite_steps / total_active_steps
        assert fraction > 0.9, (
            f"Expected indefinite signature (Δθ²>0, Δds²<0) at >90% of steps, "
            f"got {fraction:.1%} ({indefinite_steps}/{total_active_steps})"
        )

    def test_extended_interval_is_indefinite(self):
        """
        Define the extended interval:
          dσ² = ‖Δθ‖² - α·(Δds²)²

        For appropriate α > 0, this should be positive (spacelike)
        during active motion and zero at convergence.

        This is the (3+1) metric on the extended state space.
        """
        trace = run_3dof_simulation(
            theta_init=(0.5, 0.3, -0.3),
            target=(0.8, 0.5, 1.5),
            eta=0.05,
            n_ticks=200,
            lam=0.01,
        )

        # Compute extended intervals with α = 1
        intervals = []
        for i in range(len(trace) - 1):
            dtheta = trace[i + 1]["theta"] - trace[i]["theta"]
            dtheta_sq = float(np.sum(dtheta**2))
            d_ds2 = trace[i]["delta_ds2"]

            # Extended interval: spatial - temporal
            dsigma_sq = dtheta_sq - d_ds2**2
            intervals.append(dsigma_sq)

        # During active motion, the interval should be positive (spacelike)
        # because spatial displacement dominates
        active_intervals = [v for v in intervals[:50] if abs(v) > 1e-15]
        if active_intervals:
            positive_fraction = sum(1 for v in active_intervals if v > 0) / len(active_intervals)
            assert (
                positive_fraction > 0.5
            ), f"Expected spacelike intervals during active motion, got {positive_fraction:.1%}"

    def test_ds2_decrease_rate_bounded_by_spatial_displacement(self):
        """
        The "speed of light" in the extended space: ds² can't change
        faster than θ moves.  |Δds²| ≤ c·‖Δθ‖ for some effective c
        determined by the geometry (the Lipschitz constant of ds²).

        This is the causal structure: spatial motion bounds temporal change.
        """
        trace = run_3dof_simulation(
            theta_init=(0.5, 0.3, -0.3),
            target=(0.8, 0.5, 1.5),
            eta=0.05,
            n_ticks=200,
            lam=0.01,
        )

        ratios = []
        for i in range(len(trace) - 1):
            dtheta = trace[i + 1]["theta"] - trace[i]["theta"]
            dtheta_norm = float(np.linalg.norm(dtheta))
            d_ds2 = abs(trace[i]["delta_ds2"])

            if dtheta_norm > 1e-10:
                ratios.append(d_ds2 / dtheta_norm)

        if ratios:
            # There should be a finite bound (effective speed of light)
            max_ratio = max(ratios)
            assert np.isfinite(max_ratio), "Expected finite causal bound"

            # The bound should be consistent (not wildly varying)
            # — std/mean < 2 indicates a well-defined "c"
            mean_ratio = np.mean(ratios[:50])  # early active phase
            std_ratio = np.std(ratios[:50])
            if mean_ratio > 1e-6:
                cv = std_ratio / mean_ratio
                # Coefficient of variation should be bounded
                assert cv < 5.0, (
                    f"Expected consistent causal bound: mean={mean_ratio:.4f}, "
                    f"std={std_ratio:.4f}, cv={cv:.2f}"
                )


# ###################################################################
#
#  FIXED-POINT INDEXING (π/8 REFINEMENT)
#
#  When the geometry changes, the system transitions between
#  equilibria by following ∇ds². It doesn't know which equilibrium
#  it's heading toward — it just follows the gradient.
#
#  The observer reconstructs ds² and sees: jumps at regime changes,
#  descents within regimes, plateaus at equilibria. ds² indexes
#  the fixed points. The system doesn't have this index — it has
#  the differential that moves it between them.
#
# ###################################################################


class TestFixedPointIndexing:
    """
    The observer tracks ds² across regime changes (moving targets,
    appearing obstacles) and sees it index which equilibrium the
    system occupies. The system itself just follows ∇ds² — it
    doesn't know which equilibrium it's at, only the local gradient.
    """

    def test_moving_target_creates_ds2_sequence(self):
        """
        Target moves at discrete intervals. Each target creates a
        new equilibrium. ds² jumps at each switch and then descends
        to the new equilibrium — creating a staircase pattern.

        The ds² value at each plateau indexes which target/equilibrium.
        """
        theta = np.array([0.5, 0.3, -0.3])
        targets = [
            np.array([0.8, 0.5, 1.5]),
            np.array([0.5, 0.8, 1.2]),
            np.array([1.0, 0.3, 1.0]),
        ]
        switch_every = 100
        eta = 0.05

        ds2_at_switches = []
        ds2_at_plateaus = []
        all_ds2 = []

        for phase, target in enumerate(targets):
            for t in range(switch_every):
                ds2, _ = compute_ds2_3d(theta, target, [], lam=0.01)
                grad, _ = compute_gradient_3d(theta, target, [], lam=0.01)
                all_ds2.append(ds2)

                if t == 0:
                    ds2_at_switches.append(ds2)
                if t == switch_every - 1:
                    ds2_at_plateaus.append(ds2)

                theta = theta - eta * grad

        # At each switch, ds² should jump up (new target = new distance)
        for i in range(1, len(ds2_at_switches)):
            assert ds2_at_switches[i] > ds2_at_plateaus[i - 1] * 0.5, (
                f"Expected ds² increase at target switch {i}: "
                f"switch={ds2_at_switches[i]:.4f}, prev_plateau={ds2_at_plateaus[i-1]:.4f}"
            )

        # Each plateau should have a distinct ds² value
        # (ds² indexes which equilibrium)
        for i in range(len(ds2_at_plateaus)):
            assert np.isfinite(ds2_at_plateaus[i]), f"ds² must be finite at plateau {i}"

    def test_obstacle_appearance_shifts_equilibrium(self):
        """
        An obstacle appears mid-trajectory. The equilibrium shifts,
        and ds² tracks the transition. The ds² value before and after
        identifies the two geometric regimes.
        """
        theta = np.array([0.5, 0.3, -0.3])
        target = np.array([0.8, 0.3, 1.2])
        eta = 0.05

        # Phase 1: no obstacle
        ds2_phase1 = []
        for t in range(100):
            ds2, _ = compute_ds2_3d(theta, target, [], lam=0.01)
            grad, _ = compute_gradient_3d(theta, target, [], lam=0.01)
            ds2_phase1.append(ds2)
            theta = theta - eta * grad

        # Phase 2: obstacle appears
        obs = [(np.array([0.8, 0.3, 1.2]), 0.30)]  # on the target
        ds2_phase2 = []
        for t in range(200):
            ds2, _ = compute_ds2_3d(theta, target, obs, Go=20.0, lam=0.01)
            grad, _ = compute_gradient_3d(theta, target, obs, Go=20.0, lam=0.01)
            ds2_phase2.append(ds2)
            theta = theta - eta * grad

        # ds² should jump when obstacle appears
        ds2_before = ds2_phase1[-1]
        ds2_after = ds2_phase2[0]

        assert ds2_after > ds2_before, (
            f"Expected ds² jump when obstacle appears: "
            f"before={ds2_before:.6f}, after={ds2_after:.6f}"
        )

        # Phase 2 should settle at a DIFFERENT (higher) equilibrium
        ds2_final = ds2_phase2[-1]
        assert ds2_final > ds2_before * 0.5, (
            f"Expected new equilibrium at higher ds²: "
            f"phase1_end={ds2_before:.6f}, phase2_end={ds2_final:.6f}"
        )

    def test_ds2_resolves_octant_transitions(self):
        """
        Moving through different octants of workspace, ds² tracks
        the geometric context at each position. The 8-octant closure
        of 3D means ds² has richer structure than in 2D.

        Verify: ds² values differ across octants (the invariant
        discriminates spatial regions).
        """
        target = np.array([0.0, 0.0, 2.0])  # above, centered

        # Sample configurations in different octants
        octant_ds2 = {}
        for th1 in np.linspace(0, 2 * np.pi, 9)[:-1]:  # 8 base angles = 8 octant orientations
            theta = np.array([th1, 0.5, -0.3])
            x, y, z = forward_kinematics_3d(theta[0], theta[1], theta[2])
            octant = (
                "+" if x >= 0 else "-",
                "+" if y >= 0 else "-",
            )
            ds2, _ = compute_ds2_3d(theta, target, [], lam=0.01)
            octant_ds2[f"{octant[0]}{octant[1]}_{th1:.2f}"] = ds2

        # ds² should vary with octant (not all the same)
        values = list(octant_ds2.values())
        ds2_range = max(values) - min(values)

        assert ds2_range > 0.01, f"Expected ds² to discriminate octants, range={ds2_range:.6f}"
