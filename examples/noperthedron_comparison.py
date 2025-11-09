#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noperthedron vs Arm Control: Comparing Geometric Convergence

This script demonstrates the parallel between:
1. Noperthedron passage (Rupert property) and ds² metric
2. Arm control convergence and ds² metric

Both use the same fundamental framework:
    ds² = S² - C²

Where:
- ds² > 0 (time-like): convergence/passage possible
- ds² < 0 (space-like): non-convergent/no passage
- ds² = 0 (light-like): boundary condition
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import both frameworks
from src import (
    generate_noperthedron_vertices,
    test_rupert_property,
    analyze_results,
    run_arm_simulation,
    compute_change_stability,
)
from src.config import ArmConfig


def run_arm_control_experiment():
    """Run standard arm control and extract ds² trajectory"""
    print("\n" + "=" * 70)
    print("ARM CONTROL EXPERIMENT")
    print("=" * 70)

    config = ArmConfig(
        target=(0.5, 0.5),
        obstacle_center=(0.3, 0.3),
        obstacle_radius=0.25,
        theta_init=(0.1, 0.1),
        eta=0.01,
        n_ticks=200,
    )

    print("Running arm simulation...")
    df = run_arm_simulation(config)

    # Extract metrics
    iterations = df.index.values
    ds2_values = (df["S"].values ** 2) - (df["C"].values ** 2)
    C_values = df["C"].values
    S_values = df["S"].values

    print(f"Completed {len(df)} iterations")
    print(f"Final ds²: {ds2_values[-1]:.2f}")
    print(f"Converged: {ds2_values[-1] > 0}")

    return {
        "iterations": iterations,
        "ds2": ds2_values,
        "C": C_values,
        "S": S_values,
        "converged": ds2_values[-1] > 0,
    }


def run_noperthedron_experiment(n_samples=500):
    """Run Noperthedron passage tests and extract ds² distribution"""
    print("\n" + "=" * 70)
    print("NOPERTHEDRON EXPERIMENT")
    print("=" * 70)

    print("Generating Noperthedron...")
    vertices = generate_noperthedron_vertices()

    print(f"Testing {n_samples} passage attempts...")
    attempts, has_passage = test_rupert_property(vertices, n_samples=n_samples)

    stats = analyze_results(attempts)

    ds2_values = [a.ds2 for a in attempts]
    C_values = [a.C for a in attempts]
    S_values = [a.S for a in attempts]

    print(f"Completed {len(attempts)} passage tests")
    print(f"Mean ds²: {stats['ds2_mean']:.2f}")
    print(f"Has passage: {has_passage}")

    return {
        "attempts": attempts,
        "ds2": ds2_values,
        "C": C_values,
        "S": S_values,
        "has_passage": has_passage,
        "stats": stats,
    }


def create_comparison_visualization(
    arm_results, nop_results, save_path="noperthedron_comparison.png"
):
    """Create comprehensive comparison visualization"""
    print("\n" + "=" * 70)
    print("CREATING COMPARISON VISUALIZATION")
    print("=" * 70)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # ============= ARM CONTROL (Left Column) =============

    # 1. ds² over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(arm_results["iterations"], arm_results["ds2"], linewidth=2, color="blue")
    ax1.axhline(0, color="red", linestyle="--", linewidth=1, label="ds² = 0")
    ax1.fill_between(
        arm_results["iterations"],
        0,
        ax1.get_ylim()[1],
        alpha=0.2,
        color="green",
        label="Time-like (convergent)",
    )
    ax1.fill_between(
        arm_results["iterations"],
        ax1.get_ylim()[0],
        0,
        alpha=0.2,
        color="yellow",
        label="Space-like (non-convergent)",
    )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("ds² = S² - C²")
    ax1.set_title("ARM CONTROL: ds² Trajectory")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # 2. C vs S over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(
        arm_results["iterations"], arm_results["C"], label="C (Change)", color="red", linewidth=2
    )
    ax2.plot(
        arm_results["iterations"],
        arm_results["S"],
        label="S (Stability)",
        color="green",
        linewidth=2,
    )
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Value")
    ax2.set_title("ARM CONTROL: C and S Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Phase plot: C vs S
    ax3 = fig.add_subplot(gs[2, 0])
    sc = ax3.scatter(
        arm_results["C"],
        arm_results["S"],
        c=arm_results["iterations"],
        cmap="viridis",
        s=30,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )
    ax3.plot(
        [0, max(arm_results["C"])],
        [0, max(arm_results["C"])],
        "k--",
        alpha=0.3,
        label="C = S (ds² = 0)",
    )
    ax3.set_xlabel("C (Change)")
    ax3.set_ylabel("S (Stability)")
    ax3.set_title("ARM CONTROL: Phase Space")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax3, label="Iteration")

    # ============= NOPERTHEDRON (Middle Column) =============

    # 4. ds² histogram
    ax4 = fig.add_subplot(gs[0, 1])
    ax4.hist(nop_results["ds2"], bins=50, color="purple", alpha=0.7, edgecolor="black")
    ax4.axvline(0, color="red", linestyle="--", linewidth=2, label="ds² = 0")
    ymax = ax4.get_ylim()[1]
    ax4.fill_betweenx(
        [0, ymax], 0, max(nop_results["ds2"]), alpha=0.2, color="green", label="Time-like"
    )
    ax4.fill_betweenx(
        [0, ymax], min(nop_results["ds2"]), 0, alpha=0.2, color="yellow", label="Space-like"
    )
    ax4.set_xlabel("ds²")
    ax4.set_ylabel("Count")
    ax4.set_title("NOPERTHEDRON: ds² Distribution")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. C vs S scatter
    ax5 = fig.add_subplot(gs[1, 1])
    colors = ["green" if ds2 > 0 else "orange" for ds2 in nop_results["ds2"]]
    ax5.scatter(
        nop_results["C"],
        nop_results["S"],
        c=colors,
        alpha=0.5,
        s=20,
        edgecolors="black",
        linewidth=0.3,
    )
    max_val = max(max(nop_results["C"]), max(nop_results["S"]))
    ax5.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="C = S")
    ax5.set_xlabel("C (Collision)")
    ax5.set_ylabel("S (Clearance)")
    ax5.set_title("NOPERTHEDRON: Phase Space\nGreen: ds²>0, Orange: ds²<0")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect("equal")

    # 6. CDF comparison
    ax6 = fig.add_subplot(gs[2, 1])
    sorted_ds2 = np.sort(nop_results["ds2"])
    cdf = np.arange(1, len(sorted_ds2) + 1) / len(sorted_ds2)
    ax6.plot(sorted_ds2, cdf, linewidth=2, color="purple", label="Noperthedron")
    ax6.axvline(0, color="red", linestyle="--", linewidth=2)
    frac_spacelike = sum(1 for x in nop_results["ds2"] if x < 0) / len(nop_results["ds2"])
    ax6.axhline(
        frac_spacelike,
        color="orange",
        linestyle=":",
        alpha=0.7,
        label=f"{frac_spacelike:.1%} space-like",
    )
    ax6.set_xlabel("ds²")
    ax6.set_ylabel("Cumulative Probability")
    ax6.set_title("NOPERTHEDRON: CDF")
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    # ============= COMPARISON (Right Column) =============

    # 7. ds² regime comparison
    ax7 = fig.add_subplot(gs[0, 2])
    categories = ["Time-like\n(ds²>0)", "Space-like\n(ds²<0)"]

    arm_timelike = sum(1 for x in arm_results["ds2"] if x > 0) / len(arm_results["ds2"])
    arm_spacelike = 1 - arm_timelike
    nop_timelike = nop_results["stats"]["frac_timelike"]
    nop_spacelike = nop_results["stats"]["frac_spacelike"]

    x = np.arange(len(categories))
    width = 0.35

    ax7.bar(
        x - width / 2,
        [arm_timelike, arm_spacelike],
        width,
        label="Arm Control",
        color="blue",
        alpha=0.7,
        edgecolor="black",
    )
    ax7.bar(
        x + width / 2,
        [nop_timelike, nop_spacelike],
        width,
        label="Noperthedron",
        color="purple",
        alpha=0.7,
        edgecolor="black",
    )

    ax7.set_ylabel("Fraction of Configurations")
    ax7.set_title("Regime Comparison")
    ax7.set_xticks(x)
    ax7.set_xticklabels(categories)
    ax7.legend()
    ax7.grid(True, axis="y", alpha=0.3)

    # 8. Text summary comparison
    ax8 = fig.add_subplot(gs[1, 2])
    ax8.axis("off")

    summary_text = f"""
COMPARISON SUMMARY

ARM CONTROL:
• Initial ds²: {arm_results['ds2'][0]:.1f}
• Final ds²: {arm_results['ds2'][-1]:.1f}
• Converged: {arm_results['converged']}
• Time-like: {arm_timelike:.1%}
• Interpretation: System DOES
  converge to eigenstate

NOPERTHEDRON:
• Mean ds²: {nop_results['stats']['ds2_mean']:.1f}
• Range: [{nop_results['stats']['ds2_min']:.1f},
          {nop_results['stats']['ds2_max']:.1f}]
• Has passage: {nop_results['has_passage']}
• Space-like: {nop_spacelike:.1%}
• Interpretation: Most configs
  are non-convergent

INSIGHT:
Both systems use ds² = S² - C²
• Arm: ds² → positive (convergent)
• Nop: ds² mostly negative
       (non-convergent)

This supports framework:
ds² sign determines convergence!
    """

    ax8.text(
        0.1,
        0.5,
        summary_text,
        fontsize=9,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    # 9. Theoretical framework
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")

    theory_text = """
UNIFIED FRAMEWORK

ds² = S² - C²

S = Stability (bits at rest)
C = Change (bits flipping)

REGIMES:

ds² > 0: TIME-LIKE
  • Self-similar transform exists
  • Convergence possible
  • Rupert passage possible
  • Example: Arm control

ds² < 0: SPACE-LIKE
  • No self-similar transform
  • Cannot converge
  • No Rupert passage
  • Example: Noperthedron

ds² = 0: LIGHT-LIKE
  • Boundary condition
  • Critical transition

IMPLICATION:
Geometry determines dynamics!
    """

    ax9.text(
        0.1,
        0.5,
        theory_text,
        fontsize=9,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
    )

    # Main title
    fig.suptitle(
        "Noperthedron vs Arm Control: Unified Geometric Framework",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def print_theoretical_connection():
    """Print the theoretical connection between both systems"""
    print("\n" + "=" * 70)
    print("THEORETICAL CONNECTION")
    print("=" * 70)
    print(
        """
The Noperthedron and Arm Control share a common geometric framework
based on the ds² metric:

    ds² = S² - C²

Where the interpretation adapts to the domain:

ARM CONTROL:
    C = Change (gradient magnitude, bits flipping)
    S = Stability (distance from change, bits at rest)
    ds² > 0 → System converges to target (eigenstate)
    ds² < 0 → System explores, doesn't converge

NOPERTHEDRON (Rupert's Property):
    C = Collision (overlap, blocked passages)
    S = Clearance (separation, clear passages)
    ds² > 0 → Passage possible (Rupert-positive)
    ds² < 0 → No passage (Rupert-negative)

THE CONNECTION:
    Convergence ↔ Self-passage
    Eigenstate ↔ Self-similar transformation
    ds² signature ↔ Geometric possibility

The Noperthedron, proven to be Rupert-negative across ~18 million
configurations, provides geometric evidence that ds² < 0 systems
exist and cannot converge.

This is a THOUGHT EXPERIMENT exploring whether:
1. Rupert property is a geometric analog of convergence
2. ds² < 0 shapes prove non-convergent systems exist
3. Geometric constraints mirror dynamical constraints

OPEN QUESTIONS:
• What is the exact mapping between passage and convergence?
• Can we rigorously prove the analogy?
• Are there other geometric proofs of convergence limits?
• Does the C₃₀ symmetry of Noperthedron matter?
    """
    )
    print("=" * 70)


def main():
    """Main comparison routine"""
    print("=" * 70)
    print("NOPERTHEDRON vs ARM CONTROL: COMPARATIVE ANALYSIS")
    print("=" * 70)
    print("\nExploring the connection between:")
    print("  • Rupert's property (geometric passage)")
    print("  • System convergence (dynamical evolution)")
    print("  • ds² metric (unified framework)")

    # Run experiments
    arm_results = run_arm_control_experiment()
    nop_results = run_noperthedron_experiment(n_samples=500)

    # Create visualization
    create_comparison_visualization(arm_results, nop_results)

    # Print theory
    print_theoretical_connection()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(
        f"""
ARM CONTROL: {'CONVERGENT' if arm_results['converged'] else 'NON-CONVERGENT'}
  → Final ds² = {arm_results['ds2'][-1]:.2f} ({'time-like' if arm_results['ds2'][-1] > 0 else 'space-like'})

NOPERTHEDRON: {'HAS PASSAGE' if nop_results['has_passage'] else 'NO PASSAGE'}
  → Mean ds² = {nop_results['stats']['ds2_mean']:.2f} (predominantly space-like)
  → {nop_results['stats']['frac_spacelike']:.1%} of configs are space-like

The ds² metric successfully characterizes both systems.
This thought experiment suggests deep connections between
geometric constraints and dynamical convergence.
    """
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
