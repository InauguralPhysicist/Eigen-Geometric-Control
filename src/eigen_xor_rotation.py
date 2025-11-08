# -*- coding: utf-8 -*-
"""
Eigen: XOR Rotation Demo

Demonstrates discrete 90° rotation operator using XOR with fixed mask.
Creates period-2 oscillation maintaining constant C/S/ds² values.
"""

import numpy as np
import pandas as pd


def run_xor_simulation(n_ticks=32, seed=42):
    """
    Run XOR rotation simulation

    Applies fixed rotation axis to 64-bit state via XOR operation:
    state_{t+1} = state_t ⊕ axis

    Args:
        n_ticks: Number of rotation steps
        seed: Random seed for initial state

    Returns:
        DataFrame with state transitions and metrics
    """
    rng = np.random.default_rng(seed)

    # Initialize random 64-bit state
    state = np.uint64(rng.integers(0, 2**64, dtype=np.uint64))

    # Fixed 90° rotation axis
    axis = np.uint64(0x705A5661A791FFC1)

    rows = []

    for t in range(n_ticks):
        state_before = state
        state_after = state ^ axis  # XOR = 90° rotation

        # Count bit flips (change)
        C = int(bin(int(state_before ^ state_after)).count("1"))
        S = 64 - C
        ds2_CS = S * S - C * C

        rows.append(
            {
                "tick": t,
                "state_before_hex": f"0x{int(state_before):016x}",
                "state_after_hex": f"0x{int(state_after):016x}",
                "axis_hex": f"0x{int(axis):016x}",
                "C": C,
                "S": S,
                "ds2_CS": ds2_CS,
            }
        )

        state = state_after

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Example usage
    results = run_xor_simulation()

    print("Period-2 oscillation:")
    print(f"  State A: {results['state_before_hex'].iloc[0]}")
    print(f"  State B: {results['state_after_hex'].iloc[0]}")
    print(f"  C (constant): {results['C'].iloc[0]}")
    print(f"  S (constant): {results['S'].iloc[0]}")
    print(f"  ds² (constant): {results['ds2_CS'].iloc[0]}")
