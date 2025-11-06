# -*- coding: utf-8 -*-
"""
Eigen: Complete Reproduction Script

Generates all supplementary data and figures from scratch:
- XOR rotation trace (32 ticks)
- Arm control trace (140 ticks)
- Five supplementary figures (S1-S5)

Usage:
    python scripts/generate_all_results.py

Outputs:
    outputs/xor_rotation_trace.csv
    outputs/eigen_arm_trace.csv
    outputs/figS1_gradient_log.png
    outputs/figS2_θ₁.png
    outputs/figS2_θ₂.png
    outputs/figS3_energy_decomp.png
    outputs/figS4_xor_hamming.png
    outputs/figS5_phase_space_arm.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("./outputs")
out_dir.mkdir(parents=True, exist_ok=True)

# ––––– Helper –––––

def save_csv(df, name):
    path = out_dir / name
    df.to_csv(path, index=False)
    return path

# ––––– 1) XOR rotation trace –––––

rng = np.random.default_rng(42)
state0 = np.uint64(rng.integers(0, 2**64, dtype=np.uint64))
axis = np.uint64(0x705A5661A791FFC1)  # fixed 90° mask
ticks_xor = 32

states = [state0]
flips, rests, ds2_cs, C_vals, S_vals = [], [], [], [], []

for t in range(ticks_xor):
    prev = states[-1]
    nxt = prev ^ axis
    states.append(nxt)
    c = int(bin(int(prev ^ nxt)).count("1"))
    s = 64 - c
    C_vals.append(c)
    S_vals.append(s)
    flips.append(c)
    rests.append(s)
    ds2_cs.append(s*s - c*c)

xor_df = pd.DataFrame({
    "tick": np.arange(len(states)-1),
    "state_before_hex": [f"0x{int(states[i]):016x}" for i in range(len(states)-1)],
    "state_after_hex":  [f"0x{int(states[i+1]):016x}" for i in range(len(states)-1)],
    "axis_hex": [f"0x{int(axis):016x}"]*(len(states)-1),
    "C": C_vals,
    "S": S_vals,
    "ds2_CS": ds2_cs
})
xor_csv = save_csv(xor_df, "xor_rotation_trace.csv")

# ––––– 2) Arm control trace –––––

L1 = L2 = 0.9
xt, yt = 1.2, 0.3
xo, yo = 0.6, 0.1
ro = 0.25
Go = 4.0
lam = 0.02
eta = 0.12
theta1, theta2 = -1.4, 1.2
T = 140
eps_change = 1e-3

def fk(th1, th2):
    x = L1*np.cos(th1) + L2*np.cos(th1 + th2)
    y = L1*np.sin(th1) + L2*np.sin(th1 + th2)
    return x, y

def jacobian(th1, th2):
    return np.array([
        [-L1*np.sin(th1) - L2*np.sin(th1 + th2), -L2*np.sin(th1 + th2)],
        [ L1*np.cos(th1) + L2*np.cos(th1 + th2),  L2*np.cos(th1 + th2)]
    ])

rows = []
for t in range(T):
    x, y = fk(theta1, theta2)
    pos = np.array([x, y])
    target = np.array([xt, yt])
    obs = np.array([xo, yo])
    d_obs = float(np.linalg.norm(pos - obs))
    target_term = float(np.sum((pos - target)**2))
    obs_term = float(Go * max(0.0, ro - d_obs)**2)
    reg_term = float(lam * (theta1**2 + theta2**2))
    ds2_total = target_term + obs_term + reg_term

    J = jacobian(theta1, theta2)
    grad_target = 2.0 * J.T @ (pos - target)
    grad_obs = (Go * 2.0 * (ro - d_obs) * (-1.0 / d_obs) *
                (J.T @ (pos - obs))) if ro - d_obs > 0 else np.zeros(2)
    grad_reg = 2.0 * lam * np.array([theta1, theta2])
    grad = grad_target + grad_obs + grad_reg
    grad_norm = float(np.linalg.norm(grad))

    delta = -eta * grad
    theta1_new, theta2_new = float(theta1 + delta[0]), float(theta2 + delta[1])

    change1, change2 = abs(delta[0]) > eps_change, abs(delta[1]) > eps_change
    C, S = int(change1) + int(change2), 2 - (int(change1) + int(change2))
    ds2_CS = S*S - C*C

    rows.append({
        "tick": t,
        "theta1_rad": theta1,
        "theta2_rad": theta2,
        "x": x, "y": y,
        "ds2_total": ds2_total,
        "target_term": target_term,
        "obs_term": obs_term,
        "reg_term": reg_term,
        "grad_norm": grad_norm,
        "C": C, "S": S,
        "ds2_CS": ds2_CS,
        "d_obs": d_obs,
        "delta_theta1": delta[0],
        "delta_theta2": delta[1],
        "delta_theta_sq": float(np.sum(delta**2)),
    })
    theta1, theta2 = theta1_new, theta2_new

arm_df = pd.DataFrame(rows)
arm_csv = save_csv(arm_df, "eigen_arm_trace.csv")

# ––––– Supplementary Figures –––––

# S1: Gradient collapse (log scale)

plt.figure()
plt.semilogy(arm_df["tick"], arm_df["grad_norm"])
plt.xlabel("Tick"); plt.ylabel("|∇ds²|")
plt.title("S1 — Gradient collapse (log)")
plt.grid(True, which="both", alpha=0.3)
plt.savefig(out_dir / "figS1_gradient_log.png", dpi=200)

# S2a/b: Joint angles

for col, name in [("theta1_rad", "θ₁"), ("theta2_rad", "θ₂")]:
    plt.figure()
    plt.plot(arm_df["tick"], arm_df[col])
    plt.xlabel("Tick"); plt.ylabel(f"{name} (rad)")
    plt.title(f"S2 — Joint angle {name}")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / f"figS2_{name}.png", dpi=200)

# S3: Energy decomposition

plt.figure()
plt.plot(arm_df["tick"], arm_df["target_term"], label="Target")
plt.plot(arm_df["tick"], arm_df["obs_term"], label="Obstacle")
plt.plot(arm_df["tick"], arm_df["reg_term"], label="Regularizer")
plt.plot(arm_df["tick"], arm_df["ds2_total"], 'k--', linewidth=2, label="Total")
plt.xlabel("Tick"); plt.ylabel("Energy contribution")
plt.title("S3 — ds² decomposition")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(out_dir / "figS3_energy_decomp.png", dpi=200)

# S4: XOR Hamming distance

hamm = [bin(int(int(a,16) ^ int(b,16))).count('1')
        for a,b in zip(xor_df["state_before_hex"], xor_df["state_after_hex"])]
plt.figure()
plt.plot(xor_df["tick"], hamm, marker='o')
plt.xlabel("Tick"); plt.ylabel("Hamming distance (C)")
plt.title("S4 — XOR rotation constant flips")
plt.grid(True, alpha=0.3)
plt.savefig(out_dir / "figS4_xor_hamming.png", dpi=200)

# S5: Phase space (θ₁ vs θ₂)

plt.figure()
plt.plot(arm_df["theta1_rad"], arm_df["theta2_rad"], 'b-')
plt.plot(arm_df["theta1_rad"].iloc[0], arm_df["theta2_rad"].iloc[0], 'go', label="Start")
plt.plot(arm_df["theta1_rad"].iloc[-1], arm_df["theta2_rad"].iloc[-1], 'ro', label="End")
plt.xlabel("θ₁ (rad)"); plt.ylabel("θ₂ (rad)")
plt.title("S5 — Arm phase space (θ₁ vs θ₂)")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(out_dir / "figS5_phase_space_arm.png", dpi=200)

print("All supplementary outputs saved in:", out_dir.resolve())
