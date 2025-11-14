#!/usr/bin/env python3
"""
Visualize the difference between random noise and systematic oscillations
"""

import numpy as np
import matplotlib.pyplot as plt

# Create time axis
t = np.linspace(0, 10, 100)

# 1. Clean signal (moving target)
clean_signal = 2.0 + 0.1 * t

# 2. Systematic oscillation (tracking response)
systematic = clean_signal + 0.3 * np.sin(3 * t) * np.exp(-0.1 * t)

# 3. Random noise
np.random.seed(42)
random_noise = clean_signal + np.random.normal(0, 0.15, len(t))

# 4. Both combined
both = clean_signal + 0.3 * np.sin(3 * t) * np.exp(-0.1 * t) + np.random.normal(0, 0.08, len(t))

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Clean signal
axes[0, 0].plot(t, clean_signal, "b-", linewidth=2, label="True target position")
axes[0, 0].set_title("1. Clean Signal (Moving Target)", fontsize=12, fontweight="bold")
axes[0, 0].set_ylabel("Position")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Systematic oscillation
axes[0, 1].plot(t, clean_signal, "b--", alpha=0.5, label="Target")
axes[0, 1].plot(t, systematic, "r-", linewidth=2, label="Robot response")
axes[0, 1].set_title(
    "2. Systematic Oscillation (Tracking Overshoot)", fontsize=12, fontweight="bold"
)
axes[0, 1].set_ylabel("Position")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].text(
    5,
    1.5,
    "✓ Lightlike observer\ncan dampen this\n(+20% improvement)",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
    ha="center",
    fontsize=9,
)

# Plot 3: Random noise
axes[1, 0].plot(t, clean_signal, "b--", alpha=0.5, label="True position")
axes[1, 0].plot(t, random_noise, "g-", linewidth=1, label="Noisy measurement", alpha=0.7)
axes[1, 0].set_title("3. Random Noise (Sensor Error)", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("Time")
axes[1, 0].set_ylabel("Position")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].text(
    5,
    1.5,
    "✗ Lightlike observer\ncannot help\n(0% improvement)",
    bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
    ha="center",
    fontsize=9,
)

# Plot 4: Both combined
axes[1, 1].plot(t, clean_signal, "b--", alpha=0.5, label="Target")
axes[1, 1].plot(t, both, "m-", linewidth=1.5, label="Robot + noise", alpha=0.8)
axes[1, 1].set_title("4. Both Combined (Real World)", fontsize=12, fontweight="bold")
axes[1, 1].set_xlabel("Time")
axes[1, 1].set_ylabel("Position")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].text(
    5,
    1.5,
    "△ Partial benefit\nDepends on SNR\n(signal/noise ratio)",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
    ha="center",
    fontsize=9,
)

plt.suptitle("Random Noise vs Systematic Oscillations", fontsize=14, fontweight="bold", y=0.995)
plt.tight_layout()

# Save
output_path = "outputs/noise_types_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved visualization: {output_path}")

# Analyze autocorrelation
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Autocorrelation of systematic oscillation
systematic_detrend = systematic - clean_signal
random_detrend = random_noise - clean_signal

lags = range(0, 30)
autocorr_systematic = [
    (
        np.corrcoef(systematic_detrend[: -lag or None], systematic_detrend[lag:])[0, 1]
        if lag > 0
        else 1.0
    )
    for lag in lags
]
autocorr_random = [
    np.corrcoef(random_detrend[: -lag or None], random_detrend[lag:])[0, 1] if lag > 0 else 1.0
    for lag in lags
]

axes2[0].bar(lags, autocorr_systematic, color="red", alpha=0.7)
axes2[0].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
axes2[0].set_title(
    "Systematic Oscillation\n(High autocorrelation = Pattern exists)", fontweight="bold"
)
axes2[0].set_xlabel("Lag (timesteps)")
axes2[0].set_ylabel("Autocorrelation")
axes2[0].grid(True, alpha=0.3)
axes2[0].text(
    15,
    0.7,
    "✓ Pattern visible\nat lag 1-10\n→ Observer detects this",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
    ha="center",
    fontsize=9,
)

axes2[1].bar(lags, autocorr_random, color="green", alpha=0.7)
axes2[1].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
axes2[1].set_title("Random Noise\n(Low autocorrelation = No pattern)", fontweight="bold")
axes2[1].set_xlabel("Lag (timesteps)")
axes2[1].set_ylabel("Autocorrelation")
axes2[1].grid(True, alpha=0.3)
axes2[1].text(
    15,
    0.7,
    "✗ No pattern\nRapid decay to zero\n→ Observer cannot detect",
    bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
    ha="center",
    fontsize=9,
)

plt.suptitle(
    "Autocorrelation Analysis (Why Observer Works for One, Not the Other)",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()

output_path2 = "outputs/autocorrelation_comparison.png"
plt.savefig(output_path2, dpi=150, bbox_inches="tight")
print(f"Saved autocorrelation: {output_path2}")

print("\nKey Insight:")
print("=" * 60)
print("Systematic oscillations have HIGH autocorrelation")
print("  → Current state predicts future states")
print("  → Lightlike observer can detect and dampen")
print()
print("Random noise has LOW autocorrelation")
print("  → Current state doesn't predict future")
print("  → Lightlike observer cannot help")
print("=" * 60)
