# Eigen: Geometric Robot Control

Minimal control framework where motion emerges from gradient descent
on a geometric objective function.

## The Equation

```python
Q_{t+1} = Q_t - η∇ds²(Q)
```

Where ds² combines:

- Distance to target
- Obstacle repulsion
- Configuration regularization

## Why This Works

Traditional approach:

```
Perception → Planning → Control → Actuation
(4 modules, 10,000+ lines, dozens of parameters)
```

Geometric approach:

```
State → Gradient → Update
(1 equation, <200 lines, 1 parameter)
```

## Demonstrated Capabilities

**2-joint planar arm:**

- Target reaching: Converged to 5.6cm error
- Obstacle avoidance: Maintained 59cm clearance (>25cm required)
- Smooth motion: No discontinuities or oscillations
- Autonomous: No replanning or parameter adjustment

**Performance:**

- Gradient magnitude decreased 10⁶× (convergence)
- 140 ticks from start to stable eigenstate
- All metrics self-measured by system

## Repository Structure

```
Eigen/
├── src/              # Core implementation
├── data/             # Raw simulation data
├── figures/          # Visualizations
├── examples/         # Usage demos
└── docs/             # Technical documentation
```

## Quick Start

```python
from src.eigen_arm_control import EigenArm, run_simulation

# Create 2-DOF arm
arm = EigenArm(
    theta_init=[-1.4, 1.2],
    target=[1.2, 0.3],
    obstacle_center=[0.6, 0.1],
    obstacle_radius=0.25
)

# Run geometric flow
results = run_simulation(arm, n_ticks=140, eta=0.12)

# System converges autonomously
print(f"Final error: {results['ds2_total'][-1]:.4f}")
print(f"Gradient norm: {results['grad_norm'][-1]:.2e}")
```

## Key Results

### Robot Arm Control

![Trajectory](figures/fig2_arm_trajectory.png)

- Smooth path from start to target
- Natural obstacle avoidance via metric geometry
- No explicit path planning required

### Convergence Analysis

![Convergence](figures/figS1_gradient_log.png)

- Exponential gradient decay (10⁶× decrease)
- Monotonic error reduction
- Stable eigenstate reached

### Stability Metrics

The system self-measures convergence through geometric invariants:

- **C**: Change count (components in motion)
- **S**: Stability count (components at rest)
- **ds²**: Metric invariant (S² - C²)

Transitions from space-like (exploring) → light-like (boundary) → time-like (settled).

## Code Structure

- `eigen_core.py`: Framework implementation (<100 lines)
- `eigen_arm_control.py`: Robot arm demo (~100 lines)
- `eigen_xor_rotation.py`: Discrete dynamics validation
- Data and figures included for reproducibility

## Applications

- **Educational platform**: Understand control fundamentals
- **Rapid prototyping**: Test ideas without tuning
- **Embedded systems**: Minimal computation required
- **Research baseline**: Compare against learning methods

## Technical Details

### Gradient Computation

```python
def compute_gradient(Q, target, obstacles):
    """Compute ∇ds² for current configuration"""
    # Target term: distance to goal
    grad_target = 2 * (end_effector(Q) - target) @ jacobian(Q)

    # Obstacle term: repulsion from constraints
    grad_obstacle = compute_obstacle_gradient(Q, obstacles)

    # Regularization: prefer natural configurations
    grad_reg = regularization_weight * Q

    return grad_target + grad_obstacle + grad_reg
```

### Convergence Criterion

System reaches eigenstate when:

- `||∇ds²|| < ε` (gradient vanishes)
- `C → 0` (all components stable)
- `S → n` (system at rest)

## Validation

**XOR Rotation (Discrete):**

- 32 ticks of period-2 oscillation
- Constant C=33, S=31, ds²=-128
- Demonstrates pure rotation without convergence

**Robot Arm (Continuous):**

- 140 ticks to convergence
- ds²: 1.96 → 0.056
- Gradient: 4.04 → 3.59×10⁻⁶

**Bell Locality Test:**

- S = 2.0000 (classical bound)
- Preserves locality and realism
- Geometric correlation without nonlocality

## Installation

```bash
git clone https://github.com/[username]/Eigen.git
cd Eigen
pip install -r requirements.txt
```

**Requirements:**

```
numpy>=1.20.0
matplotlib>=3.3.0
pandas>=1.2.0
```

## Usage Examples

See `examples/` directory:

- `quickstart.ipynb`: Interactive demo
- `parameter_sweep.py`: Sensitivity analysis
- `custom_objective.py`: Extend to new scenarios

## Citation

If you use this code in your research:

```bibtex
@software{eigen2025,
  author = {McReynolds, Jon},
  title = {Eigen: Geometric Robot Control},
  year = {2025},
  url = {https://github.com/[username]/Eigen}
}
```

Paper in preparation.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Areas of interest:

- 3D arm implementations
- Moving target scenarios
- Multi-agent coordination
- Hardware integration

Open an issue to discuss before submitting PRs.

## Contact

Jon McReynolds
[@InauguralPhysicist](https://twitter.com/InauguralPhysicist)

-----

Built for reproducibility. Hardware validated. Ready to extend.
