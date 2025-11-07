# Examples

Practical examples demonstrating Eigen Geometric Control usage.

## Quick Start

### Basic Usage

```bash
cd examples/
python quickstart.py
```

**Output:**
- Prints simulation summary
- Shows convergence metrics
- Generates `quickstart_result.png` (if matplotlib available)

**What it demonstrates:**
- Running a simulation
- Accessing results
- Computing performance metrics
- Basic visualization

## Available Examples

### quickstart.py
**Simplest possible example** - Run simulation and view results.

**Usage:**
```bash
python examples/quickstart.py
```

**Learn:**
- How to import Eigen
- How to run simulation
- How to access results
- Basic convergence metrics

### Coming Soon

**custom_objective.py** - Extend the framework with new terms
**parameter_sweep.py** - Test sensitivity to parameters
**tutorial.ipynb** - Interactive Jupyter notebook
**visualize_trajectory.py** - Create publication-quality figures

## Running Examples

**From repository root:**
```bash
python examples/quickstart.py
```

**From examples directory:**
```bash
cd examples/
python quickstart.py
```

## Requirements

**Basic examples:**
- numpy, pandas (in requirements.txt)

**Visualization examples:**
- matplotlib (optional)

Install all dependencies:
```bash
pip install -r requirements.txt
pip install matplotlib  # For visualizations
```

## Creating Your Own

Use quickstart.py as a template:

```python
from src import run_arm_simulation

# Your custom configuration
results = run_arm_simulation(
    theta_init=(-1.0, 1.0),  # Custom initial angles
    target=(1.5, 0.5),       # Custom target
    eta=0.15,                # Custom learning rate
    n_ticks=200              # Custom duration
)

# Analyze results
print(f"Final error: {results['ds2_total'].iloc[-1]:.6f}")
print(f"Min obstacle distance: {results['d_obs'].min():.3f}m")
```

## Tips

**Faster convergence:**
- Increase η (learning rate): `eta=0.20`
- But watch for oscillations

**More exploration:**
- Decrease η: `eta=0.05`
- Takes longer but smoother

**Tighter obstacle avoidance:**
- Increase G₀ (repulsion): `Go=8.0`
- Maintains larger clearance

**Different scenarios:**
- Change target location
- Add multiple obstacles
- Modify link lengths

## Troubleshooting

**ImportError: No module named 'src'**
```bash
# Run from repository root, not examples/
cd ..
python examples/quickstart.py
```

**Matplotlib not found:**
```bash
pip install matplotlib
```

**Results look wrong:**
```bash
# Verify installation
python -c "from src import run_arm_simulation; print('OK')"
```

## Next Steps

After running examples:
1. Read `docs/overview.md` for theory
2. Explore `src/` code to understand implementation
3. Run `tests/` to verify behavior
4. Modify examples for your use case
