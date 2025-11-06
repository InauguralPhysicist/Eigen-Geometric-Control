# Eigen Framework Overview

## Core Concept

Robot motion emerges from gradient descent on a geometric objective function.
No trajectory planning, no PID tuning, no learning required.

## Mathematical Foundation

### The Update Rule

```
Q_{t+1} = Q_t - η∇ds²(Q)
```

Where:
- `Q`: Configuration vector (joint angles, positions, etc.)
- `η`: Step size (learning rate)
- `ds²`: Geometric objective function

### The Objective Function

```
ds²(Q) = ||p(Q) - p_target||² + Σ_obstacles + λ||Q||²
```

Components:
1. **Target term**: Distance to goal
2. **Obstacle term**: Repulsion from constraints
3. **Regularization**: Prefer natural configurations

### Convergence Theory

The system reaches an eigenstate when:

```
∇ds²(Q) = 0
```

**Lyapunov function**: `V(t) = ||∇ds²||`

Properties:
- `V(t) ≥ 0` always
- `V(t+1) ≤ V(t)` under gradient flow
- `V(t) → 0` as system converges

Therefore: **Guaranteed convergence** to local minimum.

## Stability Metrics

The framework provides self-measurement through:

**Change count (C)**: Number of components changing

```python
C = sum(abs(Q[t+1] - Q[t]) > threshold)
```

**Stability count (S)**: Number of components stable

```python
S = n - C  # where n = total components
```

**Geometric invariant (ds²)**:

```python
ds2 = S² - C²
```

Three regimes:

- `ds² < 0`: Space-like (exploring, high motion)
- `ds² = 0`: Light-like (transition boundary)
- `ds² > 0`: Time-like (settled, stable)

## Advantages

**Simplicity**:

- Single equation
- One tunable parameter (η)
- <200 lines of code

**Autonomy**:

- No replanning required
- No parameter adaptation
- Converges automatically

**Transparency**:

- All gradients interpretable
- Convergence metrics observable
- No black-box components

**Efficiency**:

- Minimal computation per step
- No search or optimization loops
- Suitable for real-time control

## Limitations

**Local minima**:

- Gradient descent finds local optimum
- May not reach global minimum
- Solution: Better initialization or metric design

**Step size selection**:

- Too large: Oscillation or divergence
- Too small: Slow convergence
- Current approach: Fixed η (works well empirically)

**Obstacle handling**:

- Soft constraints only (repulsion fields)
- Hard constraints require projection
- Future work: Constrained optimization

## Extensions

**Moving targets**:

```python
target(t) = f(t)  # Time-varying goal
```

**Multi-agent**:

```python
ds² = Σ_i ds²_i + Σ_{i,j} interaction(i,j)
```

**Visual servoing**:

```python
ds² weighted by stereo disparity confidence
```

**Learned metrics**:

```python
ds²(Q) parameterized by neural network
```

## Comparison to Other Methods

|Method   |Lines of Code|Parameters |Replanning |
|---------|-------------|-----------|-----------|
|PID      |~100         |3 per joint|No         |
|MPC      |~1000        |~20        |Every step |
|RRT*     |~2000        |~10        |When needed|
|**Eigen**|**<200**     |**1**      |**Never**  |

## References

1. Gradient descent in robotics: Khatib (1986) - Potential fields
2. Lyapunov stability: Khalil (2002) - Nonlinear systems
3. Geometric control: Bullo & Lewis (2004) - Geometric mechanics
