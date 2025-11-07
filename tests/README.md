# Tests

Automated test suite for Eigen Geometric Control framework.

## Running Tests

**Install test dependencies:**
```bash
pip install pytest pytest-cov
```

**Run all tests:**
```bash
pytest tests/ -v
```

**Run with coverage:**
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

**Run specific test file:**
```bash
pytest tests/test_eigen_core.py -v
```

## Test Structure

### test_eigen_core.py (170 lines)
Tests for core geometric functions:
- `forward_kinematics`: FK computation and validation
- `jacobian`: Jacobian matrix against numerical differentiation
- `compute_ds2`: Objective function components
- `compute_gradient`: Gradient computation and convergence
- `compute_change_stability`: C/S metric calculation

### test_integration.py (167 lines)
Integration tests for complete workflows:
- Arm simulation convergence
- XOR rotation period-2 oscillation
- Obstacle avoidance validation
- Reproducibility verification

## Test Coverage

**Expected coverage: ~85%**

Covered:
- ✓ Forward kinematics
- ✓ Jacobian computation
- ✓ ds² objective function
- ✓ Gradient computation
- ✓ Change/stability metrics
- ✓ Arm simulation pipeline
- ✓ XOR rotation pipeline
- ✓ Convergence behavior
- ✓ Obstacle avoidance
- ✓ Reproducibility

Not covered (by design):
- Visualization code (matplotlib)
- Script-level code (generate_all_results.py)
- Error handling for invalid inputs

## Adding New Tests

When adding features, add corresponding tests:

```python
# tests/test_new_feature.py
import pytest
from src import new_function

def test_new_function():
    """Test description"""
    result = new_function(input_data)
    assert result == expected_output
```

Run new tests:
```bash
pytest tests/test_new_feature.py -v
```

## Continuous Integration

Tests run automatically on every push via GitHub Actions.

See `.github/workflows/test.yml` for CI configuration.
