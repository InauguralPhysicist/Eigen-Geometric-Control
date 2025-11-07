# Contributing to Eigen Geometric Control

Thank you for your interest in contributing to Eigen! This project welcomes contributions from the community.

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Eigen-Geometric-Control.git
   cd Eigen-Geometric-Control
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run tests to verify setup**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-3dof-arm` - New features
- `fix/gradient-computation` - Bug fixes
- `docs/improve-readme` - Documentation
- `refactor/simplify-config` - Code improvements

### 2. Make Your Changes

**Code Style:**
- Follow PEP 8 guidelines
- Use type hints for function parameters and returns
- Add NumPy-style docstrings to all functions
- Keep functions focused and modular

**Example:**
```python
def compute_example(param1: float, param2: np.ndarray) -> Tuple[float, Dict]:
    """
    Brief description of function.

    Parameters
    ----------
    param1 : float
        Description of param1
    param2 : np.ndarray
        Description of param2

    Returns
    -------
    result : float
        Description of result
    metadata : dict
        Additional information
    """
    # Implementation
    pass
```

### 3. Add Tests

All new features must include tests in `tests/`:

```python
# tests/test_new_feature.py
import pytest
from src import new_function

class TestNewFeature:
    def test_basic_behavior(self):
        """Test basic functionality"""
        result = new_function(input_data)
        assert result == expected_output

    def test_edge_case(self):
        """Test edge case handling"""
        with pytest.raises(ValueError):
            new_function(invalid_input)
```

Run tests:
```bash
pytest tests/test_new_feature.py -v
```

### 4. Update Documentation

- Update `README.md` if adding new features
- Add docstrings to all new functions
- Update configuration examples if adding parameters
- Add usage examples to demonstrate new functionality

### 5. Commit Your Changes

Use clear, descriptive commit messages:

```bash
git commit -m "feat: Add 3-DOF arm support to eigen_core

- Extend forward_kinematics to handle 3 joints
- Update Jacobian computation for 3-DOF
- Add tests for 3-DOF configuration
- Update documentation with usage examples"
```

**Commit message format:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/updates
- `refactor:` - Code restructuring
- `ci:` - CI/CD changes

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a pull request on GitHub with:
- Clear description of changes
- Reference to related issues (if any)
- Screenshots/plots if adding visualizations
- Test results showing all tests pass

## Pull Request Guidelines

### Before Submitting

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Code follows PEP 8 style guidelines
- [ ] New functions have docstrings and type hints
- [ ] Tests added for new functionality
- [ ] Documentation updated
- [ ] No merge conflicts with main branch

### PR Description Template

```markdown
## Summary
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- List specific changes made
- Include file modifications

## Testing
- Describe tests added
- Include test output showing passes

## Screenshots (if applicable)
Add plots or visualizations

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guide
```

## Issue Reporting

### Bug Reports

Include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces
- Minimal code example

**Template:**
```markdown
**Environment:**
- Python: 3.10
- OS: Ubuntu 22.04
- Eigen version: v1.0.0

**Bug Description:**
Clear description of the bug

**Reproduction Steps:**
1. Step 1
2. Step 2
3. Expected: X, Got: Y

**Error Output:**
```
Paste error message here
```

**Minimal Example:**
```python
# Code to reproduce
```
```

### Feature Requests

Include:
- Use case description
- Proposed API or interface
- Example usage code
- Impact on existing functionality

## Code Review Process

1. **Automated Checks** - GitHub Actions runs tests automatically
2. **Maintainer Review** - Code review by project maintainer
3. **Feedback** - Address review comments
4. **Merge** - PR merged after approval

## Testing Standards

### Test Coverage

Aim for >85% code coverage:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Categories

1. **Unit Tests** - Test individual functions
2. **Integration Tests** - Test complete workflows
3. **Error Handling** - Test validation and error cases
4. **Edge Cases** - Test boundary conditions
5. **Numerical Accuracy** - Verify mathematical correctness

## Configuration Management

Use YAML configs for reproducibility:

```python
from src.config import ArmConfig

# Create custom configuration
config = ArmConfig(
    eta=0.15,
    n_ticks=100,
    theta_init=(-1.0, 1.0)
)
config.to_yaml('configs/my_experiment.yaml')
```

## Continuous Integration

GitHub Actions runs on every push:
- Installs dependencies
- Runs full test suite
- Generates coverage report

Check `.github/workflows/ci.yaml` for CI configuration.

## Questions or Help?

- **Issues**: Open an issue for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: mcreynolds.jon@gmail.com for private inquiries

## License

By contributing, you agree that your contributions will be licensed under the MIT License for non-commercial use. See [LICENSE](LICENSE) for details.

---

Thank you for contributing to Eigen Geometric Control! 🎉
