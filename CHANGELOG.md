# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Professional Python packaging configuration (`setup.py` and `pyproject.toml`)
- Code quality tool configurations (Black, isort, flake8, mypy)
- MANIFEST.in for proper package distribution
- This CHANGELOG file to track project changes

## [1.0.0] - 2025-11-08

### Fixed
- Added missing `pandas>=1.2.0` dependency to package configuration
- Fixed mypy configuration for Python 3.9+ compatibility
- Fixed package metadata for twine validation (constrained setuptools to <68)
- Removed deprecated license classifiers to prevent build warnings

### Added
- **Core Geometric Control Framework**
  - Minimal control framework where motion emerges from gradient descent
  - Single-equation approach: `Q_{t+1} = Q_t - η∇ds²(Q)`
  - Distance-based objective function combining target attraction, obstacle repulsion, and configuration regularization

- **Lorentz Transformation Framework**
  - Unified vision-control integration using Lorentz transformations
  - Stereo vision processing with XOR/XNOR geometric operations
  - Enhanced robot arm control with Lorentz-based transformations
  - Comprehensive test suite for Lorentz integration (93 tests, 77% coverage)

- **Robot Arm Control**
  - 2-joint planar arm implementation
  - Target reaching capability (converged to 5.6cm error)
  - Obstacle avoidance (maintained 59cm clearance with >25cm requirement)
  - Joint limit enforcement
  - Singularity avoidance mechanisms

- **Configuration System**
  - YAML-based configuration management
  - Multiple preset configurations (default, fast_convergence)
  - Flexible parameter tuning for different control scenarios

- **Testing Infrastructure**
  - 93 comprehensive tests with 77% code coverage
  - pytest-based test suite
  - Coverage reporting with pytest-cov
  - GitHub Actions CI/CD workflow for automated testing
  - Integration tests for stereo vision and Lorentz transformations

- **Documentation**
  - Comprehensive README with theoretical background
  - CONTRIBUTING.md guide for contributors
  - Dual licensing structure (MIT + Commercial)
  - Working examples and demonstrations
  - Performance benchmarks and validation results

- **Examples**
  - Quickstart demonstration (`examples/quickstart.py`)
  - Arm control with Lorentz integration
  - Configuration system usage examples

### Technical Details
- **Language**: Python 3.8+
- **Core Dependencies**: NumPy, SciPy, Matplotlib, PyYAML
- **Code Quality**: Type hints, NumPy docstrings, clean modular structure
- **Lines of Code**: ~1,239 production code

## [0.1.0] - 2025-10-01

### Added
- Initial project structure
- Basic geometric control algorithm
- Simple 2-joint arm implementation
- Initial test framework

---

## Version Numbering

This project uses [Semantic Versioning](https://semver.org/):
- **MAJOR** version (1.x.x): Incompatible API changes
- **MINOR** version (x.1.x): Add functionality in a backward compatible manner
- **PATCH** version (x.x.1): Backward compatible bug fixes

## Types of Changes
- `Added` - New features
- `Changed` - Changes in existing functionality
- `Deprecated` - Soon-to-be removed features
- `Removed` - Removed features
- `Fixed` - Bug fixes
- `Security` - Security vulnerability fixes

---

[Unreleased]: https://github.com/InauguralPhysicist/Eigen-Geometric-Control/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/InauguralPhysicist/Eigen-Geometric-Control/releases/tag/v1.0.0
[0.1.0]: https://github.com/InauguralPhysicist/Eigen-Geometric-Control/releases/tag/v0.1.0
