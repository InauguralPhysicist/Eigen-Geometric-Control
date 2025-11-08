"""
Setup configuration for Eigen Geometric Control.

This file makes your package installable via pip:
    pip install -e .              # Install in development mode
    pip install eigen-control     # Install from PyPI (after publishing)
"""

from pathlib import Path

from setuptools import find_packages, setup

# Read the README for the long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements from requirements.txt
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    # Basic package information
    name="eigen-geometric-control",
    version="1.0.0",
    # Author information
    author="Jonathan McReynolds",
    author_email="mcreynolds.jon@gmail.com",
    # Package description
    description="Minimal geometric robot control framework using gradient descent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # License
    license="MIT",
    # URLs
    url="https://github.com/InauguralPhysicist/Eigen-Geometric-Control",
    project_urls={
        "Bug Reports": "https://github.com/InauguralPhysicist/Eigen-Geometric-Control/issues",
        "Source": "https://github.com/InauguralPhysicist/Eigen-Geometric-Control",
        "Documentation": "https://github.com/InauguralPhysicist/Eigen-Geometric-Control#readme",
    },
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    package_dir={"": "."},
    # Python version requirement
    python_requires=">=3.8",
    # Dependencies
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "pandas>=1.2.0",
        "scipy>=1.6.0",
        "pyyaml>=5.4.0",
    ],
    # Development dependencies (installed with: pip install -e .[dev])
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    # Package classifiers (helps users find your package on PyPI)
    classifiers=[
        # Development status
        "Development Status :: 4 - Beta",
        # Intended audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Python versions
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        # Operating systems
        "Operating System :: OS Independent",
    ],
    # Keywords for PyPI search
    keywords="robotics control geometric optimization gradient-descent robot-arm",
    # Include additional files (README, LICENSE, etc.)
    include_package_data=True,
    # Zip safe
    zip_safe=False,
)
