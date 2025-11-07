# -*- coding: utf-8 -*-
"""
Configuration management for Eigen framework

Provides dataclass-based and YAML-based configuration for experiments.
Allows easy parameter management and reproducible experiment setups.
"""

from dataclasses import dataclass, asdict, field
from typing import Tuple
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class ArmConfig:
    """Configuration for 2-DOF planar arm simulation
    
    This dataclass encapsulates all parameters needed for arm control simulation.
    Can be saved/loaded from YAML files for reproducible experiments.
    
    Attributes
    ----------
    theta_init : Tuple[float, float]
        Initial joint angles (θ₁, θ₂) in radians
    target : Tuple[float, float]
        Goal position (x, y) in meters
    obstacle_center : Tuple[float, float]
        Obstacle position (x, y) in meters
    obstacle_radius : float
        Obstacle radius in meters
    L1 : float
        First link length in meters
    L2 : float
        Second link length in meters
    eta : float
        Step size (learning rate) for gradient descent
    n_ticks : int
        Number of simulation steps
    Go : float
        Obstacle repulsion strength
    lam : float
        Regularization weight
    eps_change : float
        Threshold for detecting significant change
    
    Examples
    --------
    >>> config = ArmConfig(eta=0.15, n_ticks=200)
    >>> print(config.eta)
    0.15
    
    >>> config.to_yaml('my_experiment.yaml')
    >>> loaded = ArmConfig.from_yaml('my_experiment.yaml')
    """
    
    # Initial state
    theta_init: Tuple[float, float] = (-1.4, 1.2)
    
    # Target
    target: Tuple[float, float] = (1.2, 0.3)
    
    # Obstacle
    obstacle_center: Tuple[float, float] = (0.6, 0.1)
    obstacle_radius: float = 0.25
    
    # Physical parameters
    L1: float = 0.9  # Link 1 length (m)
    L2: float = 0.9  # Link 2 length (m)
    
    # Control parameters
    eta: float = 0.12  # Step size
    n_ticks: int = 140  # Simulation length
    
    # Objective weights
    Go: float = 4.0  # Obstacle repulsion
    lam: float = 0.02  # Regularization
    
    # Change detection
    eps_change: float = 1e-3
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ArmConfig':
        """
        Load configuration from YAML file
        
        Parameters
        ----------
        path : str
            Path to YAML configuration file
        
        Returns
        -------
        config : ArmConfig
            Configuration instance
        
        Raises
        ------
        ImportError
            If PyYAML is not installed
        FileNotFoundError
            If config file doesn't exist
        
        Examples
        --------
        >>> config = ArmConfig.from_yaml('configs/default.yaml')
        >>> print(config.eta)
        0.12
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support. "
                "Install with: pip install pyyaml"
            )
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert lists to tuples for tuple fields
        if 'theta_init' in data:
            data['theta_init'] = tuple(data['theta_init'])
        if 'target' in data:
            data['target'] = tuple(data['target'])
        if 'obstacle_center' in data:
            data['obstacle_center'] = tuple(data['obstacle_center'])
        
        return cls(**data)
    
    def to_yaml(self, path: str):
        """
        Save configuration to YAML file
        
        Parameters
        ----------
        path : str
            Output path for YAML file
        
        Raises
        ------
        ImportError
            If PyYAML is not installed
        
        Examples
        --------
        >>> config = ArmConfig(eta=0.15, n_ticks=200)
        >>> config.to_yaml('configs/custom.yaml')
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support. "
                "Install with: pip install pyyaml"
            )
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def __repr__(self) -> str:
        """Pretty print configuration"""
        lines = ["ArmConfig("]
        for key, value in asdict(self).items():
            lines.append(f"  {key}: {value}")
        lines.append(")")
        return "\n".join(lines)


# Default configuration instance
DEFAULT_CONFIG = ArmConfig()


if __name__ == "__main__":
    # Example usage
    print("Default configuration:")
    print(DEFAULT_CONFIG)
    print()
    
    # Save to file (if PyYAML available)
    if YAML_AVAILABLE:
        DEFAULT_CONFIG.to_yaml('configs/default.yaml')
        print("Saved to configs/default.yaml")
        
        # Load from file
        loaded = ArmConfig.from_yaml('configs/default.yaml')
        print("\nLoaded configuration:")
        print(loaded)
    else:
        print("PyYAML not available. Install with: pip install pyyaml")
