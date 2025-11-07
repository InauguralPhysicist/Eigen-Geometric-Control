"""
Configuration management for Eigen framework

Provides dataclass-based and YAML-based configuration for experiments.
"""

from dataclasses import dataclass, asdict
from typing import Tuple, Optional
from pathlib import Path
import yaml


@dataclass
class ArmConfig:
    """Configuration for 2-DOF planar arm simulation"""

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

        Examples
        --------
        >>> config = ArmConfig.from_yaml('configs/default.yaml')
        >>> print(config.eta)
        0.12
        """
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

        Examples
        --------
        >>> config = ArmConfig(eta=0.15, n_ticks=200)
        >>> config.to_yaml('configs/custom.yaml')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

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
    config = ArmConfig()
    print("Default configuration:")
    print(config)

    # Save to file
    config.to_yaml('configs/default.yaml')
    print("\nSaved to configs/default.yaml")

    # Load from file
    loaded = ArmConfig.from_yaml('configs/default.yaml')
    print("\nLoaded configuration:")
    print(loaded)
