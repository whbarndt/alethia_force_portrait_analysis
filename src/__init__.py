"""
Aletheia Fractal Analysis Package

This package contains modular components for fractal analysis of 3D trajectory data.
"""

__version__ = "1.0.0"
__author__ = "Taylor Lab"

# Import main modules for easy access
from .interpolation import load_data, interpolate_data
from .density import compute_density
from .fractal_dimension import fractal_dimension
from .visualization import create_visualization, create_interactive_3d_plot
from .utils import load_config, setup_logging, auto_discover_data_files, save_discovered_datasets_to_csv

__all__ = [
    'load_data',
    'interpolate_data', 
    'compute_density',
    'fractal_dimension',
    'create_visualization',
    'create_interactive_3d_plot',
    'load_config',
    'setup_logging',
    'auto_discover_data_files',
    'save_discovered_datasets_to_csv'
]
