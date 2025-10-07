"""
Interpolation module for Aletheia Fractal Analysis

This module handles interpolation functionality.
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import pandas as pd
import os
import logging


def _get_interpolated_file_path(original_file_path, config):
    """Generate interpolated file path based on original file and config"""
    file_dir = os.path.dirname(original_file_path)
    file_name = os.path.basename(original_file_path)
    name_without_ext = os.path.splitext(file_name)[0]
    interpolation_method = config['interpolation'].get('method', 'cubic_spline')
    interpolation_factor = config['interpolation'].get('factor', 10000)
    interpolated_filename = f"{name_without_ext}_{interpolation_method}_{interpolation_factor}x_interpolated.csv"
    return os.path.join(file_dir, interpolated_filename)


def _load_existing_interpolated_data(interpolated_path, logger):
    """Load existing interpolated data from file"""
    try:
        interpolated_data = np.genfromtxt(interpolated_path, delimiter=',', skip_header=1)
        x_new = interpolated_data[:, 0]
        y_new = interpolated_data[:, 1]
        z_new = interpolated_data[:, 2]
        points_new = np.vstack([x_new, y_new, z_new]).T
        
        logger.info(f"Successfully loaded existing interpolated data:")
        logger.info(f"  - Number of interpolated points: {len(x_new)}")
        logger.info(f"  - X range: [{x_new.min():.6f}, {x_new.max():.6f}]")
        logger.info(f"  - Y range: [{y_new.min():.6f}, {y_new.max():.6f}]")
        logger.info(f"  - Z range: [{z_new.min():.6f}, {z_new.max():.6f}]")
        logger.info(f"  - Memory usage (MB): {points_new.nbytes / 1024 / 1024:.2f}")
        logger.info(f"  - File size: {os.path.getsize(interpolated_path) / 1024 / 1024:.2f} MB")
        
        return points_new, x_new, y_new, z_new
        
    except Exception as e:
        logger.warning(f"Failed to load existing interpolated file: {str(e)}")
        return None


def _save_interpolated_data(x_new, y_new, z_new, original_file_path, config, logger):
    """Save interpolated data to file"""
    try:
        output_path = _get_interpolated_file_path(original_file_path, config)
        
        # Save interpolated data
        interpolated_df = pd.DataFrame({
            'ax': x_new,
            'ay': y_new,
            'az': z_new
        })
        interpolated_df.to_csv(output_path, index=False)
        
        logger.info(f"  - Interpolated data saved to: {output_path}")
        logger.info(f"  - File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        logger.warning(f"  - Failed to save interpolated data: {str(e)}")


def interpolate_data(data, config, logger, original_file_path=None, current_sampling_rate=100.0, data_type='unknown'):
    """Interpolate data using cubic splines and optionally save to file"""
    
    # Check if interpolated file already exists
    if original_file_path and config['interpolation'].get('save_interpolated', False):
        interpolated_path = _get_interpolated_file_path(original_file_path, config)
        
        if os.path.exists(interpolated_path):
            logger.info(f"=== INTERPOLATED FILE ALREADY EXISTS ===")
            logger.info(f"Found existing interpolated file: {interpolated_path}")
            logger.info(f"Loading existing interpolated data instead of recomputing...")
            
            result = _load_existing_interpolated_data(interpolated_path, logger)
            if result is not None:
                return result
            else:
                logger.info("Proceeding with new interpolation...")
    
    # Proceed with interpolation if file doesn't exist or loading failed
    logger.info("=== PERFORMING NEW INTERPOLATION ===")
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    
    logger.info(f"Original data loaded:")
    logger.info(f"  - Number of data points: {len(x)}")
    logger.info(f"  - X range: [{x.min():.6f}, {x.max():.6f}] (span: {x.max() - x.min():.6f})")
    logger.info(f"  - Y range: [{y.min():.6f}, {y.max():.6f}] (span: {y.max() - y.min():.6f})")
    logger.info(f"  - Z range: [{z.min():.6f}, {z.max():.6f}] (span: {z.max() - z.min():.6f})")
    logger.info(f"  - Data shape: {data.shape}")
    logger.info(f"  - Memory usage (MB): {data.nbytes / 1024 / 1024:.2f}")
    
    logger.info("=== INTERPOLATION PHASE ===")
    t = np.arange(len(x))
    logger.info(f"  - Original time parameter range: [0, {len(x)-1}]")
    logger.info(f"  - Time step size: 1")
    
    # Create interpolation functions
    if config['interpolation']['method'] == 'cubic_spline':
        cs_x = CubicSpline(t, x)
        cs_y = CubicSpline(t, y)
        cs_z = CubicSpline(t, z)
        logger.info("  - Cubic spline interpolation functions created")
    else:
        # Linear interpolation fallback
        cs_x = interp1d(t, x, kind='linear')
        cs_y = interp1d(t, y, kind='linear')
        cs_z = interp1d(t, z, kind='linear')
        logger.info("  - Linear interpolation functions created")
    
    # Generate interpolated points
    interpolation_factor = config['interpolation']['factor']
    t_new = np.linspace(0, len(x) - 1, num=len(x) * interpolation_factor)
    logger.info(f"  - Interpolation factor: {interpolation_factor}x")
    logger.info(f"  - New time parameter range: [0, {len(x)-1}]")
    logger.info(f"  - New time points: {len(t_new)}")
    logger.info(f"  - New time step size: {(len(x)-1) / (len(t_new)-1):.6f}")
    logger.info(f"  - Estimated memory usage: {len(t_new) * 3 * 8 / 1024 / 1024:.1f} MB (float64)")
    
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    z_new = cs_z(t_new)
    points_new = np.vstack([x_new, y_new, z_new]).T
    
    logger.info(f"Interpolated data generated:")
    logger.info(f"  - Number of interpolated points: {len(x_new)}")
    logger.info(f"  - X range: [{x_new.min():.6f}, {x_new.max():.6f}] (span: {x_new.max() - x_new.min():.6f})")
    logger.info(f"  - Y range: [{y_new.min():.6f}, {y_new.max():.6f}] (span: {y_new.max() - y_new.min():.6f})")
    logger.info(f"  - Z range: [{z_new.min():.6f}, {z_new.max():.6f}] (span: {z_new.max() - z_new.min():.6f})")
    logger.info(f"  - Memory usage (MB): {points_new.nbytes / 1024 / 1024:.2f}")
    logger.info(f"  - Resolution improvement: {interpolation_factor}x finer sampling")
    
    # Save interpolated data if requested
    if config['interpolation'].get('save_interpolated', False) and original_file_path:
        _save_interpolated_data(x_new, y_new, z_new, original_file_path, config, logger)
    
    return points_new, x_new, y_new, z_new
