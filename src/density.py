"""
Density computation module for Aletheia Fractal Analysis

This module handles density analysis using Gaussian KDE for trajectory coloring.
"""

import numpy as np
from scipy.stats import gaussian_kde
import logging
import os
import pandas as pd


def _get_density_calculation_path(original_file_path, config):
    """Generate calculation file path for density data"""
    file_dir = os.path.dirname(original_file_path)
    file_name = os.path.basename(original_file_path)
    name_without_ext = os.path.splitext(file_name)[0]
    
    # Include interpolation parameters in calculation filename for uniqueness
    if config['interpolation']['enabled']:
        interpolation_method = config['interpolation'].get('method', 'cubic_spline')
        interpolation_factor = config['interpolation'].get('factor', 100)
        interpolation_suffix = f"_{interpolation_method}_{interpolation_factor}x"
    else:
        interpolation_method = ""
        interpolation_factor = 1
        interpolation_suffix = ""
    
    calculation_format = config['density'].get('calculation_format', 'csv')
    calculation_filename = f"{name_without_ext}{interpolation_suffix}_density_calculations.{calculation_format}"
    
    return os.path.join(file_dir, calculation_filename)


def _load_density_calculations(calculation_path, calculation_format, logger):
    """Load density data from calculation file"""
    try:
        if calculation_format == 'npz':
            calculation_data = np.load(calculation_path)
            dens_norm = calculation_data['dens_norm']
            logger.info(f"Successfully loaded density calculations from NPZ file:")
        elif calculation_format == 'csv':
            calculation_df = pd.read_csv(calculation_path)
            dens_norm = calculation_df['dens_norm'].values
            logger.info(f"Successfully loaded density calculations from CSV file:")
        else:
            raise ValueError(f"Unsupported calculation format: {calculation_format}")
        
        logger.info(f"  - Number of density values: {len(dens_norm)}")
        logger.info(f"  - Density range: [{dens_norm.min():.6f}, {dens_norm.max():.6f}]")
        logger.info(f"  - File size: {os.path.getsize(calculation_path) / 1024:.2f} KB")
        
        return dens_norm
        
    except Exception as e:
        logger.warning(f"Failed to load density calculations: {str(e)}")
        return None


def _save_density_calculations(dens_norm, calculation_path, calculation_format, logger):
    """Save density data to calculation file"""
    try:
        if calculation_format == 'npz':
            np.savez_compressed(calculation_path, dens_norm=dens_norm)
            logger.info(f"Density calculations saved to NPZ file: {calculation_path}")
        elif calculation_format == 'csv':
            calculation_df = pd.DataFrame({'dens_norm': dens_norm})
            calculation_df.to_csv(calculation_path, index=False)
            logger.info(f"Density calculations saved to CSV file: {calculation_path}")
        else:
            raise ValueError(f"Unsupported calculation format: {calculation_format}")
        
        logger.info(f"  - File size: {os.path.getsize(calculation_path) / 1024:.2f} KB")
        
    except Exception as e:
        logger.warning(f"Failed to save density calculations: {str(e)}")


def compute_density(points_orig, points_new, logger, config=None, original_file_path=None):
    """Compute KDE for density coloring with optional calculation saving"""
    
    # Check for existing density calculations if saving is enabled
    if config and config.get('density', {}).get('save_density_calculations', False) and original_file_path:
        calculation_path = _get_density_calculation_path(original_file_path, config)
        calculation_format = config['density'].get('calculation_format', 'csv')
        
        if os.path.exists(calculation_path):
            logger.info("=== DENSITY CALCULATIONS FOUND ===")
            logger.info(f"Found existing density calculations: {calculation_path}")
            logger.info("Loading saved density calculations instead of recomputing...")
            
            saved_dens_norm = _load_density_calculations(calculation_path, calculation_format, logger)
            if saved_dens_norm is not None:
                return saved_dens_norm
            else:
                logger.info("Calculation loading failed, proceeding with new computation...")
    
    # Proceed with density computation if no saved calculations or loading failed
    logger.info("=== PERFORMING NEW DENSITY ANALYSIS ===")
    logger.info(f"  - Using original {len(points_orig[0])} points for KDE fitting (memory efficient)")
    logger.info(f"  - Evaluating KDE on {len(points_new)} interpolated points")
    
    kde = gaussian_kde(points_orig)
    logger.info("  - Gaussian KDE fitted on original data")
    
    densities = kde.evaluate(np.vstack([points_new[:, 0], points_new[:, 1], points_new[:, 2]]))
    densities_log = np.log(densities + 1e-10)
    dens_norm = (densities_log - densities_log.min()) / (densities_log.max() - densities_log.min())
    
    logger.info(f"Density analysis completed:")
    logger.info(f"  - Raw density range: [{densities.min():.6e}, {densities.max():.6e}]")
    logger.info(f"  - Log density range: [{densities_log.min():.6f}, {densities_log.max():.6f}]")
    logger.info(f"  - Normalized density range: [{dens_norm.min():.6f}, {dens_norm.max():.6f}]")
    
    # Save density calculations if enabled
    if config and config.get('density', {}).get('save_density_calculations', False) and original_file_path:
        calculation_path = _get_density_calculation_path(original_file_path, config)
        calculation_format = config['density'].get('calculation_format', 'csv')
        _save_density_calculations(dens_norm, calculation_path, calculation_format, logger)
    
    return dens_norm
