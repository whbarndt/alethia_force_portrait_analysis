"""
Density computation module for Aletheia Fractal Analysis

This module handles density analysis using Gaussian KDE for trajectory coloring.
"""

import numpy as np
from scipy.stats import gaussian_kde
import logging


def compute_density(points_orig, points_new, logger):
    """Compute KDE for density coloring"""
    logger.info("=== DENSITY ANALYSIS PHASE ===")
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
    
    return dens_norm
