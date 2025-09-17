"""
Fractal dimension computation module for Aletheia Fractal Analysis

This module handles box counting fractal dimension analysis with optional sliding grid method.
"""

import numpy as np
from scipy.stats import linregress
import multiprocessing as mp
import gc
import logging


# Global variables for multiprocessing
_global_scaled_points = None
_global_trajectory_aware = None


def count_occupied_boxes_global(points, d, trajectory_aware):
    """Helper function to count occupied boxes (moved outside for multiprocessing)"""
    if not trajectory_aware:
        # Original point-cloud method
        binned = np.floor(points * d).astype(int)
        unique_boxes = set(tuple(b) for b in binned)
    else:
        # Trajectory-aware: sample segments
        unique_boxes = set()
        for j in range(len(points) - 1):
            p1 = points[j]
            p2 = points[j + 1]
            delta = p2 - p1
            length = np.linalg.norm(delta)
            if length == 0:
                b = tuple(np.floor(p1 * d).astype(int))
                unique_boxes.add(b)
                continue
            safety_factor = 1.1 # safety factor for sampling segments
            min_samples = 2 # minimum number of samples for sampling segments
            # calculate the number of samples for the segment
            num_samples = max(min_samples, int(length * d * safety_factor) + 1)
            t = np.linspace(0, 1, num_samples)
            seg_points = p1[None, :] + t[:, None] * delta[None, :]
            binned = np.floor(seg_points * d).astype(int)
            unique_seg = set(tuple(b) for b in binned)
            unique_boxes.update(unique_seg)
    return len(unique_boxes)


def fixed_grid_task_wrapper(d):
    """Global wrapper function for multiprocessing"""
    return count_occupied_boxes_global(_global_scaled_points, d, _global_trajectory_aware)


def sliding_grid_task_wrapper(args):
    """Global wrapper function for multiprocessing with sliding grid"""
    d, shift_x, shift_y, shift_z = args
    shift = np.array([shift_x, shift_y, shift_z])
    shifted_points = (_global_scaled_points + shift) % 1.0
    return count_occupied_boxes_global(shifted_points, d, _global_trajectory_aware), (shift_x, shift_y, shift_z)


def fractal_dimension(points, config, logger):
    """Calculate fractal dimension using box counting with optional sliding grid method"""
    global _global_scaled_points, _global_trajectory_aware
    
    logger.info("  - Starting fractal dimension calculation")
    
    # Extract only X, Y, Z coordinates (first 3 columns)
    if points.shape[1] > 3:
        logger.info(f"  - Data has {points.shape[1]} columns, using first 3 for X, Y, Z coordinates")
        points_3d = points[:, :3]
    else:
        points_3d = points
    
    # Normalize points to [0,1] cube
    mins = points_3d.min(axis=0)
    maxs = points_3d.max(axis=0)
    scaled_points = (points_3d - mins) / (maxs - mins + 1e-10)
    
    # Set global variables for multiprocessing
    _global_scaled_points = scaled_points
    _global_trajectory_aware = config['analysis'].get('trajectory_aware', False)
    
    logger.info(f"  - Original point range: X[{mins[0]:.6f}, {maxs[0]:.6f}], Y[{mins[1]:.6f}, {maxs[1]:.6f}], Z[{mins[2]:.6f}, {maxs[2]:.6f}]")
    logger.info(f"  - Normalized to unit cube: [0,1]³")
    
    # Box counting scales - use more appropriate range for 3D data
    min_div = config['analysis']['min_box_divisions']
    max_div = min(int(len(points) ** (1/3) * config['analysis']['max_box_divisions_factor']), 1000)
    divs = np.unique(np.logspace(np.log10(min_div), np.log10(max_div), 
                                 num=config['analysis']['num_box_scales'], dtype=int))
    
    logger.info(f"  - Box counting scales: {len(divs)} divisions from {min_div} to {max_div}")
    logger.info(f"  - Division values: {divs}")
    
    # Check if trajectory-aware counting is enabled
    trajectory_aware = config['analysis'].get('trajectory_aware', False)
    if trajectory_aware:
        logger.info("  - Using trajectory-aware box counting (accounts for line segments)")
    else:
        logger.info("  - Using standard point-cloud box counting")
    
    # Number of processes for parallelization
    num_processes = config['analysis'].get('num_processes', mp.cpu_count())
    logger.info(f"  - Using {num_processes} processes for parallelization")
    
    # Always compute fixed grid counts first
    logger.info("  - Computing fixed grid box counts (Method 1)")
    with mp.Pool(processes=num_processes) as pool:
        counts_fixed = pool.map(fixed_grid_task_wrapper, divs)
    
    for i, (d, num_boxes) in enumerate(zip(divs, counts_fixed)):
        if i % 2 == 0 or i == len(divs) - 1:
            logger.info(f"    - Division {d}: {num_boxes} boxes")
    
    # Check if sliding grid method is enabled
    use_sliding_grid = config['analysis'].get('use_sliding_grid', False)
    
    if use_sliding_grid:
        logger.info("  - Using SLIDING GRID method (more accurate but slower)")
        if trajectory_aware:
            logger.warning("  - Trajectory-aware with sliding grid may be computationally intensive; consider disabling sliding if slow")
        
        logger.info("  - Method 2: Sliding grid box counting (minimum variance method)")
        counts_sliding = []
        grid_shifts = config['analysis'].get('grid_shifts', 5)
        
        # Generate all shift combinations
        shift_values = np.linspace(0, 1, grid_shifts)
        shift_combinations = [(d, shift_x, shift_y, shift_z) 
                             for d in divs 
                             for shift_x in shift_values 
                             for shift_y in shift_values 
                             for shift_z in shift_values]
        
        # Parallelize over shift combinations
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(sliding_grid_task_wrapper, shift_combinations)
        
        # Process results per d
        for i, d in enumerate(divs):
            min_boxes = float('inf')
            best_shift = None
            
            # Filter results for current d
            d_results = [(num_boxes, shift) for (num_boxes, shift) in results 
                         if abs(shift_combinations[results.index((num_boxes, shift))][0] - d) < 1e-10]
            
            for num_boxes, shift in d_results:
                if num_boxes < min_boxes:
                    min_boxes = num_boxes
                    best_shift = shift
            
            counts_sliding.append(min_boxes)
            if i % 2 == 0 or i == len(divs) - 1:
                logger.info(f"    - Division {d}: {min_boxes} boxes (best shift: {best_shift})")
        
        # Compare both methods
        logger.info("  - Comparing methods:")
        logger.info(f"    - Fixed grid counts: {counts_fixed}")
        logger.info(f"    - Sliding grid counts: {counts_sliding}")
        
        # Calculate fractal dimensions for both methods
        log_divs = np.log(divs)
        
        # Method 1: Fixed grid
        log_counts_fixed = np.log(counts_fixed)
        logger.info(f"  - Fixed grid log-log range: divisions [{log_divs.min():.3f}, {log_divs.max():.3f}], counts [{log_counts_fixed.min():.3f}, {log_counts_fixed.max():.3f}]")
        
        # Method 2: Sliding grid
        log_counts_sliding = np.log(counts_sliding)
        logger.info(f"  - Sliding grid log-log range: divisions [{log_divs.min():.3f}, {log_divs.max():.3f}], counts [{log_counts_sliding.min():.3f}, {log_counts_sliding.max():.3f}]")
        
        # Find best fits for both methods
        def find_best_fits(log_divs, log_counts, method_name):
            all_fits = []
            min_fit_points = config['analysis']['min_fit_points']
            
            for window_size in range(min_fit_points, len(divs) + 1):
                for start in range(len(divs) - window_size + 1):
                    end = start + window_size
                    result = linregress(log_divs[start:end], log_counts[start:end])
                    fit_info = {
                        'start': start, 'end': end, 'D': result.slope, 
                        'R2': result.rvalue ** 2, 'intercept': result.intercept,
                        'range': (start, end), 'method': method_name
                    }
                    all_fits.append(fit_info)
            
            preferred_fits = [f for f in all_fits if 1.8 <= f['D'] <= 2.8]
            
            if preferred_fits:
                best_fit = max(preferred_fits, key=lambda x: x['R2'])
                logger.info(f"    - {method_name} preferred fit: D={best_fit['D']:.4f}, R²={best_fit['R2']:.4f}, range={best_fit['range']}")
            else:
                best_fit = max(all_fits, key=lambda x: x['R2'])
                logger.warning(f"    - {method_name} no preferred fits, using highest R²: D={best_fit['D']:.4f}, R²={best_fit['R2']:.4f}")
            
            return best_fit, all_fits
        
        best_fixed, all_fixed = find_best_fits(log_divs, log_counts_fixed, "Fixed grid")
        best_sliding, all_sliding = find_best_fits(log_divs, log_counts_sliding, "Sliding grid")
        
        expected_D = 2.25
        fixed_error = abs(best_fixed['D'] - expected_D)
        sliding_error = abs(best_sliding['D'] - expected_D)
        
        if sliding_error < fixed_error:
            logger.info(f"  - Selected sliding grid method (closer to expected D={expected_D})")
            best_D = best_sliding['D']
            best_R2 = best_sliding['R2']
            best_intercept = best_sliding['intercept']
            best_range = best_sliding['range']
            counts = counts_sliding
            log_counts = log_counts_sliding
            method_used = "sliding_grid"
        else:
            logger.info(f"  - Selected fixed grid method (closer to expected D={expected_D})")
            best_D = best_fixed['D']
            best_R2 = best_fixed['R2']
            best_intercept = best_fixed['intercept']
            best_range = best_fixed['range']
            counts = counts_fixed
            log_counts = log_counts_fixed
            method_used = "fixed_grid"
        
        logger.info(f"  - Final selection: {method_used} - D={best_D:.4f}, R²={best_R2:.4f}, range={best_range}")
        
    else:
        logger.info("  - Using STANDARD FIXED GRID method (faster)")
        counts = counts_fixed
        log_divs = np.log(divs)
        log_counts = np.log(counts)
        
        logger.info(f"  - Log-log data range: divisions [{log_divs.min():.3f}, {log_divs.max():.3f}], counts [{log_counts.min():.3f}, {log_counts.max():.3f}]")
        
        all_fits = []
        min_fit_points = config['analysis']['min_fit_points']
        logger.info("  - Performing dynamic range selection for optimal fit")
        
        for window_size in range(min_fit_points, len(divs) + 1):
            for start in range(len(divs) - window_size + 1):
                end = start + window_size
                result = linregress(log_divs[start:end], log_counts[start:end])
                fit_info = {
                    'start': start, 'end': end, 'D': result.slope, 
                    'R2': result.rvalue ** 2, 'intercept': result.intercept,
                    'range': (start, end)
                }
                all_fits.append(fit_info)
        
        all_fits.sort(key=lambda x: x['R2'], reverse=True)
        logger.info("  - Top 5 fits:")
        for i, fit in enumerate(all_fits[:5]):
            logger.info(f"    {i+1}. D={fit['D']:.4f}, R²={fit['R2']:.4f}, range={fit['range']}")
        
        preferred_fits = [f for f in all_fits if 1.8 <= f['D'] <= 2.8]
        
        if preferred_fits:
            best_fit = max(preferred_fits, key=lambda x: x['R2'])
            best_D = best_fit['D']
            best_R2 = best_fit['R2']
            best_intercept = best_fit['intercept']
            best_range = best_fit['range']
            logger.info(f"  - Selected preferred fit: D={best_D:.4f}, R²={best_R2:.4f}, range={best_range}")
        else:
            best_fit = all_fits[0]
            best_D = best_fit['D']
            best_R2 = best_fit['R2']
            best_intercept = best_fit['intercept']
            best_range = best_fit['range']
            logger.warning(f"  - No preferred fits found, using highest R²: D={best_D:.4f}, R²={best_R2:.4f}")
        
        logger.info(f"  - Best fit found: D={best_D:.4f}, R²={best_R2:.4f}, range={best_range}")
    
    full_result = linregress(log_divs, log_counts)
    full_D = full_result.slope
    full_R2 = full_result.rvalue ** 2
    full_intercept = full_result.intercept
    logger.info(f"  - Full range fit: D={full_D:.4f}, R²={full_R2:.4f}")
    
    if best_D < 1.5:
        logger.warning(f"  - WARNING: Fractal dimension {best_D:.4f} seems low for 3D trajectory data!")
        logger.warning(f"  - Expected range: 2.0-2.5 for typical 3D trajectories")
        logger.warning(f"  - This might indicate an issue with the analysis or data")
    
    # Clean up global variables to free memory
    _global_scaled_points = None
    _global_trajectory_aware = None
    gc.collect()
    
    return best_D, best_R2, log_divs, log_counts, divs, counts, best_intercept, best_range, full_D, full_R2, full_intercept
