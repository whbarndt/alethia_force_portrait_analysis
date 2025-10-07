"""
Fractal dimension computation module for Aletheia Fractal Analysis

This module handles box counting fractal dimension analysis with optional sliding grid method.
"""

import numpy as np
from scipy.stats import linregress
import gc
import time
import pandas as pd
import os
from .utils import generate_suffixes


def _count_occupied_boxes(points, d, trajectory_aware):
    """Helper function to count occupied boxes"""
    if not trajectory_aware:
        # Original point-cloud method
        binned = np.floor(points * d).astype(int)
        unique_boxes = set(tuple(b) for b in binned)
    else:
        # Trajectory-aware: sample segments with memory optimization
        unique_boxes = set()
        # Process in chunks to reduce memory usage
        chunk_size = min(10000, len(points) - 1)  # Process max 10k segments at a time
        for start_idx in range(0, len(points) - 1, chunk_size):
            end_idx = min(start_idx + chunk_size, len(points) - 1)
            chunk_boxes = set()
            
            for j in range(start_idx, end_idx):
                p1 = points[j]
                p2 = points[j + 1]
                delta = p2 - p1
                length = np.linalg.norm(delta)
                if length == 0:
                    b = tuple(np.floor(p1 * d).astype(int))
                    chunk_boxes.add(b)
                    continue
                
                # Reduced safety factor and max samples for efficiency
                safety_factor = 1.05
                min_samples = 2
                max_samples = 50
                num_samples = min(max_samples, max(min_samples, int(length * d * safety_factor) + 1))
                
                t = np.linspace(0, 1, num_samples)
                seg_points = p1[None, :] + t[:, None] * delta[None, :]
                binned = np.floor(seg_points * d).astype(int)
                unique_seg = set(tuple(b) for b in binned)
                chunk_boxes.update(unique_seg)
            
            unique_boxes.update(chunk_boxes)
            del chunk_boxes

    return len(unique_boxes)

def _calculate_fits(log_divs, log_counts, method_name, divs, counts, analysis_config, logger):
    """Helper function for finding best fits across different ranges"""
    all_fits = []
    # Get config values
    min_fit_points = analysis_config['min_fit_points']
    minimum_kept_D = analysis_config['minimum_kept_D']
    maximum_kept_D = analysis_config['maximum_kept_D']
    min_span_orders = analysis_config.get('min_span_orders', 1.0)  # Minimum orders of magnitude span
    
    logger.info(f"  - Calculating all possible fits for {method_name}...")
    
    for window_size in range(min_fit_points, len(divs) + 1):
        for start in range(len(divs) - window_size + 1):
            end = start + window_size
            result = linregress(log_divs[start:end], log_counts[start:end])
            
            # Get the actual division values and counts for this range
            range_divs = divs[start:end]
            range_counts = counts[start:end]
            
            fit_info = {
                'method': method_name,
                'start_idx': start,
                'end_idx': end,
                'num_points': window_size,
                'D': result.slope, 
                'R2': result.rvalue ** 2, 
                'intercept': result.intercept,
                'range_divs': ','.join(map(str, range_divs)),
                'range_counts': ','.join(map(str, range_counts)),
                'min_div': np.array(range_divs).min(),
                'max_div': np.array(range_divs).max(),
                'min_count': np.array(range_counts).min(),
                'max_count': np.array(range_counts).max()
            }
            all_fits.append(fit_info)
    
    # Filter fits by D range and minimum span requirement
    preferred_fits = []
    for f in all_fits:
        if minimum_kept_D <= f['D'] <= maximum_kept_D:
            # Check if span covers at least min_span_orders orders of magnitude
            span_orders = np.log10(f['max_div'] / f['min_div'])
            if span_orders >= min_span_orders:
                f['span_orders'] = span_orders
                preferred_fits.append(f)

    return preferred_fits, all_fits


def fractal_dimension(points, config, logger, output_dir='.', dataset_info=None, interpolation_factor=1, data_source='', run_timestamp=None):
    """
    Calculate fractal dimensions using box counting with optional sliding grid method
    
    Args:
        points: numpy array of points
        config: dictionary of configuration
        logger: logger object
        output_dir: directory to output the results
        dataset_info: string of dataset information
        interpolation_factor: integer of interpolation factor
        data_source: string of data source
        run_timestamp: string of run timestamp
    
    Returns:
        best_fits: list of best fits above the minimum magnitude requirements
        all_fits: list of all fits above the minimum points requirement
        log_divs: numpy array of log division values
        log_counts: numpy array of log count values
        divs: list of division values
        counts: list of count values
        method_used: string of method used
    """
    logger.info("  - Starting fractal dimension calculation")
    
    # Extract only X, Y, Z coordinates (first 3 columns)
    if points.shape[1] > 3:
        logger.info(f"  - Data has {points.shape[1]} columns, using first 3 for fractal analysis")
        logger.info(f"  - Fourth column looks like this: {points[:, 3]}")
        points_3d = points[:, :3]
    else:
        points_3d = points
    
    # Normalize points to [0,1] cube
    mins = points_3d.min(axis=0)
    maxs = points_3d.max(axis=0)
    scaled_points = (points_3d - mins) / (maxs - mins + 1e-10)
    
    logger.info(f"  - Original point range: X[{mins[0]:.6f}, {maxs[0]:.6f}], Y[{mins[1]:.6f}, {maxs[1]:.6f}], Z[{mins[2]:.6f}, {maxs[2]:.6f}]")
    logger.info(f"  - Normalized to unit cube: [0,1]³")
    
    # Box counting scales - use more appropriate range for 3D data
    min_div = config['analysis']['min_box_divisions']
    max_div = config['analysis']['max_box_divisions']
    divs = np.unique(np.logspace(np.log10(min_div), np.log10(max_div), num=config['analysis']['num_box_scales'], dtype=int))
    
    logger.info(f"  - Box counting scales: {len(divs)} divisions from {min_div} to {max_div}")
    logger.info(f"  - Division values: {divs}")
    
    # Check if trajectory-aware counting is enabled
    trajectory_aware = config['analysis'].get('trajectory_aware', False)
    if trajectory_aware:
        logger.info("  - Using trajectory-aware box counting (accounts for line segments)")
        trajectory_aware_string = "trajectory-aware"
    else:
        logger.info("  - Using standard point-cloud box counting")
        trajectory_aware_string = "standard"
    
    # Always compute fixed grid counts first
    logger.info("  - Computing fixed grid box counts (Method 1)")
    count_occupied_boxes_start_time = time.time()
    counts_fixed = []
    for d in divs:
        count = _count_occupied_boxes(scaled_points, d, trajectory_aware)
        counts_fixed.append(count)
    count_occupied_boxes_computation_time = time.time() - count_occupied_boxes_start_time      

    logger.info(f"  - {trajectory_aware_string} computation completed in {count_occupied_boxes_computation_time:.2f} seconds")
    
    # Log the counts for each division
    for i, (d, num_boxes) in enumerate(zip(divs, counts_fixed)):
        logger.info(f"    - Division {d}: {num_boxes} boxes")
    
    # Always compute fixed grid analysis first
    log_divs = np.log(divs)
    log_counts_fixed = np.log(counts_fixed)
    logger.info(f"  - Fixed grid log-log range: divisions [{log_divs.min():.3f}, {log_divs.max():.3f}], counts [{log_counts_fixed.min():.3f}, {log_counts_fixed.max():.3f}]")
    
    # Find best fits for fixed grid
    best_fits_fixed, all_fits_fixed = _calculate_fits(log_divs, log_counts_fixed, "Fixed grid", divs, counts_fixed, config['analysis'], logger)
    
    # Initialize with fixed grid results
    counts = counts_fixed
    log_counts = log_counts_fixed
    best_fits = best_fits_fixed
    all_fits = all_fits_fixed
    method_used = "fixed_grid"
    
    # Check if sliding grid method is enabled
    use_sliding_grid = config['analysis'].get('use_sliding_grid', False)
    if use_sliding_grid:
        logger.info("  - Using SLIDING GRID method (more accurate but slower)")
        if trajectory_aware:
            logger.warning("  - Trajectory-aware with sliding grid may be computationally intensive; consider disabling sliding if slow")
        
        logger.info("  - Method 2: Sliding grid box counting (minimum variance method)")
        grid_shifts = config['analysis'].get('grid_shifts', 5)
        
        # Generate all shift combinations
        shift_values = np.linspace(0, 1, grid_shifts)
        
        # Sequential computation for sliding grid
        sliding_start_time = time.time()
        counts_sliding = []
        
        for i, d in enumerate(divs):
            min_boxes = float('inf')
            best_shift = None
            
            # Try all shift combinations for this division
            for shift_x in shift_values:
                for shift_y in shift_values:
                    for shift_z in shift_values:
                        shift = np.array([shift_x, shift_y, shift_z])
                        
                        shifted_points = (scaled_points + shift) % 1.0
                        num_boxes = _count_occupied_boxes(shifted_points, d, trajectory_aware)
                        
                        if num_boxes < min_boxes:
                            min_boxes = num_boxes
                            best_shift = (shift_x, shift_y, shift_z)
            
            counts_sliding.append(min_boxes)
            logger.info(f"    - Division {d}: {min_boxes} boxes (best shift: {best_shift})")
        
        sliding_computation_time = time.time() - sliding_start_time
        logger.info(f"  - Sliding grid computation completed in {sliding_computation_time:.2f} seconds")
        
        # Compare both methods
        logger.info("  - Comparing methods:")
        logger.info(f"    - Fixed grid counts: {counts_fixed}")
        logger.info(f"    - Sliding grid counts: {counts_sliding}")
        
        # Calculate fractal dimensions for sliding grid
        log_counts_sliding = np.log(counts_sliding)
        logger.info(f"  - Sliding grid log-log range: divisions [{log_divs.min():.3f}, {log_divs.max():.3f}], counts [{log_counts_sliding.min():.3f}, {log_counts_sliding.max():.3f}]")
        
        # Find best fits for sliding grid
        best_fits_sliding, all_fits_sliding = _calculate_fits(log_divs, log_counts_sliding, "Sliding grid", divs, counts_sliding, config['analysis'], logger)
        
        counts = counts_sliding
        log_counts = log_counts_sliding
        best_fits = best_fits_sliding
        all_fits = all_fits_sliding
        method_used = "sliding_grid"
    
    # Force garbage collection to free memory
    gc.collect()
    
    # Export all fits to CSV if enabled
    if config['analysis'].get('export_all_fits_csv', True):
        # Create fits directory
        fits_dir = os.path.join(output_dir, 'fits')
        os.makedirs(fits_dir, exist_ok=True)
        
        split_dataset_info = dataset_info.split('_')
        person = split_dataset_info[0]
        speed = split_dataset_info[1]
        run_id = split_dataset_info[2]

        # Generate standardized suffixes
        output_suffixes = generate_suffixes(
            dataset_info=dataset_info,
            interpolation_factor=interpolation_factor,
            data_source=data_source,
            use_sliding_grid=use_sliding_grid,
            trajectory_aware=trajectory_aware
        )
        
        # Helper function to add metadata columns to DataFrame
        def _add_metadata_columns(df, method_name, traj_aware, interp_factor, data_source, run_timestamp):
            df_copy = df.copy()
            df_copy['person'] = person
            df_copy['speed'] = speed
            df_copy['run_id'] = run_id
            df_copy['method'] = method_name
            df_copy['trajectory_aware'] = traj_aware
            df_copy['interpolation_factor'] = interp_factor
            df_copy['data_source'] = data_source
            df_copy['run_timestamp'] = run_timestamp if run_timestamp else 'unknown'
            return df_copy

        # Create dataframes for all and best fits
        df_all_fits = _add_metadata_columns(pd.DataFrame(all_fits), method_used, trajectory_aware, interpolation_factor, data_source, run_timestamp)
        df_best_fits = _add_metadata_columns(pd.DataFrame(best_fits), method_used, trajectory_aware, interpolation_factor, data_source, run_timestamp)
        
        # Name and create paths for all and best fits
        all_fits_csv_filename = f"all_fits_{method_used}_{output_suffixes}.csv"
        best_fits_csv_filename = f"best_fits_{method_used}_{output_suffixes}.csv"
        all_fits_csv_path = os.path.join(fits_dir, all_fits_csv_filename)
        best_fits_csv_path = os.path.join(fits_dir, best_fits_csv_filename)

        # Export all and best fits to CSV files
        df_all_fits.to_csv(all_fits_csv_path, index=False)
        df_best_fits.to_csv(best_fits_csv_path, index=False)
        logger.info(f"  - Exported {len(all_fits)} {method_used} fits to: {all_fits_csv_path} and {best_fits_csv_path}")
    
    return best_fits, log_divs, log_counts, divs, counts, method_used
