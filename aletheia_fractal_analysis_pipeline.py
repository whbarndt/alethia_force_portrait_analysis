"""
Refactored Aletheia Fractal Analysis Pipeline

This is the main wrapper script that orchestrates all the analysis modules.
"""

import argparse
import os
import json
import gc
import numpy as np
from datetime import datetime

# Import our refactored modules
from src.utils import load_config, setup_logging, auto_discover_data_files, save_discovered_datasets_to_csv
from src.interpolation import load_data, interpolate_data
from src.density import compute_density
from src.fractal_dimension import fractal_dimension
from src.visualization import create_visualization, create_interactive_3d_plot


def process_single_dataset(config, dataset_info, output_dir, logger):
    """Process a single dataset with the given configuration"""
    # Create a copy of config and update with dataset info
    dataset_config = config.copy()
    dataset_config['data'].update(dataset_info)
    
    # Create dataset-specific output directory
    person = dataset_info['person']
    speed = dataset_info['speed']
    run_id = dataset_info['run_id']
    
    # Use flat structure: {person}_{speed}_{run_id}/
    dataset_output_dir = os.path.join(output_dir, f"{person}_{speed}_{run_id}")
    
    # Create date-based plots directory
    current_date = datetime.now().strftime('%Y%m%d')
    plots_dir = os.path.join('plots', current_date)
    
    os.makedirs(dataset_output_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    logger.info(f"=== PROCESSING DATASET: {person}_{speed}_{run_id} ===")
    logger.info(f"Output directory: {dataset_output_dir}")
    
    try:
        # Load data
        data, needs_interpolation, file_path, current_sampling_rate, data_type = load_data(dataset_config, dataset_info, logger)
        
        if needs_interpolation:
            # Interpolate the data
            points_new, x_new, y_new, z_new = interpolate_data(data, dataset_config, logger, dataset_info['file_path'])
        else:
            # Use pre-interpolated data directly
            x_new = data[:, 0]
            y_new = data[:, 1]
            z_new = data[:, 2]
            points_new = data
            logger.info(f"Using pre-interpolated data with {len(points_new)} points")
        
        # Print number of points to monitor scale
        print(f"Number of points: {len(points_new)}")
        
        # Compute density (use original data points for KDE if interpolating)
        if config['visualization'].get('density_coloring', True):
            if needs_interpolation:
                points_orig = np.vstack([data[:, 0], data[:, 1], data[:, 2]])
            else:
                points_orig = np.vstack([x_new, y_new, z_new])
            
            dens_norm = compute_density(points_orig, points_new, logger)
        else:
            # Skip density analysis - use uniform coloring
            dens_norm = np.ones(len(points_new))  # Uniform density for single color
            logger.info("Skipping density analysis - using uniform coloring")
        
        # Compute fractal dimension
        logger.info("=== FRACTAL ANALYSIS PHASE ===")
        best_D, best_R2, log_divs, log_counts, divs, counts, best_intercept, best_range, full_D, full_R2, full_intercept = fractal_dimension(points_new, dataset_config, logger)
        
        logger.info(f"Fractal analysis completed:")
        logger.info(f"  - Fractal Dimension D: {best_D:.4f}")
        logger.info(f"  - R² goodness of fit: {best_R2:.4f}")
        logger.info(f"  - Best fit range: {best_range}")
        
        # Create visualizations
        plot_filenames = create_visualization(points_new, best_D, best_R2, log_divs, log_counts, divs, counts,
                                          best_intercept, best_range, full_D, full_R2, full_intercept, None, None, dens_norm, x_new, y_new, z_new,
                                          dataset_config, logger, plots_dir)
        
        # Create interactive 3D plot if requested
        if dataset_config['visualization'].get('interactive_3d', False):
            html_filename = create_interactive_3d_plot(points_new, dens_norm, x_new, y_new, z_new, dataset_config, logger, dataset_output_dir)
            if html_filename:
                logger.info(f"  - Interactive 3D plot saved to: {html_filename}")
        
        # Save results summary
        results_summary = {
            'person': person,
            'speed': speed,
            'run_id': run_id,
            'fractal_dimension': best_D,
            'r_squared': best_R2,
            'best_range': best_range,
            'full_range_D': full_D,
            'full_range_R2': full_R2,
            'num_points': len(points_new),
            'processing_time': datetime.now().isoformat()
        }
        
        # Clean up all variables to free memory after results summary
        del log_divs, log_counts, divs, counts, best_intercept, full_D, full_R2, full_intercept
        del plot_filenames
        if 'html_filename' in locals():
            del html_filename
        gc.collect()
        
        # Save results to JSON file
        results_file = os.path.join(dataset_output_dir, f"results_{person}_{speed}_{run_id}.json")
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"  - Results saved to: {results_file}")
        logger.info(f"  - Dataset {person}_{speed}_{run_id} completed successfully")
        
        # Clean up large variables to free memory
        del points_new, x_new, y_new, z_new
        if 'dens_norm' in locals():
            del dens_norm
        gc.collect()
        
        return results_summary
        
    except Exception as e:
        logger.error(f"Error processing dataset {person}_{speed}_{run_id}: {str(e)}")
        return None


def main():
    """Main function to run the fractal analysis pipeline"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Aletheia Fractal Analysis (Refactored)')
    parser.add_argument('--config', '-c', default='config.yaml', 
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--no-interpolate', action='store_true',
                       help='Use pre-interpolated data instead of interpolating raw data')
    parser.add_argument('--custom-start', type=int, default=None,
                        help='Start index for custom range plot')
    parser.add_argument('--custom-end', type=int, default=None,
                        help='End index for custom range plot')
    parser.add_argument('--output-dir', default='.', help='Base directory for output logs and plots')
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Please create a config.yaml file or specify the correct path.")
        return
    
    # Override interpolation setting if specified
    if args.no_interpolate:
        config['interpolation']['enabled'] = False
        print("Using pre-interpolated data (interpolation disabled)")
    else:
        print(f"Interpolation enabled: {config['interpolation']['enabled']}")
    
    # Check if batch processing is enabled
    batch_enabled = config.get('batch', {}).get('enabled', False)
    
    if batch_enabled:
        # Batch processing mode
        print("=== BATCH PROCESSING MODE ENABLED ===")
        
        try:
            # Always auto-discover data files in batch mode
            print("=== AUTO-DISCOVERING DATA FILES ===")
            base_path = config['data']['base_path']
            target_file = config['data'].get('data_file', 'interpdata.csv')
            
            print(f"Searching for {target_file} files in: {base_path}")
            datasets = auto_discover_data_files(base_path, target_file)
            
            if not datasets:
                print("No datasets found! Please check the base_path and data_type settings.")
                return
            
            # Save discovered datasets to CSV
            csv_filename = save_discovered_datasets_to_csv(datasets, 'discovered_datasets.csv')
            print(f"Discovered datasets saved to: {csv_filename}")
            
            # Set up base output directory
            batch_config = config.get('batch', {})
            base_output_dir = batch_config.get('output_base_dir', 'test')
            os.makedirs(base_output_dir, exist_ok=True)
            
            # Set up logging for batch processing
            logger, log_filename = setup_logging(config)
            logger.info("=== STARTING BATCH PROCESSING PIPELINE ===")
            logger.info(f"Log file: {log_filename}")
            logger.info(f"Configuration file: {args.config}")
            logger.info(f"Batch processing enabled: {batch_enabled}")
            logger.info(f"Data file: {target_file}")
            if config['interpolation']['enabled']:
                interpolation_method = config['interpolation'].get('method', 'cubic_spline')
                logger.info(f"Interpolation method: {interpolation_method}")
            logger.info(f"Number of datasets: {len(datasets)}")
            logger.info(f"Base output directory: {base_output_dir}")
            
            # Process each dataset
            results = []
            successful = 0
            failed = 0
            
            for i, dataset_info in enumerate(datasets, 1):
                logger.info(f"=== PROCESSING DATASET {i}/{len(datasets)} ===")
                print(f"Processing dataset {i}/{len(datasets)}: {dataset_info['person']}_{dataset_info['speed']}_{dataset_info['run_id']}")
                
                try:
                    result = process_single_dataset(config, dataset_info, base_output_dir, logger)
                    if result:
                        results.append(result)
                        successful += 1
                        logger.info(f"Dataset {i} completed successfully")
                        
                        # Force garbage collection to free memory
                        gc.collect()
                        logger.info(f"Memory cleanup completed for dataset {i}")
                    else:
                        failed += 1
                        logger.warning(f"Dataset {i} failed")
                except Exception as e:
                    failed += 1
                    logger.error(f"Error processing dataset {i}: {str(e)}")
                    print(f"Error processing dataset {i}: {str(e)}")
            
            # Create batch summary
            logger.info("=== BATCH PROCESSING COMPLETE ===")
            logger.info(f"Total datasets: {len(datasets)}")
            logger.info(f"Successful: {successful}")
            logger.info(f"Failed: {failed}")
            
            # Save batch results summary
            batch_summary = {
                'total_datasets': len(datasets),
                'successful': successful,
                'failed': failed,
                'processing_time': datetime.now().isoformat(),
                'results': results
            }
            
            batch_results_file = os.path.join(base_output_dir, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(batch_results_file, 'w') as f:
                json.dump(batch_summary, f, indent=2)
            
            logger.info(f"Batch results saved to: {batch_results_file}")
            print(f"=== BATCH PROCESSING COMPLETE ===")
            print(f"Successful: {successful}/{len(datasets)}")
            print(f"Results saved to: {batch_results_file}")
            
        except Exception as e:
            print(f"Error during batch processing: {str(e)}")
            raise
            
    else:
        # Single dataset processing mode (original functionality)
        print("=== SINGLE DATASET PROCESSING MODE ===")
        
        # Set up logging
        logger, log_filename = setup_logging(config)
        
        logger.info("=== STARTING SINGLE DATASET PROCESSING PIPELINE ===")
        logger.info(f"Log file: {log_filename}")
        logger.info(f"Configuration file: {args.config}")
        logger.info(f"Interpolation enabled: {config['interpolation']['enabled']}")
        if config['interpolation']['enabled']:
            data_file = config['data'].get('data_file', 'interpdata.csv')
            interpolation_method = config['interpolation'].get('method', 'cubic_spline')
            logger.info(f"Data file: {data_file}")
            logger.info(f"Interpolation method: {interpolation_method}")
        logger.info(f"Sliding grid method enabled: {config['analysis'].get('use_sliding_grid', False)}")
        
        try:
            # Load data
            data, needs_interpolation, file_path, current_sampling_rate, data_type = load_data(config, None, logger)
            
            if needs_interpolation:
                # Interpolate the data
                points_new, x_new, y_new, z_new = interpolate_data(data, config, logger, file_path)
            else:
                # Use pre-interpolated data directly
                x_new = data[:, 0]
                y_new = data[:, 1]
                z_new = data[:, 2]
                points_new = data
                logger.info(f"Using pre-interpolated data with {len(points_new)} points")
            
            # Print number of points to monitor scale
            print(f"Number of points: {len(points_new)}")
            
            # Compute density (use original data points for KDE if interpolating)
            if config['visualization'].get('density_coloring', True):
                if needs_interpolation:
                    points_orig = np.vstack([data[:, 0], data[:, 1], data[:, 2]])
                else:
                    points_orig = np.vstack([x_new, y_new, z_new])
                
                dens_norm = compute_density(points_orig, points_new, logger)
            else:
                # Skip density analysis - use uniform coloring
                dens_norm = np.ones(len(points_new))  # Uniform density for single color
                logger.info("Skipping density analysis - using uniform coloring")
            
            # Compute fractal dimension
            logger.info("=== FRACTAL ANALYSIS PHASE ===")
            best_D, best_R2, log_divs, log_counts, divs, counts, best_intercept, best_range, full_D, full_R2, full_intercept = fractal_dimension(points_new, config, logger)
            
            logger.info(f"Fractal analysis completed:")
            logger.info(f"  - Fractal Dimension D: {best_D:.4f}")
            logger.info(f"  - R² goodness of fit: {best_R2:.4f}")
            logger.info(f"  - Best fit range: {best_range}")
            
            # Add clarification about the linear relationship span
            if best_range:
                start, end = best_range
                start_div = divs[start]
                end_div = divs[end]
                start_count = counts[start]
                end_count = counts[end]
                
                # Calculate orders of magnitude in log space
                log_span_divs = log_divs[end] - log_divs[start]
                log_span_counts = log_counts[end] - log_counts[start]
                
                logger.info(f"  - Linear relationship span:")
                logger.info(f"    - Box divisions: {start_div} to {end_div} ({log_span_divs:.2f} orders of magnitude)")
                logger.info(f"    - Box counts: {start_count} to {end_count} ({log_span_counts:.2f} orders of magnitude)")
                logger.info(f"    - This linear relationship spans from {start_div} to {end_div} box divisions")
                logger.info(f"    - And from {start_count} to {end_count} box counts")
                logger.info(f"    - The power law N(ε) ∝ ε^(-D) holds over this range with D = {best_D:.4f}")
                
                # Add terminology definitions and explanations
                logger.info(f"  - Terminology definitions:")
                logger.info(f"    - Box Divisions: The number of grid cells along each axis (e.g., {start_div} means {start_div}×{start_div}×{start_div} = {start_div**3} total grid cells)")
                logger.info(f"    - Box Counts: The actual number of grid cells that contain trajectory points (occupancy count)")
                logger.info(f"    - Orders of Magnitude: The logarithmic span (log10(end/start)) indicating the scale range of the analysis")
                logger.info(f"    - Power Law: N(ε) ∝ ε^(-D) where N = box count, ε = box size (1/divisions), D = fractal dimension")
                logger.info(f"    - Fractal Dimension D: Measures how completely a trajectory fills 3D space (D=1=line, D=2=surface, D=3=volume)")
                logger.info(f"    - R²: Coefficient of determination, measures how well the power law fits (1.0 = perfect fit)")
                logger.info(f"    - Best Fit Range: The subset of data points that give the most reliable fractal dimension estimate")
            
            print(f"Fractal Dimension D: {best_D:.4f}, R²: {best_R2:.4f} (best range)")

            print(f"Magnification range (box divisions): {start_div} to {end_div}")

            print(f"Orders of magnitude: {log_span_divs / np.log(10):.2f}")
            
            # Create visualizations with date-based directory
            current_date = datetime.now().strftime('%Y%m%d')
            plots_dir = os.path.join('plots', current_date)
            os.makedirs(plots_dir, exist_ok=True)
            
            plot_filenames = create_visualization(points_new, best_D, best_R2, log_divs, log_counts, divs, counts,
                                              best_intercept, best_range, full_D, full_R2, full_intercept, args.custom_start, args.custom_end, dens_norm, x_new, y_new, z_new,
                                              config, logger, plots_dir)
            
            # Create interactive 3D plot if requested
            if config['visualization'].get('interactive_3d', False):
                html_filename = create_interactive_3d_plot(points_new, dens_norm, x_new, y_new, z_new, config, logger, args.output_dir)
                if html_filename:
                    logger.info(f"  - Interactive 3D plot saved to: {html_filename}")
            
            logger.info("=== PROCESSING PIPELINE COMPLETE ===")
            logger.info(f"Final output:")
            logger.info(f"  - Total points processed: {len(points_new)}")
            if needs_interpolation:
                logger.info(f"  - Data scale: {config['interpolation']['factor']}x original resolution")
            logger.info(f"  - Fractal dimension: {best_D:.4f} ± {1-best_R2:.4f} uncertainty")
            logger.info(f"  - Memory usage: {points_new.nbytes / 1024 / 1024:.2f} MB")
            logger.info(f"  - Log file saved to: {log_filename}")
            for plot_filename in plot_filenames:
                if plot_filename:
                    logger.info(f"  - Plot saved to: {plot_filename}")
            
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            raise


if __name__ == "__main__":
    main()
