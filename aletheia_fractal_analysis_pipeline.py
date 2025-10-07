"""
Refactored Aletheia Fractal Analysis Pipeline

This is the main wrapper script that orchestrates all the analysis modules.
"""

import argparse
import os
import json
import gc
import numpy as np
import logging
from datetime import datetime

# Import our refactored modules
from src.utils import load_config, setup_logging, auto_discover_data_files, save_discovered_datasets_to_csv, load_data
from src.interpolation import interpolate_data
from src.density import compute_density
from src.fractal_dimension import fractal_dimension
from src.visualization import create_visualization, create_interactive_3d_plot, visualize_feature_space_at_div_depths, visualize_individual_boxes

def process_single_dataset(config, dataset_info, base_output_dir, logger, run_timestamp):
    """Process a single dataset with the given configuration"""
    # Create a copy of config and update with dataset info
    dataset_config = config.copy()

    # Create dataset-specific output directory
    person = dataset_info['person']
    speed = dataset_info['speed']
    run_id = dataset_info['run_id']
    file_path = dataset_info['file_path']
    
    # Use flat structure: {person}_{speed}_{run_id}/
    dataset_output_dir = os.path.join(base_output_dir, f"{person}_{speed}_{run_id}")
    
    # Create plots directory within dataset output directory with shared timestamp
    plots_dir = os.path.join(dataset_output_dir, 'plots', run_timestamp)
    os.makedirs(plots_dir, exist_ok=True)
    
    logger.info(f"=== PROCESSING DATASET: {person}_{speed}_{run_id} ===")
    logger.info(f"Output directory: {dataset_output_dir}")
    
    # Load data
    data = load_data(dataset_config, dataset_info, logger)

    logger.info(f"Data loaded: {data.shape}")

    needs_interpolation = dataset_config['interpolation']['enabled']
    if needs_interpolation:
        # Interpolate the data
        points_new, x_new, y_new, z_new = interpolate_data(data, dataset_config, logger, file_path)
    else:
        # Use pre-interpolated data directly
        x_new = data[:, 0]
        y_new = data[:, 1]
        z_new = data[:, 2]
        points_new = data
        logger.info(f"Using pre-interpolated data with {len(points_new)} points")
    
    logger.info(f"X new: {x_new.shape}")
    logger.info(f"Y new: {y_new.shape}")
    logger.info(f"Z new: {z_new.shape}")
    logger.info(f"Points new: {points_new.shape}")
    logger.info(f"Number of points: {len(points_new)}")
    
    # Compute density (use original data points for KDE if interpolating)
    if config['visualization'].get('density_coloring', True):
        if needs_interpolation:
            points_orig = np.vstack([data[:, 0], data[:, 1], data[:, 2]])
        else:
            points_orig = np.vstack([x_new, y_new, z_new])
        
        dens_norm = compute_density(points_orig, points_new, logger, config, file_path)
    else:
        # Skip density analysis - use uniform coloring
        dens_norm = np.ones(len(points_new))  # Uniform density for single color
        logger.info("Skipping density analysis - using uniform coloring")
    
    ################## STOPPED HERE ##################

    # Compute fractal dimension
    logger.info("=== FRACTAL ANALYSIS PHASE ===")
    interpolation_factor = dataset_config['interpolation']['factor'] if needs_interpolation else 1
    data_source = dataset_config['data']['data_file_to_use'].split('_')[0]
    best_fits, log_divs, log_counts, divs, counts, method_used = fractal_dimension(points_new, dataset_config, logger, dataset_output_dir, f"{person}_{speed}_{run_id}", interpolation_factor, data_source, run_timestamp)
    
    logger.info(f"Fractal analysis completed:")
    logger.info(f" Generated {len(best_fits)} best fits, fits above the minimum magnitude requirements")
    
    # Create analysis results dictionary
    analysis_results = {
        'points_new': points_new,
        'best_fits': best_fits,
        'log_divs': log_divs,
        'log_counts': log_counts,
        'divs': divs,
        'counts': counts,
        'method_used': method_used,
        'dens_norm': dens_norm,
        'x_new': x_new,
        'y_new': y_new,
        'z_new': z_new,
        'interpolation_factor': dataset_config['interpolation']['factor'] if needs_interpolation else 1,
        'data_source': dataset_config['data']['data_file_to_use'].split('_')[0],
        'trajectory_aware': dataset_config['analysis'].get('trajectory_aware', False)
    }
    
    plot_filenames = create_visualization(analysis_results, dataset_config, logger, plots_dir, run_timestamp)
    
    # Create interactive 3D plot if requested
    if dataset_config['visualization'].get('interactive_3d', False):
        html_filename = create_interactive_3d_plot(analysis_results, dataset_config, logger, dataset_output_dir, run_timestamp)
        if html_filename:
            logger.info(f"  - Interactive 3D plot saved to: {html_filename}")
    
    # Create feature space div depth visualization if requested
    if dataset_config['visualization'].get('div_depth_visualization', {}).get('enabled', False):
        div_depth_plots = visualize_feature_space_at_div_depths(analysis_results, dataset_config, logger, plots_dir, run_timestamp)
        if div_depth_plots:
            logger.info(f"  - Feature space div depth visualizations saved: {len(div_depth_plots)} plots")
    
    # Create individual box contents visualization if requested
    if dataset_config['visualization'].get('individual_box_visualization', {}).get('enabled', False):
        individual_box_plots = visualize_individual_boxes(analysis_results, dataset_config, logger, plots_dir, run_timestamp)
        if individual_box_plots:
            logger.info(f"  - Individual box visualizations saved: {len(individual_box_plots)} plots")
    
    # Save results summary
    results_summary = {
        'person': person,
        'speed': speed,
        'run_id': run_id,
        'num_points': len(points_new),
        'trajectory_aware': config['analysis'].get('trajectory_aware', False),
        'processing_time': datetime.now().isoformat()
    }
    
    # Clean up variables to free memory
    del log_divs, log_counts, divs, counts
    del plot_filenames, dens_norm, points_new, x_new, y_new, z_new, data
    if 'points_orig' in locals():
        del points_orig
    if 'html_filename' in locals():
        del html_filename
    gc.collect()
    
    # Save results to JSON file
    results_file = os.path.join(dataset_output_dir, f"results_{person}_{speed}_{run_id}.json")
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Add results file path to summary for single dataset mode
    results_summary['results_file'] = results_file
    
    logger.info(f"  - Results saved to: {results_file}")
    logger.info(f"  - Dataset {person}_{speed}_{run_id} completed successfully")
    
    return results_summary


def main():
    """Main function to run the fractal analysis pipeline"""
    # File paths to config.yaml
    base_project_path = "/home/echo/Desktop/Taylor_Lab/Aletheia_Code/aletheia_force_portrait_analysis"
    config_file_name = 'aletheia_pipeline_config.yaml'
    config_file_path = os.path.join('config', config_file_name)
    base_log_path = 'logs'
    
    # Create single timestamp for entire run to keep all outputs grouped
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load configuration
    try:
        config = load_config(config_file_path)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_file_path}")
        print("Please create a config.yaml file or specify the correct path.")
        exit(1)
    
    # Get base output directory from config
    base_output_directory = config.get('output', {}).get('base_output_directory', '.')
    base_data_path = config['data']['base_path']
    
    # Check if batch processing is enabled
    batch_enabled = config.get('batch', {}).get('enabled', False)
    
    # Set up logging - one logger for the entire run using shared timestamp
    if batch_enabled:
        log_filename = base_output_directory + base_log_path + f"batch_pipeline_{run_timestamp}.log"
    else:
        log_filename = base_output_directory + base_log_path + f"single_pipeline_{run_timestamp}.log"
    
    # Set up logging
    logger, log_filename = setup_logging(config)
    
    logger.info("=== STARTING FRACTAL ANALYSIS PIPELINE ===")
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Configuration file: {config_file_path}")
    logger.info(f"Using base output directory: {base_output_directory}")
    logger.info(f"Interpolation enabled: {config['interpolation']['enabled']}")
    
    if batch_enabled:
        # Batch processing mode
        logger.info("=== BATCH PROCESSING MODE ENABLED ===")
        
        # Always auto-discover data files in batch mode
        logger.info("=== AUTO-DISCOVERING DATA FILES ===")
        target_data_file = config['data'].get('data_file', 'interpdata.csv')
        
        logger.info(f"Searching for {target_data_file} files in: {base_data_path}")
        list_of_all_datasets_info = auto_discover_data_files(base_data_path, target_data_file)
        
        if not list_of_all_datasets_info:
            logger.error("No datasets found! Please check the base_path and data_type settings.")
            exit(1)
        
        # Save discovered datasets to CSV
        name_of_datasets_csv_file = 'discovered_datasets.csv'
        csv_filename = save_discovered_datasets_to_csv(list_of_all_datasets_info, base_output_directory, name_of_datasets_csv_file)
        logger.info(f"Discovered datasets saved to: {csv_filename}")
        
        # Set up batch output directory (subdirectory of base output directory)
        batch_output_dir = os.path.join(base_output_directory, "batches", f"batch_{run_timestamp}")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        logger.info(f"Number of datasets: {len(list_of_all_datasets_info)}")
        logger.info(f"Batch output directory: {batch_output_dir}")
        logger.info(f"Trajectory-aware mode: {config['analysis'].get('trajectory_aware', False)}")
        
        for i, dataset_info in enumerate(list_of_all_datasets_info, 1):
            dataset_name = f"{dataset_info['person']}_{dataset_info['speed']}_{dataset_info['run_id']}"
            logger.info(f"=== PROCESSING DATASET {i}/{len(list_of_all_datasets_info)}: {dataset_name} ===")
            logger.info(f"Processing dataset {i}/{len(list_of_all_datasets_info)}: {dataset_name}")
            
            result = process_single_dataset(config, dataset_info, batch_output_dir, logger, run_timestamp)
            
            logger.info("=== PROCESSING PIPELINE COMPLETE ===")
            logger.info(f"Final output:")
            logger.info(f"  - Total points processed: {result['num_points']}")
            logger.info(f"  - Log file saved to: {log_filename}")
            logger.info(f"  - Results saved to: {result.get('results_file', 'N/A')}")

            # Basic garbage collection to free memory
            gc.collect()
        
        # Create batch summary
        logger.info("=== BATCH PROCESSING COMPLETE ===")
        logger.info(f"Total datasets: {len(list_of_all_datasets_info)}")
        logger.info(f"Total Processing time: {datetime.now().isoformat()}")
        logger.info(f"Log file: {log_filename}")
    else:
        # Single dataset processing mode - use process_single_dataset function
        logger.info("=== SINGLE DATASET PROCESSING MODE ===")
        logger.info(f"Log file: {log_filename}")
        logger.info(f"Configuration file: {config_file_path}")
        
        # Get dataset information from config
        single_config = config.get('single', {})
        person = single_config.get('person', 'unknown')
        speed = single_config.get('speed', 'unknown')
        run_id = single_config.get('run_id', 'unknown')
        
        # The structure should be: person/speed/run_id/DBFiles/filename
        data_file_to_use = config['data'].get('data_file_to_use', 'interpdata.csv')
        file_path = os.path.join(base_data_path, person, speed, run_id, 'DBFiles', data_file_to_use)

        # Create dataset_info dictionary for process_single_dataset
        dataset_info = {
            'person': person,
            'speed': speed,
            'run_id': run_id,
            'file_path': file_path
        }
        
        # Use process_single_dataset function to avoid code duplication
        result = process_single_dataset(config, dataset_info, base_output_directory, logger, run_timestamp)
        
        logger.info("=== PROCESSING PIPELINE COMPLETE ===")
        logger.info(f"Final output:")
        logger.info(f"  - Total points processed: {result['num_points']}")
        logger.info(f"  - Log file saved to: {log_filename}")
        logger.info(f"  - Results saved to: {result.get('results_file', 'N/A')}")

if __name__ == "__main__":
    main()
