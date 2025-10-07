"""
Utility functions for Aletheia Fractal Analysis

This module contains common utility functions used across the analysis pipeline.
"""

import os
import yaml
import logging
import pandas as pd
import glob
import numpy as np
from datetime import datetime


def load_data(config, dataset_info, logger):
    """Load data based on configuration and interpolation setting"""
    file_path = dataset_info['file_path']
    
    logger.info(f"Loading data from: {file_path}")
    
    # Check if data file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    return data

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def setup_logging(config, dataset_info=None):
    """Set up logging based on configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Get dataset information for log filename
    if dataset_info:
        # For batch processing, use the dataset info
        person = dataset_info['person']
        speed = dataset_info['speed']
        run_id = dataset_info['run_id']
    else:
        # For single dataset processing, use config values
        single_config = config.get('single', {})
        person = single_config.get('person', 'unknown')
        speed = single_config.get('speed', 'unknown')
        run_id = single_config.get('run_id', 'unknown')
    
    dataset_info_str = f"{person}_{speed}_{run_id}"
    
    # Set up logging to both file and console
    log_filename = f"logs/processing_{dataset_info_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Clear any existing handlers to prevent log mixing
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    
    handlers = []
    if config['logging']['include_file']:
        handlers.append(logging.FileHandler(log_filename))
    if config['logging']['include_console']:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Force reconfiguration
    )
    logger = logging.getLogger(__name__)
    
    return logger, log_filename


def auto_discover_data_files(base_path, target_file='interpdata.csv'):
    """
    Automatically discover all data files in the base path and extract path parameters.
    
    Args:
        base_path (str): Base path to search for data files
        target_file (str): File to search for (e.g., 'interpdata.csv' or 'rotatedata.csv')
    
    Returns:
        list: List of dictionaries with path parameters and file paths
    """
    discovered_datasets = []
    
    # Use glob to find all matching files recursively
    search_pattern = os.path.join(base_path, '**', target_file)
    matching_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(matching_files)} {target_file} files")
    
    for file_path in matching_files:
        # Extract path components
        # Remove base_path and target_file from the path
        relative_path = os.path.relpath(file_path, base_path)
        path_parts = relative_path.split(os.sep)
        
        # The structure should be: person/speed/run_id/DBFiles/filename
        if len(path_parts) >= 4 and path_parts[-2] == 'DBFiles':
            person = path_parts[0]
            speed = path_parts[1]
            run_id = path_parts[2]
            
            dataset_info = {
                'person': person,
                'speed': speed,
                'run_id': run_id,
                'file_path': file_path
            }
            
            discovered_datasets.append(dataset_info)
            print(f"  - {person}_{speed}_{run_id}: {file_path}")
        else:
            print(f"  - Skipping {file_path} (unexpected path structure)")
    
    return discovered_datasets


def save_discovered_datasets_to_csv(datasets, output_dir='./outputs', filename='discovered_datasets.csv'):
    """
    Save discovered datasets to a CSV file for later use.
    
    Args:
        datasets (list): List of discovered dataset dictionaries
        output_dir (str): Base output directory to save the CSV in outputs subdir
        filename (str): Name of the CSV file
    """

    # Create outputs directory
    exports_dir = os.path.join(output_dir, 'exports')
    os.makedirs(exports_dir, exist_ok=True)
    output_file = os.path.join(exports_dir, filename)
    
    # Create DataFrame
    df_data = []
    for dataset in datasets:
        df_data.append({
            'person': dataset['person'],
            'speed': dataset['speed'],
            'runid': dataset['run_id'],  # Note: using 'runid' to match existing format
            'file_path': dataset['file_path']
        })
    
    df = pd.DataFrame(df_data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved {len(datasets)} discovered datasets to {output_file}")
    
    return output_file

def generate_suffixes(dataset_info=None, interpolation_factor=1, data_source='', 
                     use_sliding_grid=False, trajectory_aware=False):
    """
    Generate standardized suffixes for file naming and plot titles.
    
    Args:
        dataset_info (str): Dataset identifier string (e.g., "P036_05-0mph_1339")
        interpolation_factor (int): Interpolation factor used
        data_source (str): Data source identifier
        use_sliding_grid (bool): Whether sliding grid method was used
        trajectory_aware (bool): Whether trajectory-aware counting was used
    
    Returns:
        string: string containing the standardized suffix combination
    """

    dataset_info_suffix = f"_{dataset_info}" if dataset_info else ""
    data_source_suffix = f"_{data_source}" if data_source else ""
    interpolation_factor_suffix = f"_{interpolation_factor}x" if interpolation_factor >= 1 else ""
    use_sliding_grid_suffix = "_sliding_grid" if use_sliding_grid else ""
    trajectory_aware_suffix = "_trajectory_aware" if trajectory_aware else ""

    suffixes = f"{dataset_info_suffix}{data_source_suffix}{interpolation_factor_suffix}{use_sliding_grid_suffix}{trajectory_aware_suffix}"
    return suffixes