"""
Utility functions for Aletheia Fractal Analysis

This module contains common utility functions used across the analysis pipeline.
"""

import os
import yaml
import logging
import pandas as pd
import glob
from datetime import datetime


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
    
    handlers = []
    if config['logging']['include_file']:
        handlers.append(logging.FileHandler(log_filename))
    if config['logging']['include_console']:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
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
        try:
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
                    'file_path': file_path,
                    'relative_path': relative_path
                }
                
                discovered_datasets.append(dataset_info)
                print(f"  - {person}_{speed}_{run_id}: {relative_path}")
            else:
                print(f"  - Skipping {file_path} (unexpected path structure)")
                
        except Exception as e:
            print(f"  - Error processing {file_path}: {str(e)}")
    
    return discovered_datasets


def save_discovered_datasets_to_csv(datasets, output_file='discovered_datasets.csv'):
    """
    Save discovered datasets to a CSV file for later use.
    
    Args:
        datasets (list): List of discovered dataset dictionaries
        output_file (str): Output CSV filename
    """
    if not datasets:
        print("No datasets to save")
        return
    
    # Create DataFrame
    df_data = []
    for dataset in datasets:
        df_data.append({
            'person': dataset['person'],
            'speed': dataset['speed'],
            'runid': dataset['run_id'],  # Note: using 'runid' to match existing format
            'file_path': dataset['file_path'],
            'relative_path': dataset['relative_path']
        })
    
    df = pd.DataFrame(df_data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved {len(datasets)} discovered datasets to {output_file}")
    
    return output_file
