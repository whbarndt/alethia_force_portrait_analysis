#!/usr/bin/env python3
"""
Plot D vs Velocity for Person Datasets

This script creates plots showing fractal dimension (D) vs velocity for individual persons
or batch processing of multiple persons. It reads CSV files containing fit results and
creates visualization plots.

Usage:
    python plot_person_d_vs_velocity.py --config config/plot_config.yaml
    python plot_person_d_vs_velocity.py --config config/plot_config.yaml --batch
"""

import os
import sys
import yaml
import argparse
import glob
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

# Import with proper module handling
from src.visualization import plot_d_vs_velocity

def setup_logging(config):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # Create logger
    logger = logging.getLogger('d_vs_velocity_plotter')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if log_config.get('include_console', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_config.get('include_file', True):
        log_dir = os.path.join(config.get('output', {}).get('base_output_directory', '.'), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"d_vs_velocity_plotting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)


def find_csv_files(output_dir, person=None, method_filter=None):
    """
    Find CSV files containing fit results
    
    Args:
        output_dir: Base output directory to search
        person: Optional person ID to filter (e.g., 'P036')
        method_filter: Optional method filter (e.g., 'sliding_grid', 'fixed_grid')
    
    Returns:
        List of CSV file paths
    """
    # Search patterns for different locations
    search_patterns = [
        os.path.join(output_dir, 'fits', '*.csv'),
        os.path.join(output_dir, 'batch_*', '*', 'fits', '*.csv'),
    ]
    
    csv_files = []
    for pattern in search_patterns:
        csv_files.extend(glob.glob(pattern))
    
    # Filter by person if specified
    if person:
        csv_files = [f for f in csv_files if person in os.path.basename(f)]
    
    # Filter by method if specified
    if method_filter:
        csv_files = [f for f in csv_files if method_filter in os.path.basename(f)]
    
    return sorted(csv_files)


def extract_person_data_from_csv(csv_file, d_selection='best', plot_top_n_fits=1, min_order_of_magnitude=1.0):
    """
    Extract person data from a CSV file - returns top N D values per speed
    
    Args:
        csv_file: Path to CSV file
        d_selection: Which D value to use ('best', 'highest_r2', 'median', 'mean')
        plot_top_n_fits: Number of top R² fits to return (default: 1 for best fit only)
        min_order_of_magnitude: Minimum order of magnitude span (default: 1.0 = 10x range)
    
    Returns:
        Dictionary with person, speed, d_value(s), r2_value(s), filtered_df
    """
    try:
        df = pd.read_csv(csv_file)
        
        # Filter by order of magnitude span (max_div/min_div >= 10^min_order_of_magnitude)
        min_span = 10 ** min_order_of_magnitude
        df_filtered = df[df['max_div'] / df['min_div'] >= min_span].copy()
        
        if df_filtered.empty:
            print(f"No fits span {min_span}x range in {csv_file}")
            return None
        
        print(f"Filtered {len(df)} fits to {len(df_filtered)} fits spanning ≥{min_span}x range")
        
        # Extract person info from filename
        filename = os.path.basename(csv_file)
        parts = filename.split('_')
        
        # Look for person ID pattern (P###)
        person = 'Unknown'
        speed = 'Unknown'
        
        for part in parts:
            if part.startswith('P') and len(part) >= 4 and part[1:].isdigit():
                person = part
                break
        
        # Look for speed pattern (##-0mph)
        for part in parts:
            if '-0mph' in part:
                speed = part
                break
        
        person_data = {
            'person': person,
            'speed': speed,
            'csv_file': csv_file,
            'filtered_df': df_filtered
        }

        if plot_top_n_fits > 1:
            # Get top N R² values from filtered data
            top_n_idx = df_filtered['R2'].nlargest(plot_top_n_fits).index
            d_values = df_filtered.loc[top_n_idx, 'D'].tolist()
            r2_values = df_filtered.loc[top_n_idx, 'R2'].tolist()
            
            person_data['d_values'] = d_values
            person_data['r2_values'] = r2_values

            return person_data

        else:
            # Select the BEST D value (highest R²) from filtered data
            best_idx = df_filtered['R2'].idxmax()
            d_value = df_filtered.loc[best_idx, 'D']
            r2_value = df_filtered.loc[best_idx, 'R2']
            
            person_data['d_value'] = d_value
            person_data['r2_value'] = r2_value

            return person_data
    
    except Exception as e:
        print(f"Error processing CSV file {csv_file}: {e}")
        exit(1)

def export_filtered_csv(data_list, output_dir, logger, min_order_of_magnitude=1.0):
    """
    Export filtered data to CSV files with '_filtered' after timestamp and filtering value
    
    Args:
        data_list: List of person data dictionaries containing filtered_df
        output_dir: Output directory for CSV files
        logger: Logger instance
        min_order_of_magnitude: Minimum order of magnitude span for filename
    """
    logger.info("Exporting filtered data to CSV files...")
    
    # Create filtering value string for filename
    filter_value_str = f"om{min_order_of_magnitude:.1f}".replace('.', 'p')  # e.g., "om1p0" for 1.0
    
    for data in data_list:
        if data is None or 'filtered_df' not in data:
            continue
        
        # Get original CSV filename
        original_csv = data['csv_file']
        original_filename = os.path.basename(original_csv)
        
        # Create filtered filename by inserting '_filtered_om1p0'
        # Pattern: original_name.csv -> original_name_filtered_om1p0.csv
        filtered_filename = original_filename.replace('.csv', f'_filtered_{filter_value_str}.csv')
        
        # Create output path
        filtered_csv_path = os.path.join(output_dir, 'fits', 'filtered', filtered_filename)
        os.makedirs(os.path.dirname(filtered_csv_path), exist_ok=True)
        
        # Export filtered data
        data['filtered_df'].to_csv(filtered_csv_path, index=False)
        logger.info(f"Exported filtered data to: {filtered_csv_path}")
    
    logger.info("Filtered CSV export completed")


def group_data_by_person(data_list, plot_top_n_fits=1):
    """
    Group data by person and organize by speed - top N D values per speed
    
    Args:
        data_list: List of person data dictionaries
        plot_top_n_fits: Number of top R² fits to handle (default: 1 for single best)
    
    Returns:
        Dictionary with person IDs as keys and organized data as values
    """
    person_groups = {}
    
    for data in data_list:
        person = data['person']
        speed = data['speed']
        
        if person not in person_groups:
            person_groups[person] = {}
        
        if plot_top_n_fits > 1:
            # For top N R² mode, store all values
            if speed not in person_groups[person]:
                person_groups[person][speed] = {
                    'd_values': data['d_values'],
                    'r2_values': data['r2_values'],
                    'csv_file': data['csv_file']
                }
        else:
            # If this speed doesn't exist yet, or if this R² is better, use this data
            if speed not in person_groups[person] or data['r2_value'] > person_groups[person][speed]['r2_value']:
                person_groups[person][speed] = {
                    'd_value': data['d_value'],
                    'r2_value': data['r2_value'],
                    'csv_file': data['csv_file']
                }
    
    # Convert to the expected format and sort by speed
    result = {}
    for person, speed_data in person_groups.items():
        # Create list of tuples for sorting
        speed_list = []
        for speed, data in speed_data.items():
            if plot_top_n_fits > 1:
                speed_list.append((speed, data['d_values'], data['r2_values'], data['csv_file']))
            else:
                speed_list.append((speed, data['d_value'], data['r2_value'], data['csv_file']))
        
        # Sort by speed (extract numeric part)
        def speed_key(item):
            try:
                return float(item[0].split('-')[0])  # No division by 10
            except:
                return 0.0
        
        speed_list.sort(key=speed_key)

        result[person] = {
            'person': person,
            'speeds': [item[0] for item in speed_list],
            'csv_files': [item[3] for item in speed_list]
        }
        
        # Unpack sorted data
        if plot_top_n_fits > 1: 
            result[person]['d_values_list'] = [item[1] for item in speed_list]
            result[person]['r2_values_list'] = [item[2] for item in speed_list]
        else:
            result[person]['d_values'] = [item[1] for item in speed_list]
            result[person]['r2_values'] = [item[2] for item in speed_list]
    
    return result

def plot_single_person(person_data, config, logger, output_dir):
    """Plot D vs velocity for a single person"""
    logger.info(f"Plotting D vs velocity for person {person_data['person']}")
    
    if len(person_data['speeds']) < 2:
        logger.warning(f"Person {person_data['person']} has only {len(person_data['speeds'])} speed(s). Need at least 2 for meaningful plot.")
        return None
    
    return plot_d_vs_velocity(person_data, config, logger, output_dir)


def plot_batch_persons(person_groups, config, logger, output_dir):
    """Plot D vs velocity for multiple persons"""
    logger.info(f"Batch plotting D vs velocity for {len(person_groups)} persons")
    
    plot_files = []
    for person, person_data in person_groups.items():
        logger.info(f"Processing person {person} with {len(person_data['speeds'])} speeds")
        plot_file = plot_single_person(person_data, config, logger, output_dir)
        if plot_file:
            plot_files.append(plot_file)
    
    return plot_files


def main():
    parser = argparse.ArgumentParser(description='Plot D vs Velocity for Person Datasets')
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    parser.add_argument('--batch', action='store_true', help='Process all persons in batch mode')
    parser.add_argument('--person', help='Specific person ID to plot (e.g., P036)')
    parser.add_argument('--method', help='Method filter (sliding_grid, fixed_grid)')
    parser.add_argument('--d-selection', default='best', 
                       choices=['best', 'highest_r2', 'median', 'mean'],
                       help='Which D value to use from each CSV file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("=== D vs Velocity Plotting Started ===")
    
    # Get output directory
    output_dir = config.get('output', {}).get('base_output_directory', './output')
    logger.info(f"Output directory: {output_dir}")
    
    # Find CSV files
    csv_files = find_csv_files(output_dir, person=args.person, method_filter=args.method)
    logger.info(f"Found {len(csv_files)} CSV files")
    
    if not csv_files:
        logger.error("No CSV files found. Check output directory and filters.")
        sys.exit(1)
    
    # Get plotting configuration
    plot_top_n_fits = config.get('plotting', {}).get('plot_top_n_fits', 1)
    min_order_of_magnitude = config.get('plotting', {}).get('min_order_of_magnitude', 1.0)
    logger.info(f"Plot top {plot_top_n_fits} R² fits per velocity")
    logger.info(f"Filter fits to span ≥{10**min_order_of_magnitude}x range (order of magnitude ≥{min_order_of_magnitude})")
    
    # Extract data from CSV files
    logger.info("Extracting data from CSV files...")
    data_list = []
    for csv_file in csv_files:
        logger.info(f"Processing: {csv_file}")
        data = extract_person_data_from_csv(csv_file, args.d_selection, plot_top_n_fits, min_order_of_magnitude)
        if data:
            data_list.append(data)
    
    if not data_list:
        logger.error("No valid data extracted from CSV files.")
        sys.exit(1)
    
    # Export filtered CSV files
    export_filtered_csv(data_list, output_dir, logger, min_order_of_magnitude)
    
    # Group data by person
    person_groups = group_data_by_person(data_list, plot_top_n_fits)
    logger.info(f"Grouped data for {len(person_groups)} persons: {list(person_groups.keys())}")
    
    # Create plots
    if args.batch:
        # Batch mode: plot all persons
        plot_files = plot_batch_persons(person_groups, config, logger, output_dir)
        logger.info(f"Created {len(plot_files)} plots in batch mode")
    else:
        # Single mode: plot specific person or first person found
        if args.person and args.person in person_groups:
            person_data = person_groups[args.person]
            plot_file = plot_single_person(person_data, config, logger, output_dir)
            if plot_file:
                logger.info(f"Created plot for person {args.person}: {plot_file}")
        else:
            # Plot first person found
            if person_groups:
                first_person = list(person_groups.keys())[0]
                person_data = person_groups[first_person]
                plot_file = plot_single_person(person_data, config, logger, output_dir)
                if plot_file:
                    logger.info(f"Created plot for person {first_person}: {plot_file}")
            else:
                logger.error("No person data available for plotting")
                sys.exit(1)
    
    logger.info("=== D vs Velocity Plotting Completed ===")


if __name__ == "__main__":
    main()
