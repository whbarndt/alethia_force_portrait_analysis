"""
Visualization module for Aletheia Fractal Analysis

This module handles all plotting and visualization functionality.
"""

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os
from datetime import datetime
from .utils import generate_suffixes

# Try to import plotly for interactive 3D plots
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

def create_visualization(analysis_results, config, logger, plots_dir=None, run_timestamp=None):
    """Create and save visualization plots"""
    logger.info("=== VISUALIZATION PHASE ===")
    
    # Extract analysis results from dictionary
    points_new = analysis_results['points_new']
    best_fits = analysis_results['best_fits']
    log_divs = analysis_results['log_divs']
    log_counts = analysis_results['log_counts']
    divs = analysis_results['divs']
    counts = analysis_results['counts']
    method_used = analysis_results['method_used']
    dens_norm = analysis_results['dens_norm']
    x_new = analysis_results['x_new']
    y_new = analysis_results['y_new']
    z_new = analysis_results['z_new']
    interpolation_factor = analysis_results.get('interpolation_factor', 1)
    data_source = analysis_results.get('data_source', '')
    
    # For backward compatibility, extract best fit info
    if best_fits:
        best_fit = best_fits[0]  # First fit is typically the best
        best_D = best_fit['D']
        best_R2 = best_fit['R2']
        best_intercept = best_fit['intercept']
        best_range = (best_fit['start_idx'], best_fit['end_idx'])
    else:
        logger.warning("No best fits available")
        return []
    
    # Calculate full range fit for comparison
    full_result = linregress(log_divs, log_counts)
    full_D = full_result.slope
    full_R2 = full_result.rvalue ** 2
    full_intercept = full_result.intercept
    
    custom_start = analysis_results.get('custom_start')
    custom_end = analysis_results.get('custom_end')
    
    # Get dataset information for titles and filenames
    try:
        single_config = config.get('single', {})
        person = single_config['person']
        speed = single_config['speed']
        run_id = single_config['run_id']
    except Exception as e:
        logger.error(f"Missing dataset information in configuration! Error:{e}")
        exit(1)
    dataset_info = f"{person}_{speed}_{run_id}"
    
    # Use provided plots_dir if available, otherwise use base_output_directory from config with shared timestamp
    if plots_dir is None:
        base_output_dir = config.get('output', {}).get('base_output_directory', '.')
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plots_dir = os.path.join(base_output_dir, 'plots', run_timestamp)
        os.makedirs(plots_dir, exist_ok=True)
    
    # Ensure we have a timestamp for filenames
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    fig_size = tuple(config['visualization']['figure_size'])
    save_format = config['visualization']['save_format']
    dpi = config['visualization']['dpi']
    
    # Get analysis method flags
    use_sliding_grid = method_used == "sliding_grid"
    trajectory_aware = analysis_results.get('trajectory_aware', False)
    
    # Generate standardized suffixes
    suffixes = generate_suffixes(
        dataset_info=dataset_info,
        interpolation_factor=interpolation_factor,
        data_source=data_source,
        use_sliding_grid=use_sliding_grid,
        trajectory_aware=trajectory_aware
    )
    
    # Generate plot title text components
    resolution_text = f" ({interpolation_factor}x resolution)" if interpolation_factor > 1 else ""
    sliding_grid_text = " (Sliding Grid)" if use_sliding_grid else ""
    trajectory_text = " (Trajectory-Aware)" if trajectory_aware else ""
    
    def _plot_box_counting(ax, plot_divs, plot_counts, fit_D, fit_intercept, fit_R2, title_suffix, highlight_range=None, annotate=True):
        ax.set_facecolor('black')
        
        # Plot the actual values (not log-transformed)
        ax.plot(plot_divs, plot_counts, 'o', color='orange', markersize=8, label='Data')
        
        # Add the fitted line (convert back from log space)
        x_fit = np.array([plot_divs.min(), plot_divs.max()])
        y_fit = np.exp(fit_intercept) * (x_fit ** fit_D)  # Convert from log space: y = e^intercept * x^D
        ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Power Law Fit: D={fit_D:.4f}, R²={fit_R2:.4f}')
        
        # Highlight range if provided
        if highlight_range:
            start, end = highlight_range
            ax.plot(plot_divs[start:end+1], plot_counts[start:end+1], 'o-', color='yellow', linewidth=3, markersize=10,
                    label=f'Highlighted Range')
        
        # Set logarithmic scales on both axes
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlabel('Box Divisions', color='white', fontsize=12)
        ax.set_ylabel('Box Counts', color='white', fontsize=12)
        ax.tick_params(colors='white', labelsize=10)
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=11)
        ax.set_title(f'Fractal Box Counting {title_suffix}: {dataset_info}{resolution_text}{sliding_grid_text}{trajectory_text}', 
                     color='white', fontsize=14)
        
        # Add annotations for each point
        if annotate:
            for i in range(len(plot_divs)):
                ax.annotate(f'({plot_divs[i]}, {plot_counts[i]})', (plot_divs[i], plot_counts[i]), 
                            textcoords="offset points", xytext=(0,10), ha='center', color='white', fontsize=8)
    
    # Plot 1: Full range
    fig_full = plt.figure(figsize=fig_size, facecolor='black')
    ax_full = fig_full.add_subplot(111)
    _plot_box_counting(ax_full, divs, counts, full_D, full_intercept, full_R2, "Full Range")
    full_plot_filename = f"{plots_dir}/full_range_fractal_analysis{suffixes}_{run_timestamp}.{save_format}"
    plt.tight_layout()
    fig_full.savefig(full_plot_filename, dpi=dpi, bbox_inches='tight', facecolor='black')
    logger.info(f"  - Saved full range plot to: {full_plot_filename}")
    plt.close(fig_full)
    
    # Plot 2: Best range (plot only the best range data with its fit)
    if best_range:
        start, end = best_range
        plot_divs_best = divs[start:end+1]
        plot_counts_best = counts[start:end+1]
        fig_best = plt.figure(figsize=fig_size, facecolor='black')
        ax_best = fig_best.add_subplot(111)
        _plot_box_counting(ax_best, plot_divs_best, plot_counts_best, best_D, best_intercept, best_R2, "Best Range")
        best_plot_filename = f"{plots_dir}/best_range_fractal_analysis{suffixes}_{run_timestamp}.{save_format}"
        plt.tight_layout()
        fig_best.savefig(best_plot_filename, dpi=dpi, bbox_inches='tight', facecolor='black')
        logger.info(f"  - Saved best range plot to: {best_plot_filename}")
        plt.close(fig_best)
    else:
        logger.warning("No best range available for plotting.")
    
    # Plot 3: Custom range (if provided)
    if custom_start is not None and custom_end is not None:
        if custom_start < 0 or custom_end >= len(divs) or custom_start > custom_end:
            logger.warning("Invalid custom range provided; skipping custom plot.")
        else:
            custom_result = linregress(log_divs[custom_start:custom_end+1], log_counts[custom_start:custom_end+1])
            custom_D = custom_result.slope
            custom_R2 = custom_result.rvalue ** 2
            custom_intercept = custom_result.intercept
            logger.info(f"  - Custom range fit: D={custom_D:.4f}, R²={custom_R2:.4f}, range=({custom_start}, {custom_end})")
            
            plot_divs_custom = divs[custom_start:custom_end+1]
            plot_counts_custom = counts[custom_start:custom_end+1]
            fig_custom = plt.figure(figsize=fig_size, facecolor='black')
            ax_custom = fig_custom.add_subplot(111)
            _plot_box_counting(ax_custom, plot_divs_custom, plot_counts_custom, custom_D, custom_intercept, custom_R2, f"Custom Range ({custom_start}-{custom_end})")
            custom_plot_filename = f"{plots_dir}/custom_range_fractal_analysis{suffixes}_{run_timestamp}.{save_format}"
            plt.tight_layout()
            fig_custom.savefig(custom_plot_filename, dpi=dpi, bbox_inches='tight', facecolor='black')
            logger.info(f"  - Saved custom range plot to: {custom_plot_filename}")
            plt.close(fig_custom)
    else:
        logger.info("No custom range provided; skipping custom plot.")
    
    # Plot 4: Top 6 fits comparison (top 2 R² + top 2 span + top 2 ~1.5 span)
    if len(best_fits) >= 2:
        # Sort fits by R² and span
        fits_by_r2 = sorted(best_fits, key=lambda x: x['R2'], reverse=True)
        fits_by_span = sorted(best_fits, key=lambda x: x.get('span_orders', 0), reverse=True)
        
        # Filter fits with span close to 1.5 orders of magnitude
        target_span = 1.5
        tolerance = 0.3  # Allow fits within ±0.3 span of target
        fits_near_1_5_span = [fit for fit in best_fits 
                              if abs(fit.get('span_orders', 0) - target_span) <= tolerance]
        
        # Sort 1.5 span fits by R²
        fits_for_1_5_span = sorted(fits_near_1_5_span, key=lambda x: x['R2'], reverse=True)
        
        logger.info(f"  - Found {len(fits_near_1_5_span)} fits with span ~1.5 orders of magnitude (±{tolerance})")
        
        # Get top 2 from each category, avoiding duplicates
        top_fits = []
        seen_fits = set()
        
        # Add top 2 R² fits
        for fit in fits_by_r2[:2]:
            fit_id = (fit['start_idx'], fit['end_idx'], fit['D'])
            if fit_id not in seen_fits:
                top_fits.append(('R²', fit))
                seen_fits.add(fit_id)
        
        # Add top 2 span fits
        for fit in fits_by_span[:2]:
            fit_id = (fit['start_idx'], fit['end_idx'], fit['D'])
            if fit_id not in seen_fits:
                top_fits.append(('Span', fit))
                seen_fits.add(fit_id)
        
        # Add top 2 fits near 1.5 span
        for fit in fits_for_1_5_span[:2]:
            fit_id = (fit['start_idx'], fit['end_idx'], fit['D'])
            if fit_id not in seen_fits:
                top_fits.append(('~1.5 Span', fit))
                seen_fits.add(fit_id)
        
        if len(top_fits) >= 2:
            fig_comparison = plt.figure(figsize=fig_size, facecolor='black')
            ax_comparison = fig_comparison.add_subplot(111)
            ax_comparison.set_facecolor('black')
            
            # Plot the actual data points
            ax_comparison.plot(divs, counts, 'o', color='orange', markersize=8, label='Data', zorder=5)
            
            # Define colors for different fits
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
            linestyles = ['-', '--', '-.', ':', '-', '--']
            
            # Plot each top fit
            for i, (fit_type, fit) in enumerate(top_fits[:6]):  # Limit to 6 fits
                start_idx, end_idx = fit['start_idx'], fit['end_idx']
                fit_divs = divs[start_idx:end_idx+1]
                fit_counts = counts[start_idx:end_idx+1]
                
                # Plot the fitted line
                x_fit = np.array([fit_divs.min(), fit_divs.max()])
                y_fit = np.exp(fit['intercept']) * (x_fit ** fit['D'])
                
                color = colors[i % len(colors)]
                linestyle = linestyles[i % len(linestyles)]
                
                ax_comparison.plot(x_fit, y_fit, color=color, linewidth=2, linestyle=linestyle,
                                 label=f'{fit_type} #{i+1}: D={fit["D"]:.4f}, R²={fit["R2"]:.4f}, span={fit.get("span_orders", 0):.2f}')
                
                # Highlight the data points used in this fit
                ax_comparison.plot(fit_divs, fit_counts, 'o', color=color, markersize=6, alpha=0.7, zorder=4)
            
            # Set logarithmic scales
            ax_comparison.set_xscale('log')
            ax_comparison.set_yscale('log')
            
            ax_comparison.set_xlabel('Box Divisions', color='white', fontsize=12)
            ax_comparison.set_ylabel('Box Counts', color='white', fontsize=12)
            ax_comparison.tick_params(colors='white', labelsize=10)
            ax_comparison.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=10)
            ax_comparison.set_title(f'Top Fits Comparison: {dataset_info}{resolution_text}{sliding_grid_text}{trajectory_text}', 
                                  color='white', fontsize=14)
            
            comparison_plot_filename = f"{plots_dir}/top_fits_comparison{suffixes}_{run_timestamp}.{save_format}"
            plt.tight_layout()
            fig_comparison.savefig(comparison_plot_filename, dpi=dpi, bbox_inches='tight', facecolor='black')
            logger.info(f"  - Saved top fits comparison plot to: {comparison_plot_filename}")
            plt.close(fig_comparison)
        else:
            logger.warning("Not enough unique fits for comparison plot")
    else:
        logger.warning("Not enough best fits for comparison plot")
    
    # Original combined visualization (optional, keeping for compatibility)
    if config['visualization'].get('create_combined_plot', True):
        fig = plt.figure(figsize=fig_size, facecolor='black')
        logger.info(f"  - Created figure: {fig.get_size_inches()} inches")
        
        if config['visualization']['plot_3d_trajectory']:
            # 3D Trajectory Plot
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.set_facecolor('black')
            logger.info("  - Created 3D trajectory subplot")
            
            # Get 3D plot settings from config
            scatter_alpha = config['visualization'].get('scatter_alpha', 0.1)
            scatter_size = config['visualization'].get('scatter_size', 0.05)
            line_width = config['visualization'].get('line_width', 0.2)
            colormap = config['visualization'].get('colormap', 'plasma')
            
            # Create 3D scatter plot with density coloring (if enabled)
            if config['visualization'].get('density_coloring', True):
                scatter = ax1.scatter(x_new, y_new, z_new, c=dens_norm, cmap=colormap, 
                                     s=scatter_size, alpha=scatter_alpha)
                logger.info(f"  - Added scatter plot: {len(x_new)} points with density coloring")
            else:
                scatter = ax1.scatter(x_new, y_new, z_new, c='blue', 
                                     s=scatter_size, alpha=scatter_alpha)
                logger.info(f"  - Added scatter plot: {len(x_new)} points with uniform coloring")
            
            # Add trajectory line
            ax1.plot(x_new, y_new, z_new, color='orange', linewidth=line_width)
            logger.info("  - Added trajectory line plot")
            
            # Configure 3D plot styling with dataset info
            ax1.set_axis_off()
            ax1.grid(False)
            if config['visualization'].get('density_coloring', True):
                ax1.set_title(f'3D Trajectory: {dataset_info}{resolution_text}{sliding_grid_text}{trajectory_text}\nDensity Coloring', color='white', fontsize=12)
            else:
                ax1.set_title(f'3D Trajectory: {dataset_info}{resolution_text}{sliding_grid_text}{trajectory_text}\nUniform Coloring', color='white', fontsize=12)
            
            # Add colorbar for density reference (only if density coloring is enabled)
            if config['visualization'].get('density_coloring', True):
                cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8, aspect=20)
                cbar.set_label('Normalized Density', color='white', fontsize=10)
                cbar.ax.tick_params(colors='white')
                logger.info("  - Configured 3D plot styling with colorbar")
            else:
                logger.info("  - Configured 3D plot styling without colorbar")
            
            # Box Counting Plot (subplot)
            ax = fig.add_subplot(122)
        else:
            # Only Box Counting Plot
            ax = fig.add_subplot(111)
        
        _plot_box_counting(ax, divs, counts, best_D, best_intercept, best_R2, "Combined", best_range)
        
        plt.tight_layout()
        logger.info("  - Created box counting scaling plot")
        
        # Save the plot with dataset info in filename
        combined_plot_filename = f"{plots_dir}/combined_fractal_analysis{suffixes}_{run_timestamp}.{save_format}"
        plt.savefig(combined_plot_filename, dpi=dpi, bbox_inches='tight', facecolor='black')
        logger.info(f"  - Saved combined plot to: {combined_plot_filename}")
        plt.close(fig)
    
    # Collect all plot filenames
    plot_filenames = [full_plot_filename, best_plot_filename]
    if 'custom_plot_filename' in locals():
        plot_filenames.append(custom_plot_filename)
    if 'comparison_plot_filename' in locals():
        plot_filenames.append(comparison_plot_filename)
    
    return plot_filenames


def create_interactive_3d_plot(analysis_results, config, logger, output_dir, run_timestamp=None):
    """Create an interactive 3D plot using plotly"""
    if not PLOTLY_AVAILABLE:
        logger.warning("  - Plotly not available. Install with: pip install plotly")
        return None
    
    logger.info("  - Creating interactive 3D plot with plotly")
    
    # Extract data from analysis_results
    points_new = analysis_results['points_new']
    dens_norm = analysis_results['dens_norm']
    x_new = analysis_results['x_new']
    y_new = analysis_results['y_new']
    z_new = analysis_results['z_new']
    interpolation_factor = analysis_results.get('interpolation_factor', 1)
    data_source = analysis_results.get('data_source', '')
    
    # Get dataset information for title and filename
    try:
        single_config = config.get('single', {})
        person = single_config['person']
        speed = single_config['speed']
        run_id = single_config['run_id']
    except Exception as e:        
        logger.error(f"Missing dataset information in configuration! Error:{e}")
        exit(1)
    dataset_info = f"{person}_{speed}_{run_id}"
    
    # Generate standardized suffixes (only data-related parameters matter for interactive plots)
    suffixes = generate_suffixes(
        dataset_info=dataset_info,
        interpolation_factor=interpolation_factor,
        data_source=data_source,
        use_sliding_grid=False,  # Not relevant for data visualization
        trajectory_aware=False    # Not relevant for data visualization
    )
    resolution_text = f" ({interpolation_factor}x resolution)" if interpolation_factor > 1 else ""
    data_source_text = f" ({data_source})" if data_source else ""
    
    # Create interactive 3D scatter plot
    fig = go.Figure()
    
    # Add scatter plot with density coloring
    scatter = go.Scatter3d(
        x=x_new,
        y=y_new,
        z=z_new,
        mode='markers',
        marker=dict(
            size=config['visualization'].get('scatter_size', 0.05) * 100,  # Scale up for plotly
            color=dens_norm,
            colorscale=config['visualization'].get('colormap', 'plasma'),
            opacity=config['visualization'].get('scatter_alpha', 0.1),
            colorbar=dict(title="Normalized Density")
        ),
        name='Trajectory Points'
    )
    
    # Add trajectory line
    line = go.Scatter3d(
        x=x_new,
        y=y_new,
        z=z_new,
        mode='lines',
        line=dict(
            color='orange',
            width=config['visualization'].get('line_width', 0.2) * 10  # Scale up for plotly
        ),
        name='Trajectory Path'
    )
    
    fig.add_trace(scatter)
    fig.add_trace(line)
    
    # Update layout with dataset info
    fig.update_layout(
        title=f'Interactive 3D Trajectory: {dataset_info}{resolution_text}{data_source_text}<br>Density Coloring',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600,
        showlegend=True
    )
    
    # Save as HTML file with dataset info in filename in dedicated interactive_3d_plots directory
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    interactive_plots_dir = os.path.join(output_dir, 'plots', 'interactive_3d_plots')
    os.makedirs(interactive_plots_dir, exist_ok=True)
    html_filename = os.path.join(interactive_plots_dir, f"interactive_3d_plot{suffixes}.html")
    fig.write_html(html_filename)
    logger.info(f"  - Interactive 3D plot saved to: {html_filename}")
    
    return html_filename


def visualize_individual_boxes(analysis_results, config, logger, plots_dir=None, run_timestamp=None):
    """
    Visualize the contents of individual boxes at a specific division depth.
    
    This function creates 3D plots showing the actual data points contained within
    specific boxes, allowing you to see the granularity and structure of the data
    at different box sizes. This helps understand whether boxes contain individual
    point clouds, lines, or clusters of points.
    
    Args:
        analysis_results: Dictionary containing analysis results
        config: Configuration dictionary
        logger: Logger instance
        plots_dir: Directory to save plots (optional)
        run_timestamp: Timestamp for filename (optional)
    
    Returns:
        list: Paths to saved plot files
    """
    logger.info("=== CREATING INDIVIDUAL BOX CONTENTS VISUALIZATION ===")
    
    # Extract data from analysis_results
    points_new = analysis_results['points_new']
    divs = analysis_results['divs']
    counts = analysis_results['counts']
    dens_norm = analysis_results['dens_norm']
    x_new = analysis_results['x_new']
    y_new = analysis_results['y_new']
    z_new = analysis_results['z_new']
    interpolation_factor = analysis_results.get('interpolation_factor', 1)
    data_source = analysis_results.get('data_source', '')
    
    # Get dataset information for titles and filenames
    try:
        single_config = config.get('single', {})
        person = single_config['person']
        speed = single_config['speed']
        run_id = single_config['run_id']
    except Exception as e:        
        logger.error(f"Missing dataset information in configuration! Error:{e}")
        return []
    dataset_info = f"{person}_{speed}_{run_id}"
    
    # Use provided plots_dir if available, otherwise use base_output_directory from config
    if plots_dir is None:
        base_output_dir = config.get('output', {}).get('base_output_directory', '.')
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plots_dir = os.path.join(base_output_dir, 'plots', run_timestamp)
        os.makedirs(plots_dir, exist_ok=True)
    
    # Ensure we have a timestamp for filenames
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get visualization settings
    fig_size = tuple(config['visualization']['figure_size'])
    save_format = config['visualization']['save_format']
    dpi = config['visualization']['dpi']
    
    # Generate standardized suffixes
    suffixes = generate_suffixes(
        dataset_info=dataset_info,
        interpolation_factor=interpolation_factor,
        data_source=data_source,
        use_sliding_grid=False,  # Not relevant for data visualization
        trajectory_aware=False    # Not relevant for data visualization
    )
    
    resolution_text = f" ({interpolation_factor}x resolution)" if interpolation_factor > 1 else ""
    data_source_text = f" ({data_source})" if data_source else ""
    
    # Get configuration for individual box visualization
    box_config = config.get('visualization', {}).get('individual_box_visualization', {})
    target_div = box_config.get('target_div', None)
    max_boxes_to_show = box_config.get('max_boxes_to_show', 4)
    box_selection_method = box_config.get('box_selection_method', 'most_points')  # 'most_points', 'random', 'spread'
    
    # If no target div specified, use a middle-range div
    if target_div is None:
        target_div = divs[len(divs)//2]  # Use middle div
        logger.info(f"  - No target div specified, using middle div: {target_div}")
    
    logger.info(f"  - Visualizing individual boxes at div={target_div}")
    
    # Normalize points to [0,1] cube (same as in fractal analysis)
    points_3d = points_new[:, :3] if points_new.shape[1] > 3 else points_new
    mins = points_3d.min(axis=0)
    maxs = points_3d.max(axis=0)
    scaled_points = (points_3d - mins) / (maxs - mins + 1e-10)
    
    # Calculate which boxes are occupied at this div depth
    binned = np.floor(scaled_points * target_div).astype(int)
    
    # Group points by box
    box_contents = {}
    for i, box_coord in enumerate(binned):
        box_tuple = tuple(box_coord)
        if box_tuple not in box_contents:
            box_contents[box_tuple] = []
        box_contents[box_tuple].append(i)  # Store point indices
    
    logger.info(f"  - Found {len(box_contents)} occupied boxes at div={target_div}")
    
    # Select boxes to visualize based on method
    if box_selection_method == 'most_points':
        # Select boxes with most points
        sorted_boxes = sorted(box_contents.items(), key=lambda x: len(x[1]), reverse=True)
        selected_boxes = sorted_boxes[:max_boxes_to_show]
        logger.info(f"  - Selected {len(selected_boxes)} boxes with most points")
    elif box_selection_method == 'random':
        # Select random boxes
        import random
        selected_boxes = random.sample(list(box_contents.items()), min(max_boxes_to_show, len(box_contents)))
        logger.info(f"  - Selected {len(selected_boxes)} random boxes")
    elif box_selection_method == 'spread':
        # Select boxes that are spread across the space
        box_coords = list(box_contents.keys())
        if len(box_coords) >= max_boxes_to_show:
            # Simple spread: take boxes from different regions
            step = len(box_coords) // max_boxes_to_show
            selected_boxes = [(box_coords[i*step], box_contents[box_coords[i*step]]) for i in range(max_boxes_to_show)]
        else:
            selected_boxes = list(box_contents.items())
        logger.info(f"  - Selected {len(selected_boxes)} spread boxes")
    else:
        # Default to most points
        sorted_boxes = sorted(box_contents.items(), key=lambda x: len(x[1]), reverse=True)
        selected_boxes = sorted_boxes[:max_boxes_to_show]
        logger.info(f"  - Selected {len(selected_boxes)} boxes with most points (default)")
    
    plot_filenames = []
    
    # Create plots for each selected box
    for i, (box_coord, point_indices) in enumerate(selected_boxes):
        logger.info(f"  - Creating visualization for box {box_coord} with {len(point_indices)} points")
        
        # Create figure with subplots
        fig = plt.figure(figsize=fig_size, facecolor='black')
        
        # Main 3D plot showing all data with highlighted box
        ax_main = fig.add_subplot(121, projection='3d')
        ax_main.set_facecolor('black')
        
        # Plot all points in light gray
        ax_main.scatter(x_new, y_new, z_new, c='lightgray', 
                      s=config['visualization'].get('scatter_size', 0.05) * 50,
                      alpha=0.1, label='All points')
        
        # Highlight points in the selected box
        box_points_x = x_new[point_indices]
        box_points_y = y_new[point_indices]
        box_points_z = z_new[point_indices]
        box_points_density = dens_norm[point_indices]
        
        scatter_box = ax_main.scatter(box_points_x, box_points_y, box_points_z, 
                                   c=box_points_density,
                                   cmap=config['visualization'].get('colormap', 'plasma'),
                                   s=config['visualization'].get('scatter_size', 0.05) * 200,
                                   alpha=0.8, label=f'Box {box_coord}')
        
        # Add trajectory line for box points
        if len(point_indices) > 1:
            # Sort points by their order in the trajectory
            sorted_indices = sorted(point_indices)
            ax_main.plot(x_new[sorted_indices], y_new[sorted_indices], z_new[sorted_indices], 
                        color='orange', linewidth=3, alpha=0.8)
        
        ax_main.set_title(f'All Data with Highlighted Box\nBox {box_coord}: {len(point_indices)} points', 
                         color='white', fontsize=12)
        ax_main.set_xlabel('X', color='white')
        ax_main.set_ylabel('Y', color='white')
        ax_main.set_zlabel('Z', color='white')
        ax_main.legend()
        
        # Detailed view of just the box contents
        ax_detail = fig.add_subplot(122, projection='3d')
        ax_detail.set_facecolor('black')
        
        # Plot only the points in this box
        scatter_detail = ax_detail.scatter(box_points_x, box_points_y, box_points_z, 
                                        c=box_points_density,
                                        cmap=config['visualization'].get('colormap', 'plasma'),
                                        s=config['visualization'].get('scatter_size', 0.05) * 300,
                                        alpha=0.9)
        
        # Add trajectory line for box points
        if len(point_indices) > 1:
            sorted_indices = sorted(point_indices)
            ax_detail.plot(x_new[sorted_indices], y_new[sorted_indices], z_new[sorted_indices], 
                          color='orange', linewidth=4, alpha=0.9)
        
        # Add colorbar
        cbar = plt.colorbar(scatter_detail, ax=ax_detail, shrink=0.8, aspect=20)
        cbar.set_label('Density', color='white', fontsize=10)
        cbar.ax.tick_params(colors='white')
        
        ax_detail.set_title(f'Box Contents Detail\nBox {box_coord}: {len(point_indices)} points', 
                           color='white', fontsize=12)
        ax_detail.set_xlabel('X', color='white')
        ax_detail.set_ylabel('Y', color='white')
        ax_detail.set_zlabel('Z', color='white')
        
        # Add text annotation with box statistics
        box_size = 1.0 / target_div
        box_center_x = (box_coord[0] + 0.5) * box_size * (maxs[0] - mins[0]) + mins[0]
        box_center_y = (box_coord[1] + 0.5) * box_size * (maxs[1] - mins[1]) + mins[1]
        box_center_z = (box_coord[2] + 0.5) * box_size * (maxs[2] - mins[2]) + mins[2]
        
        stats_text = (f'Box: {box_coord}\n'
                    f'Points: {len(point_indices)}\n'
                    f'Box size: {box_size:.3f}\n'
                    f'Center: ({box_center_x:.3f}, {box_center_y:.3f}, {box_center_z:.3f})')
        
        fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, 
                fontsize=10, color='white', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{plots_dir}/individual_box_{box_coord[0]}_{box_coord[1]}_{box_coord[2]}_div_{target_div}{suffixes}_{run_timestamp}.{save_format}"
        fig.savefig(plot_filename, dpi=dpi, bbox_inches='tight', facecolor='black')
        logger.info(f"    - Saved box {box_coord} visualization to: {plot_filename}")
        plot_filenames.append(plot_filename)
        
        plt.close(fig)
    
    # Create a summary plot showing all selected boxes together
    if len(selected_boxes) > 1:
        logger.info("  - Creating summary comparison plot")
        fig_summary = plt.figure(figsize=(fig_size[0] * 1.5, fig_size[1]), facecolor='black')
        
        # Create subplots for each box
        num_cols = min(2, len(selected_boxes))
        num_rows = (len(selected_boxes) + num_cols - 1) // num_cols
        
        for i, (box_coord, point_indices) in enumerate(selected_boxes):
            ax = fig_summary.add_subplot(num_rows, num_cols, i+1, projection='3d')
            ax.set_facecolor('black')
            
            # Plot box points
            box_points_x = x_new[point_indices]
            box_points_y = y_new[point_indices]
            box_points_z = z_new[point_indices]
            box_points_density = dens_norm[point_indices]
            
            ax.scatter(box_points_x, box_points_y, box_points_z, 
                      c=box_points_density, cmap=config['visualization'].get('colormap', 'plasma'),
                      s=50, alpha=0.8)
            
            # Add trajectory line
            if len(point_indices) > 1:
                sorted_indices = sorted(point_indices)
                ax.plot(x_new[sorted_indices], y_new[sorted_indices], z_new[sorted_indices], 
                        color='orange', linewidth=2, alpha=0.8)
            
            ax.set_title(f'Box {box_coord}\n{len(point_indices)} points', color='white', fontsize=10)
            ax.set_xlabel('X', color='white', fontsize=8)
            ax.set_ylabel('Y', color='white', fontsize=8)
            ax.set_zlabel('Z', color='white', fontsize=8)
        
        # Add main title
        fig_summary.suptitle(f'Individual Box Contents at Div={target_div}: {dataset_info}{resolution_text}{data_source_text}', 
                            color='white', fontsize=14)
        
        plt.tight_layout()
        
        # Save summary plot
        summary_filename = f"{plots_dir}/individual_boxes_summary_div_{target_div}{suffixes}_{run_timestamp}.{save_format}"
        fig_summary.savefig(summary_filename, dpi=dpi, bbox_inches='tight', facecolor='black')
        logger.info(f"  - Saved summary plot to: {summary_filename}")
        plot_filenames.append(summary_filename)
        
        plt.close(fig_summary)
    
    logger.info(f"  - Individual box visualization completed: {len(plot_filenames)} plots created")
    return plot_filenames


def visualize_feature_space_at_div_depths(analysis_results, config, logger, plots_dir=None, run_timestamp=None):
    """
    Visualize how the feature space looks at different division depths (div values).
    
    This function creates 3D plots showing the data at specific div depths to understand
    the granularity of the box counting representation. At low div values, boxes are large
    and may contain many points. At high div values, boxes are small and may contain
    individual point clouds or even single points.
    
    Args:
        analysis_results: Dictionary containing analysis results
        config: Configuration dictionary
        logger: Logger instance
        plots_dir: Directory to save plots (optional)
        run_timestamp: Timestamp for filename (optional)
    
    Returns:
        list: Paths to saved plot files
    """
    logger.info("=== CREATING FEATURE SPACE DIV DEPTH VISUALIZATION ===")
    
    # Extract data from analysis_results
    points_new = analysis_results['points_new']
    divs = analysis_results['divs']
    counts = analysis_results['counts']
    dens_norm = analysis_results['dens_norm']
    x_new = analysis_results['x_new']
    y_new = analysis_results['y_new']
    z_new = analysis_results['z_new']
    interpolation_factor = analysis_results.get('interpolation_factor', 1)
    data_source = analysis_results.get('data_source', '')
    
    # Get dataset information for titles and filenames
    try:
        single_config = config.get('single', {})
        person = single_config['person']
        speed = single_config['speed']
        run_id = single_config['run_id']
    except Exception as e:        
        logger.error(f"Missing dataset information in configuration! Error:{e}")
        return []
    dataset_info = f"{person}_{speed}_{run_id}"
    
    # Use provided plots_dir if available, otherwise use base_output_directory from config
    if plots_dir is None:
        base_output_dir = config.get('output', {}).get('base_output_directory', '.')
        if run_timestamp is None:
            run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plots_dir = os.path.join(base_output_dir, 'plots', run_timestamp)
        os.makedirs(plots_dir, exist_ok=True)
    
    # Ensure we have a timestamp for filenames
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get visualization settings
    fig_size = tuple(config['visualization']['figure_size'])
    save_format = config['visualization']['save_format']
    dpi = config['visualization']['dpi']
    
    # Generate standardized suffixes
    suffixes = generate_suffixes(
        dataset_info=dataset_info,
        interpolation_factor=interpolation_factor,
        data_source=data_source,
        use_sliding_grid=False,  # Not relevant for data visualization
        trajectory_aware=False    # Not relevant for data visualization
    )
    
    resolution_text = f" ({interpolation_factor}x resolution)" if interpolation_factor > 1 else ""
    data_source_text = f" ({data_source})" if data_source else ""
    
    # Get configuration for div depth visualization
    div_depth_config = config.get('visualization', {}).get('div_depth_visualization', {})
    selected_divs = div_depth_config.get('selected_divs', [])
    max_plots = div_depth_config.get('max_plots', 6)
    
    # If no specific divs selected, choose representative ones
    if not selected_divs:
        # Select divs that span the range: low, medium-low, medium, medium-high, high
        num_divs = len(divs)
        if num_divs >= 5:
            indices = [0, num_divs//4, num_divs//2, 3*num_divs//4, num_divs-1]
        else:
            indices = list(range(num_divs))
        selected_divs = [divs[i] for i in indices[:max_plots]]
    
    logger.info(f"  - Visualizing feature space at div depths: {selected_divs}")
    
    # Normalize points to [0,1] cube (same as in fractal analysis)
    points_3d = points_new[:, :3] if points_new.shape[1] > 3 else points_new
    mins = points_3d.min(axis=0)
    maxs = points_3d.max(axis=0)
    scaled_points = (points_3d - mins) / (maxs - mins + 1e-10)
    
    plot_filenames = []
    
    # Create plots for each selected div depth
    for i, div in enumerate(selected_divs):
        logger.info(f"  - Creating visualization for div={div}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=fig_size, facecolor='black')
        
        # Main 3D plot showing the data
        ax_main = fig.add_subplot(121, projection='3d')
        ax_main.set_facecolor('black')
        
        # Plot the original data points
        scatter = ax_main.scatter(x_new, y_new, z_new, c=dens_norm, 
                                cmap=config['visualization'].get('colormap', 'plasma'),
                                s=config['visualization'].get('scatter_size', 0.05) * 100,
                                alpha=config['visualization'].get('scatter_alpha', 0.1))
        
        # Add trajectory line
        ax_main.plot(x_new, y_new, z_new, color='orange', 
                    linewidth=config['visualization'].get('line_width', 0.2) * 10)
        
        ax_main.set_title(f'Original Data\nDiv={div}', color='white', fontsize=12)
        ax_main.set_xlabel('X', color='white')
        ax_main.set_ylabel('Y', color='white')
        ax_main.set_zlabel('Z', color='white')
        
        # Box visualization subplot
        ax_boxes = fig.add_subplot(122, projection='3d')
        ax_boxes.set_facecolor('black')
        
        # Calculate which boxes are occupied at this div depth
        binned = np.floor(scaled_points * div).astype(int)
        unique_boxes = set(tuple(b) for b in binned)
        
        logger.info(f"    - Div {div}: {len(unique_boxes)} occupied boxes out of {div**3} total boxes")
        
        # Convert box coordinates back to real coordinates for visualization
        box_size = 1.0 / div
        box_centers = []
        box_colors = []
        
        for box_coord in unique_boxes:
            # Convert box coordinate to center point in [0,1] space
            center_x = (box_coord[0] + 0.5) * box_size
            center_y = (box_coord[1] + 0.5) * box_size
            center_z = (box_coord[2] + 0.5) * box_size
            
            # Convert back to original coordinate space
            orig_x = center_x * (maxs[0] - mins[0]) + mins[0]
            orig_y = center_y * (maxs[1] - mins[1]) + mins[1]
            orig_z = center_z * (maxs[2] - mins[2]) + mins[2]
            
            box_centers.append([orig_x, orig_y, orig_z])
            
            # Count points in this box for coloring
            points_in_box = np.sum((binned == box_coord).all(axis=1))
            box_colors.append(points_in_box)
        
        if box_centers:
            box_centers = np.array(box_centers)
            box_colors = np.array(box_colors)
            
            # Normalize colors for better visualization
            if len(np.unique(box_colors)) > 1:
                box_colors_norm = (box_colors - box_colors.min()) / (box_colors.max() - box_colors.min())
            else:
                box_colors_norm = np.ones_like(box_colors)
            
            # Plot boxes as scatter points (centers)
            scatter_boxes = ax_boxes.scatter(box_centers[:, 0], box_centers[:, 1], box_centers[:, 2],
                                           c=box_colors_norm, cmap='viridis', s=50, alpha=0.8)
            
            # Add colorbar for box point counts
            cbar = plt.colorbar(scatter_boxes, ax=ax_boxes, shrink=0.8, aspect=20)
            cbar.set_label('Points per Box', color='white', fontsize=10)
            cbar.ax.tick_params(colors='white')
        
        ax_boxes.set_title(f'Box Centers (Div={div})\n{len(unique_boxes)} occupied boxes', 
                          color='white', fontsize=12)
        ax_boxes.set_xlabel('X', color='white')
        ax_boxes.set_ylabel('Y', color='white')
        ax_boxes.set_zlabel('Z', color='white')
        
        # Add text annotation with statistics
        stats_text = f'Div: {div}\nBoxes: {len(unique_boxes)}\nTotal possible: {div**3}\nCoverage: {len(unique_boxes)/div**3*100:.1f}%'
        fig.text(0.02, 0.98, stats_text, transform=fig.transFigure, 
                fontsize=10, color='white', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{plots_dir}/feature_space_div_{div}{suffixes}_{run_timestamp}.{save_format}"
        fig.savefig(plot_filename, dpi=dpi, bbox_inches='tight', facecolor='black')
        logger.info(f"    - Saved div {div} visualization to: {plot_filename}")
        plot_filenames.append(plot_filename)
        
        plt.close(fig)
    
    # Create a summary plot showing all div depths together
    logger.info("  - Creating summary comparison plot")
    fig_summary = plt.figure(figsize=(fig_size[0] * 1.5, fig_size[1]), facecolor='black')
    
    # Create subplots for each div depth
    num_cols = min(3, len(selected_divs))
    num_rows = (len(selected_divs) + num_cols - 1) // num_cols
    
    for i, div in enumerate(selected_divs):
        ax = fig_summary.add_subplot(num_rows, num_cols, i+1, projection='3d')
        ax.set_facecolor('black')
        
        # Calculate occupied boxes
        binned = np.floor(scaled_points * div).astype(int)
        unique_boxes = set(tuple(b) for b in binned)
        
        # Convert to real coordinates
        box_size = 1.0 / div
        box_centers = []
        box_colors = []
        
        for box_coord in unique_boxes:
            center_x = (box_coord[0] + 0.5) * box_size
            center_y = (box_coord[1] + 0.5) * box_size
            center_z = (box_coord[2] + 0.5) * box_size
            
            orig_x = center_x * (maxs[0] - mins[0]) + mins[0]
            orig_y = center_y * (maxs[1] - mins[1]) + mins[1]
            orig_z = center_z * (maxs[2] - mins[2]) + mins[2]
            
            box_centers.append([orig_x, orig_y, orig_z])
            points_in_box = np.sum((binned == box_coord).all(axis=1))
            box_colors.append(points_in_box)
        
        if box_centers:
            box_centers = np.array(box_centers)
            box_colors = np.array(box_colors)
            
            if len(np.unique(box_colors)) > 1:
                box_colors_norm = (box_colors - box_colors.min()) / (box_colors.max() - box_colors.min())
            else:
                box_colors_norm = np.ones_like(box_colors)
            
            ax.scatter(box_centers[:, 0], box_centers[:, 1], box_centers[:, 2],
                      c=box_colors_norm, cmap='viridis', s=30, alpha=0.8)
        
        ax.set_title(f'Div={div}\n{len(unique_boxes)} boxes', color='white', fontsize=10)
        ax.set_xlabel('X', color='white', fontsize=8)
        ax.set_ylabel('Y', color='white', fontsize=8)
        ax.set_zlabel('Z', color='white', fontsize=8)
    
    # Add main title
    fig_summary.suptitle(f'Feature Space at Different Division Depths: {dataset_info}{resolution_text}{data_source_text}', 
                        color='white', fontsize=14)
    
    plt.tight_layout()
    
    # Save summary plot
    summary_filename = f"{plots_dir}/feature_space_div_summary{suffixes}_{run_timestamp}.{save_format}"
    fig_summary.savefig(summary_filename, dpi=dpi, bbox_inches='tight', facecolor='black')
    logger.info(f"  - Saved summary plot to: {summary_filename}")
    plot_filenames.append(summary_filename)
    
    plt.close(fig_summary)
    
    logger.info(f"  - Feature space div depth visualization completed: {len(plot_filenames)} plots created")
    return plot_filenames


def plot_d_vs_velocity(person_data, config, logger, output_dir, run_timestamp=None):
    """
    Plot D vs velocity for a person's dataset
    
    Args:
        person_data: Dictionary with keys 'person', 'speeds', 'd_values'/'d_values_list', 'r2_values'/'r2_values_list'
        config: Configuration dictionary
        logger: Logger instance
        output_dir: Output directory for plots
        run_timestamp: Optional timestamp for filename
    
    Returns:
        str: Path to saved plot file
    """
    logger.info(f"Creating D vs velocity plot for person {person_data['person']}")
    
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Extract data
    person = person_data['person']
    speeds = person_data['speeds']
    
    # Check if we have multiple D values per speed (top N R² mode)
    plot_top_n_fits = config.get('plotting', {}).get('plot_top_n_fits', 1)
    
    if plot_top_n_fits > 1 and 'd_values_list' in person_data:
        # Multiple D values per speed
        d_values_list = person_data['d_values_list']
        r2_values_list = person_data.get('r2_values_list', [])
        d_values = [d_list[0] for d_list in d_values_list]  # Use best D value for main line
        r2_values = [r2_list[0] for r2_list in r2_values_list]  # Use best R² value for main line
    else:
        # Single D value per speed
        d_values = person_data['d_values']
        r2_values = person_data.get('r2_values', [])
        d_values_list = None
        r2_values_list = None
    
    # Convert speed strings to numeric values (extract mph values)
    numeric_speeds = []
    for speed in speeds:
        # Extract numeric part from speed string like "05-0mph" -> 5.0
        try:
            speed_num = float(speed.split('-')[0])
            numeric_speeds.append(speed_num)
        except (ValueError, IndexError):
            logger.warning(f"Could not parse speed '{speed}', using 0.0")
            numeric_speeds.append(0.0)
    
    # Create plot
    fig_size = tuple(config.get('visualization', {}).get('figure_size', [10, 6]))
    save_format = config.get('visualization', {}).get('save_format', 'png')
    dpi = config.get('visualization', {}).get('dpi', 300)
    
    fig, ax = plt.subplots(figsize=fig_size, facecolor='black')
    ax.set_facecolor('black')
    
    # Plot D values vs velocity
    ax.plot(numeric_speeds, d_values, 'o-', color='orange', markersize=8, linewidth=2, 
            label=f'Fractal Dimension (D) - Best R²')
    
    # Plot additional D values if in top N R² mode
    if plot_top_n_fits > 1 and d_values_list:
        colors = ['red', 'green', 'blue', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        max_fits = min(plot_top_n_fits, len(d_values_list[0]) if d_values_list else 0)
        for i in range(1, max_fits):
            additional_d_values = [d_list[i] if i < len(d_list) else None for d_list in d_values_list]
            additional_r2_values = [r2_list[i] if i < len(r2_list) else None for r2_list in r2_values_list]
            
            # Filter out None values
            valid_indices = [j for j, val in enumerate(additional_d_values) if val is not None]
            if valid_indices:
                valid_speeds = [numeric_speeds[j] for j in valid_indices]
                valid_d_values = [additional_d_values[j] for j in valid_indices]
                valid_r2_values = [additional_r2_values[j] for j in valid_indices]
                
                color = colors[(i-1) % len(colors)]  # Cycle through colors if N > 3
                ax.plot(valid_speeds, valid_d_values, 'o', color=color, markersize=6, 
                       alpha=0.7, label=f'D - R² #{i+1}')
    
    # Add R² values as secondary y-axis if available
    if r2_values:
        ax2 = ax.twinx()
        ax2.plot(numeric_speeds, r2_values, 's-', color='cyan', markersize=6, linewidth=1.5,
                alpha=0.7, label='R² Values')
        ax2.set_ylabel('R² Values', color='cyan', fontsize=12)
        ax2.tick_params(colors='cyan', labelsize=10)
        ax2.set_ylim(0, 1)
    
    # Configure main plot
    ax.set_xlabel('Velocity (mph)', color='white', fontsize=12)
    ax.set_ylabel('Fractal Dimension (D)', color='white', fontsize=12)
    ax.tick_params(colors='white', labelsize=10)
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=11)
    ax.set_title(f'Fractal Dimension vs Velocity: {person}', color='white', fontsize=14)
    ax.grid(True, alpha=0.3, color='gray')
    
    # Add annotations for each point
    for i, (speed, d_val) in enumerate(zip(numeric_speeds, d_values)):
        ax.annotate(f'D={d_val:.3f}', (speed, d_val), 
                   textcoords="offset points", xytext=(0,10), ha='center', 
                   color='white', fontsize=8)
    
    # Add annotations for additional D values if in top N R² mode
    if plot_top_n_fits > 1 and d_values_list:
        for i in range(1, max_fits):
            additional_d_values = [d_list[i] if i < len(d_list) else None for d_list in d_values_list]
            additional_r2_values = [r2_list[i] if i < len(r2_list) else None for r2_list in r2_values_list]
            
            for j, (speed, d_val, r2_val) in enumerate(zip(numeric_speeds, additional_d_values, additional_r2_values)):
                if d_val is not None:
                    color = colors[(i-1) % len(colors)]  # Cycle through colors if N > 3
                    ax.annotate(f'D={d_val:.3f}\nR²={r2_val:.3f}', (speed, d_val), 
                               textcoords="offset points", xytext=(0,-20), ha='center', 
                               color=color, fontsize=6)
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(output_dir, 'plots', 'd_vs_velocity')
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename = os.path.join(plots_dir, f"d_vs_velocity_{person}_{run_timestamp}.{save_format}")
    fig.savefig(plot_filename, dpi=dpi, bbox_inches='tight', facecolor='black')
    logger.info(f"Saved D vs velocity plot to: {plot_filename}")
    
    plt.close(fig)
    return plot_filename
