"""
Visualization module for Aletheia Fractal Analysis

This module handles all plotting and visualization functionality.
"""

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime
import logging

# Optional import for interactive 3D plotting
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def create_visualization(points_new, best_D, best_R2, log_divs, log_counts, divs, counts, 
                        best_intercept, best_range, full_D, full_R2, full_intercept, custom_start, custom_end, dens_norm, x_new, y_new, z_new, 
                        config, logger, plots_dir=None):
    """Create and save visualization plots"""
    logger.info("=== VISUALIZATION PHASE ===")
    
    # Get dataset information for titles and filenames
    # Try to get from single config first, then from data config as fallback
    single_config = config.get('single', {})
    if single_config.get('person') and single_config.get('speed') and single_config.get('run_id'):
        person = single_config['person']
        speed = single_config['speed']
        run_id = single_config['run_id']
    else:
        # Fallback to data config (for backward compatibility)
        data_config = config['data']
        person = data_config.get('person', 'unknown')
        speed = data_config.get('speed', 'unknown')
        run_id = data_config.get('run_id', 'unknown')
    
    dataset_info = f"{person}_{speed}_{run_id}"
    
    # Use provided plots_dir if available, otherwise create plots directory with date subfolder
    if plots_dir is None:
        # Create plots directory with current date subfolder
        current_date = datetime.now().strftime('%Y%m%d')
        plots_dir = os.path.join('plots', current_date)
        os.makedirs(plots_dir, exist_ok=True)
    
    fig_size = tuple(config['visualization']['figure_size'])
    save_format = config['visualization']['save_format']
    dpi = config['visualization']['dpi']
    
    # Get interpolation factor for plot naming
    interpolation_factor = config['interpolation'].get('factor', 1)
    resolution_suffix = f"_{interpolation_factor}x" if interpolation_factor > 1 else ""
    resolution_text = f" ({interpolation_factor}x resolution)" if interpolation_factor > 1 else ""
    
    def plot_box_counting(ax, plot_divs, plot_counts, fit_D, fit_intercept, fit_R2, title_suffix, highlight_range=None, annotate=True):
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
        ax.set_title(f'Fractal Box Counting {title_suffix}: {dataset_info}{resolution_text}', 
                     color='white', fontsize=14)
        
        # Add annotations for each point
        if annotate:
            for i in range(len(plot_divs)):
                ax.annotate(f'({plot_divs[i]}, {plot_counts[i]})', (plot_divs[i], plot_counts[i]), 
                            textcoords="offset points", xytext=(0,10), ha='center', color='white', fontsize=8)
    
    # Plot 1: Full range
    fig_full = plt.figure(figsize=fig_size, facecolor='black')
    ax_full = fig_full.add_subplot(111)
    plot_box_counting(ax_full, divs, counts, full_D, full_intercept, full_R2, "Full Range")
    full_plot_filename = f"{plots_dir}/full_range_fractal_analysis_{dataset_info}{resolution_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{save_format}"
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
        plot_box_counting(ax_best, plot_divs_best, plot_counts_best, best_D, best_intercept, best_R2, "Best Range")
        best_plot_filename = f"{plots_dir}/best_range_fractal_analysis_{dataset_info}{resolution_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{save_format}"
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
            plot_box_counting(ax_custom, plot_divs_custom, plot_counts_custom, custom_D, custom_intercept, custom_R2, f"Custom Range ({custom_start}-{custom_end})")
            custom_plot_filename = f"{plots_dir}/custom_range_fractal_analysis_{dataset_info}{resolution_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{save_format}"
            plt.tight_layout()
            fig_custom.savefig(custom_plot_filename, dpi=dpi, bbox_inches='tight', facecolor='black')
            logger.info(f"  - Saved custom range plot to: {custom_plot_filename}")
            plt.close(fig_custom)
    else:
        logger.info("No custom range provided; skipping custom plot.")
    
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
                ax1.set_title(f'3D Trajectory: {dataset_info}{resolution_text}\nDensity Coloring', color='white', fontsize=12)
            else:
                ax1.set_title(f'3D Trajectory: {dataset_info}{resolution_text}\nUniform Coloring', color='white', fontsize=12)
            
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
        
        plot_box_counting(ax, divs, counts, best_D, best_intercept, best_R2, "Combined", best_range)
        
        plt.tight_layout()
        logger.info("  - Created box counting scaling plot")
        
        # Save the plot with dataset info in filename
        combined_plot_filename = f"{plots_dir}/combined_fractal_analysis_{dataset_info}{resolution_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{save_format}"
        plt.savefig(combined_plot_filename, dpi=dpi, bbox_inches='tight', facecolor='black')
        logger.info(f"  - Saved combined plot to: {combined_plot_filename}")
        plt.close(fig)
    
    return [full_plot_filename, best_plot_filename, custom_plot_filename] if 'custom_plot_filename' in locals() else [full_plot_filename, best_plot_filename]


def create_interactive_3d_plot(points_new, dens_norm, x_new, y_new, z_new, config, logger, output_dir):
    """Create an interactive 3D plot using plotly"""
    if not PLOTLY_AVAILABLE:
        logger.warning("  - Plotly not available. Install with: pip install plotly")
        return None
    
    logger.info("  - Creating interactive 3D plot with plotly")
    
    # Get dataset information for title and filename
    data_config = config['data']
    person = data_config['person']
    speed = data_config['speed']
    run_id = data_config['run_id']
    dataset_info = f"{person}_{speed}_{run_id}"
    
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
        title=f'Interactive 3D Trajectory: {dataset_info}<br>Density Coloring',
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
    
    # Save as HTML file with dataset info in filename
    os.makedirs(os.path.join(output_dir, 'plots/interactive_3d_plots'), exist_ok=True)
    html_filename = os.path.join(output_dir, f"plots/interactive_3d_plots/interactive_3d_plot_{dataset_info}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    fig.write_html(html_filename)
    logger.info(f"  - Interactive 3D plot saved to: {html_filename}")
    
    return html_filename
