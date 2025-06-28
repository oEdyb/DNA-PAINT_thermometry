"""
Plotting utilities for DNA-PAINT thermometry analysis.

This module contains reusable plotting functions to reduce code duplication
and standardize visualization across the analysis pipeline.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from auxiliary_functions import manage_save_directory, plot_vs_time_with_hist
from constants import FIGURE_DPI, FIGURE_DPI_LOW, PLOT_MARKER_SIZE, PLOT_ALPHA, ANALYSIS_RADIUS


def setup_matplotlib():
    """Configure matplotlib for non-interactive plotting."""
    plt.ioff()
    plt.close("all")


def plot_circle(center, radius, **kwargs):
    """Create a circle patch for plotting."""
    return Circle(center, radius, **kwargs)


def save_figure(figure_path, dpi=FIGURE_DPI):
    """Save current figure with standard settings."""
    plt.savefig(figure_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_pick_time_series(bin_centers_minutes, locs_vs_time, pick_id, bin_size, 
                         figures_folder, exp_time):
    """Plot localization count vs time for a single pick."""
    plt.figure()
    plt.step(bin_centers_minutes, locs_vs_time, where='mid', label=f'Pick {pick_id:04d}')
    plt.xlabel('Time (min)')
    plt.ylabel('Locs')
    plt.ylim([0, 80])
    ax = plt.gca()
    ax.axvline(x=10, ymin=0, ymax=1, color='k', linewidth='2', linestyle='--')
    ax.set_title(f'Number of locs per pick vs time. Bin size {bin_size*0.1/60:.1f} min')
    
    aux_folder = manage_save_directory(figures_folder, 'locs_vs_time_per_pick')
    figure_name = f'locs_per_pick_vs_time_pick_{pick_id:02d}'
    figure_path = os.path.join(aux_folder, f'{figure_name}.png')
    save_figure(figure_path, dpi=FIGURE_DPI_LOW)


def plot_scatter_with_np(x_positions, y_positions, x_np_positions=None, y_np_positions=None,
                        x_avg_np=None, y_avg_np=None, pick_id=0, figures_folder="", 
                        plot_type="raw"):
    """Plot scatter plot of localizations with optional NP data."""
    plt.figure()
    plt.scatter(x_positions, y_positions, color='C0', label='Fluorophore Emission', s=PLOT_MARKER_SIZE)
    
    if x_np_positions is not None and y_np_positions is not None:
        plt.scatter(x_np_positions, y_np_positions, color='C1', s=PLOT_MARKER_SIZE, alpha=0.2)
    
    if x_avg_np is not None and y_avg_np is not None:
        plt.scatter(x_avg_np, y_avg_np, color='C1', label='NP Scattering', s=PLOT_MARKER_SIZE, alpha=1)
        plt.plot(x_avg_np, y_avg_np, 'x', color='k', label='Center of NP')
        plt.legend(loc='upper left')
    
    plt.ylabel(r'y ($\mu$m)')
    plt.xlabel(r'x ($\mu$m)')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.axis('square')
    ax = plt.gca()
    ax.set_title(f'Position of locs per pick. Pick {pick_id:02d}')
    
    aux_folder = manage_save_directory(figures_folder, 'scatter_plots')
    figure_name = f'xy_pick_scatter_{plot_type}_{pick_id:02d}'
    figure_path = os.path.join(aux_folder, f'{figure_name}.png')
    save_figure(figure_path)


def plot_fine_2d_histogram(z_hist, x_centers, y_centers, peak_coords=None, 
                          x_avg_np=None, y_avg_np=None, pick_id=0, figures_folder="",
                          cmap='viridis'):
    """Plot fine 2D histogram with peaks and NP markers."""
    plt.figure()
    plt.imshow(z_hist, interpolation='none', origin='lower',
              extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]])
    ax = plt.gca()
    ax.set_facecolor(plt.cm.get_cmap(cmap)(0))
    
    if peak_coords and len(peak_coords) > 0:
        peak_x = [coord[0] for coord in peak_coords]
        peak_y = [coord[1] for coord in peak_coords]
        plt.plot(peak_x, peak_y, 'x', markersize=9, 
                color='white', markeredgecolor='black', mew=1, label='binding sites')
        
        for k, (circle_x, circle_y) in enumerate(peak_coords):
            circ = plot_circle((circle_x, circle_y), radius=ANALYSIS_RADIUS, 
                              facecolor='none', edgecolor='white', linewidth=1)
            ax.add_patch(circ)
            
            label_x = circle_x + ANALYSIS_RADIUS * 1.25
            label_y = circle_y + ANALYSIS_RADIUS * 0.25
            ax.text(label_x, label_y, f"{k}", ha='center', va='center', 
                   rotation=0, fontsize=12, color='white')
    
    if x_avg_np is not None and y_avg_np is not None:
        plt.plot(x_avg_np, y_avg_np, 'o', markersize=8, markerfacecolor='C1', 
                markeredgecolor='white', label='NP')
        plt.legend(loc='upper right')
    
    plt.ylabel(r'y ($\mu$m)')
    plt.xlabel(r'x ($\mu$m)')
    cbar = plt.colorbar()
    cbar.ax.set_title(u'Locs', fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax.set_title(f'Pick {pick_id:02d}', fontsize=10)
    
    aux_folder = manage_save_directory(figures_folder, 'image_FINE')
    figure_name = f'xy_pick_image_PAINT_{pick_id:02d}'
    figure_path = os.path.join(aux_folder, f'{figure_name}.png')
    save_figure(figure_path)


def plot_peak_detection_process(z_hist_smooth, x_centers, y_centers, peak_coords,
                               x_avg_np=None, y_avg_np=None, pick_id=0, figures_folder=""):
    """Plot peak detection process visualization."""
    plt.figure()
    plt.imshow(z_hist_smooth, interpolation='none', origin='lower',
              extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]], 
              cmap='viridis')
    ax = plt.gca()
    ax.set_facecolor(plt.cm.get_cmap('viridis')(0))
    
    if peak_coords and len(peak_coords) > 0:
        peak_x = [coord[0] for coord in peak_coords]
        peak_y = [coord[1] for coord in peak_coords]
        plt.scatter(peak_x, peak_y, c='red', s=150, marker='o', 
                  edgecolors='white', linewidths=2, label=f'{len(peak_coords)} detected peaks')
        
        for k, (px, py) in enumerate(peak_coords):
            ax.text(px, py, f'{k}', ha='center', va='center', 
                   fontsize=10, color='white', fontweight='bold')
    
    if x_avg_np is not None and y_avg_np is not None:
        plt.plot(x_avg_np, y_avg_np, 's', markersize=10, markerfacecolor='yellow', 
                markeredgecolor='black', label='NP')
    
    plt.legend(loc='upper right')
    plt.ylabel(r'y ($\mu$m)')
    plt.xlabel(r'x ($\mu$m)')
    cbar = plt.colorbar()
    cbar.ax.set_title(u'Density')
    ax.set_title(f'Peak detection - Pick {pick_id:02d}', fontsize=10)
    
    aux_folder = manage_save_directory(figures_folder, 'image_peak_detection')
    figure_name = f'peak_detection_process_{pick_id:02d}'
    figure_path = os.path.join(aux_folder, f'{figure_name}.png')
    save_figure(figure_path)


def plot_distance_matrix(matrix, pick_id, figures_folder, matrix_type="distance"):
    """Plot distance or standard deviation matrix."""
    plt.figure()
    plt.imshow(matrix, interpolation='none', cmap='spring')
    ax = plt.gca()
    
    for l in range(matrix.shape[0]):
        for m in range(matrix.shape[1]):
            if matrix_type == "distance" and l == m:
                ax.text(m, l, '-', ha="center", va="center", color=[0,0,0], fontsize=18)
            elif matrix_type == "std_dev" and l != m:
                ax.text(m, l, '-', ha="center", va="center", color=[0,0,0], fontsize=18)
            else:
                ax.text(m, l, '%.0f' % matrix[l, m],
                       ha="center", va="center", color=[0,0,0], fontsize=18)
    
    ax.xaxis.tick_top()
    ax.set_xticks(np.array(range(matrix.shape[1])))
    ax.set_yticks(np.array(range(matrix.shape[0])))
    
    axis_string = ['NP']
    for j in range(matrix.shape[0] - 1):
        axis_string.append(f'Site {j+1}')
    ax.set_xticklabels(axis_string)
    ax.set_yticklabels(axis_string)
    
    aux_folder = manage_save_directory(figures_folder, f'matrix_{matrix_type}')
    figure_name = f'matrix_{matrix_type}_{pick_id:02d}'
    figure_path = os.path.join(aux_folder, f'{figure_name}.png')
    save_figure(figure_path, dpi=FIGURE_DPI_LOW)


def plot_global_position_histogram(positions_data, hist_range, figures_folder, 
                                  plot_type="binding_sites"):
    """Plot histogram of relative positions."""
    number_of_bins = 16
    bin_size = (hist_range[-1] - hist_range[0]) / number_of_bins
    position_bins, bin_edges = np.histogram(positions_data, bins=number_of_bins, range=hist_range)
    bin_centers = bin_edges[:-1] + bin_size / 2
    
    plt.figure()
    plt.bar(bin_centers, position_bins, width=0.8*bin_size, align='center')
    plt.xlabel('Position (nm)')
    plt.ylabel('Counts')
    
    figure_name = f'relative_positions_{plot_type}'
    figure_path = os.path.join(figures_folder, f'{figure_name}.png')
    save_figure(figure_path)


def plot_kinetics_analysis(tons_all, toffs_all, tstarts_all, sum_photons_all, 
                          std_photons_all, SNR_filtered, SBR_filtered, 
                          tstarts_filtered, figures_folder):
    """Plot comprehensive kinetics analysis."""
    
    if len(tons_all) > 0 and len(tstarts_all) > 0:
        plt.figure()
        ax, slope, intercept = plot_vs_time_with_hist(tons_all, tstarts_all/60, order=1, fit_line=True)
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Binding time [s]')
        ax.set_title(f'Binding time vs. start time\nSlope: {slope:.3f}, Intercept: {intercept:.3f}')
        figure_path = os.path.join(figures_folder, 'binding_time_vs_time.png')
        save_figure(figure_path)
    
    if len(sum_photons_all) > 0 and len(tstarts_all) > 0:
        plt.figure()
        ax = plot_vs_time_with_hist(sum_photons_all, tstarts_all/60)
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Total photons [photons]')
        ax.set_title('Total photons per binding event vs time')
        figure_path = os.path.join(figures_folder, 'total_photons_vs_time.png')
        save_figure(figure_path)
    
    # Photon count histogram
    if len(sum_photons_all) > 0:
        plt.figure()
        bin_edges = np.histogram_bin_edges(sum_photons_all, 'fd')
        plt.hist(sum_photons_all, bins=bin_edges)
        plt.xlabel('Photon count [photons]')
        plt.ylabel('Frequency')
        plt.title('Histogram of photon count per binding event')
        plt.yscale('log')
        figure_path = os.path.join(figures_folder, 'photon_histogram.png')
        save_figure(figure_path)
    
    # Standard deviation histogram
    if len(std_photons_all) > 0:
        plt.figure()
        bin_edges = np.histogram_bin_edges(std_photons_all, 'fd')
        plt.hist(std_photons_all, bins=bin_edges)
        plt.xlabel('Standard deviation [photons]')
        plt.ylabel('Frequency')
        plt.title('Histogram of photon standard deviation per binding event')
        plt.yscale('log')
        figure_path = os.path.join(figures_folder, 'std_photons_histogram.png')
        save_figure(figure_path)
    
    # SNR and SBR scatter plot
    if len(SNR_filtered) > 0 and len(tstarts_filtered) > 0:
        plt.figure()
        plt.scatter(tstarts_filtered/60, SNR_filtered, s=0.85, alpha=PLOT_ALPHA, label='SNR')
        plt.scatter(tstarts_filtered/60, SBR_filtered, s=0.85, alpha=PLOT_ALPHA, label='SBR')
        plt.yscale('log')
        plt.xlabel('Time [min]')
        plt.ylabel('SNR / SBR')
        plt.title('Signal-to-Noise and Signal-to-Background Ratios vs Time')
        plt.legend()
        figure_path = os.path.join(figures_folder, 'SNR_SBR_scatter_plot.png')
        save_figure(figure_path)


def plot_binding_site_photon_distributions(all_traces_per_site, figures_folder):
    """Plot photon distributions for each binding site."""
    if not all_traces_per_site:
        return
    
    keys = sorted(all_traces_per_site.keys(), key=float)
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'black', 'gray']
    line_styles = ['-'] * 10
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    try:
        global_min = min([data.min() for data in all_traces_per_site.values()])
        global_max = max([data.max() for data in all_traces_per_site.values()])
        
        x_lim_lower = global_min - 1
        x_lim_upper = global_max + 1
        
        for key, color, line_style in zip(keys, colors, line_styles):
            data = all_traces_per_site[key]
            bins = int(np.ceil(np.sqrt(len(data))))
            print(key, len(data))
            ax.hist(data, bins=bins, label=f'{key}', histtype='step', density=False, 
                   color=color, linestyle=line_style, linewidth=1, alpha=PLOT_ALPHA)
        
        ax.set_xlim(x_lim_lower, x_lim_upper)
        ax.set_xlabel("Photons", fontsize=24)
        ax.set_ylabel("Counts", fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.legend(title='Binding site')
        ax.set_title("Binding site photon distributions", fontsize=24)
        plt.tight_layout()
        
        figure_path = os.path.join(figures_folder, 'binding_site_photon_distributions.png')
        save_figure(figure_path)
    except Exception as e:
        print(f"Error plotting binding site distributions: {e}")
