"""
Created on Tuesday July 2nd 2025

@author: Devin AI (refactored from original code by Mariano Barella)

This module contains extracted functions from the step2 processing files to make
the code more modular and maintainable. These functions handle various aspects
of DNA-PAINT data processing including folder setup, data loading, peak detection,
plotting, and kinetics analysis.

Jabba dabba doo - refactoring the massive step2 functions!
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Circle as plot_circle
import re
from auxiliary_functions import detect_peaks, detect_peaks_improved, get_peak_detection_histogram, distance, fit_linear, \
    perpendicular_distance, manage_save_directory, plot_vs_time_with_hist, update_pkl, \
    calculate_tau_on_times_average
from sklearn.mixture import GaussianMixture
import time
from auxiliary_functions_gaussian import plot_gaussian_2d
import scipy
import glob

# ================ MATPLOTLIB CONFIGURATION ================
cmap = plt.cm.get_cmap('viridis')
bkg_color = [0.1, 0.1, 0.1]


def setup_step2_folders(main_folder, method_name):
    """Create the folder structure for step2 analysis with method-specific separation."""
    analysis_folder = os.path.join(main_folder, 'analysis')
    step2_base_folder = manage_save_directory(analysis_folder, 'step2')
    step2_method_folder = manage_save_directory(step2_base_folder, method_name)
    
    folders = {
        'step2_method_folder': step2_method_folder,
        'figures_folder': manage_save_directory(step2_method_folder, 'figures'),
        'data_folder': manage_save_directory(step2_method_folder, 'data'),
    }
    
    folders['figures_per_pick_folder'] = manage_save_directory(folders['figures_folder'], 'per_pick')
    folders['traces_per_pick_folder'] = manage_save_directory(folders['data_folder'], 'traces')
    folders['traces_per_site_folder'] = manage_save_directory(folders['traces_per_pick_folder'], 'traces_per_site')
    folders['kinetics_folder'] = manage_save_directory(folders['data_folder'], 'kinetics_data')
    folders['gaussian_folder'] = manage_save_directory(folders['kinetics_folder'], 'gaussian_data')
    
    return folders

def cleanup_existing_traces(traces_per_site_folder):
    """Clean up existing trace files in the traces per site folder."""
    if os.path.exists(traces_per_site_folder):
        for f in os.listdir(traces_per_site_folder):
            file_path = os.path.join(traces_per_site_folder, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                print(f"{file_path} is not a file, skipping.")

def load_step2_data(working_folder, NP_flag, pixel_size):
    """Load all required data files for step2 processing."""
    list_of_files = os.listdir(working_folder)
    list_of_files = [f for f in list_of_files if re.search('.dat', f)]
    list_of_files.sort()
    
    if NP_flag:
        list_of_files_origami = [f for f in list_of_files if re.search('NP_subtracted',f)]
        list_of_files_NP = [f for f in list_of_files if re.search('raw',f)]
    else:
        list_of_files_origami = list_of_files
        list_of_files_NP = []
    
    data = {}
    
    frame_file = [f for f in list_of_files_origami if re.search('_frame',f)][0]
    frame_filepath = os.path.join(working_folder, frame_file)
    data['frame'] = np.loadtxt(frame_filepath)
    
    photons_file = [f for f in list_of_files_origami if re.search('_photons',f)][0]
    photons_filepath = os.path.join(working_folder, photons_file)
    data['photons'] = np.loadtxt(photons_filepath)
    if NP_flag:
        photons_file_NP = [f for f in list_of_files_NP if re.search('_photons', f)][0]
        photons_filepath_NP = os.path.join(working_folder, photons_file_NP)
        data['photons'] = np.loadtxt(photons_filepath_NP)
    
    bkg_file = [f for f in list_of_files_origami if re.search('_bkg',f)][0]
    bkg_filepath = os.path.join(working_folder, bkg_file)
    data['bkg'] = np.loadtxt(bkg_filepath)
    
    position_file = [f for f in list_of_files_origami if re.search('_xy',f)][0]
    position_filepath = os.path.join(working_folder, position_file)
    position = np.loadtxt(position_filepath)
    data['x'] = position[:,0]*pixel_size
    data['y'] = position[:,1]*pixel_size
    
    if NP_flag:
        position_file_NP = [f for f in list_of_files_NP if re.search('_xy',f)][0]
        position_filepath_NP = os.path.join(working_folder, position_file_NP)
        xy_NP = np.loadtxt(position_filepath_NP)
        data['x_NP'] = xy_NP[:,0]*pixel_size
        data['y_NP'] = xy_NP[:,1]*pixel_size
    
    pick_file = [f for f in list_of_files_origami if re.search('_pick_number',f)][0]
    pick_filepath = os.path.join(working_folder, pick_file)
    data['pick_list'] = np.loadtxt(pick_filepath)
    
    if NP_flag:
        pick_file_NP = [f for f in list_of_files_NP if re.search('_pick_number',f)][0]
        pick_filepath_NP = os.path.join(working_folder, pick_file_NP)
        data['pick_list_NP'] = np.loadtxt(pick_filepath_NP)
    
    return data

def detect_peaks_adaptive(x_positions, y_positions, hist_bounds, docking_sites, threshold_start=1):
    """Adaptive peak detection with automatic threshold adjustment."""
    total_peaks_found = 0
    threshold_COARSE = threshold_start
    bins_COARSE = 1 if docking_sites == 1 else 20
    docking_sites_temp = docking_sites
    site_goal = docking_sites
    z_hist_COARSE = None
    x_hist_COARSE = None
    y_hist_COARSE = None
    detected_peaks = None
    index_peaks = None
    
    while total_peaks_found != site_goal:
        if docking_sites_temp == docking_sites - 1 or total_peaks_found == docking_sites - 1:
            docking_sites_temp = docking_sites - 1
            if total_peaks_found == docking_sites - 1:
                break
        
        z_hist_COARSE, x_hist_COARSE, y_hist_COARSE = np.histogram2d(
            x_positions, y_positions, 
            bins=bins_COARSE, range=hist_bounds, density=True
        )
        z_hist_COARSE = z_hist_COARSE.T
        z_hist_COARSE = np.where(z_hist_COARSE < threshold_COARSE, 0, z_hist_COARSE)
        
        # Peak detection
        detected_peaks = detect_peaks(z_hist_COARSE)
        index_peaks = np.where(detected_peaks == True)
        total_peaks_found = len(index_peaks[0])
        
        threshold_COARSE += 5
        if threshold_COARSE > 5000:
            break
    
    peak_coords = []
    if total_peaks_found > 0 and x_hist_COARSE is not None:
        x_hist_COARSE_centers = x_hist_COARSE[:-1] + np.diff(x_hist_COARSE)/2
        y_hist_COARSE_centers = y_hist_COARSE[:-1] + np.diff(y_hist_COARSE)/2
        
        peak_coords = [(x_hist_COARSE_centers[index_peaks[1][j]], 
                       y_hist_COARSE_centers[index_peaks[0][j]]) 
                      for j in range(total_peaks_found)]
    
    return peak_coords, total_peaks_found, z_hist_COARSE, x_hist_COARSE, y_hist_COARSE, detected_peaks, index_peaks

def calculate_binding_site_stats(x_filtered, y_filtered):
    """Calculate statistics for a binding site from filtered positions."""
    cm_x = np.mean(x_filtered)
    cm_y = np.mean(y_filtered)
    std_x = np.std(x_filtered, ddof=1)
    std_y = np.std(y_filtered, ddof=1)
    return cm_x, cm_y, std_x, std_y

def process_binding_site_traces(frame_filtered, photons_filtered, number_of_frames):
    """Create trace data for a binding site."""
    trace = np.zeros(number_of_frames)
    np.add.at(trace, frame_filtered.astype(int), photons_filtered)
    return trace

def calculate_distance_matrices(cm_sites_x, cm_sites_y, x_avg_NP, y_avg_NP, std_devs_x, std_devs_y, NP_flag):
    """Calculate distance and standard deviation matrices."""
    total_peaks_found = len(cm_sites_x)
    matrix_distance = np.zeros([total_peaks_found + 1, total_peaks_found + 1])
    matrix_std_dev = np.zeros([total_peaks_found + 1, total_peaks_found + 1])
    
    np_to_binding_distances = np.array([])
    
    # NP to binding sites distances
    if NP_flag and x_avg_NP is not None:
        np_to_binding_distances = np.sqrt(
            (cm_sites_x - x_avg_NP)**2 + 
            (cm_sites_y - y_avg_NP)**2) * 1e3
        
        matrix_distance[0, 1:] = np_to_binding_distances
        matrix_distance[1:, 0] = np_to_binding_distances
        matrix_std_dev[0, 0] = max(np.std([x_avg_NP]), np.std([y_avg_NP])) * 1e3
    
    # Binding site to binding site distances
    peak_distances = np.array([])
    for j in range(total_peaks_found):
        x_binding_row = cm_sites_x[j]
        y_binding_row = cm_sites_y[j]
        matrix_std_dev[j + 1, j + 1] = max(std_devs_x[j], std_devs_y[j]) * 1e3
        
        for k in range(j + 1, total_peaks_found):
            x_binding_col = cm_sites_x[k]
            y_binding_col = cm_sites_y[k]
            distance_between_locs_CM = np.sqrt(
                (x_binding_col - x_binding_row)**2 + 
                (y_binding_col - y_binding_row)**2) * 1e3
            
            matrix_distance[j + 1, k + 1] = distance_between_locs_CM
            matrix_distance[k + 1, j + 1] = distance_between_locs_CM
            peak_distances = np.append(peak_distances, distance_between_locs_CM)
    
    return matrix_distance, matrix_std_dev, peak_distances, np_to_binding_distances

def plot_pick_time_series(bin_centers_minutes, locs_vs_time, pick_id, figures_per_pick_folder, bin_size):
    """Plot localization count vs time for a single pick."""
    plt.figure()
    plt.step(bin_centers_minutes, locs_vs_time, where='mid', label=f'Pick {pick_id:04d}')
    plt.xlabel('Time (min)')
    plt.ylabel('Locs')
    plt.ylim([0, 80])
    ax = plt.gca()
    ax.axvline(x=10, ymin=0, ymax=1, color='k', linewidth='2', linestyle='--')
    ax.set_title(f'Number of locs per pick vs time. Bin size {bin_size*0.1/60:.1f} min')
    aux_folder = manage_save_directory(figures_per_pick_folder, 'locs_vs_time_per_pick')
    figure_name = f'locs_per_pick_vs_time_pick_{pick_id:02d}'
    figure_path = os.path.join(aux_folder, f'{figure_name}.png')
    plt.savefig(figure_path, dpi=100, bbox_inches='tight')
    plt.close()

def plot_scatter_with_np(x_pos, y_pos, x_np_data, y_np_data, x_avg_NP, y_avg_NP, NP_flag, pick_id, figures_per_pick_folder):
    """Plot scatter plots with NP data if available."""
    # RAW scatter plot
    plt.figure(100)
    plt.scatter(x_pos, y_pos, color='C0', label='Fluorophore Emission', s=0.5)
    
    if NP_flag and x_np_data is not None:
        plt.scatter(x_np_data, y_np_data, color='C1', s=0.5, alpha=0.2)
        plt.scatter(x_avg_NP, y_avg_NP, color='C1', label='NP Scattering', s=0.5, alpha=1)
        plt.plot(x_avg_NP, y_avg_NP, 'x', color='k', label='Center of NP')
        plt.legend(loc='upper left')
    
    plt.ylabel(r'y ($\mu$m)')
    plt.xlabel(r'x ($\mu$m)')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.axis('square')
    ax = plt.gca()
    ax.set_title(f'Position of locs per pick. Pick {pick_id:02d}')
    aux_folder = manage_save_directory(figures_per_pick_folder, 'scatter_plots')
    figure_name = f'xy_pick_scatter_NP_and_PAINT_{pick_id:02d}'
    figure_path = os.path.join(aux_folder, f'{figure_name}.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(2)
    plt.plot(x_pos, y_pos, '.', color='C0', label='PAINT', linewidth=0.5)
    
    if NP_flag and x_avg_NP is not None:
        plt.plot(x_avg_NP, y_avg_NP, 'o', markersize=10, markerfacecolor='C1', 
                markeredgecolor='k', label='NP')
        plt.legend(loc='upper left')
    
    plt.ylabel(r'y ($\mu$m)')
    plt.xlabel(r'x ($\mu$m)')
    plt.axis('square')
    ax = plt.gca()
    ax.set_title(f'Position of locs per pick. Pick {pick_id:02d}')
    aux_folder = manage_save_directory(figures_per_pick_folder, 'scatter_plots')
    figure_name = f'xy_pick_scatter_PAINT_{pick_id:02d}'
    figure_path = os.path.join(aux_folder, f'{figure_name}.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_fine_2d_image(z_hist, x_centers, y_centers, cm_sites_x, cm_sites_y, analysis_radius, 
                      ranks, slope, x_avg_NP, y_avg_NP, NP_flag, pick_id, figures_per_pick_folder, 
                      peaks_flag, paper_version=False):
    """Plot fine 2D histogram image with binding sites and NP."""
    plt.figure(3)
    plt.imshow(z_hist, interpolation='none', origin='lower',
              extent=[x_centers[0], x_centers[-1], 
                      y_centers[0], y_centers[-1]])
    ax = plt.gca()
    ax.set_facecolor(bkg_color)
    
    if peaks_flag and len(cm_sites_x) > 1:
        alpha_val = 0.65 if paper_version else 1.0
        marker_size = 5 if paper_version else 9
        line_width = 1 if paper_version else 2
        
        plt.plot(cm_sites_x, cm_sites_y, 'x', markersize=marker_size,
                color='white', mew=line_width, label='Binding Sites' if paper_version else 'binding sites', 
                alpha=alpha_val)
        
        if not paper_version:
            plt.plot(x_centers, y_centers, '--', linewidth=1, color='white')
        
        for k, (circle_x, circle_y) in enumerate(zip(cm_sites_x, cm_sites_y)):
            circ = plot_circle((circle_x, circle_y), radius=analysis_radius, 
                              color='white', fill=False, linewidth=line_width, alpha=alpha_val)
            ax.add_patch(circ)
            
            # Peak labels
            if slope is not None:
                theta = np.arctan(-abs(slope) if paper_version else slope)
                perpendicular_x = circle_x + analysis_radius*1.25*np.cos(theta+np.pi/2)
                perpendicular_y = circle_y + -1/slope * (perpendicular_x-circle_x)
                text_position = (perpendicular_x, perpendicular_y)
                text_content = f"{ranks[k]}"
                ax.text(*text_position, text_content, ha='center', va='center', 
                       rotation=0, fontsize=11 if paper_version else 12, 
                       color='white', alpha=alpha_val)
    
    if NP_flag and x_avg_NP is not None:
        face_color = 'white' if paper_version else 'C1'
        edge_color = 'black' if paper_version else 'white'
        plt.plot(x_avg_NP, y_avg_NP, 'o', markersize=8, markerfacecolor=face_color, 
                markeredgecolor=edge_color, label='Center of NP' if paper_version else 'NP')
        plt.legend(loc='upper left' if paper_version else 'upper right')
    
    if paper_version:
        scalebar = ScaleBar(1e3, 'nm', location='lower left') 
        ax.add_artist(scalebar)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        cbar = plt.colorbar()
        cbar.ax.set_title(u'Locs')
        cbar.ax.tick_params()
        figure_name = f'PAPER_xy_pick_image_PAINT_{pick_id:02d}'
    else:
        plt.ylabel(r'y ($\mu$m)')
        plt.xlabel(r'x ($\mu$m)')
        cbar = plt.colorbar()
        cbar.ax.set_title(u'Locs', fontsize=16)
        cbar.ax.tick_params(labelsize=16)
        ax.set_title(f'Position of locs per pick. Pick {pick_id:02d}')
        figure_name = f'xy_pick_image_PAINT_{pick_id:02d}'
    
    aux_folder = manage_save_directory(figures_per_pick_folder, 'image_FINE')
    figure_path = os.path.join(aux_folder, f'{figure_name}.png')
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_distance_matrices(matrix_distance, matrix_std_dev, total_peaks_found, pick_id, figures_per_pick_folder):
    """Plot distance and standard deviation matrices."""
    if len(matrix_distance) > 1:
        # Matrix distance plot
        plt.figure(10)
        plt.imshow(matrix_distance, interpolation='none', cmap='spring')
        ax = plt.gca()
        
        for l in range(matrix_distance.shape[0]):
            for m in range(matrix_distance.shape[1]):
                if l == m:
                    ax.text(m, l, '-', ha="center", va="center", color=[0,0,0], fontsize=18)
                else:
                    ax.text(m, l, '%.0f' % matrix_distance[l, m],
                           ha="center", va="center", color=[0,0,0], fontsize=18)
        
        ax.xaxis.tick_top()
        ax.set_xticks(np.array(range(matrix_distance.shape[1])))
        ax.set_yticks(np.array(range(matrix_distance.shape[0])))
        axis_string = ['NP']
        for j in range(total_peaks_found):
            axis_string.append(f'Site {j+1}')
        ax.set_xticklabels(axis_string)
        ax.set_yticklabels(axis_string)
        aux_folder = manage_save_directory(figures_per_pick_folder, 'matrix_distance')
        figure_name = f'matrix_distance_{pick_id:02d}'
        figure_path = os.path.join(aux_folder, f'{figure_name}.png')
        plt.savefig(figure_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Matrix std dev plot
        plt.figure(11)
        plt.imshow(matrix_std_dev, interpolation='none', cmap='spring')
        ax = plt.gca()
        
        for l in range(matrix_distance.shape[0]):
            for m in range(matrix_distance.shape[1]):
                if l != m:
                    ax.text(m, l, '-', ha="center", va="center", color=[0,0,0], fontsize=18)
                else:
                    ax.text(m, l, '%.0f' % matrix_std_dev[l, m],
                           ha="center", va="center", color=[0,0,0], fontsize=18)
        
        ax.xaxis.tick_top()
        ax.set_xticks(np.array(range(matrix_distance.shape[1])))
        ax.set_yticks(np.array(range(matrix_distance.shape[0])))
        axis_string = ['NP']
        for j in range(total_peaks_found):
            axis_string.append(f'Site {j+1}')
        ax.set_xticklabels(axis_string)
        ax.set_yticklabels(axis_string)
        aux_folder = manage_save_directory(figures_per_pick_folder, 'matrix_std_dev')
        figure_name = f'matrix_std_dev_{pick_id:02d}'
        figure_path = os.path.join(aux_folder, f'{figure_name}.png')
        plt.savefig(figure_path, dpi=100, bbox_inches='tight')

def process_position_averaging(pick_trace, photons_of_picked, bkg_of_picked, exp_time, 
                              x_position_of_picked_raw, y_position_of_picked_raw, frame_of_picked, 
                              pick_id, verbose_flag=False):
    """Process position averaging for binding events in the averaging method."""
    # Set parameters for binding event detection
    photons_threshold = np.mean(photons_of_picked) * 0.1  # Simple threshold
    background_level = np.mean(bkg_of_picked)
    mask_level = 1  # Simple masking
    mask_singles = False
    
    # Get averaged positions for binding events
    tau_results = calculate_tau_on_times_average(
        pick_trace, photons_threshold, background_level, exp_time,
        mask_level, mask_singles, False, pick_id,  # verbose_flag=False
        x_position_of_picked_raw, y_position_of_picked_raw, frame_of_picked
    )
    
    # Extract averaged positions from results
    if tau_results[0] is not False and len(tau_results) >= 14:
        avg_x_positions = tau_results[12]  # average_x_positions
        avg_y_positions = tau_results[13]  # average_y_positions
        
        # Filter out NaN values
        valid_mask = ~(np.isnan(avg_x_positions) | np.isnan(avg_y_positions))
        if np.any(valid_mask):
            x_position_of_picked = avg_x_positions[valid_mask]
            y_position_of_picked = avg_y_positions[valid_mask]
            
            # Create corresponding frame and photon data for averaged positions
            # Use the start times from the binding events
            binding_start_times = tau_results[3]  # start_time
            valid_start_times = binding_start_times[valid_mask]
            frame_of_picked = (valid_start_times / exp_time).astype(int)
            
            # Use sum photons for each event instead of individual photons
            sum_photons_events = tau_results[6]  # sum_photons
            photons_of_picked = sum_photons_events[valid_mask]
            
            if verbose_flag:
                print(f'Using {len(x_position_of_picked)} averaged positions from {len(x_position_of_picked_raw)} raw localizations')
        else:
            # Fall back to raw data if no valid averaged positions
            x_position_of_picked = x_position_of_picked_raw
            y_position_of_picked = y_position_of_picked_raw
            if verbose_flag:
                print('No valid averaged positions found, using raw localizations')
    else:
        # Fall back to raw data if averaging failed
        x_position_of_picked = x_position_of_picked_raw
        y_position_of_picked = y_position_of_picked_raw
        if verbose_flag:
            print('Position averaging failed, using raw localizations')
    
    return x_position_of_picked, y_position_of_picked, frame_of_picked, photons_of_picked


def plot_relative_positions(positions_concat_NP, positions_concat_origami, figures_folder):
    """Plot relative positions for NP and binding sites."""
    # Plot NP relative positions
    number_of_bins = 16
    hist_range = [25, 160]
    bin_size = (hist_range[-1] - hist_range[0])/number_of_bins
    position_bins, bin_edges = np.histogram(positions_concat_NP, bins=number_of_bins, 
                                            range=hist_range)
    bin_centers = bin_edges[:-1] + bin_size/2
    plt.figure()
    plt.bar(bin_centers, position_bins, width=0.8*bin_size, align='center')
    plt.xlabel('Position (nm)')
    plt.ylabel('Counts')
    figure_name = 'relative_positions_NP_sites'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot binding site relative positions
    position_bins, bin_edges = np.histogram(positions_concat_origami, bins=number_of_bins, 
                                            range=hist_range)
    bin_centers = bin_edges[:-1] + bin_size/2
    plt.figure()
    plt.bar(bin_centers, position_bins, width=0.8*bin_size, align='center')
    plt.xlabel('Position (nm)')
    plt.ylabel('Counts')
    figure_name = 'relative_positions_binding_sites'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_global_time_series(locs_of_picked_vs_time, bin_centers_minutes, total_time_min, 
                           total_number_of_picks, figures_folder):
    """Plot global time series analysis."""
    sum_of_locs_of_picked_vs_time = np.sum(locs_of_picked_vs_time, axis=0)
    plt.figure()
    plt.step(bin_centers_minutes, sum_of_locs_of_picked_vs_time, where='mid')
    plt.xlabel('Time (min)')
    plt.ylabel('Locs')
    x_limit = [0, total_time_min]
    plt.xlim(x_limit)
    ax = plt.gca()
    bin_size_seconds = (bin_centers_minutes[1] - bin_centers_minutes[0]) * 60 if len(bin_centers_minutes) > 1 else 10
    ax.set_title('Sum of localizations vs time. Binning time %d s. %d picks. ' 
                 % (bin_size_seconds, total_number_of_picks))
    figure_name = 'locs_vs_time_all'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_kinetics_data(photons_concat, time_concat, kinetics_folder):
    """Save kinetics data to files."""
    new_filename = 'PHOTONS.dat'
    new_filepath = os.path.join(kinetics_folder, new_filename)
    np.savetxt(new_filepath, photons_concat)

    new_filename = 'TIME.dat'
    new_filepath = os.path.join(kinetics_folder, new_filename)
    np.savetxt(new_filepath, time_concat)


def plot_background_analysis(bkg_concat, time_concat, figures_folder):
    """Plot background signal analysis."""
    ax = plot_vs_time_with_hist(bkg_concat, time_concat, order=2)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Total background [photons]')
    ax.set_title('Total background received vs time.')
    figure_name = 'bkg_vs_time'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_site_photon_distributions(all_traces_per_site, figures_folder):
    """Plot site-specific photon distributions."""
    keys = sorted(all_traces_per_site.keys(), key=float)
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'black', 'gray']
    line_styles = ['-']*10

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
                   color=color, linestyle=line_style, linewidth=1, alpha=0.6)

        ax.set_xlim(x_lim_lower, x_lim_upper)
        ax.set_xlabel("Photons", fontsize=24)
        ax.set_ylabel("Counts", fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.legend(title='Binding site')
        ax.set_title("Binding site photon distributions", fontsize=24)
        plt.tight_layout()
        figure_name = 'binding_site_photon_distributions'
        figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error plotting site photon distributions: {e}")
        plt.close()
