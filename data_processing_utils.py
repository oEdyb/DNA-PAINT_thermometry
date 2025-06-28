"""
Utility functions for data processing and file operations.

This module contains reusable functions for data extraction, filtering,
and saving operations used across the analysis pipeline.
"""

import os
import numpy as np
from auxiliary_functions import manage_save_directory
from constants import FOLDER_NAMES, DAT_EXTENSION


def save_data_array(data, folder_path, filename, fmt='%05d'):
    """Save numpy array to file with proper formatting."""
    if len(data) > 0:
        filepath = os.path.join(folder_path, filename)
        np.savetxt(filepath, data, fmt=fmt)


def save_coordinate_data(x_data, y_data, folder_path, filename):
    """Save x,y coordinate data to file."""
    if len(x_data) > 0 and len(y_data) > 0:
        filepath = os.path.join(folder_path, filename)
        np.savetxt(filepath, np.column_stack((x_data, y_data)))


def create_analysis_folders(main_folder):
    """Create all necessary folders for analysis output."""
    folders = {}
    
    for key, folder_name in FOLDER_NAMES.items():
        folders[key] = manage_save_directory(main_folder, folder_name)
    
    return folders


def filter_localizations(x, y, photons, sigma_x, sigma_y, frame, 
                        sigma_threshold=1.5, uncertainty_threshold=30e-3, 
                        photon_threshold=50):
    """
    Filter localizations based on quality criteria.
    
    Args:
        x, y: position arrays
        photons: photon counts
        sigma_x, sigma_y: localization uncertainties
        frame: frame numbers
        sigma_threshold: threshold for sigma filtering
        uncertainty_threshold: threshold for uncertainty filtering (in um)
        photon_threshold: minimum photon count
    
    Returns:
        Filtered arrays and indices
    """
    sigma_mask = (sigma_x < sigma_threshold) & (sigma_y < sigma_threshold)
    uncertainty_mask = (sigma_x < uncertainty_threshold) & (sigma_y < uncertainty_threshold)
    photon_mask = photons > photon_threshold
    
    combined_mask = sigma_mask & uncertainty_mask & photon_mask
    
    filtered_data = {
        'x': x[combined_mask],
        'y': y[combined_mask], 
        'photons': photons[combined_mask],
        'sigma_x': sigma_x[combined_mask],
        'sigma_y': sigma_y[combined_mask],
        'frame': frame[combined_mask],
        'indices': np.where(combined_mask)[0]
    }
    
    return filtered_data


def extract_hdf5_data(hdf5_file, dataset_names):
    """
    Extract data from HDF5 file.
    
    Args:
        hdf5_file: opened HDF5 file object
        dataset_names: list of dataset names to extract
    
    Returns:
        Dictionary with extracted data arrays
    """
    data = {}
    for name in dataset_names:
        if name in hdf5_file:
            data[name] = hdf5_file[name][:]
        else:
            print(f"Warning: Dataset '{name}' not found in HDF5 file")
            data[name] = np.array([])
    
    return data


def save_step1_results(filtered_data, output_folder, file_prefix=""):
    """
    Save Step 1 processing results to files.
    
    Args:
        filtered_data: dictionary with filtered data arrays
        output_folder: folder to save results
        file_prefix: optional prefix for filenames
    """
    save_coordinate_data(
        filtered_data['x'], filtered_data['y'],
        output_folder, f"{file_prefix}xy_coordinates{DAT_EXTENSION}"
    )
    
    data_files = {
        'photons': f"{file_prefix}photons{DAT_EXTENSION}",
        'sigma_x': f"{file_prefix}sigma_x{DAT_EXTENSION}",
        'sigma_y': f"{file_prefix}sigma_y{DAT_EXTENSION}",
        'frame': f"{file_prefix}frame{DAT_EXTENSION}"
    }
    
    for key, filename in data_files.items():
        if key in filtered_data:
            save_data_array(filtered_data[key], output_folder, filename)


def calculate_histogram_bounds(x_data, y_data, margin_factor=0.1):
    """Calculate histogram bounds with margin."""
    x_min, x_max = np.min(x_data), np.max(x_data)
    y_min, y_max = np.min(y_data), np.max(y_data)
    
    x_margin = (x_max - x_min) * margin_factor
    y_margin = (y_max - y_min) * margin_factor
    
    bounds = [
        [x_min - x_margin, x_max + x_margin],
        [y_min - y_margin, y_max + y_margin]
    ]
    
    return bounds


def load_dat_file(filepath):
    """Load data from .dat file with error handling."""
    try:
        if os.path.exists(filepath):
            return np.loadtxt(filepath)
        else:
            print(f"Warning: File not found: {filepath}")
            return np.array([])
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return np.array([])


def get_pick_files(working_folder, file_pattern="pick_*.dat"):
    """Get list of pick files in working folder."""
    import glob
    pattern = os.path.join(working_folder, file_pattern)
    files = glob.glob(pattern)
    return sorted(files)
