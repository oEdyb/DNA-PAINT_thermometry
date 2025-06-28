"""
Constants and configuration values for DNA-PAINT thermometry analysis.

This module centralizes all magic numbers, default values, and configuration
parameters used across the analysis pipeline.
"""

import os

DEFAULT_BASE_FOLDER = "C:\\Users\\olled\\Documents\\DNA-PAINT\\Data\\single_channel_DNA-PAINT_example\\Week_4\\All_DNA_Origami\\17_picks"
DEFAULT_PIXEL_SIZE = 0.130  # in micrometers
DEFAULT_PICK_SIZE = 3  # in camera pixels
DEFAULT_RADIUS_OF_PICK_TO_AVERAGE = 0.25  # in camera pixel size

DEFAULT_DOCKING_SITES = 3
DEFAULT_NP_FLAG = False
DEFAULT_THRESHOLD = 1
DEFAULT_NUMBER_OF_FRAMES = 12000
DEFAULT_EXPOSURE_TIME = 0.1  # in seconds
DEFAULT_PLOT_FLAG = True

HIST_2D_BIN_SIZE = 20  # bins for 2D histogram
HIST_FINE_BIN_SIZE = 50  # bins for fine histogram
NUMBER_OF_BINS_TIME = 120  # bins for time histogram
NUMBER_OF_BINS_POSITION = 16  # bins for position histogram

PHOTON_FILTER_MULTIPLIER = 0.5  # multiplier for photon filtering
PERPENDICULAR_FILTER_DISTANCE = 45e-3  # arbitrary filter distance
MIN_DISTANCE_BETWEEN_PEAKS_NM = 10  # minimum distance between peaks in nm

FIGURE_DPI = 300
FIGURE_DPI_LOW = 100
ANALYSIS_RADIUS = 0.05  # radius for analysis circles
PLOT_MARKER_SIZE = 0.5
PLOT_ALPHA = 0.6

# ================ KINETICS ANALYSIS CONSTANTS ================
PHOTON_THRESHOLD_DEFAULT = 50
BACKGROUND_LEVEL_DEFAULT = 10
MASK_LEVEL_DEFAULT = 1
MASK_SINGLES_DEFAULT = True

POSITION_HIST_RANGE = [25, 160]  # range for position histograms in nm

# ================ MATPLOTLIB CONFIGURATION ================
MATPLOTLIB_BACKEND = 'Agg'  # non-interactive backend
COLORMAP = 'viridis'

DAT_EXTENSION = '.dat'
PNG_EXTENSION = '.png'
HDF5_EXTENSION = '.hdf5'
PICKLE_EXTENSION = '.pkl'

GAUSSIAN_SIGMA = 0.8  # sigma for gaussian filtering
EXPECTED_PEAKS_DEFAULT = 3
DOUBLE_EVENT_THRESHOLD = 1.7  # threshold for double event detection
DOUBLE_EVENT_MIN_FRAMES = 5  # minimum frames for double event analysis

STEP1_FILTER_SIGMA = 1.5
STEP1_FILTER_UNCERTAINTY = 30e-3  # in micrometers
STEP1_FILTER_PHOTONS = 50

FOLDER_NAMES = {
    'data': 'data',
    'figures': 'figures',
    'kinetics': 'kinetics',
    'traces_per_pick': 'traces_per_pick',
    'traces_per_site': 'traces_per_site',
    'gaussian': 'gaussian',
    'figures_per_pick': 'figures_per_pick',
    'ton_per_site': 'ton_per_site',
    'toff_per_site': 'toff_per_site',
    'mean_photons_per_site': 'mean_photons_per_site',
    'std_photons_per_site': 'std_photons_per_site',
    'sum_photons_per_site': 'sum_photons_per_site',
    'photons_per_site': 'photons_per_site'
}

def get_default_config():
    """Return default configuration dictionary."""
    return {
        'base_folder': DEFAULT_BASE_FOLDER,
        'docking_sites': DEFAULT_DOCKING_SITES,
        'NP_flag': DEFAULT_NP_FLAG,
        'pixel_size': DEFAULT_PIXEL_SIZE,
        'pick_size': DEFAULT_PICK_SIZE,
        'radius_of_pick_to_average': DEFAULT_RADIUS_OF_PICK_TO_AVERAGE,
        'threshold': DEFAULT_THRESHOLD,
        'number_of_frames': DEFAULT_NUMBER_OF_FRAMES,
        'exp_time': DEFAULT_EXPOSURE_TIME,
        'plot_flag': DEFAULT_PLOT_FLAG,
        'photon_threshold': PHOTON_THRESHOLD_DEFAULT,
        'background_level': BACKGROUND_LEVEL_DEFAULT,
        'mask_level': MASK_LEVEL_DEFAULT,
        'mask_singles': MASK_SINGLES_DEFAULT
    }
