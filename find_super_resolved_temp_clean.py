# -*- coding: utf-8 -*-
"""
Analysis of single AuNPs temperature profile using DNA-PAINT super-resolution technique

Mariano Barella

5 July 2023

"""

import os
import re
import tkinter as tk
import tkinter.filedialog as fd

import numpy as np

# from auxiliary_functions import save_parameters
import step1_extract_and_save_data_from_hdf5_picasso_files as step1
import step2_process_picasso_extracted_data as step2
import step3_calculate_tonoff_with_mask as step3
import step4_estimate_binding_time_using_MLE as step4
import time
import argparse
from ast import literal_eval

#####################################################################
# TODO: Print all relevant measurements at the end of this script.
def run_analysis(selected_file, working_folder, step, params):

    number_of_frames = params.get('number_of_frames', 0)
    exp_time = params.get('exposure_time', 0)
    docking_sites = params.get('docking_sites', 0)
    NP_flag = params.get('checkNP', 0)
    pixel_size = params.get('pixel_size', 0)*1e-3
    pick_size = params.get('pick_size', 0)
    radius_of_pick_to_average = params.get('size_to_average', 0)
    th = params.get('th', 0)
    plot_flag = params.get('checkPlot', 0)
    photons_threshold = params.get('photon_threshold', 0)
    background_level = params.get('background_level', 0)
    mask_level = params.get('mask_level', 0)
    rango = params.get('range', 0)
    initial_params = params.get('initial_params', {})
    likelihood_err_param = params.get('likelihood_error', 0)
    opt_display_flag = params.get('checkOptimizationDisplay', 0)
    hyper_exponential_flag = params.get('checkHyperExponential', 0)
    recursive_flag = params.get('recursive', 0)
    rectangles_flag = params.get('rectangle', 0)
    verbose_flag = params.get('verbose', 0)
    lpx_filter = params.get('lpx_filter', 0)
    lpy_filter = params.get('lpy_filter', 0)
    mask_singles = params.get('mask_singles', 0)
    photon_threshold_flag = False

    # print("Analysis Parameters:")
    # print(f"Selected File: {selected_file}")
    # print(f"Working Folder: {working_folder}")
    # print(f"Step: {step}")
    # print(f"Verbose: {verbose_flag}")
    # print(f"lpy filter: {lpx_filter}")
    # print(f"lpx filter: {lpy_filter}")
    # print(f"Number of Frames: {number_of_frames}")
    # print(f"Exposure Time: {exp_time}")
    # print(f"Docking Sites: {docking_sites}")
    # print(f"NP Flag: {NP_flag}")
    # print(f"Pixel Size: {pixel_size}")
    # print(f"Pick Size: {pick_size}")
    # print(f"Radius to Average: {radius_of_pick_to_average}")
    # print(f"Initial threshold: {th}")
    # print(f"Plot Flag: {plot_flag}")
    # print(f"Photons Threshold: {photons_threshold}")
    # print(f"Background Level: {background_level}")
    # print(f"Mask Level: {mask_level}")
    # print(f"Range: {rango}")
    # print(f"Initial Parameters: {initial_params}")
    # print(f"Likelihood Error Parameter: {likelihood_err_param}")
    # print(f"Optimization Display Flag: {opt_display_flag}")
    # print(f"Hyper Exponential Flag: {hyper_exponential_flag}")
    # print(f"Recursive Flag: {recursive_flag}")
    # print(f"Rectangles Flag: {rectangles_flag}")

    # mask_level = 10



    if step[0] == 'True':
        # run step
        step1.split_hdf5(selected_file, working_folder, recursive_flag, rectangles_flag,
                         lpx_filter, lpy_filter, verbose_flag, NP_flag)
    else:
        print('\nSTEP 1 was not executed.')
        
    #####################################################################
    
    # folder and file management
    step2_working_folder = os.path.join(working_folder, 'split_data')
    if step[1] == 'True':
        # run step
        step2.process_dat_files(number_of_frames, exp_time, step2_working_folder, \
                              docking_sites, NP_flag, pixel_size, pick_size, \
                              radius_of_pick_to_average, th, plot_flag, verbose_flag)
    else:
        print('\nSTEP 2 was not executed.')
        
    #####################################################################
        
    # folder and file management
    step3_working_folder = os.path.join(step2_working_folder, 'kinetics_data')
    if step[2] == 'True':
        # run step
        list_of_files_step3 = os.listdir(step3_working_folder)
        all_traces_filename = [f for f in list_of_files_step3 if re.search('TRACES_ALL',f)][0]
        bkg_filename = [f for f in os.listdir(step2_working_folder) if re.search('bkg', f)][0]
        photons_filename = [f for f in os.listdir(step2_working_folder) if re.search('photons', f)][0]
        bkg = np.loadtxt(os.path.join(step2_working_folder, bkg_filename))
        photons = np.loadtxt(os.path.join(step2_working_folder, photons_filename))
        background_level = np.mean(bkg, axis=None)
        step3.calculate_kinetics(exp_time, photons_threshold, background_level, photons,\
                                 step2_working_folder, \
                                 all_traces_filename, mask_level, mask_singles, number_of_frames, verbose_flag,
                                 photon_threshold_flag)
    else:
        print('\nSTEP 3 was not executed.')
    
    #####################################################################
        
    # folder and file management
    step4_working_folder = step3_working_folder
    if step[3] == 'True':
        # run step
        step4.estimate_binding_unbinding_times(exp_time, rango, step4_working_folder,
                                        initial_params, likelihood_err_param,
                                        opt_display_flag, hyper_exponential_flag, verbose_flag)
    else:
        print('\nSTEP 4 was not executed.')
    
    #####################################################################
    
    print('\nProcess done.')
    
    return


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Run analysis on HDF5 file.')
    parser.add_argument('--selected_file', type=str, required=True, help='Path to selected HDF5 file')
    parser.add_argument('--working_folder', type=str, required=True, help='Working folder')
    parser.add_argument('--step', type=str, nargs=4, required=True,
                        help='List of steps to execute (1 or 0)')
    parser.add_argument('--params', type=str, required=True, help='Parameters')

    args = parser.parse_args()
    params = literal_eval(args.params)
    if params.get('recursive', 0):
        for directory in os.listdir(os.path.dirname(args.working_folder)):
            dir = os.path.join(os.path.dirname(args.working_folder), directory)
            files = os.listdir(dir)

            # Filter the files to only those containing '_picked' in their name
            picked_files = [f for f in files if '_picked' in f]

            # Sort or process to ensure consistent selection of the first file
            picked_files.sort()

            # Take the first file from the filtered list, if any
            picked_file = picked_files[0] if picked_files else None
            path_to_picked_file = os.path.join(dir, picked_file)
            try:
                run_analysis(path_to_picked_file, dir, args.step, params)
            except:
                print(f'An error occurred with {directory}')
                pass
    else:
        run_analysis(args.selected_file, args.working_folder, args.step, params)
    print("Total execution time: --- %s seconds ---" % (time.time() - start_time))
