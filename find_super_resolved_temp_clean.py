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
import pandas as pd
import json
from datetime import datetime

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

def save_consolidated_results(results_dict, metadata_dict, working_folder):
    """Save consolidated results to CSV and metadata to JSON"""
    
    # ================ SAVE RESULTS TO CSV ================
    results_df = pd.DataFrame([results_dict])
    csv_path = os.path.join(working_folder, 'results.csv')
    results_df.to_csv(csv_path, index=False)
    
    # ================ SAVE METADATA TO JSON ================
    metadata_dict['analysis_timestamp'] = datetime.now().isoformat()
    metadata_path = os.path.join(working_folder, 'analysis_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    
    print(f'\n📊 Results saved to: {csv_path}')
    print(f'📋 Metadata saved to: {metadata_path}')

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

    # ================ INITIALIZE RESULTS AND METADATA DICTIONARIES ================
    results_dict = {
        'sample_name': os.path.basename(selected_file),
        'working_folder': working_folder
    }
    
    metadata_dict = {
        'analysis_parameters': params,
        'file_path': selected_file,
        'steps_executed': [str(i+1) for i, enabled in enumerate(step) if enabled == 'True']
    }

    # ================ ANALYSIS PARAMETERS SUMMARY ================
    print('\n' + '='*18 + '⚙️ ANALYSIS PARAMETERS ⚙️' + '='*18)
    print(f'   Exposure Time: {exp_time}s | Frames: {number_of_frames} | Docking Sites: {docking_sites}')
    print(f'   Photon Threshold: {photons_threshold} | Background: {background_level} | Mask Level: {mask_level}')
    print(f'   Pixel Size: {pixel_size*1000:.0f}nm | Pick Size: {pick_size}px | LP Filters: {lpx_filter}/{lpy_filter}')
    if hyper_exponential_flag:
        print(f'   Mode: Hyper-Exponential | Range: {rango} | Error: {likelihood_err_param}')
    else:
        print(f'   Mode: Mono-Exponential | Range: {rango} | Error: {likelihood_err_param}')
    steps_enabled = [f'Step {i+1}' for i, enabled in enumerate(step) if enabled == 'True']
    print(f'   Steps: {", ".join(steps_enabled)}')
    print('='*65)

    if step[0] == 'True':
        # ================ RUN STEP 1 ================
        step1_results = step1.split_hdf5(selected_file, working_folder, recursive_flag, rectangles_flag,
                         lpx_filter, lpy_filter, verbose_flag, NP_flag)
        if step1_results:
            results_dict.update({f'step1_{k}': v for k, v in step1_results.items()})
    else:
        print('\nSTEP 1 was not executed.')
        
    #####################################################################
    
    # folder and file management
    step2_working_folder = os.path.join(working_folder, 'analysis', 'step1', 'data')
    if step[1] == 'True':
        # ================ RUN STEP 2 ================
        step2_results = step2.process_dat_files(number_of_frames, exp_time, step2_working_folder, \
                              docking_sites, NP_flag, pixel_size, pick_size, \
                              radius_of_pick_to_average, th, plot_flag, verbose_flag)
        if step2_results:
            results_dict.update({f'step2_{k}': v for k, v in step2_results.items()})
    else:
        print('\nSTEP 2 was not executed.')
        
    #####################################################################
        
    # folder and file management
    step3_working_folder = os.path.join(working_folder, 'analysis', 'step2', 'data', 'kinetics_data')
    step2_main_folder = os.path.join(working_folder, 'analysis', 'step2', 'data')
    if step[2] == 'True':
        # ================ RUN STEP 3 ================
        # Check if the required folders and files exist
        if not os.path.exists(step3_working_folder):
            print(f'\nERROR: Step 3 cannot run - kinetics data folder not found: {step3_working_folder}')
            print('Make sure Step 2 completed successfully.')
        elif not os.path.exists(step2_main_folder):
            print(f'\nERROR: Step 3 cannot run - step2 data folder not found: {step2_main_folder}')
            print('Make sure Step 2 completed successfully.')
        else:
            try:
                list_of_files_step3 = os.listdir(step3_working_folder)
                if verbose_flag:
                    print(f'Debug: Found {len(list_of_files_step3)} files in kinetics folder: {list_of_files_step3}')
                
                all_traces_filename = [f for f in list_of_files_step3 if re.search('TRACES_ALL',f)][0]
                bkg_filename = [f for f in list_of_files_step3 if re.search('BKG', f)][0]
                photons_filename = [f for f in list_of_files_step3 if re.search('PHOTONS', f)][0]
                
                if verbose_flag:
                    print(f'Debug: Using files - TRACES: {all_traces_filename}, BKG: {bkg_filename}, PHOTONS: {photons_filename}')
                
                bkg = np.loadtxt(os.path.join(step3_working_folder, bkg_filename))
                photons = np.loadtxt(os.path.join(step3_working_folder, photons_filename))
                background_level = np.mean(bkg, axis=None)
                step3_results = step3.calculate_kinetics(exp_time, photons_threshold, background_level, photons,\
                                         working_folder, \
                                         all_traces_filename, mask_level, mask_singles, number_of_frames, verbose_flag,
                                         photon_threshold_flag)
                if step3_results:
                    results_dict.update({f'step3_{k}': v for k, v in step3_results.items()})
            except FileNotFoundError as e:
                print(f'\nERROR: Step 3 cannot run - required file not found: {e}')
                print('Make sure Step 2 completed successfully and generated all required files.')
            except IndexError as e:
                print(f'\nERROR: Step 3 cannot run - required files missing in kinetics folder')
                print(f'Looking for TRACES_ALL, BKG, and PHOTONS files')
                print(f'Files found in {step3_working_folder}: {list_of_files_step3 if "list_of_files_step3" in locals() else "Could not list files"}')
    else:
        print('\nSTEP 3 was not executed.')
    
    #####################################################################
        
    # folder and file management
    step4_working_folder = os.path.join(working_folder, 'analysis', 'step3', 'data')
    if step[3] == 'True':
        # ================ RUN STEP 4 ================
        # Check if the required folder exists
        if not os.path.exists(step4_working_folder):
            print(f'\nERROR: Step 4 cannot run - step3 data folder not found: {step4_working_folder}')
            print('Make sure Step 3 completed successfully.')
        else:
            try:
                step4_results = step4.estimate_binding_unbinding_times(exp_time, rango, step4_working_folder,
                                                initial_params, likelihood_err_param,
                                                opt_display_flag, hyper_exponential_flag, verbose_flag)
                if step4_results:
                    results_dict.update({f'step4_{k}': v for k, v in step4_results.items()})
            except FileNotFoundError as e:
                print(f'\nERROR: Step 4 cannot run - required file not found: {e}')
                print('Make sure Step 3 completed successfully and generated t_on.dat and t_off.dat files.')
            except Exception as e:
                print(f'\nERROR: Step 4 failed with error: {e}')
                print('Check that Step 3 generated valid binding time data.')
    else:
        print('\nSTEP 4 was not executed.')
    
    #####################################################################
    
    # ================ SAVE CONSOLIDATED RESULTS ================
    save_consolidated_results(results_dict, metadata_dict, working_folder)
    
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
