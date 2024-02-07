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
# from auxiliary_functions import save_parameters
import step1_extract_and_save_data_from_hdf5_picasso_files as step1
import step2_process_picasso_extracted_data as step2
import step3_calculate_tonoff_with_mask as step3
import step4_estimate_binding_time_using_MLE as step4
import time
import argparse
from ast import literal_eval

#####################################################################

def run_analysis(selected_file, working_folder, step, params):
    number_of_frames = params.get('number_of_frames', 0)
    exp_time = params.get('exposure_time', 0)
    docking_sites = params.get('docking_sites', 0)
    NP_flag = params.get('checkNP', 0)
    pixel_size = params.get('pixel_size', 0)
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


    if step[0] == 'True':
        # run step
        step1.split_hdf5(selected_file, working_folder, recursive_flag, rectangles_flag)
    else:
        print('\nSTEP 1 was not executed.')
        
    #####################################################################
    
    # folder and file management
    step2_working_folder = os.path.join(working_folder, 'split_data')
    if step[1] == 'True':
        # run step
        step2.process_dat_files(number_of_frames, exp_time, step2_working_folder, \
                              docking_sites, NP_flag, pixel_size, pick_size, \
                              radius_of_pick_to_average, th, plot_flag)
    else:
        print('\nSTEP 2 was not executed.')
        
    #####################################################################
        
    # folder and file management
    step3_working_folder = os.path.join(step2_working_folder, 'kinetics_data')
    if step[2] == 'True':
        # run step
        list_of_files_step3 = os.listdir(step3_working_folder)
        all_traces_filename = [f for f in list_of_files_step3 if re.search('^TRACES_ALL_',f)][0]
        step3.calculate_kinetics(exp_time, photons_threshold, background_level, \
                                 step3_working_folder, \
                                 all_traces_filename, mask_level)
    else:
        print('\nSTEP 3 was not executed.')
    
    #####################################################################
        
    # folder and file management
    step4_working_folder = step3_working_folder
    if step[3] == 'True':
        # run step
        step4.estimate_binding_unbinding_times(exp_time, rango, step4_working_folder,
                                        initial_params, likelihood_err_param,
                                        opt_display_flag, hyper_exponential_flag)
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
    run_analysis(args.selected_file, args.working_folder, args.step, params)
    print("Total execution time: --- %s seconds ---" % (time.time() - start_time))
