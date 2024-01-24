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

#####################################################################

def run_analysis(selected_file, working_folder):
    # selected_file is a dat file
        
    # INPUTS for STEP 1    
    # if picks are rectangles set TRUE
    rectangles_flag = False
    # set TRUE to run the script for all the files inside the selected folder
    recursive_flag = False
    
    # INPUTS for STEP 2
    # docking sites per origami
    docking_sites = 1
    # is there any NP (hybridized structure)
    NP_flag = False
    # camera pixel size
    pixel_size = 0.130 # in um
    # size of the pick used in picasso
    pick_size = 0.75 # in camera pixels (put the same number used in Picasso)
    # size of the pick to include locs around the detected peaks
    radius_of_pick_to_average = 0.75 # in camera pixel size
    # set an intensity threshold to avoid dumb peak detection in the background
    # this threshold is arbitrary, don't worry about this parameter, the code 
    # change it automatically to detect the number of docking sites set above
    th = 1
    # time parameters
    # number_of_frames = 7375 # for PPT
    number_of_frames = 7646 # for COT
    exp_time = 0.1 # in s
    plot_flag = True
    hyper_exponential_flag = False

    # INPUTS for STEP 3
    # photon threshold to identify a binding event
    # should be lager than the background
    photons_threshold = 300
    background_level = 150 # in photons
    mask_level = 2
    
    # INPUTS for STEP 4
    # estimation of binding times
    rango = [0, 13]
    tau_long_init = 5 # in s
    tau_short_init = 0.1 # in s, not relevant for mono
    set_ratio = 1 # long/short or amplitude if mono
    initial_params = [tau_long_init, tau_short_init, set_ratio]
    opt_display_flag = True
    likelihood_err_param = 2 # corresponds to a 95% likelihood interval

    # WHICH STEPS DO YOU WANNA RUN?
    # do_step1, do_step2, do_step3, do_step4 = 1,0,0,0
    do_step1, do_step2, do_step3, do_step4 = 0,0,0,1
    # do_step1, do_step2, do_step3, do_step4 = 1,1,1,1
    # do_step1, do_step2, do_step3, do_step4 = 0,0,0,1

    #####################################################################
    #####################################################################
    #####################################################################
    
    if do_step1:
        # run step
        step1.split_hdf5(selected_file, working_folder, recursive_flag, rectangles_flag)
    else:
        print('\nSTEP 1 was not executed.')
        
    #####################################################################
    
    # folder and file management
    step2_working_folder = os.path.join(working_folder, 'split_data')
    if do_step2:
        # run step
        step2.process_dat_files(number_of_frames, exp_time, step2_working_folder, \
                              docking_sites, NP_flag, pixel_size, pick_size, \
                              radius_of_pick_to_average, th, plot_flag)
    else:
        print('\nSTEP 2 was not executed.')
        
    #####################################################################
        
    # folder and file management
    step3_working_folder = os.path.join(step2_working_folder, 'kinetics_data')
    if do_step3:
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
    if do_step4:
        # run step
        step4.estimate_binding_unbinding_times(exp_time, rango, step4_working_folder, \
                                        initial_params, likelihood_err_param, \
                                        opt_display_flag, hyper_exponential_flag)
    else:
        print('\nSTEP 4 was not executed.')
    
    #####################################################################
    
    print('\nProcess done.')
    
    return

#####################################################################
#####################################################################
#####################################################################

if __name__ == '__main__':

    # load and open folder and file
    base_folder = 'C:\\datos_mariano\\posdoc\\unifr\\smartphone_project\\buffer_performance'
    root = tk.Tk()
    selected_file = fd.askopenfilename(initialdir = base_folder, \
                                       filetypes=(("", "*.hdf5") , ("", "*.")))
    root.withdraw()
    working_folder = os.path.dirname(selected_file)
    
    print('\nProcessing %s' % working_folder)

    run_analysis(selected_file, working_folder)
