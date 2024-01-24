# -*- coding: utf-8 -*-
"""
Created on Fri July 21 2023

@author: Mariano Barella

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
import tkinter.filedialog as fd
from auxiliary_functions import calculate_tau_on_times

plt.close("all")

##############################################################################

def calculate_kinetics(exp_time, photons_threshold, background_level, folder, filename, mask_level):
    # exp_time in ms
    print('\nStarting STEP 3.')

    # filepath
    traces_file = os.path.join(folder, filename)
    # load data
    traces = np.loadtxt(traces_file)
    
    tons = np.array([])
    toffs = np.array([])
    tstarts = np.array([])
    SNR_all = np.array([])
    SBR_all = np.array([])
    number_of_traces = int(traces.shape[1])
    
    # calculate binding times
    for i in range(number_of_traces):
    # for i in range(1):
        trace = traces[:,i]
        [ton, toff, binary, tstart, SNR, SBR] = calculate_tau_on_times(trace, photons_threshold, \
                                                                  background_level, \
                                                                  exp_time, mask_level)     
        tons = np.append(tons, ton)
        toffs = np.append(toffs, toff)
        tstarts = np.append(tstarts, tstart)
        SNR_all = np.append(SNR_all, SNR)
        SBR_all = np.append(SBR_all, SBR)
    
    # remove zeros
    tons = np.trim_zeros(tons)
    toffs = np.trim_zeros(toffs) 
    SNR_all = np.trim_zeros(SNR_all) 
    SBR_all = np.trim_zeros(SBR_all) 
     
    # save data
    
    t_on_filename = os.path.join(folder, 't_on_' + filename[:-4]+'.dat')
    t_off_filename = os.path.join(folder, 't_off_' + filename[:-4] + '.dat')
    t_start_filename = os.path.join(folder, 't_start_' + filename[:-4] + '.dat')
    snr_filename = os.path.join(folder, 'snr_' + filename[:-4] + '.dat')
    sbr_filename = os.path.join(folder, 'sbr_' + filename[:-4] + '.dat')
    np.savetxt(t_on_filename, tons, fmt = '%.3f')
    np.savetxt(t_off_filename, toffs, fmt = '%.3f')
    np.savetxt(t_start_filename, tstarts, fmt = '%.3f')
    np.savetxt(snr_filename, SNR_all, fmt = '%.3f')
    np.savetxt(sbr_filename, SBR_all, fmt = '%.3f')
    
    print('\nt_on, t_off, t_start, SNR, and SBR data saved.')
    
    print('\nTotal number of t_on data points', len(tons))
    
    print('\nDone with STEP 3.')
    
    return

##############----------------------###############-------------------
##############----------------------###############-------------------
##############----------------------###############-------------------


if __name__ == '__main__':

    # load and open folder and file
    base_folder = 'C:\\datos_mariano\\posdoc\\unifr\\DNA-PAINT_nanothermometry\\data_fribourg'
    root = tk.Tk()
    selected_file = fd.askopenfilename(initialdir = base_folder, \
                                       filetypes=(("", "*.dat"), ("", "*.")))
    root.withdraw()
    folder = os.path.dirname(selected_file)
    filename = os.path.basename(selected_file)
    
    # time parameters☺
    photons_threshold = 50
    exp_time = 0.1 # in s
    mask_level = 2
    background_level = 150 # in photons 
    
    calculate_kinetics(exp_time, photons_threshold, background_level, folder, filename, mask_level)