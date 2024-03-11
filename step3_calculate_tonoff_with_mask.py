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
from auxiliary_functions import calculate_tau_on_times, manage_save_directory, fit_linear, plot_vs_time_with_hist
import scipy


plt.close("all")

##############################################################################

def calculate_kinetics(exp_time, photons_threshold, background_level, folder_main, filename, mask_level, mask_singles, nr_of_frames,
                       verbose_flag):
    # exp_time in ms
    print('\nStarting STEP 3.')

    # filepath
    folder = os.path.join(folder_main, 'kinetics_data')
    figures_folder = os.path.join(folder_main, 'figures_global')
    traces_file = os.path.join(folder, filename)
    # load data
    traces = np.loadtxt(traces_file)
    
    tons = np.array([])
    toffs = np.array([])
    tstarts = np.array([])
    SNR_all = np.array([])
    SBR_all = np.array([])
    sum_photons_all = np.array([])

    number_of_traces = int(traces.shape[1])

    
    # calculate binding times
    for i in range(number_of_traces):
    # for i in range(1):
        # For every single peak (docking sites * picks), get their trace through all frames.
        trace = traces[:, i]

        [ton, toff, binary, tstart, SNR, SBR, sum_photons] = calculate_tau_on_times(trace, photons_threshold, \
                                                                  background_level, \
                                                                  exp_time, mask_level, mask_singles, verbose_flag, i)

        if ton.any() is not False:
            tons = np.append(tons, ton)
            toffs = np.append(toffs, toff)
            tstarts = np.append(tstarts, tstart)
            SNR_all = np.append(SNR_all, SNR)
            SBR_all = np.append(SBR_all, SBR)
            sum_photons_all = np.append(sum_photons_all, sum_photons)
    
    # remove zeros
    tons = np.trim_zeros(tons)
    toffs = np.trim_zeros(toffs) 
    SNR_all = np.trim_zeros(SNR_all) 
    SBR_all = np.trim_zeros(SBR_all)

    # Plot binding time vs start time.
    figure_name = 'binding_time_vs_time'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    ax, slope, intercept = plot_vs_time_with_hist(tons, tstarts/60, order=1, fit_line=True)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Binding time [s]')
    ax.set_title(
        'Binding time (s) vs. start time (min)' + f"\nSlope: {round(slope, 3)}" + f'\nIntercept: {round(intercept, 3)}')

    # The inset_axes parameters are [left, bottom, width, height] in figure fraction.
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # PHOTONS
    ax = plot_vs_time_with_hist(sum_photons_all, tstarts/60)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Total photons [photons]')
    ax.set_title('Total photons received vs time.')

    figure_name = 'total_photons_vs_time'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # HISTOGRAM
    fig, ax = plt.subplots(1, 1)
    bin_edges = np.histogram_bin_edges(sum_photons_all, 'fd')
    # Create the marginal plot on the right of the main plot, sharing the y-axis with the main plot.
    ax.hist(sum_photons_all, bins=bin_edges)
    ax.set_xlabel('Photon count [photons]')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of photon count per binding event.')
    ax.set_yscale('log')
    figure_name = 'photon_histogram'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Print the number of events
    # print(f'The number of events were: {len(tstarts)}')
    # print(f'with {number_of_traces} traces.')
    #
    # print(f'Nr of events / (nr of traces * time) = {len(tstarts)/(number_of_traces * nr_of_frames * exp_time)}.')




    # save data

    try:
        t_on_filename = os.path.join(folder, 't_on_' + filename[:-4]+'.dat')
        t_off_filename = os.path.join(folder, 't_off_' + filename[:-4] + '.dat')
        t_start_filename = os.path.join(folder, 't_start_' + filename[:-4] + '.dat')
        snr_filename = os.path.join(folder, 'snr_' + filename[:-4] + '.dat')
        sbr_filename = os.path.join(folder, 'sbr_' + filename[:-4] + '.dat')
        sum_photons_filename = os.path.join(folder, 'sum_photons_' + filename[:-4] + '.dat')
        np.savetxt(t_on_filename, tons, fmt = '%.3f')
        np.savetxt(t_off_filename, toffs, fmt = '%.3f')
        np.savetxt(t_start_filename, tstarts, fmt = '%.3f')
        np.savetxt(snr_filename, SNR_all, fmt = '%.3f')
        np.savetxt(sbr_filename, SBR_all, fmt = '%.3f')
        np.savetxt(sum_photons_filename, sum_photons_all, fmt='%.3f')
    except:
        t_on_filename = os.path.join(folder, 't_on.dat')
        t_off_filename = os.path.join(folder, 't_off.dat')
        t_start_filename = os.path.join(folder, 't_start.dat')
        snr_filename = os.path.join(folder, 'snr.dat')
        sbr_filename = os.path.join(folder, 'sbr.dat')
        sum_photons_filename = os.path.join(folder, 'sum_photons.dat')
        np.savetxt(t_on_filename, tons, fmt = '%.3f')
        np.savetxt(t_off_filename, toffs, fmt = '%.3f')
        np.savetxt(t_start_filename, tstarts, fmt = '%.3f')
        np.savetxt(snr_filename, SNR_all, fmt = '%.3f')
        np.savetxt(sbr_filename, SBR_all, fmt = '%.3f')
        np.savetxt(sum_photons_filename, sum_photons_all, fmt='%.3f')


    # Summary of STEP 3
    print("\n------ Summary of STEP 3 ------")
    print(f"Total Events Processed: {len(tstarts)}")
    print(f"Total Traces Analyzed: {number_of_traces}")
    print(f"Events per Trace-Time Ratio: {len(tstarts) / (number_of_traces * nr_of_frames * exp_time):.3f}")

    # Slope and intercept from the fitted linear model
    print(f"Slope of Binding Time vs. Time: {slope:.3f}")
    print(f"Intercept of Binding Time vs. Time: {intercept:.3f}")

    # Confirmation of saved data
    print("t_on, t_off, t_start, SNR, and SBR data saved to respective files.")
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