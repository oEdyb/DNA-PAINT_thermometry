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
from auxiliary_functions import *
import scipy
import glob


plt.close("all")
plt.rc('font', size=20)  # controls default text sizes
plt.rc('axes', titlesize=24)  # fontsize of the axes title
plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)  # fontsize of the tick labels
plt.rc('ytick', labelsize=20)  # fontsize of the tick labels
plt.rc('legend', fontsize=12)  # legend fontsize
plt.rc('figure', titlesize=24)  # fontsize of the figure title

##############################################################################

def calculate_kinetics(exp_time, photons_threshold, background_level, photons, folder_main, filename, mask_level, mask_singles, nr_of_frames,
                       verbose_flag, photon_threshold_flag):
    # exp_time in ms
    print('\nStarting STEP 3.')

    # filepath
    folder = os.path.join(folder_main, 'kinetics_data')
    figures_folder = os.path.join(folder_main, 'figures_global')
    traces_file = os.path.join(folder, filename)
    traces_per_site_file = os.path.join(folder_main, 'per_pick\\traces\\traces per site')

    ton_per_site_path = manage_save_directory(folder, 'ton_per_site')
    toff_per_site_path = manage_save_directory(folder, 'toff_per_site')
    mean_photons_per_site_path = manage_save_directory(folder, 'mean_photons_per_site')
    std_photons_per_site_path = manage_save_directory(folder, 'std_photons_per_site')
    sum_photons_per_site_path = manage_save_directory(folder, 'sum_photons_per_site')
    photons_per_site_path = manage_save_directory(folder, 'photons_per_site')

    folders_to_remove_files_from = [ton_per_site_path, toff_per_site_path, mean_photons_per_site_path, std_photons_per_site_path,
                                    sum_photons_per_site_path, photons_per_site_path]

    for directory in folders_to_remove_files_from:
        for f in os.listdir(directory):
            file_path = os.path.join(directory, f)  # Combine directory path and file name
            if os.path.isfile(file_path):  # Ensure it's a file (not a directory)
                os.remove(file_path)  # Remove the file
            else:
                print(f"{file_path} is not a file, skipping.")

    # The following dict is a dict of dicts, pick and then site
    traces_per_pick_and_site = {}

    for file_name in os.listdir(traces_per_site_file):
        if file_name.endswith('.dat'):
            parts = file_name.split('_')
            pick_nr = parts[2]
            site_nr = parts[4]
            distance_to_NP = parts[6].split('.')[0]

            # Load the data from the file
            data = np.loadtxt(os.path.join(traces_per_site_file, file_name))

            # Store the data in a nested dictionary
            if pick_nr not in traces_per_pick_and_site:
                traces_per_pick_and_site[pick_nr] = {}
            traces_per_pick_and_site[pick_nr][site_nr] = {
                'distance_to_NP': distance_to_NP,
                'data': data
            }
    # load data
    traces = np.loadtxt(traces_file)
    
    tons = np.array([])
    toffs = np.array([])
    tstarts = np.array([])
    SNR_all = np.array([])
    SBR_all = np.array([])
    sum_photons_all = np.array([])
    tstart_SNR_all = np.array([])
    double_event_count_all = np.array([])
    avg_photons_all = np.array([])
    ton_dict = {}
    number_of_traces = int(traces.shape[1])
    photon_intensity_all = np.array([])
    std_photons_all = np.array([])


    photon_mean = np.mean(photons)
    photon_std = np.std(photons)
    photons_plot = photons[photons < photon_mean + 4 * photon_std]
    bin_edges_photons = np.histogram_bin_edges(photons_plot, 'fd')
    # Fit Gaussian to photons per localization in order to filter.
    photon_threshold_flag = False
    if photon_threshold_flag:


        def gaussian_1d(x, a, mu, sigma):
            return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        counts, bins = np.histogram(photons_plot, bins=bin_edges_photons)
        middle_of_bins = bins - (bins[1]-bins[0])/2
        popt = scipy.optimize.curve_fit(gaussian_1d, middle_of_bins[1:], counts, p0=[np.max(photons, axis=None), photon_mean, photon_std])
        photons_threshold = popt[0][1] - popt[0][2]
    else:
        photons_threshold = 0.02
    # calculate binding times
    for i in range(number_of_traces):
    # for i in range(1):
        # For every single peak (docking sites * picks), get their trace through all frames.
        trace = traces[:, i]

        [ton, toff, binary, tstart, SNR, SBR, sum_photons, mean_photons, photon_intensity, std_photons, tstart_SNR, double_event_count] = calculate_tau_on_times(trace, photons_threshold, \
                                                                  background_level, \
                                                                  exp_time, mask_level, mask_singles, verbose_flag, i)

        if ton.any() is not False:
            # ton_dict[f'{i}'] = ton
            tons = np.append(tons, ton)
            toffs = np.append(toffs, toff)
            tstarts = np.append(tstarts, tstart)
            SNR_all = np.append(SNR_all, SNR)
            SBR_all = np.append(SBR_all, SBR)
            avg_photons_all = np.append(avg_photons_all, mean_photons)
            sum_photons_all = np.append(sum_photons_all, sum_photons)
            tstart_SNR_all = np.append(tstart_SNR_all, tstart_SNR)
            double_event_count_all = np.append(double_event_count_all, double_event_count)
            photon_intensity_all = np.append(photon_intensity_all, photon_intensity)
            std_photons_all = np.append(std_photons_all, std_photons)
    
    # remove zeros
    tons = np.trim_zeros(tons)
    toffs = np.trim_zeros(toffs)

    filter_indices = np.logical_and.reduce((
        ~np.isnan(SNR_all),
        ~np.isnan(SBR_all),
        ~np.isinf(SNR_all),
        ~np.isinf(SBR_all)
    ))

    # Apply the filter
    tstart_SNR_all = tstart_SNR_all[filter_indices]
    SNR_all = SNR_all[filter_indices]
    SBR_all = SBR_all[filter_indices]

    # PER SITE

    for pick_key in traces_per_pick_and_site.keys():
        for site_key in traces_per_pick_and_site[pick_key].keys():
            trace = traces_per_pick_and_site[pick_key][site_key]['data']

            [ton, toff, binary, tstart, SNR, SBR, sum_photons, mean_photons, photon_intensity, std_photons, tstart_SNR, double_event_count] = calculate_tau_on_times(
                trace, photons_threshold, \
                background_level, \
                exp_time, mask_level, mask_singles, verbose_flag, i)

            if ton.any() is not False:
                # ton_dict[f'{i}'] = ton
                # tons = np.append(tons, ton)
                # toffs = np.append(toffs, toff)
                # tstarts = np.append(tstarts, tstart)
                # SNR_all = np.append(SNR_all, SNR)
                # avg_photons_all = np.append(avg_photons_all, mean_photons)
                # SBR_all = np.append(SBR_all, SBR)
                # sum_photons_all = np.append(sum_photons_all, sum_photons)
                # photon_intensity_all = np.append(photon_intensity_all, photon_intensity)
                # tstart_SNR_all = np.append(tstart_SNR_all, tstart_SNR)
                # double_event_count_all = np.append(double_event_count_all, double_event_count)

                ton = np.trim_zeros(ton)
                distance_to_NP_current = traces_per_pick_and_site[pick_key][site_key]['distance_to_NP']
                try:
                    int(distance_to_NP_current)
                except:
                    distance_to_NP_current = 9999999
                mean_photons_per_site_filename = os.path.join(mean_photons_per_site_path,
                                                     f'meanphotons_pick_{int(pick_key)}_site_{int(site_key)}_dist_{int(distance_to_NP_current)}.dat')
                std_photons_per_site_filename = os.path.join(std_photons_per_site_path,
                                                              f'stdphotons_pick_{int(pick_key)}_site_{int(site_key)}_dist_{int(distance_to_NP_current)}.dat')
                photons_per_site_filename = os.path.join(photons_per_site_path,
                                                              f'photons_pick_{int(pick_key)}_site_{int(site_key)}_dist_{int(distance_to_NP_current)}.dat')
                sum_photons_per_site_filename = os.path.join(sum_photons_per_site_path,
                                                              f'sumphotons_pick_{int(pick_key)}_site_{int(site_key)}_dist_{int(distance_to_NP_current)}.dat')
                ton_per_site_filename = os.path.join(ton_per_site_path, f'ton_pick_{int(pick_key)}_site_{int(site_key)}_dist_{int(distance_to_NP_current)}.dat')
                toff_per_site_filename = os.path.join(toff_per_site_path,
                                                     f'toff_pick_{int(pick_key)}_site_{int(site_key)}_dist_{int(distance_to_NP_current)}.dat')

                np.savetxt(ton_per_site_filename, ton, fmt='%.3f')
                np.savetxt(toff_per_site_filename, toff, fmt='%.3f')
                np.savetxt(mean_photons_per_site_filename, mean_photons, fmt='%.3f')
                np.savetxt(std_photons_per_site_filename, std_photons, fmt='%.3f')
                np.savetxt(sum_photons_per_site_filename, sum_photons, fmt='%.3f')
                np.savetxt(photons_per_site_filename, photon_intensity, fmt='%.3f')


    # remove zeros
    tons = np.trim_zeros(tons)
    toffs = np.trim_zeros(toffs)

    filter_indices = np.logical_and.reduce((
        ~np.isnan(SNR_all),
        ~np.isnan(SBR_all),
        ~np.isinf(SNR_all),
        ~np.isinf(SBR_all)
    ))

    # Apply the filter
    tstart_SNR_all = tstart_SNR_all[filter_indices]
    SNR_all = SNR_all[filter_indices]
    SBR_all = SBR_all[filter_indices]

    # Plot binding time vs start time.
    figure_name = 'binding_time_vs_time'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    ax, slope, intercept = plot_vs_time_with_hist(tons, tstarts/60, order=1, fit_line=True)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Binding time [s]')
    ax.set_title(
        'Binding time (s) vs. start time (min)' + f"\nSlope: {round(slope, 3)}" + f'\nIntercept: {round(intercept, 3)}')

    # The inset_axes parameters are [left, bottom, width, height] in figure fraction.
    plt.tight_layout()
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
    plt.tight_layout()
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
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # HISTOGRAM for PHOTONS per localization
    fig, ax = plt.subplots(1, 1)
    bin_edges_photons = np.histogram_bin_edges(photons_plot, 'fd')
    # Create the marginal plot on the right of the main plot, sharing the y-axis with the main plot.
    ax.hist(photons_plot, bins=bin_edges_photons)
    if photon_threshold_flag:
        y_gaussian = gaussian_1d(middle_of_bins, *popt[0])
        ax.plot(middle_of_bins, y_gaussian, label='Gaussian fit')
        plt.axvline(x=photons_threshold, color='r', linestyle='--', label='Threshold')
        plt.legend(loc='best')
    ax.set_xlabel('Photon count [photons]')
    ax.set_ylabel('Counts')
    ax.set_title('Photons per localization.')
    # Compute and plot mean
    median = np.median(photons_plot)
    ax.axvline(median, color='red', linestyle='dashed', linewidth=1)
    ax.text(median + 20, ax.get_ylim()[1] * 0.8, f'Median: {median:.2f}', color='red')

    figure_name = 'photons_localization'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Plots of SNR and SBR
    fig, ax = plt.subplots(1, 1)
    ax.scatter(tstart_SNR_all/60, SNR_all, s=0.85, alpha=0.6, label='SNR')
    ax.scatter(tstart_SNR_all / 60, SBR_all, s=0.85, alpha=0.6, label='SBR')
    ax.set_yscale('log')
    ax.set_xlabel('Time [min]')
    ax.set_title('Log of SNR & SBR over time.')
    plt.legend()
    figure_name = 'SNR_SBR_scatter_plot'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



    # Print the number of events
    # print(f'The number of events were: {len(tstarts)}')
    # print(f'with {number_of_traces} traces.')
    #
    # print(f'Nr of events / (nr of traces * time) = {len(tstarts)/(number_of_traces * nr_of_frames * exp_time)}.')




    # save data
    t_on_filename = os.path.join(folder, 't_on.dat')
    t_off_filename = os.path.join(folder, 't_off.dat')
    t_start_filename = os.path.join(folder, 't_start.dat')
    snr_filename = os.path.join(folder, 'snr.dat')
    sbr_filename = os.path.join(folder, 'sbr.dat')
    sum_photons_filename = os.path.join(folder, 'sum_photons.dat')
    photons_filename = os.path.join(folder, 'photons.dat')
    std_photons_filename = os.path.join(folder, 'std_photons.dat')
    double_event_filename = os.path.join(folder, 'double_event_count.dat')
    np.savetxt(t_on_filename, tons, fmt = '%.3f')
    np.savetxt(t_off_filename, toffs, fmt = '%.3f')
    np.savetxt(t_start_filename, tstarts, fmt = '%.3f')
    np.savetxt(snr_filename, SNR_all, fmt = '%.3f')
    np.savetxt(sbr_filename, SBR_all, fmt = '%.3f')
    np.savetxt(sum_photons_filename, sum_photons_all, fmt='%.3f')
    np.savetxt(std_photons_filename, std_photons_all, fmt='%.3f')
    np.savetxt(photons_filename, photon_intensity_all, fmt='%.3f')
    np.savetxt(double_event_filename, double_event_count_all, fmt='%.3f')


    # Summary of STEP 3
    print("\n------ Summary of STEP 3 ------")
    print(f"Total Events Processed: {len(tstarts)}")
    print(f"Total Traces Analyzed: {number_of_traces}")
    print(f"Mask level: {mask_level}")
    print(f"Events per Trace-Time Ratio: {len(tstarts) / (number_of_traces * nr_of_frames * exp_time):.3f}")

    # Slope and intercept from the fitted linear model
    print(f"Slope of Binding Time vs. Time: {slope:.3f}")
    print(f"Intercept of Binding Time vs. Time: {intercept:.3f}")
    print(f"SNR: {np.mean(SNR_all, axis=None)}")
    print(f"SBR: {np.mean(SBR_all, axis=None)}")
    print(f"Double events per event: {np.sum(double_event_count_all)/len(double_event_count_all)}")


    update_pkl(folder_main, 'SNR', np.mean(SNR_all, axis=None))
    update_pkl(folder_main, 'SBR', np.mean(SBR_all, axis=None))
    update_pkl(folder_main, 'Events', len(tstarts))
    update_pkl(folder_main, 'Nr of traces', number_of_traces)
    update_pkl(folder_main, 'Events per Trace-Time Ratio', len(tstarts) / (number_of_traces * nr_of_frames * exp_time))
    update_pkl(folder_main, 'Double Events', double_event_count_all)
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