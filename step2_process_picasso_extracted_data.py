# -*- coding: utf-8 -*-
"""
Created on Tuesday April 4th of 2023

@author: Mariano Barella

Version 3. Changes:
    - new flag that select if structures are origamis or hybridized structures
    - automatically selects the best threshold of the peak finding algorithm

This script analyzes already-processed Picasso data. It opens .dat files that 
were generated with "extract_and_save_data_from_hdf5_picasso_files.py".

When the program starts select ANY .dat file. This action will determine the 
working folder.

As input it uses:
    - main folder
    - number of frames
    - exposure time
    - if NP is present (hybridized structure)
    - pixel size of the original video
    - size of the pick you used in picasso analysis pipeline
    - desired radius of analysis to average localization position
    - number of docking sites you are looking for (defined by origami design)
    
Outputs are:
    - plots per pick (scatter plot of locs, fine and coarse 2D histograms,
                      binary image showing center of docking sites,
                      matrix of relative distances, matrix of localization precision)
    - traces per pick
    - a single file with ALL traces of the super-resolved image
    - global figures (photons vs time, localizations vs time and background vs time)

Warning: the program is coded to follow filename convention of the script
"extract_and_save_data_from_hdf5_picasso_files.py".

"""
import os
os.environ["OMP_NUM_THREADS"] = '2'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Circle as plot_circle
import tkinter as tk
import tkinter.filedialog as fd
import re
from auxiliary_functions import detect_peaks, distance, fit_linear, \
    perpendicular_distance, manage_save_directory
from sklearn.mixture import GaussianMixture
import time
from auxiliary_functions_gaussian import plot_gaussian_2d
    
plt.ioff()
plt.close("all")
cmap = plt.cm.get_cmap('viridis')
bkg_color = cmap(0)

##############################################################################

def process_dat_files(number_of_frames, exp_time, working_folder,
                      docking_sites, NP_flag, pixel_size, pick_size, 
                      radius_of_pick_to_average, th, plot_flag, verbose_flag):
    
    print('\nStarting STEP 2.')
    
    total_time_sec = number_of_frames*exp_time # in sec
    total_time_min = total_time_sec/60 # in min
    print('Total time %.1f min' % total_time_min)
        
    # create folder to save data
    # global figures folder
    figures_folder = manage_save_directory(working_folder, 'figures_global')
    # figures per pick folder
    per_pick_folder = os.path.join(working_folder, 'per_pick')
    figures_per_pick_folder = manage_save_directory(per_pick_folder, 'figures')
    traces_per_pick_folder = manage_save_directory(per_pick_folder, 'traces')
    
    kinetics_folder = manage_save_directory(working_folder, 'kinetics_data')
    gaussian_folder = manage_save_directory(kinetics_folder, 'gaussian_data')

    # list files
    list_of_files = os.listdir(working_folder)
    list_of_files = [f for f in list_of_files if re.search('.dat',f)]
    list_of_files.sort()
    if NP_flag:
        list_of_files_origami = [f for f in list_of_files if re.search('COMBINED',f)]
        list_of_files_NP = [f for f in list_of_files if re.search('Combined Stacks',f)]
    else:
        list_of_files_origami = list_of_files
    
    ##############################################################################
    # load data
    
    # frame number, used for time estimation
    frame_file = [f for f in list_of_files_origami if re.search('_frame',f)][0]
    frame_filepath = os.path.join(working_folder, frame_file)
    frame = np.loadtxt(frame_filepath)
    
    # photons
    photons_file = [f for f in list_of_files_origami if re.search('_photons',f)][0]
    photons_filepath = os.path.join(working_folder, photons_file)
    photons = np.loadtxt(photons_filepath)
    
    # bkg
    bkg_file = [f for f in list_of_files_origami if re.search('_bkg',f)][0]
    bkg_filepath = os.path.join(working_folder, bkg_file)
    bkg = np.loadtxt(bkg_filepath)
    
    # xy positions
    # origami
    position_file = [f for f in list_of_files_origami if re.search('_xy',f)][0]
    position_filepath = os.path.join(working_folder, position_file)
    position = np.loadtxt(position_filepath)
    x = position[:,0]*pixel_size
    y = position[:,1]*pixel_size
    # NP
    if NP_flag:
        position_file_NP = [f for f in list_of_files_NP if re.search('_xy',f)][0]
        position_filepath_NP = os.path.join(working_folder, position_file_NP)
        xy_NP = np.loadtxt(position_filepath_NP)
        x_NP = xy_NP[:,0]*pixel_size
        y_NP = xy_NP[:,1]*pixel_size
    
    # number of pick
    # origami
    pick_file = [f for f in list_of_files_origami if re.search('_pick_number',f)][0]
    pick_filepath = os.path.join(working_folder, pick_file)
    pick_list = np.loadtxt(pick_filepath)
    # NP
    if NP_flag:
        pick_file_NP = [f for f in list_of_files_NP if re.search('_pick_number',f)][0]
        pick_filepath_NP = os.path.join(working_folder, pick_file_NP)
        pick_list_NP = np.loadtxt(pick_filepath_NP)
    
    ##############################################################################
    
    # how many picks?
    pick_number = np.unique(pick_list)
    total_number_of_picks = len(pick_number)
    print('Total picks', total_number_of_picks)
    
    # allocate arrays for statistics
    locs_of_picked = np.zeros(total_number_of_picks)
    # number of bins for temporal binning
    number_of_bins = 60
    locs_of_picked_vs_time = np.zeros([total_number_of_picks, number_of_bins])
    photons_concat = np.array([])
    bkg_concat = np.array([])
    frame_concat = np.array([])
    positions_concat_NP = np.array([])
    positions_concat_origami = np.array([])
    gmm_stds = []
    gmm_stds_x = np.array([])
    gmm_stds_y = np.array([])
    all_traces = np.zeros(number_of_frames)
    
    # set number of bins for FINE histograming 
    N = int(2*pick_size*pixel_size*1000/10)
    hist_2D_bin_size = pixel_size*1000*pick_size/N # this should be around 5 nm
    print('2D histogram bin size', hist_2D_bin_size, 'nm')
    
    ########################################################################
    ########################################################################
    ########################################################################
    
    # data assignment per pick
    for i in range(total_number_of_picks):
        pick_id = pick_number[i]
        if verbose_flag:
            print('\n---------- Pick number %d of %d\n' % (i+1, total_number_of_picks))

        index_picked = np.where(pick_list == pick_id)
        # for origami
        frame_of_picked = frame[index_picked]
        photons_of_picked = photons[index_picked]
        bkg_of_picked = bkg[index_picked]
        x_position_of_picked = x[index_picked]
        y_position_of_picked = y[index_picked]
        # make FINE 2D histogram of locs
        x_min = min(x_position_of_picked)
        y_min = min(y_position_of_picked)
        x_max = x_min + pick_size*pixel_size
        y_max = y_min + pick_size*pixel_size
        z_hist, x_hist, y_hist = np.histogram2d(x_position_of_picked, 
                                                y_position_of_picked, 
                                                bins = N, 
                                                range = [[x_min, x_max], \
                                                         [y_min, y_max]])
        # Histogram does not follow Cartesian convention (see Notes),
        # therefore transpose z_hist for visualization purposes.
        z_hist = z_hist.T
        x_hist_step = np.diff(x_hist)
        y_hist_step = np.diff(y_hist)
        x_hist_centers = x_hist[:-1] + x_hist_step/2
        y_hist_centers = y_hist[:-1] + y_hist_step/2
        
        total_peaks_found = 0
        threshold_COARSE = th
        if docking_sites == 1:
            bins_COARSE = 1
        else:
            bins_COARSE = 20
        while total_peaks_found != docking_sites:
            # make COARSE 2D histogram of locs
            # number of bins is arbitrary, determined after trial and error
            x_min = min(x_position_of_picked)
            y_min = min(y_position_of_picked)
            x_max = x_min + pick_size*pixel_size
            y_max = y_min + pick_size*pixel_size
            z_hist_COARSE, x_hist_COARSE, y_hist_COARSE = np.histogram2d(x_position_of_picked, 
                                                                         y_position_of_picked, 
                                                                         bins = bins_COARSE,
                                                                         range = [[x_min, x_max], \
                                                                                  [y_min, y_max]],
                                                                         density = True)
            z_hist_COARSE = z_hist_COARSE.T
            x_hist_step_COARSE = np.diff(x_hist_COARSE)
            y_hist_step_COARSE = np.diff(y_hist_COARSE)
            x_hist_COARSE_centers = x_hist_COARSE[:-1] + x_hist_step_COARSE/2
            y_hist_COARSE_centers = y_hist_COARSE[:-1] + y_hist_step_COARSE/2
            z_hist_COARSE = np.where(z_hist_COARSE < threshold_COARSE, 0, z_hist_COARSE)
            
            # peak detection for Center of Mass localization
            detected_peaks = detect_peaks(z_hist_COARSE)
            # find Center of Mass of locs near the peaks that were found
            index_peaks = np.where(detected_peaks == True) # this is a tuple
            total_peaks_found = len(index_peaks[0])
            threshold_COARSE += 5
            if threshold_COARSE > 5000:
                # this MAX value is arbitrary
                break
        if docking_sites != 1:
            if verbose_flag:
                print('threshold_COARSE reached', threshold_COARSE)
                print(total_peaks_found, 'total peaks found\n')
        if total_peaks_found == 0:
            peaks_flag = False
        else:
            peaks_flag = True
            
        analysis_radius = radius_of_pick_to_average*pixel_size
        cm_binding_sites_x = np.array([])
        cm_binding_sites_y = np.array([])
        cm_std_dev_binding_sites_x = np.array([])
        cm_std_dev_binding_sites_y = np.array([])
        # array where traces are going to be saved
        all_traces_per_pick = np.zeros(number_of_frames)
        inv_cov_init = []
        for j in range(total_peaks_found):
            if docking_sites != 1:
                if verbose_flag:
                    print('Binding site %d of %d' % (j+1, total_peaks_found))
            index_x_peak = index_peaks[1][j] # first element of the tuple are rows
            index_y_peak = index_peaks[0][j] # second element of the tuple are columns
            x_peak = x_hist_COARSE_centers[index_x_peak]
            y_peak = y_hist_COARSE_centers[index_y_peak]
            # grab all locs inside the selected circle,
            # circle = selected radius around the (x,y) of the detected peak
            # 1) calculate distance of all locs with respect to the (x,y) of the peak
            d = distance(x_position_of_picked, y_position_of_picked, x_peak, y_peak)
            # 2) filter by the radius
            index_inside_radius = np.where(d < analysis_radius)
            x_position_of_picked_filtered = x_position_of_picked[index_inside_radius]
            y_position_of_picked_filtered = y_position_of_picked[index_inside_radius]
            # 3) calculate average position of the binding site
            cm_binding_site_x = np.mean(x_position_of_picked_filtered)
            cm_binding_site_y = np.mean(y_position_of_picked_filtered)
            cm_std_dev_binding_site_x = np.std(x_position_of_picked_filtered, ddof = 1)
            cm_std_dev_binding_site_y = np.std(y_position_of_picked_filtered, ddof = 1)
            # print('CM binding in nm (x y): %.3f %.3f' % (cm_binding_site_x*1e3, \
            #       cm_binding_site_y*1e3))
            # print('std dev CM binding in nm (x y): %.3f %.3f' % (cm_std_dev_binding_site_x*1e3, \
            #       cm_std_dev_binding_site_y*1e3))
            # 4) save the averaged position in a new array
            cm_binding_sites_x = np.append(cm_binding_sites_x, cm_binding_site_x)
            cm_binding_sites_y = np.append(cm_binding_sites_y, cm_binding_site_y)
            cm_std_dev_binding_sites_x = np.append(cm_std_dev_binding_sites_x, cm_std_dev_binding_site_x)
            cm_std_dev_binding_sites_y = np.append(cm_std_dev_binding_sites_y, cm_std_dev_binding_site_y)
            # 5) export the trace of the binding site
            frame_of_picked_filtered = np.array(frame_of_picked[index_inside_radius], dtype=int)
            photons_of_picked_filtered = photons_of_picked[index_inside_radius]
            empty_trace = np.zeros(number_of_frames)
            empty_trace[frame_of_picked_filtered] = photons_of_picked_filtered
            trace = empty_trace
            # compile traces of the pick in one array
            all_traces_per_pick = np.vstack([all_traces_per_pick, trace])
            # compile all traces of the image in one array
            all_traces = np.vstack([all_traces, trace])

            # First guess of inverse of cov matrix for GMM
            inv_cov_init.append(np.linalg.inv(np.cov(np.array([x_position_of_picked_filtered, y_position_of_picked_filtered]))))
        # delete first fake and empty trace (needed to make the proper array)
        all_traces_per_pick = np.delete(all_traces, 0, axis = 0)
        all_traces_per_pick = all_traces_per_pick.T
        # save traces per pick
        if peaks_flag:
            new_filename = 'TRACE_pick_%02d_%s.dat' % (i, frame_file[:-10])
            new_filepath = os.path.join(traces_per_pick_folder, new_filename)
            np.savetxt(new_filepath, all_traces_per_pick, fmt='%05d')
        
        # get NP coords in um
        if NP_flag:
            index_picked_NP = np.where(pick_list_NP == pick_id)
            x_position_of_picked_NP = x_NP[index_picked_NP]
            y_position_of_picked_NP = y_NP[index_picked_NP]
            x_avg_NP = np.mean(x_position_of_picked_NP)
            y_avg_NP = np.mean(y_position_of_picked_NP)
            x_std_dev_NP = np.std(x_position_of_picked_NP, ddof = 1)
            y_std_dev_NP = np.std(y_position_of_picked_NP, ddof = 1)
            # print them in nm
            # print('\nCM NP in nm (x y): %.3f %.3f' % (x_avg_NP*1e3, y_avg_NP*1e3))
            # print('std dev CM NP in nm (x y): %.3f %.3f' % (x_std_dev_NP*1e3, y_std_dev_NP*1e3))
        
        # fit linear (origami direction) of the binding sites 
        # to find the perpendicular distance to the NP
        x_fitted, y_fitted, slope, intercept, Rsquared = fit_linear(cm_binding_sites_x, 
                                                                    cm_binding_sites_y)
        # distance between NP and the line fitted by the three binding sites
        if NP_flag:
            distance_to_NP = perpendicular_distance(slope, intercept, 
                                                    x_avg_NP, y_avg_NP)
            distance_to_NP_nm = distance_to_NP*1e3
            # print('Perpendicular distance to NP: %.1f nm' % distance_to_NP_nm)
        # TODO: Fix good distance value or add as parameter.
        perpendicular_dist_of_picked = perpendicular_distance(slope, intercept, x_position_of_picked, y_position_of_picked)
        # Filtering the localizations based on the perpendicular distance from the fitted line.
        # Relax the distance a little when Rsquared is large.
        filter_dist = 40e-3/Rsquared  # Arbitrary at the moment.
        x_filtered_perpendicular = x_position_of_picked[perpendicular_dist_of_picked < filter_dist]
        y_filtered_perpendicular = y_position_of_picked[perpendicular_dist_of_picked < filter_dist]

        new_filename = 'xy_perpendicular_filtered_' + str(i) + '.dat'
        new_filepath = os.path.join(gaussian_folder, new_filename)
        np.savetxt(new_filepath, np.array([x_filtered_perpendicular, y_filtered_perpendicular]).T)


        # calculate relative distances between all points
        # ------------------ in nanometers -----------------------
        # allocate: total size = number of detected peaks + 1 for NP
        matrix_distance = np.zeros([total_peaks_found + 1, total_peaks_found + 1])
        matrix_std_dev = np.zeros([total_peaks_found + 1, total_peaks_found + 1])
        # calcualte first row of the matrix distance
        if NP_flag:
            for j in range(total_peaks_found):
                x_binding = cm_binding_sites_x[j]
                y_binding = cm_binding_sites_y[j]
                distance_between_locs_CM = distance(x_binding, y_binding, x_avg_NP, y_avg_NP)*1e3
                matrix_distance[0, j + 1] = distance_between_locs_CM
                matrix_distance[j + 1, 0] = distance_between_locs_CM
            matrix_std_dev[0, 0] = max(x_std_dev_NP, y_std_dev_NP)*1e3
            positions_concat_NP = np.append(positions_concat_NP, distance_between_locs_CM)
        # calcualte the rest of the rows of the matrix distance
        peak_distances = np.array([])
        for j in range(total_peaks_found):
            x_binding_row = cm_binding_sites_x[j]
            y_binding_row = cm_binding_sites_y[j]
            matrix_std_dev[j + 1, j + 1] = max(cm_std_dev_binding_sites_x[j], \
                                               cm_std_dev_binding_sites_y[j])*1e3
            for k in range(j + 1, total_peaks_found):
                x_binding_col = cm_binding_sites_x[k]
                y_binding_col = cm_binding_sites_y[k]
                distance_between_locs_CM = distance(x_binding_col, y_binding_col, \
                                                  x_binding_row, y_binding_row)*1e3
                matrix_distance[j + 1, k + 1] = distance_between_locs_CM
                matrix_distance[k + 1, j + 1] = distance_between_locs_CM
                peak_distances = np.append(peak_distances, distance_between_locs_CM)
                positions_concat_origami = np.append(positions_concat_origami, distance_between_locs_CM)

        # Assigning peak labels using the distances between the peaks.
        peak_mean_distance = np.zeros([total_peaks_found])
        for l in range(1, total_peaks_found+1):
            peak_mean_distance[l-1] = np.mean(matrix_distance[l, :])
        ascending_index = peak_mean_distance.argsort()
        ranks = ascending_index.argsort()

        # TODO: Maybe n_init is more robust? But then we would need to change the way the stds are labeled for each peak.
        # TODO: Return sx and sy relative the fitted line.
        # TODO: Make the algorithm guess near the fitted line?

        try:
            # Fitting the GMM.
            start_time = time.time()
            x_filtered, y_filtered = x_filtered_perpendicular, y_filtered_perpendicular

            # gmm = GaussianMixture(n_components=3, covariance_type='full', means_init=np.array([cm_binding_sites_x, cm_binding_sites_y]).T, max_iter=100)
            gmm = GaussianMixture(n_components=3, covariance_type='full',
                                  n_init=10, max_iter=100, precisions_init=inv_cov_init)
            gmm.fit(X=np.array([x_filtered, y_filtered]).T)

            # Labeling the standard deviations according to ascending distances.
            sx_labeled = gmm.covariances_[:, 0, 0][ranks]
            sy_labeled = gmm.covariances_[:, 1, 1][ranks]
            gmm_stds.append([np.append(sx_labeled.reshape(-1, 1), sy_labeled.reshape(-1, 1), axis=1)])
            gmm_stds_x = np.append(gmm_stds_x, gmm.covariances_[:, 0, 0][ranks])
            gmm_stds_y = np.append(gmm_stds_y, gmm.covariances_[:, 1, 1][ranks])

            # Plotting
            aux_folder = manage_save_directory(figures_per_pick_folder, 'GMM')
            figure_name = 'GMM_%02d' % i
            figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
            if plot_flag:
                fig, ax = plt.subplots(1, 1)
                ax.scatter(x_filtered, y_filtered, s=0.8)
                ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], s=10, c='r')
                for m in range(len(gmm.means_[:, 1])):
                    rho = gmm.covariances_[m][0, 1] / (np.sqrt(gmm.covariances_[m][0, 0]) * np.sqrt(gmm.covariances_[m][1, 1]))
                    plot_gaussian_2d([min(x_filtered), max(x_filtered)], [min(y_filtered), max(y_filtered)], gmm.means_[m, 0],
                                     gmm.means_[m, 1], 1, np.sqrt(gmm.covariances_[m][0, 0]),
                                     np.sqrt(gmm.covariances_[m][1, 1]), rho, 0, color='r')
                plt.savefig(figure_path, dpi = 100, bbox_inches='tight')
            if verbose_flag:
                print("Total execution time of GMM: --- %s seconds ---" % (time.time() - start_time))
        except:
            pass




        
        # designed distances
        # asymmetric origami (fourth origami): 
        #   - 136 nm between ends
        #   - 83 nm between center and far end
        #   - 54 nm between center and closer end
        # symmetric origami (third and fifth origami): 
        #   - 125 nm between ends
        #   - 54 nm between centers and far end
        #   - 18 nm between the two center peaks
        # print(matrix_distance)
        # print(matrix_std_dev)
        
        # plot matrix distance
        if plot_flag:
            plt.figure(10)
            plt.imshow(matrix_distance, interpolation='none', cmap='spring')
            ax = plt.gca()
            for l in range(matrix_distance.shape[0]):
                for m in range(matrix_distance.shape[1]):
                    if l == m:
                        ax.text(m, l, '-' ,
                            ha="center", va="center", color=[0,0,0], 
                            fontsize = 18)
                    else:
                        ax.text(m, l, '%.0f' % matrix_distance[l, m],
                            ha="center", va="center", color=[0,0,0], 
                            fontsize = 18)
            ax.xaxis.tick_top()
            ax.set_xticks(np.array(range(matrix_distance.shape[1])))
            ax.set_yticks(np.array(range(matrix_distance.shape[0])))
            axis_string = ['NP']
            for j in range(total_peaks_found):
                axis_string.append('Site %d' % (j+1))
            ax.set_xticklabels(axis_string)
            ax.set_yticklabels(axis_string)
            aux_folder = manage_save_directory(figures_per_pick_folder, 'matrix_distance')
            figure_name = 'matrix_distance_%02d' % i
            figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 100, bbox_inches='tight')
            plt.close()
        
            # plot matrix of max std dev    
            plt.figure(11)
            plt.imshow(matrix_std_dev, interpolation='none', cmap='spring')
            ax = plt.gca()
            for l in range(matrix_distance.shape[0]):
                for m in range(matrix_distance.shape[1]):
                    if not l == m:
                        ax.text(m, l, '-' ,
                            ha="center", va="center", color=[0,0,0], 
                            fontsize = 18)
                    else:
                        ax.text(m, l, '%.0f' % matrix_std_dev[l, m],
                            ha="center", va="center", color=[0,0,0], 
                            fontsize = 18)
            ax.xaxis.tick_top()
            ax.set_xticks(np.array(range(matrix_distance.shape[1])))
            ax.set_yticks(np.array(range(matrix_distance.shape[0])))
            axis_string = ['NP']
            for j in range(total_peaks_found):
                axis_string.append('Site %d' % (j+1))
            ax.set_xticklabels(axis_string)
            ax.set_yticklabels(axis_string)
            aux_folder = manage_save_directory(figures_per_pick_folder,'matrix_std_dev')
            figure_name = 'matrix_std_dev_%02d' % i
            figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 100, bbox_inches='tight')
            plt.close()
        
        # plots of the binding sites
        photons_concat = np.concatenate([photons_concat, photons_of_picked])
        bkg_concat = np.concatenate([bkg_concat, bkg_of_picked])
        frame_concat = np.concatenate([frame_concat, frame_of_picked])
        locs_of_picked[i] = len(frame_of_picked)
        hist_range = [0, number_of_frames]
        bin_size = (hist_range[-1] - hist_range[0])/number_of_bins
        locs_of_picked_vs_time[i,:], bin_edges = np.histogram(frame_of_picked, bins = number_of_bins, range = hist_range)
        bin_centers = bin_edges[:-1] + bin_size/2
        bin_centers_minutes = bin_centers*exp_time/60  
        # plot when the pick was bright vs time
        if plot_flag:
            plt.figure()
            # plt.plot(bin_centers, locs_of_picked_vs_time, label = 'Pick %04d' % i)
            plt.step(bin_centers_minutes, locs_of_picked_vs_time[i,:], where = 'mid', label = 'Pick %04d' % i)
            # plt.legend(loc='upper right')
            plt.xlabel('Time (min)')
            plt.ylabel('Locs')
            plt.ylim([0, 80])
            ax = plt.gca()
            ax.axvline(x=10, ymin=0, ymax=1, color = 'k', linewidth = '2', linestyle = '--')
            ax.set_title('Number of locs per pick vs time. Bin size %.1f min' % (bin_size*0.1/60))
            aux_folder = manage_save_directory(figures_per_pick_folder,'locs_vs_time_per_pick')
            figure_name = 'locs_per_pick_vs_time_pick_%02d' % i
            figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 100, bbox_inches='tight')
            plt.close()
            
        # plot xy coord of the pick in several ways, including the peaks detected
        if plot_flag:
            # plot all RAW
            plt.figure(1)
            plt.plot(x_position_of_picked, y_position_of_picked, '.', color = 'C0', label = 'PAINT')
            if NP_flag:
                plt.plot(x_position_of_picked_NP, y_position_of_picked_NP, '.', color = 'C1', label = 'NP')
                plt.plot(x_avg_NP, y_avg_NP, 'x', color = 'k', label = 'Avg position NP')
                plt.legend(loc='upper right')
            plt.ylabel('y ($\mu$m)')
            plt.xlabel('x ($\mu$m)')
            plt.axis('square')
            ax = plt.gca()
            ax.set_title('Position of locs per pick. Pick %02d' % i)
            aux_folder = manage_save_directory(figures_per_pick_folder,'scatter_plots')        
            figure_name = 'xy_pick_scatter_NP_and_PAINT_%02d' % i
            figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
            plt.close()
            
        if plot_flag:
            # plot SCATTER + NP
            plt.figure(2)
            plt.plot(x_position_of_picked, y_position_of_picked, '.', color = 'C0', label = 'PAINT')
            if NP_flag:
                plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 10, markerfacecolor = 'C1', 
                         markeredgecolor = 'k', label = 'NP')
                plt.legend(loc='upper right')
            plt.ylabel('y ($\mu$m)')
            plt.xlabel('x ($\mu$m)')
            plt.axis('square')
            ax = plt.gca()
            ax.set_title('Position of locs per pick. Pick %02d' % i)
            aux_folder = manage_save_directory(figures_per_pick_folder,'scatter_plots')        
            figure_name = 'xy_pick_scatter_PAINT_%02d' % i
            figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
            plt.close()
            # Save x and y positions for every pick.
            # TODO: Save with corresponding labels.
            # data_to_save = np.asarray([x_position_of_picked, y_position_of_picked]).T
            # new_filename = 'pick_' + str(i) + '_xy_.dat'
            # new_filepath = os.path.join(kinetics_folder, new_filename)
            # np.savetxt(new_filepath, data_to_save)
            #
            # # save ALL traces in one file
            # new_filename = 'pick_' + str(i) + '_cm_xy.dat'
            # new_filepath = os.path.join(kinetics_folder, new_filename)
            # np.savetxt(new_filepath, np.asarray([cm_binding_sites_x, cm_binding_sites_y]).T)



            
        if plot_flag:
            # plot FINE 2d image
            plt.figure(3)
            plt.imshow(z_hist, interpolation='none', origin='lower',
                       extent=[x_hist_centers[0], x_hist_centers[-1], 
                               y_hist_centers[0], y_hist_centers[-1]])
            ax = plt.gca()
            ax.set_facecolor(bkg_color)
            plt.plot(cm_binding_sites_x, cm_binding_sites_y, 'x', markersize = 9, 
                     color = 'white', label = 'binding sites')
            plt.plot(x_fitted, y_fitted, '--', linewidth = 1, color = 'white')

            # TODO: Plot both PAPER version and normal version here.
            for k, (circle_x, circle_y) in enumerate(zip(cm_binding_sites_x, cm_binding_sites_y)):
                circ = plot_circle((circle_x, circle_y), radius = analysis_radius, 
                            color = 'white', fill = False)
                ax.add_patch(circ)

                # Peak labels in ascending order
                theta = np.arctan(slope)
                perpendicular_x = circle_x + analysis_radius*1.25*np.cos(theta+np.pi/2)
                perpendicular_y = circle_y + -1/slope * (perpendicular_x-circle_x)
                text_position = (perpendicular_x, perpendicular_y)
                text_content = f"{ranks[k]}"
                ax.text(*text_position, text_content, ha='center', va='center', rotation=0, fontsize=12, color='white')

            if NP_flag:
                plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 8, markerfacecolor = 'C1', 
                         markeredgecolor = 'white', label = 'NP')
                plt.legend(loc='upper right')
            plt.ylabel('y ($\mu$m)')
            plt.xlabel('x ($\mu$m)')
            cbar = plt.colorbar()
            cbar.ax.set_title(u'Locs', fontsize = 16)
            cbar.ax.tick_params(labelsize = 16)
            ax.set_title('Position of locs per pick. Pick %02d' % i)
            aux_folder = manage_save_directory(figures_per_pick_folder,'image_FINE')        
            figure_name = 'xy_pick_image_PAINT_%02d' % i
            figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
            plt.close()
            
        if plot_flag:
            # plot FINE 2d image FOR PAPER
            plt.figure(3)
            plt.imshow(z_hist, interpolation='none', origin='lower',
                       extent=[x_hist_centers[0], x_hist_centers[-1], 
                               y_hist_centers[0], y_hist_centers[-1]])
            plt.plot(cm_binding_sites_x, cm_binding_sites_y, 'x', markersize = 9, 
                     color = 'black', mew = 2, label = 'binding sites')
            plt.plot(x_fitted, y_fitted, '--', linewidth = 1, color = 'wheat')
            ax = plt.gca()
            ax.set_facecolor(bkg_color)
            for k, (circle_x, circle_y) in enumerate(zip(cm_binding_sites_x, cm_binding_sites_y)):
                circ = plot_circle((circle_x, circle_y), radius = analysis_radius, 
                            color = 'white', fill = False)
                ax.add_patch(circ)
                # Peak labels in ascending order
                theta = np.arctan(-abs(slope))
                perpendicular_x = circle_x + analysis_radius*1.25*np.cos(theta+np.pi/2)
                perpendicular_y = circle_y + -1/slope * (perpendicular_x-circle_x)
                text_position = (perpendicular_x, perpendicular_y)
                text_content = f"{ranks[k]}"
                ax.text(*text_position, text_content, ha='center', va='center', rotation=0, fontsize=12, color='white')
            if NP_flag:
                plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 8, markerfacecolor = 'white', 
                         markeredgecolor = 'black', label = 'NP')
                plt.legend(loc='upper right')
            scalebar = ScaleBar(1e3, 'nm', location = 'lower left') 
            ax.add_artist(scalebar)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            aux_folder = manage_save_directory(figures_per_pick_folder,'image_FINE')        
            figure_name = 'PAPER_xy_pick_image_PAINT_%02d' % i
            figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
            plt.close()
            
        if plot_flag and (docking_sites != 1):
            # plot COARSE 2d image
            plt.figure(4)
            plt.imshow(z_hist_COARSE, interpolation='none', origin='lower',
                       extent=[x_hist_COARSE_centers[0], x_hist_COARSE_centers[-1], 
                               y_hist_COARSE_centers[0], y_hist_COARSE_centers[-1]])
            if NP_flag:
                plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 10, markerfacecolor = 'white', 
                         markeredgecolor = 'k', label = 'NP')
                plt.legend(loc='upper right')
            plt.ylabel('y ($\mu$m)')
            plt.xlabel('x ($\mu$m)')
            ax = plt.gca()
            ax.set_facecolor(bkg_color)
            cbar = plt.colorbar()
            cbar.ax.set_title(u'Locs')
            cbar.ax.tick_params()
            ax.set_title('Position of locs per pick. Pick %02d' % i)
            aux_folder = manage_save_directory(figures_per_pick_folder,'image_COARSE')
            figure_name = 'xy_pick_image_COARSE_PAINT_%02d' % i
            figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
            plt.close()
            
        if plot_flag and (docking_sites != 1):
            # plot BINARY 2d image
            plt.figure(5)
            plt.imshow(detected_peaks, interpolation='none', origin='lower', cmap = 'binary',
                       extent=[x_hist_COARSE_centers[0], x_hist_COARSE_centers[-1], 
                               y_hist_COARSE_centers[0], y_hist_COARSE_centers[-1]])
            if NP_flag:
                plt.plot(x_avg_NP, y_avg_NP, 'o', markersize = 10, markerfacecolor = 'C1', 
                         markeredgecolor = 'k', label = 'NP')
                plt.legend(loc='upper right')
            plt.ylabel('y ($\mu$m)')
            plt.xlabel('x ($\mu$m)')
            ax = plt.gca()
            ax.set_title('Position of locs per pick. Pick %02d' % i)
            aux_folder = manage_save_directory(figures_per_pick_folder,'binary_image')
            figure_name = 'xy_pick_image_peaks_PAINT_%02d' % i
            figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
            plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
            plt.close()


    ## plot relative positions of the binding sites with respect NP
    number_of_bins = 16
    hist_range = [25, 160]
    bin_size = (hist_range[-1] - hist_range[0])/number_of_bins
    print('\nRelative position NP-sites bin size', bin_size)
    position_bins, bin_edges = np.histogram(positions_concat_NP, bins = number_of_bins, \
                                            range = hist_range)
    bin_centers = bin_edges[:-1] + bin_size/2
    plt.figure()
    plt.bar(bin_centers, position_bins, width = 0.8*bin_size, align = 'center')
    plt.xlabel('Position (nm)')
    plt.ylabel('Counts')
    figure_name = 'relative_positions_NP_sites'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    plt.close()
    
    ## plot relative positions between binding sites
    number_of_bins = 16
    hist_range = [25, 160]
    bin_size = (hist_range[-1] - hist_range[0])/number_of_bins
    print('\nRelative position between binding sites bin size', bin_size)
    position_bins, bin_edges = np.histogram(positions_concat_origami, bins = number_of_bins, \
                                            range = hist_range)
    bin_centers = bin_edges[:-1] + bin_size/2
    plt.figure()
    plt.bar(bin_centers, position_bins, width = 0.8*bin_size, align = 'center')
    plt.xlabel('Position (nm)')
    plt.ylabel('Counts')
    figure_name = 'relative_positions_binding_sites'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    plt.close()
    
    # plot global variables, all the picks of the video
    time_concat = frame_concat*exp_time/60
    
    ## LOCS
    sum_of_locs_of_picked_vs_time = np.sum(locs_of_picked_vs_time, axis=0)
    plt.figure()
    plt.step(bin_centers_minutes, sum_of_locs_of_picked_vs_time, where = 'mid')
    plt.xlabel('Time (min)')
    plt.ylabel('Locs')
    x_limit = [0, total_time_min]
    plt.xlim(x_limit)
    ax = plt.gca()
    ax.set_title('Sum of localizations vs time. Binning time %d s. %d picks. ' \
                 % ((bin_size*0.1), total_number_of_picks))
    figure_name = 'locs_vs_time_all'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    plt.close()
    
    ## PHOTONS
    fig, ax = plt.subplots(1, 1)
    ax.hist(time_concat, bins=len(bin_centers_minutes), weights=photons_concat, histtype='step')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Photons')
    ax.set_title('Photons vs time')
    x_limit = [0, total_time_min]
    plt.xlim(x_limit)
    figure_name = 'photons_vs_time'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    plt.close()

    new_filename = 'PHOTONS_%s.dat' % (frame_file[:-10])
    new_filepath = os.path.join(kinetics_folder, new_filename)
    np.savetxt(new_filepath, photons_concat)

    new_filename = 'TIME_%s.dat' % (frame_file[:-10])
    new_filepath = os.path.join(kinetics_folder, new_filename)
    np.savetxt(new_filepath, time_concat)
    ## BACKGROUND
    fig, ax = plt.subplots(1, 1)
    ax.hist(time_concat, bins=len(bin_centers_minutes), weights=bkg_concat, histtype='step')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Background')
    ax.set_title('Background vs time')
    x_limit = [0, total_time_min]
    plt.xlim(x_limit)
    figure_name = 'bkg_vs_time'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    plt.close()
    
    ################################### save data
    
    # delete first fake and empty trace (needed to make the proper array)
    all_traces = np.delete(all_traces, 0, axis = 0)
    all_traces = all_traces.T
    # save ALL traces in one file
    new_filename = 'TRACES_ALL_%s.dat' % (frame_file[:-10])
    new_filepath = os.path.join(kinetics_folder, new_filename)
    np.savetxt(new_filepath, all_traces_per_pick, fmt='%05d')
    # compile all traces of the image in one array

    # GMM stds
    plt.figure()
    fig, ax = plt.subplots(1, 1)
    #ax.hist(np.sqrt(np.array(gmm_stds).reshape(-1, 2))*1e3, bins=len(gmm_stds), stacked=True, label=['s_x', 's_y'])
    gmm_stds_total = np.concatenate((gmm_stds_x.reshape(-1, 1), gmm_stds_y.reshape(-1, 1)), axis=1)
    ax.hist(np.sqrt(gmm_stds_total)*1e3, bins=len(gmm_stds), stacked=True, label=['s_x', 's_y'])
    ax.set_xlabel('Standard deviation (um))')
    ax.set_ylabel('Counts')
    plt.legend()
    ax.set_title('Distribution of the GMMs standard deviation.')
    figure_name = 'GMM_stds'
    figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()

    new_filename = 'gmm_stds.dat'
    new_filepath = os.path.join(gaussian_folder, new_filename)
    np.savetxt(new_filepath, np.array(gmm_stds).reshape(-1, 2))


    # number of locs
    data_to_save = np.asarray([pick_number, locs_of_picked]).T
    new_filename = 'number_of_locs_per_pick.dat'
    new_filepath = os.path.join(figures_folder, new_filename)
    np.savetxt(new_filepath, data_to_save, fmt='%i')
    
    # relative positions
    data_to_save = positions_concat_NP
    new_filename = 'relative_positions_NP_sites_in_nm.dat'
    new_filepath = os.path.join(figures_folder, new_filename)
    np.savetxt(new_filepath, data_to_save, fmt='%.1f')
    data_to_save = positions_concat_origami
    new_filename = 'relative_positions_binding_sites_in_nm.dat'
    new_filepath = os.path.join(figures_folder, new_filename)
    np.savetxt(new_filepath, data_to_save, fmt='%.1f')

    print('\nDone with STEP 2.')

    return
        
#####################################################################
#####################################################################
#####################################################################

if __name__ == '__main__':
    
    # load and open folder and file
    base_folder = "C:\\Users\\olled\\Documents\\DNA-PAINT\\Data\\single_channel_DNA-PAINT_example\\Week_4\\All_DNA_Origami\\17_picks"
    root = tk.Tk()
    selected_file = fd.askopenfilename(initialdir = base_folder,
                                          filetypes=(("", "*.dat") , ("", "*.")))   
    root.withdraw()
    working_folder = os.path.dirname(selected_file)
    
    # docking site per origami
    docking_sites = 3
    # is there any NP (hybridized structure)
    NP_flag = False
    # camera pixel size
    pixel_size = 0.130 # in um
    # size of the pick used in picasso
    pick_size = 3 # in camera pixels (put the same number used in Picasso)
    # size of the pick to include locs around the detected peaks
    radius_of_pick_to_average = 0.25 # in camera pixel size
    # set an intensity threshold to avoid dumb peak detection in the background
    # this threshold is arbitrary, don't worry about this parameter, the code 
    # change it automatically to detect the number of docking sites set above
    th = 1
    # time parameters☺
    number_of_frames = 12000
    exp_time = 0.1 # in s
    plot_flag = True
    
    process_dat_files(number_of_frames, exp_time, working_folder,
                          docking_sites, NP_flag, pixel_size, pick_size, 
                          radius_of_pick_to_average, th, plot_flag, True)
