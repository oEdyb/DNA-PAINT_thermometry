# -*- coding: utf-8 -*-
"""
Created on Tue May 11 2021

Finding the kinetic binding time of an imager probe in a DNA-PAINT experiment
    hyperexponential case: two exponential distributions, two possible binding 
    events (long and short with a ratio between their probabilities).
Maximum Likelihood Estimation (MLE) is performed to estimate the
best parameters that fit the data. Instead of maximizing the 
likelihood, -log(likelihood) is minimized.

@author: Mariano Barella

Fribourg, Switzerland
"""
# ================ IMPORT LIBRARIES ================
import pickle
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as opt
import re
import os
import tkinter as tk
import tkinter.filedialog as fd
from auxiliary_functions import (log_likelihood_hyper, log_likelihood_hyper_with_error, \
                            hyperexp_func_with_error, hyperexp_func, \
                            monoexp_func, \
                            monoexp_func_with_error, log_likelihood_mono_with_error,
                                 log_likelihood_mono_with_error_one_param, update_pkl, manage_save_directory)

# ================ CONFIGURATION SETTINGS ================
# ignore divide by zero warning and scipy optimization warnings
np.seterr(divide='ignore')
warnings.filterwarnings('ignore', message='delta_grad == 0.0')
warnings.filterwarnings('ignore', category=UserWarning, module='scipy.optimize')

plt.close('all')
plt.ioff()

########################################################################
###################### HYPEREXPONENTIAL PROBLEM ########################
########################################################################

# ================ MAIN FUNCTION FOR BINDING/UNBINDING TIME ESTIMATION ================
def estimate_binding_unbinding_times(exp_time, rango, working_folder, \
                                    initial_params, likelihood_err_param, \
                                    opt_display_flag, hyper_exponential_flag, verbose_flag):
    
    print('\nStarting STEP 4.')
    
    # ================ CREATE STEP4 FOLDER STRUCTURE ================
    # Determine main folder based on input path structure
    if 'kinetics_data' in working_folder:
        # For position averaging method: working_folder is .../analysis/step2/position_averaging_method/data/kinetics_data
        main_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(working_folder)))))  # Go back 5 directories
    else:
        # For original method: working_folder is .../analysis/step3/data  
        main_folder = os.path.dirname(os.path.dirname(os.path.dirname(working_folder)))  # Go back 3 directories
    
    analysis_folder = os.path.join(main_folder, 'analysis')
    step4_data_folder = manage_save_directory(analysis_folder, 'step4/data')
    step4_figures_folder = manage_save_directory(analysis_folder, 'step4/figures')
    
    if verbose_flag:
        print(f'Debug: working_folder = {working_folder}')
        print(f'Debug: main_folder = {main_folder}')
        print(f'Debug: step4_data_folder = {step4_data_folder}')
        print(f'Debug: step4_figures_folder = {step4_figures_folder}')
    
    # ================ LOCATE AND LOAD DATA FILES ================
    list_of_files = os.listdir(working_folder)
    
    t_on_datafile = [f for f in list_of_files if re.search('t_on', f)][0]
    t_off_datafile = [f for f in list_of_files if re.search('t_off', f)][0]

    t_on_full_filepath = os.path.join(working_folder, t_on_datafile)
    t_off_full_filepath = os.path.join(working_folder, t_off_datafile)

    ########################################################################

    # ================ ANALYZE BINDING TIME (t_on) ================
    solutions_on = find_best_tau_using_MLE(t_on_full_filepath, rango, \
                                            exp_time, initial_params, \
                                            likelihood_err_param, \
                                            hyper_exponential_flag, \
                                            opt_display_flag, 1, verbose_flag, step4_figures_folder, 'tau_on', plot_flag=False)
    
    #print('\nSolution for tau_on')
    # print(solutions_on)
    
    ########################################################################

    # ================ ANALYZE UNBINDING TIME (t_off) ================
    solutions_off = find_best_tau_using_MLE(t_off_full_filepath, rango, \
                                            exp_time, [230, 2, 1], \
                                            likelihood_err_param, \
                                            True, \
                                            opt_display_flag, 1, verbose_flag, step4_figures_folder, 'tau_off', plot_flag=False)

    # print('\nSolution for tau_off')
    # # print(solutions_off)

    # ================ COMPILE RESULTS AND SAVE ================
    parameters = {"tau_long": solutions_on[0][0], "tau_long_error_plus": solutions_on[0][1],
                  "tau_long_error_minus": solutions_on[0][2], "ratio": solutions_on[2][0],
                  "tau_long_off": solutions_off[0][0], "tau_long_off_error_plus": solutions_off[0][1],
                  "tau_long_off_error_minus": solutions_off[0][2], "ratio_off": solutions_off[2][0],
                  "tau_short": solutions_on[1][0], "tau_short_error_plus": solutions_on[1][1],
                  "tau_short_error_minus": solutions_on[1][2], "tau_short_off": solutions_on[1][0],
                  "tau_short_off_error_plus": solutions_on[1][1], "tau_short_off_error_minus": solutions_on[1][2]}
    path = os.path.dirname(working_folder)
    for parameter in parameters.keys():
        update_pkl(path, parameter, parameters[parameter])


    # ================ SAVE PARAMETERS TO PICKLE FILE ================
    dict_path = os.path.join(step4_data_folder, "MLE_parameters.pkl")
    with open(dict_path, 'wb') as f:
        pickle.dump(parameters, f)

    ########################################################################

    # ================ FINAL PROMINENT TAU_LONG OUTPUT ================

    print('\n' + '='*23 + ' TAU LONG ' + '='*23)
    print(f'   TAU_LONG = {solutions_on[0][0]:.3f} seconds')
    print(f'   Error: +{solutions_on[0][1]:.3f} / -{solutions_on[0][2]:.3f} seconds')
    print('='*60)
    if verbose_flag:
        if hyper_exponential_flag and solutions_on[1][0] > 0:
            print(f'   Additional info: tau_short = {solutions_on[1][0]:.3f} s, ratio = {solutions_on[2][0]:.3f}')
        print(f'   Full results saved to: {dict_path}')
        print('='*60)
    print('\nDone with STEP 4.')

    # ================ RETURN RESULTS FOR CONSOLIDATION ================
    results = {
        'tau_long_seconds': solutions_on[0][0],
        'tau_long_error_plus': solutions_on[0][1], 
        'tau_long_error_minus': solutions_on[0][2],
        'tau_long_off_seconds': solutions_off[0][0],
        'tau_long_off_error_plus': solutions_off[0][1],
        'tau_long_off_error_minus': solutions_off[0][2],
        'ratio_on': solutions_on[2][0],
        'ratio_off': solutions_off[2][0]
    }
    
    # Add hyperexponential-specific results if applicable
    if hyper_exponential_flag:
        results.update({
            'tau_short_seconds': solutions_on[1][0],
            'tau_short_error_plus': solutions_on[1][1],
            'tau_short_error_minus': solutions_on[1][2],
            'tau_short_off_seconds': solutions_off[1][0],
            'tau_short_off_error_plus': solutions_off[1][1],
            'tau_short_off_error_minus': solutions_off[1][2]
        })
    
    return results

########################################################################
########################################################################
########################################################################

# ================ FUNCTION TO FIND BEST TAU USING MLE ================
def find_best_tau_using_MLE(full_filepath, rango, exp_time, initial_params, \
                            likelihood_err_param, hyper_exponential_flag, \
                            opt_display_flag, factor, verbose_flag, figures_folder, plot_name, sample_data = None, plot_flag=False):
    # factor is to be used in case you're estimating tau_off which is significantly
    # larger than tau_on, typically
    from step4_functions import (load_sample_data, setup_mle_optimization, 
                                perform_mle_optimization, calculate_parameter_errors,
                                prepare_histogram_data, plot_mle_histogram, 
                                plot_mle_paper_figure, compile_mle_results, print_mle_summary)

    # ================ PREPARE INPUT DATA ================
    rango = factor*np.array(rango)
    sample = load_sample_data(full_filepath, sample_data)
    st = time.time()

    # ================ SET OPTIMIZATION PARAMETERS ================
    bounds, initial_guess, constraints = setup_mle_optimization(hyper_exponential_flag, initial_params, factor)

        # numerical approximation of the log_ML function
    # problem: hyperexponential/monoexponential
    # probe variables' space
    # tau_long_array = factor*np.arange(0.05, tau_long_init*1.1, 0.05)
    # if hyper_exponential_flag:
    #     tau_short_array = np.arange(0.05, 2, 0.05)
    #
    # else:
    #     tau_short_array = np.array([0])
    # l = len(tau_long_array)
    # m = len(tau_short_array)
    # log_likelihood_matrix = np.zeros((l, m))
    # for i in range(l):
    #     for j in range(m):
    #         # if j >= i:
    #             # likelihood_matrix[i,j] = np.nan
    #         # else:
    #             theta_param = [tau_long_array[i], tau_short_array[j], set_ratio]
    #             # apply log to plot colormap with high-contrast
    #             if hyper_exponential_flag:
    #                 log_likelihood_matrix[i,j] = log_likelihood_hyper_with_error(theta_param, sample)
    #             else:
    #                 log_likelihood_matrix[i,j] = log_likelihood_mono_with_error(theta_param, sample)
    # log_log_likelihood_matrix = np.log(log_likelihood_matrix)
    
    # ================ PERFORM MLE OPTIMIZATION ================
    road_to_convergence = []
    road_to_convergence.append(initial_params)
    out_estimator = perform_mle_optimization(sample, hyper_exponential_flag, bounds, 
                                           initial_guess, constraints, opt_display_flag, 
                                           road_to_convergence)
    road_to_convergence = np.array(road_to_convergence)
    
    # ================ EXTRACT OPTIMIZED PARAMETERS ================
    tau_long_MLE = out_estimator.x[0]
    tau_short_MLE = out_estimator.x[1]
    ratio_MLE = out_estimator.x[2]
    
    tau_long_amplitude = ratio_MLE/(ratio_MLE + 1)*(1/tau_long_MLE)
    tau_short_amplitude = 1/(ratio_MLE + 1)*(1/tau_short_MLE)


    #print('\nInitial values were:')
    #print('tau_long', tau_long_init, ', tau_short', tau_short_init, \
    #      ', ratio', set_ratio)
    
    #print('\nFit values are:')
    #print('tau_long %.3f, tau_short %.3f, ratio %.4f' % \
    #      (tau_long_MLE, tau_short_MLE, ratio_MLE))
    
    #print('tau_long_amplitude %.3f, tau_short_amplitude %.3f' % \
    #     (tau_long_amplitude, tau_short_amplitude))

    # ================ VISUALIZATION OF FITTING RESULTS ================
    # plot output and minimization map
    # Takes too long with tau off.
    # plt.figure(1)
    # ax = plt.gca()
    # min_map = ax.imshow(log_log_likelihood_matrix, interpolation = 'none', cmap = cm.jet,
    #                     origin='lower', aspect = 1/(factor*tau_long_init),
    #                     extent=[tau_short_array[0], tau_short_array[-1],
    #                     tau_long_array[0], tau_long_array[-1]])
    # ax.plot(road_to_convergence[:,1], road_to_convergence[:,0], marker='.', \
    #         color='w', markeredgecolor='k', markeredgewidth=0.8, linewidth=0.8)
    # ax.plot(road_to_convergence[0,1], road_to_convergence[0,0], marker='o', \
    #         color='C3', markeredgecolor='w', markeredgewidth=0.8, linewidth=0.8)
    # ax.plot(road_to_convergence[-1,1], road_to_convergence[-1,0], marker='o', \
    #         color='C2', markeredgecolor='w', markeredgewidth=0.8, linewidth=0.8)
    # ax.set_ylabel('Long time (s)')
    # ax.set_xlabel('Short time (s)')
    # cbar = plt.colorbar(min_map, ax = ax)
    # plt.grid(False)
    # cbar.ax.set_title('log( -log( likelihood ) )', fontsize = 13)
    
    # ================ PREPARE HISTOGRAM FOR VISUALIZATION ================
    bin_center, counts_norm, counts_norm_MLE, bin_size = prepare_histogram_data(
        sample, rango, exp_time, factor, hyper_exponential_flag, 
        tau_long_MLE, tau_short_MLE, ratio_MLE)


    # ================ GENERATE PLOTS OF MLE RESULTS ================
    # Residual plot
    # min_prob_threshold = 5e-2
    #
    # plt.figure()
    # bin_edges_fd = np.histogram_bin_edges(sample, bins='fd')
    # counts_res, _ = np.histogram(sample, bins=bin_edges_fd)
    # integral_res = np.sum(counts_res * np.diff(bin_edges_fd))
    # counts_res_norm = counts_res / integral_res
    # time_values = bin_edges_fd[1:] - (bin_edges_fd[1] - bin_edges_fd[0]) / 2
    #
    # if hyper_exponential_flag:
    #     model_pred = hyperexp_func(time_values, tau_long_MLE, tau_short_MLE, ratio_MLE)
    # else:
    #     model_pred = monoexp_func(time_values, tau_long_MLE, tau_short_MLE, ratio_MLE)
    #
    # # Apply the cutoff based on model predictions
    # valid_indices = model_pred >= min_prob_threshold
    # filtered_time_values = time_values[valid_indices]
    # filtered_model_pred = model_pred[valid_indices]
    # filtered_counts_res_norm = counts_res_norm[valid_indices]
    #
    # # Calculate residuals for the filtered data
    # res_percent = (-filtered_counts_res_norm + filtered_model_pred) * 100 / filtered_model_pred
    #
    # plt.scatter(filtered_time_values, res_percent, s=1.5)
    # plt.plot(filtered_time_values, res_percent, color='k', linewidth=0.5)
    # plt.title('Model residual in percentage vs. binding time (with tail cutoff).')
    # plt.xlabel('Binding time [s]')
    # plt.ylabel('Deviation from data [%]')
    # plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.6)
    # 
    # plt.close()

    # ================ PLOT DATA AND MLE FIT ================
    # plot data and MLE output
    if True:
        plt.figure(99)
        plt.bar(bin_center, counts_norm, width = bin_size, color = 'lightgrey', edgecolor = 'k', label='Data')
        plt.plot(bin_center, counts_norm_MLE, '-', linewidth = 0.1, color = '#638bf9', label='MLE')
        plt.grid()
        ax = plt.gca()
        ax.set_xlabel('Binding time (s)', fontsize=18)
        ax.set_ylabel('Normalized frequency', fontsize=18)
        ax.set_yscale('log')
        ax.set_axisbelow(True)
        plt.title('bin size %.3f s' % bin_size)
        plt.legend()
        figure_name = f'binding_time_hist_MLE_{plot_name}'
        figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
        plt.tight_layout()
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')


        # ================ PLOT FORMATTED PUBLICATION-READY FIGURE ================
        # plot data and MLE output °°°°°° FOR PAPER°°°°°°°°°°°
        plt.figure(100)
        plt.bar(bin_center, counts_norm, width = bin_size, color = 'lightgrey', edgecolor = 'k', label='Data')
        plt.plot(bin_center, counts_norm_MLE, '-', linewidth = 3, color = '#638bf9', label='MLE')
        plt.grid()
        ax = plt.gca()
        ax.set_xlabel('Binding time (s)', fontsize=20)
        ax.set_ylabel('Normalized frequency', fontsize=20)
        ax.set_yscale('log')
        ax.set_axisbelow(True)
        ax.set_xlim([0, 10])
        ax.set_ylim([1e-4, 10])

        plt.legend(loc='upper right', prop={'size':12})
        plt.tick_params(axis='both', which='major', labelsize=18)

        if hyper_exponential_flag:
            tau_long_text = rf'$\tau_{{\mathrm{{long}}}} = {tau_long_MLE:.2f}\,s$'
            tau_short_text = rf'$\tau_{{\mathrm{{short}}}} = {tau_short_MLE:.2f}\,s$'
            plt.text(0.05, 0.95, tau_long_text + '\n' + tau_short_text, transform=ax.transAxes, fontsize=14,
                     verticalalignment='top')
        else:
            tau_long_text = rf'$\tau_{{\mathrm{{long}}}} = {tau_long_MLE:.2f}\,s$'
            plt.text(0.05, 0.95, tau_long_text, transform=ax.transAxes, fontsize=14,
                     verticalalignment='top')

        figure_name = f'binding_time_hist_MLE_paper_{plot_name}'
        figure_path = os.path.join(figures_folder, '%s.png' % figure_name)
        plt.tight_layout()
        plt.savefig(figure_path, dpi = 300, bbox_inches='tight')


    # ================ ESTIMATE ERROR INTERVALS ================
    # estimate error as a likelihood interval
    # see "Probability and statistics in particle physics" of A. G. Froedsen, 
    # 1979 ed., page 221 and on, Chapter 9.6 (particularly, page 233)
    # variable assumed to be independent 
    a = likelihood_err_param
    if hyper_exponential_flag:
        MLE_value = log_likelihood_hyper_with_error(out_estimator.x, sample)
    else:
        MLE_value = log_likelihood_mono_with_error(out_estimator.x, sample)

    # ================ CALCULATE ERROR FOR TAU_LONG ================
    # define functions as f = abs( MLE - (MLE_value + a)) to find the roots
    # FOR TAU_ON #########################################################
    if hyper_exponential_flag:
        log_likelihood_tau_on_var = lambda x : np.abs(log_likelihood_hyper_with_error([x, tau_short_MLE, ratio_MLE], \
                                                                           sample) - (MLE_value + a))
    else:
        log_likelihood_tau_on_var = lambda x : np.abs(log_likelihood_mono_with_error([x, tau_short_MLE, ratio_MLE], \
                                                                           sample) - (MLE_value + a))
    initial_guess = tau_long_MLE + 0.1
    out_tau_on_error_estimator_plus = opt.root(log_likelihood_tau_on_var,
                                               initial_guess,
                                               method = 'hybr',
                                               options = {'xtol':1e-16,
                                                          'maxfev':2000})
    initial_guess = tau_long_MLE - 0.1
    out_tau_on_error_estimator_minus = opt.root(log_likelihood_tau_on_var,
                                               initial_guess,
                                               method = 'hybr',
                                               options = {'xtol':1e-16,
                                                          'maxfev':2000})

    # ================ CALCULATE ERROR FOR TAU_SHORT ================
    # FOR TAU_SHORT #########################################################
    if hyper_exponential_flag:
        log_likelihood_tau_short_var = lambda x : np.abs(log_likelihood_hyper_with_error([tau_long_MLE, x, ratio_MLE], \
                                                                               sample) - (MLE_value + a))
        initial_guess = tau_short_MLE + 0.1
        out_tau_short_error_estimator_plus = opt.root(log_likelihood_tau_short_var,
                                                 initial_guess,
                                                 method = 'hybr',
                                                 options = {'xtol':1e-16,
                                                            'maxfev':2000})
        initial_guess = tau_short_MLE - 0.1
        out_tau_short_error_estimator_minus = opt.root(log_likelihood_tau_short_var,
                                                       initial_guess,
                                                       method = 'hybr',
                                                       options = {'xtol':1e-16,
                                                                  'maxfev':2000})

    # ================ CALCULATE ERROR FOR RATIO ================
    # FOR RATIO #########################################################
    if hyper_exponential_flag:
        log_likelihood_ratio_var = lambda x : np.abs(log_likelihood_hyper_with_error([tau_long_MLE, tau_short_MLE, x], \
                                                                           sample) - (MLE_value + a))
    else:
        log_likelihood_ratio_var = lambda x : np.abs(log_likelihood_mono_with_error([tau_long_MLE, tau_short_MLE, x], \
                                                                           sample) - (MLE_value + a))
    initial_guess = ratio_MLE + 0.01
    out_ratio_error_estimator_plus = opt.root(log_likelihood_ratio_var,
                                              initial_guess,
                                              method = 'hybr',
                                              options = {'xtol':1e-16,
                                                         'maxfev':2000})
    initial_guess = ratio_MLE - 0.01
    out_ratio_error_estimator_minus = opt.root(log_likelihood_ratio_var,
                                               initial_guess,
                                               method = 'hybr',
                                               options = {'xtol':1e-16,
                                                          'maxfev':2000})
    
    # ================ CALCULATE FINAL ERROR VALUES ================
    # calculate errors: 
    tau_long_MLE_error_plus = np.abs(out_tau_on_error_estimator_plus.x[0] - tau_long_MLE)
    tau_long_MLE_error_minus = np.abs(out_tau_on_error_estimator_minus.x[0] - tau_long_MLE)
    #print('\nError on tau_long (s): +%.3f / -%.3f' % (tau_long_MLE_error_plus, tau_long_MLE_error_minus))
    if hyper_exponential_flag:
        tau_short_MLE_error_plus = np.abs(out_tau_short_error_estimator_plus.x[0] - tau_short_MLE)
        tau_short_MLE_error_minus = np.abs(out_tau_short_error_estimator_minus.x[0] - tau_short_MLE)
    #    print('Error on tau_short (s): +%.3f / -%.3f' % (tau_short_MLE_error_plus, tau_short_MLE_error_minus))
    ratio_MLE_error_plus = np.abs(out_ratio_error_estimator_plus.x[0] - ratio_MLE)
    ratio_MLE_error_minus = np.abs(out_ratio_error_estimator_minus.x[0] - ratio_MLE)
    #print('Error on ratio (s): +%.3f / -%.3f' % (ratio_MLE_error_plus, ratio_MLE_error_minus))
    
    ########################################################################
    
    # ================ COMPILE RESULTS ================
    solutions = compile_mle_results(hyper_exponential_flag, tau_long_MLE, tau_short_MLE, ratio_MLE,
                                   tau_long_MLE_error_plus, tau_long_MLE_error_minus,
                                   tau_short_MLE_error_plus if hyper_exponential_flag else 0,
                                   tau_short_MLE_error_minus if hyper_exponential_flag else 0,
                                   ratio_MLE_error_plus, ratio_MLE_error_minus)

    # ================ OUTPUT EXECUTION TIME AND SUMMARY ================
    et = time.time()
    elapsed_time = et - st
    if verbose_flag:
        print('Execution time of minimizer:', elapsed_time, 'seconds')
        print_mle_summary(hyper_exponential_flag, tau_long_MLE, tau_short_MLE, ratio_MLE,
                         tau_long_MLE_error_plus, tau_long_MLE_error_minus,
                         tau_short_MLE_error_plus if hyper_exponential_flag else 0,
                         tau_short_MLE_error_minus if hyper_exponential_flag else 0,
                         ratio_MLE_error_plus, ratio_MLE_error_minus,
                         tau_short_amplitude, bin_size, initial_params)
    ########################################################################
    
    # ================ ADDITIONAL LIKELIHOOD ANALYSIS (COMMENTED) ================
    # uncomment the following lines to calculate and plot the
    # independant log likelihood and check the independent variable assumption
    # does not includes the monoexp version
    
    # log_likelihood_matrix_for_tau_on_error = np.zeros((l))
    # f = np.zeros((l))
    # log_likelihood_matrix_for_tau_short_error = np.zeros((m))
    # for i in range(l):
    #     theta_param = [tau_on_array[i], tau_short_MLE, ratio_MLE]
    #     log_likelihood_matrix_for_tau_on_error[i] = log_likelihood_hyper_with_error(theta_param, sample)
    #     f[i] = log_likelihood_tau_on_var(tau_on_array[i])
    # for i in range(m):
    #     theta_param = [tau_on_MLE, short_time_array[i], ratio_MLE]
    #     log_likelihood_matrix_for_tau_short_error[i] = log_likelihood_hyper_with_error(theta_param, sample)
    
    # plt.figure(19)
    # plt.plot(tau_long_array, f, '-', linewidth = 2, color = 'C0', label='')
    # plt.grid()
    # ax = plt.gca()
    # ax.set_xlabel('Binding time (s)')
    # ax.set_ylabel('Log likelihood - (Log likelihood min + a)')
    # # ax.set_yscale('log')
    # ax.set_axisbelow(True)
    # plt.legend()
    
    # plt.figure(30)
    # plt.plot(taulongn_array, log_likelihood_matrix_for_tau_on_error, '-', linewidth = 2, color = 'C0', label='')
    # plt.grid()
    # ax = plt.gca()
    # ax.set_xlabel('Binding time (s)')
    # ax.set_ylabel('Log likelihood')
    # # ax.set_yscale('log')
    # ax.set_axisbelow(True)
    # plt.legend()
    
    # plt.figure(31)
    # plt.plot(tau_short_array, log_likelihood_matrix_for_tau_short_error, '-', linewidth = 2, color = 'C0', label='')
    # plt.grid()
    # ax = plt.gca()
    # ax.set_xlabel('Short time (s)')
    # ax.set_ylabel('Log likelihood')
    # # ax.set_yscale('log')
    # ax.set_axisbelow(True)
    # plt.legend()

    ########################################################################
    if plot_flag:
        plt.show()
    else:
        plt.close()

    return solutions

#########################################################################
#########################################################################
#########################################################################

# ================ SCRIPT EXECUTION ENTRY POINT ================
if __name__ == '__main__':
    
    # load and open folder and file
    base_folder = 'C:\\datos_mariano\\posdoc\\unifr\\DNA-PAINT_nanothermometry\\data_fribourg'
    root = tk.Tk()
    selected_file = fd.askopenfilename(initialdir = base_folder, \
                                       filetypes=(("", "*.dat") , ("", "*.")))   
    root.withdraw()    
    working_folder = os.path.dirname(selected_file)
    
    # input parameters
    exp_time = 0.1 # in ms
    rango = [0, 15]
    tau_long_init = 5 # in s
    tau_short_init = 0.1 # in s
    set_ratio = 0.5
    opt_display_flag = True
    likelihood_err_param = 0.5
    initial_params = [tau_long_init, tau_short_init, set_ratio]

    estimate_binding_unbinding_times(exp_time, rango, working_folder, \
                                    initial_params, likelihood_err_param, \
                                    opt_display_flag, False)
