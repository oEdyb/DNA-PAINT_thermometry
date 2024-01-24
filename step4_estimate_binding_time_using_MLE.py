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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.optimize as opt
import re
import os
import tkinter as tk
import tkinter.filedialog as fd
from auxiliary_functions import log_likelihood_hyper, log_likelihood_hyper_with_error, \
                            hyperexp_func_with_error, hyperexp_func, \
                            monoexp_func, \
                            monoexp_func_with_error, log_likelihood_mono_with_error

# ignore divide by zero warning
np.seterr(divide='ignore')

plt.close('all')
plt.ioff()

########################################################################
###################### HYPEREXPONENTIAL PROBLEM ########################
########################################################################

def estimate_binding_unbinding_times(exp_time, rango, working_folder, \
                                    initial_params, likelihood_err_param, \
                                    opt_display_flag, hyper_exponential_flag):
    
    print('\nStarting STEP 4.')
        
    list_of_files = os.listdir(working_folder)
    
    t_on_datafile = [f for f in list_of_files if re.search('t_on_TRACES_ALL', f)][0]
    t_off_datafile = [f for f in list_of_files if re.search('t_off_TRACES_ALL', f)][0]

    t_on_full_filepath = os.path.join(working_folder, t_on_datafile)
    t_off_full_filepath = os.path.join(working_folder, t_off_datafile)

    ########################################################################

    solutions_on = find_best_tau_using_MLE(t_on_full_filepath, rango, \
                                            exp_time, initial_params, \
                                            likelihood_err_param, \
                                            hyper_exponential_flag, \
                                            opt_display_flag, 1)
    
    print('\nSolution for tau_on')
    # print(solutions_on)
    
    ########################################################################

    # solutions_off = find_best_tau_using_MLE(t_off_full_filepath, rango, \
    #                                         exp_time, initial_params, \
    #                                         likelihood_err_param, \
    #                                         opt_display_flag, 5)

    # print('\nSolution for tau_off')
    # print(solutions_off)

    ########################################################################

    print('\nDone with STEP 4.')

    return

########################################################################
########################################################################
########################################################################

def find_best_tau_using_MLE(full_filepath, rango, exp_time, initial_params, \
                            likelihood_err_param, hyper_exponential_flag, \
                            opt_display_flag, factor):
    # factor is to be used in case you're estimating tau_off which is significantly
    # larger than tau_on, typically
    rango = factor*np.array(rango)
    # load data
    sample = np.loadtxt(full_filepath)
    
    # numerical approximation of the log_ML function using scipy.optimize
    
    # if hyperexponential define bounds and initial params for two exponential
    # otherwise set them for a single exponential
    if hyper_exponential_flag:
        # bounds
        tau_short_lower_bound = 0.01
        tau_short_upper_bound = 5
        ratio_lower_bound = 0
        ratio_upper_bound = 100
        # initial parameters, input of the minimizer
        [tau_long_init, tau_short_init, set_ratio] = initial_params
    else:
        tau_short_lower_bound = 1e-4
        tau_short_upper_bound = 1e-3
        ratio_lower_bound = 0
        ratio_upper_bound = np.inf
        tau_long_init = initial_params[0]
        tau_short_init = 1 # not relevant
        set_ratio = initial_params[2] # initial amplitude

    # numerical approximation of the log_ML function
    # problem: hyperexponential/monoexponential
    # probe variables' space
    tau_long_array = factor*np.arange(0.05, tau_long_init*1.1, 0.05)
    if hyper_exponential_flag:
        tau_short_array = np.arange(0.05, 2, 0.05)

    else:
        tau_short_array = np.array([0])
    l = len(tau_long_array)
    m = len(tau_short_array)
    log_likelihood_matrix = np.zeros((l, m))
    for i in range(l):
        for j in range(m):
            # if j >= i:
                # likelihood_matrix[i,j] = np.nan
            # else:
                theta_param = [tau_long_array[i], tau_short_array[j], set_ratio]
                # apply log to plot colormap with high-contrast
                if hyper_exponential_flag:
                    log_likelihood_matrix[i,j] = log_likelihood_hyper_with_error(theta_param, sample)
                else:
                    log_likelihood_matrix[i,j] = log_likelihood_mono_with_error(theta_param, sample)
    log_log_likelihood_matrix = np.log(log_likelihood_matrix)
    
    # before plotting the MLE map === Minimize!!!
    # prepare function to store points the method pass through
    road_to_convergence = list()
    road_to_convergence.append(initial_params)
    def callback_fun_trust(X, log_ouput):
        road_to_convergence.append(list(X))
        return 
    def callback_fun(X):
        road_to_convergence.append(list(X))
        return 
    # define bounds of the minimization problem (any bounded method)    
    bnds = opt.Bounds([0.01, tau_short_lower_bound, ratio_lower_bound], \
                      [factor*100, tau_short_upper_bound, ratio_upper_bound]) # [lower bound array], [upper bound array]
    
    # now minimize
    print('Optimization process started...')
    ################# constrained and bounded methods
    
    # define constraint of the minimization problem (for trust-constr method)
    # that is real_binding_time > short_on_time
    constr_array = np.array([1, -1, 0])
    constr = opt.LinearConstraint(constr_array, 0, np.inf, keep_feasible = True)
    if hyper_exponential_flag:
        out_estimator = opt.minimize(log_likelihood_hyper_with_error, 
                                    initial_params, 
                                    args = (sample), 
                                    method = 'trust-constr',
                                    bounds = bnds,
                                    constraints = constr,
                                    callback = callback_fun_trust,
                                    options = {'maxiter':2000, 
                                                'xtol':1e-16,
                                                'gtol': 1e-16,
                                                'disp':opt_display_flag})
    else:
        out_estimator = opt.minimize(log_likelihood_mono_with_error, 
                                    initial_params, 
                                    args = (sample), 
                                    method = 'trust-constr',
                                    bounds = bnds,
                                    constraints = constr,
                                    callback = callback_fun_trust,
                                    options = {'maxiter':2000, 
                                                'xtol':1e-16,
                                                'gtol': 1e-16,
                                                'disp':opt_display_flag})
    road_to_convergence = np.array(road_to_convergence)
    print(out_estimator)
    
    # assign variables
    tau_long_MLE = out_estimator.x[0]
    tau_short_MLE = out_estimator.x[1]
    ratio_MLE = out_estimator.x[2]
    
    tau_long_amplitude = ratio_MLE/(ratio_MLE + 1)*(1/tau_long_MLE)
    tau_short_amplitude = 1/(ratio_MLE + 1)*(1/tau_short_MLE)
    
    print('\nInitial values were:')
    print('tau_long', tau_long_init, ', tau_short', tau_short_init, \
          ', ratio', set_ratio)
    
    print('\nFit values are:')
    print('tau_long %.3f, tau_short %.3f, ratio %.4f' % \
          (tau_long_MLE, tau_short_MLE, ratio_MLE))
    
    print('tau_long_amplitude %.3f, tau_short_amplitude %.3f' % \
          (tau_long_amplitude, tau_short_amplitude))

    # plot output and minimization map
    plt.figure(1)
    ax = plt.gca()
    min_map = ax.imshow(log_log_likelihood_matrix, interpolation = 'none', cmap = cm.jet,
                        origin='lower', aspect = 1/(factor*tau_long_init), 
                        extent=[tau_short_array[0], tau_short_array[-1], 
                        tau_long_array[0], tau_long_array[-1]])
    ax.plot(road_to_convergence[:,1], road_to_convergence[:,0], marker='.', \
            color='w', markeredgecolor='k', markeredgewidth=0.8, linewidth=0.8)
    ax.plot(road_to_convergence[0,1], road_to_convergence[0,0], marker='o', \
            color='C3', markeredgecolor='w', markeredgewidth=0.8, linewidth=0.8)
    ax.plot(road_to_convergence[-1,1], road_to_convergence[-1,0], marker='o', \
            color='C2', markeredgecolor='w', markeredgewidth=0.8, linewidth=0.8)
    ax.set_ylabel('Long time (s)')
    ax.set_xlabel('Short time (s)')
    cbar = plt.colorbar(min_map, ax = ax)
    plt.grid(False)
    cbar.ax.set_title('log( -log( likelihood ) )', fontsize = 13)
    
    # show how the solution fits the histogram
    # prepare histogram binning
    bin_size = factor*exp_time*2 #factor 2 is arbitrary for PAPER
    number_of_bins = int((rango[1] - rango[0])/bin_size)

    counts, bin_edges = np.histogram(sample, bins=number_of_bins, range=rango)
    integral = np.sum(counts*np.diff(bin_edges))
    counts_norm = counts/integral
    bin_center = bin_edges[1:] - bin_size/2
    if hyper_exponential_flag:
        counts_norm_MLE = hyperexp_func_with_error(bin_center, tau_long_MLE, tau_short_MLE, ratio_MLE)
    else:
        counts_norm_MLE = monoexp_func(bin_center, tau_long_MLE, tau_short_MLE, ratio_MLE)
    
    # plot data and MLE output
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
    figure_name = 'binding_time_hist_MLE'
    aux_folder = os.path.dirname(full_filepath)        
    figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    
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
    ax.set_xlim([0,10])
    ax.set_ylim([1e-4,10])
    plt.legend(loc='upper right', prop={'size':12})
    plt.tick_params(axis='both', which='major', labelsize=18)
    figure_name = 'binding_time_hist_MLE'
    aux_folder = os.path.dirname(full_filepath)        
    figure_path = os.path.join(aux_folder, '%s.png' % figure_name)
    plt.savefig(figure_path, dpi = 300, bbox_inches='tight')
    
    
    # estimate error as a likelihood interval
    # see "Probability and statistics in particle physics" of A. G. Froedsen, 
    # 1979 ed., page 221 and on, Chapter 9.6 (particularly, page 233)
    # variable assumed to be independent 
    a = likelihood_err_param
    if hyper_exponential_flag:
        MLE_value = log_likelihood_hyper_with_error(out_estimator.x, sample)
    else:
        MLE_value = log_likelihood_mono_with_error(out_estimator.x, sample)

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
    #######################################################################
    # calculate errors: 
    tau_long_MLE_error_plus = np.abs(out_tau_on_error_estimator_plus.x[0] - tau_long_MLE)
    tau_long_MLE_error_minus = np.abs(out_tau_on_error_estimator_minus.x[0] - tau_long_MLE)
    print('\nError on tau_long (s): +%.3f / -%.3f' % (tau_long_MLE_error_plus, tau_long_MLE_error_minus))
    if hyper_exponential_flag:
        tau_short_MLE_error_plus = np.abs(out_tau_short_error_estimator_plus.x[0] - tau_short_MLE)
        tau_short_MLE_error_minus = np.abs(out_tau_short_error_estimator_minus.x[0] - tau_short_MLE)
        print('Error on tau_short (s): +%.3f / -%.3f' % (tau_short_MLE_error_plus, tau_short_MLE_error_minus))
    ratio_MLE_error_plus = np.abs(out_ratio_error_estimator_plus.x[0] - ratio_MLE)
    ratio_MLE_error_minus = np.abs(out_ratio_error_estimator_minus.x[0] - ratio_MLE)
    print('Error on ratio (s): +%.3f / -%.3f' % (ratio_MLE_error_plus, ratio_MLE_error_minus))
    
    ########################################################################
    
    # make solutions array
    if hyper_exponential_flag:
        solutions = np.array([[tau_long_MLE, tau_long_MLE_error_plus, tau_long_MLE_error_minus],
                          [tau_short_MLE, tau_short_MLE_error_plus, tau_short_MLE_error_minus],
                          [ratio_MLE, ratio_MLE_error_plus, ratio_MLE_error_minus]])
    else:
        solutions = np.array([[tau_long_MLE, tau_long_MLE_error_plus, tau_long_MLE_error_minus],
                            [0, 0, 0],
                            [ratio_MLE, ratio_MLE_error_plus, ratio_MLE_error_minus]])
        
    
    ########################################################################
    
    # uncomment the following lines to calculate and plot the
    # independant log likelihood and check the independent variable assumption
    # does not include the monoexp version
    
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

    plt.show()

    return solutions

#########################################################################
#########################################################################
#########################################################################

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
                                    opt_display_flag)
