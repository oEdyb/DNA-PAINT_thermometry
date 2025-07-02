"""
Created on Tuesday July 2nd 2025

@author: Devin AI (refactored from original code by Mariano Barella)

This module contains extracted functions from the step4 MLE estimation files to make
the code more modular and maintainable. These functions handle various aspects
of Maximum Likelihood Estimation including optimization setup, error calculations,
plotting, and results compilation.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import time
from auxiliary_functions import (hyperexp_func, monoexp_func, hyperexp_func_with_error,
                               monoexp_func_with_error, log_likelihood_hyper_with_error, 
                               log_likelihood_mono_with_error)


def setup_mle_optimization(hyper_exponential_flag, initial_params):
    """Setup optimization parameters and constraints for MLE."""
    if hyper_exponential_flag:
        tau_long_init, tau_short_init, set_ratio = initial_params
        bounds = [(0.01, 100), (0.01, 100), (0.01, 0.99)]
        initial_guess = [tau_long_init, tau_short_init, set_ratio]
        
        constr_array = np.array([1, -1, 0])
        constr_dict = {'type': 'ineq', 'fun': lambda x: np.dot(constr_array, x)}
        constraints = [constr_dict]
    else:
        tau_long_init, tau_short_init, set_ratio = initial_params
        bounds = [(0.01, 100), (0.01, 100), (0.01, 0.99)]
        initial_guess = [tau_long_init, tau_short_init, set_ratio]
        constraints = []
    
    return bounds, initial_guess, constraints


def perform_mle_optimization(sample, hyper_exponential_flag, bounds, initial_guess, 
                           constraints, opt_display_flag, road_to_convergence):
    """Perform the core MLE optimization process."""
    
    def callback_fun_trust(X, log_output):
        road_to_convergence.append(list(X))
        return 
    
    def callback_fun(X):
        road_to_convergence.append(list(X))
        return 
    
    if hyper_exponential_flag:
        objective_func = lambda x: -log_likelihood_hyper_with_error(x, sample)
    else:
        objective_func = lambda x: -log_likelihood_mono_with_error(x, sample)
    
    if opt_display_flag:
        print('Starting MLE optimization...')
    
    out_estimator = opt.minimize(
        objective_func,
        initial_guess,
        method='trust-constr',
        bounds=bounds,
        constraints=constraints,
        callback=callback_fun_trust,
        options={'disp': opt_display_flag, 'maxiter': 2000}
    )
    
    if opt_display_flag:
        print('Optimization completed.')
        print(f'Success: {out_estimator.success}')
        print(f'Message: {out_estimator.message}')
    
    return out_estimator


def calculate_parameter_errors(out_estimator, sample, hyper_exponential_flag, likelihood_err_param):
    """Calculate error intervals for all MLE parameters."""
    tau_long_MLE, tau_short_MLE, ratio_MLE = out_estimator.x
    
    if hyper_exponential_flag:
        MLE_value = log_likelihood_hyper_with_error(out_estimator.x, sample)
    else:
        MLE_value = log_likelihood_mono_with_error(out_estimator.x, sample)
    
    a = likelihood_err_param
    
    if hyper_exponential_flag:
        log_likelihood_tau_on_var = lambda x: np.abs(
            log_likelihood_hyper_with_error([x, tau_short_MLE, ratio_MLE], sample) - (MLE_value + a)
        )
    else:
        log_likelihood_tau_on_var = lambda x: np.abs(
            log_likelihood_mono_with_error([x, tau_short_MLE, ratio_MLE], sample) - (MLE_value + a)
        )
    
    initial_guess = tau_long_MLE + 0.1
    out_tau_on_error_estimator_plus = opt.root(
        log_likelihood_tau_on_var, initial_guess, method='hybr',
        options={'xtol': 1e-16, 'maxfev': 2000}
    )
    
    initial_guess = tau_long_MLE - 0.1
    out_tau_on_error_estimator_minus = opt.root(
        log_likelihood_tau_on_var, initial_guess, method='hybr',
        options={'xtol': 1e-16, 'maxfev': 2000}
    )
    
    tau_long_MLE_error_plus = np.abs(out_tau_on_error_estimator_plus.x[0] - tau_long_MLE)
    tau_long_MLE_error_minus = np.abs(out_tau_on_error_estimator_minus.x[0] - tau_long_MLE)
    
    tau_short_MLE_error_plus = 0
    tau_short_MLE_error_minus = 0
    
    if hyper_exponential_flag:
        log_likelihood_tau_short_var = lambda x: np.abs(
            log_likelihood_hyper_with_error([tau_long_MLE, x, ratio_MLE], sample) - (MLE_value + a)
        )
        
        initial_guess = tau_short_MLE + 0.1
        out_tau_short_error_estimator_plus = opt.root(
            log_likelihood_tau_short_var, initial_guess, method='hybr',
            options={'xtol': 1e-16, 'maxfev': 2000}
        )
        
        initial_guess = tau_short_MLE - 0.1
        out_tau_short_error_estimator_minus = opt.root(
            log_likelihood_tau_short_var, initial_guess, method='hybr',
            options={'xtol': 1e-16, 'maxfev': 2000}
        )
        
        tau_short_MLE_error_plus = np.abs(out_tau_short_error_estimator_plus.x[0] - tau_short_MLE)
        tau_short_MLE_error_minus = np.abs(out_tau_short_error_estimator_minus.x[0] - tau_short_MLE)
    
    if hyper_exponential_flag:
        log_likelihood_ratio_var = lambda x: np.abs(
            log_likelihood_hyper_with_error([tau_long_MLE, tau_short_MLE, x], sample) - (MLE_value + a)
        )
    else:
        log_likelihood_ratio_var = lambda x: np.abs(
            log_likelihood_mono_with_error([tau_long_MLE, tau_short_MLE, x], sample) - (MLE_value + a)
        )
    
    initial_guess = ratio_MLE + 0.01
    out_ratio_error_estimator_plus = opt.root(
        log_likelihood_ratio_var, initial_guess, method='hybr',
        options={'xtol': 1e-16, 'maxfev': 2000}
    )
    
    initial_guess = ratio_MLE - 0.01
    out_ratio_error_estimator_minus = opt.root(
        log_likelihood_ratio_var, initial_guess, method='hybr',
        options={'xtol': 1e-16, 'maxfev': 2000}
    )
    
    ratio_MLE_error_plus = np.abs(out_ratio_error_estimator_plus.x[0] - ratio_MLE)
    ratio_MLE_error_minus = np.abs(out_ratio_error_estimator_minus.x[0] - ratio_MLE)
    
    return {
        'tau_long_error_plus': tau_long_MLE_error_plus,
        'tau_long_error_minus': tau_long_MLE_error_minus,
        'tau_short_error_plus': tau_short_MLE_error_plus,
        'tau_short_error_minus': tau_short_MLE_error_minus,
        'ratio_error_plus': ratio_MLE_error_plus,
        'ratio_error_minus': ratio_MLE_error_minus
    }


def plot_mle_histogram(bin_center, counts_norm, counts_norm_MLE, bin_size, 
                      figures_folder, plot_name, plot_flag=True):
    """Plot standard MLE histogram with data and fit."""
    if not plot_flag:
        return None
        
    plt.figure(99)
    plt.bar(bin_center, counts_norm, width=bin_size, color='lightgrey', 
            edgecolor='k', label='Data')
    plt.plot(bin_center, counts_norm_MLE, '-', linewidth=0.1, color='#638bf9', 
             label='MLE')
    plt.grid()
    ax = plt.gca()
    ax.set_xlabel('Binding time (s)', fontsize=18)
    ax.set_ylabel('Normalized frequency', fontsize=18)
    ax.set_yscale('log')
    ax.set_axisbelow(True)
    plt.title('bin size %.3f s' % bin_size)
    plt.legend()
    
    figure_name = f'binding_time_hist_MLE_{plot_name}'
    figure_path = f'{figures_folder}/{figure_name}.png'
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return figure_path


def plot_mle_paper_figure(bin_center, counts_norm, counts_norm_MLE, bin_size,
                         tau_long_MLE, tau_short_MLE, hyper_exponential_flag,
                         figures_folder, plot_name, plot_flag=True):
    """Plot publication-ready MLE figure."""
    if not plot_flag:
        return None
        
    plt.figure(100)
    plt.bar(bin_center, counts_norm, width=bin_size, color='lightgrey', 
            edgecolor='k', label='Data')
    plt.plot(bin_center, counts_norm_MLE, '-', linewidth=3, color='#638bf9', 
             label='MLE')
    plt.grid()
    ax = plt.gca()
    ax.set_xlabel('Binding time (s)', fontsize=20)
    ax.set_ylabel('Normalized frequency', fontsize=20)
    ax.set_yscale('log')
    ax.set_axisbelow(True)
    ax.set_xlim(0, 10)
    ax.set_ylim(1e-4, 10)
    
    plt.legend(loc='upper right', prop={'size': 12})
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    if hyper_exponential_flag:
        tau_long_text = rf'$\tau_{{\mathrm{{long}}}} = {tau_long_MLE:.2f}\,s$'
        tau_short_text = rf'$\tau_{{\mathrm{{short}}}} = {tau_short_MLE:.2f}\,s$'
        plt.text(0.05, 0.95, tau_long_text + '\n' + tau_short_text, 
                transform=ax.transAxes, fontsize=14, verticalalignment='top')
    else:
        tau_long_text = rf'$\tau_{{\mathrm{{long}}}} = {tau_long_MLE:.2f}\,s$'
        plt.text(0.05, 0.95, tau_long_text, transform=ax.transAxes, 
                fontsize=14, verticalalignment='top')
    
    figure_name = f'binding_time_hist_MLE_paper_{plot_name}'
    figure_path = f'{figures_folder}/{figure_name}.png'
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return figure_path


def compile_mle_results(out_estimator, error_results, hyper_exponential_flag):
    """Compile final MLE results into a structured format."""
    tau_long_MLE, tau_short_MLE, ratio_MLE = out_estimator.x
    
    if hyper_exponential_flag:
        solutions = np.array([
            [tau_long_MLE, error_results['tau_long_error_plus'], error_results['tau_long_error_minus']],
            [tau_short_MLE, error_results['tau_short_error_plus'], error_results['tau_short_error_minus']],
            [ratio_MLE, error_results['ratio_error_plus'], error_results['ratio_error_minus']]
        ])
    else:
        solutions = np.array([
            [tau_long_MLE, error_results['tau_long_error_plus'], error_results['tau_long_error_minus']],
            [0, 0, 0],
            [ratio_MLE, error_results['ratio_error_plus'], error_results['ratio_error_minus']]
        ])
    
    return solutions


def prepare_histogram_data(sample, rango, exp_time, factor, hyper_exponential_flag, 
                          tau_long_MLE, tau_short_MLE, ratio_MLE):
    """Prepare histogram data for MLE visualization."""
    bin_size = factor * exp_time * 2  # factor 2 is arbitrary for PAPER
    number_of_bins = int((rango[1] - rango[0]) / bin_size)
    counts, bin_edges = np.histogram(sample, bins=number_of_bins, range=rango)
    integral = np.sum(counts * np.diff(bin_edges))
    counts_norm = counts / integral
    bin_center = bin_edges[1:] - bin_size / 2
    
    if hyper_exponential_flag:
        counts_norm_MLE = hyperexp_func_with_error(bin_center, tau_long_MLE, tau_short_MLE, ratio_MLE)
    else:
        counts_norm_MLE = monoexp_func(bin_center, tau_long_MLE, tau_short_MLE, ratio_MLE)
    
    return bin_center, counts_norm, counts_norm_MLE, bin_size


def load_sample_data(full_filepath, sample_data=None):
    """Load sample data from file or use provided data."""
    if sample_data is not None:
        sample = sample_data
    else:
        sample = np.loadtxt(full_filepath)
    
    sample = sample[~np.isnan(sample)]
    return sample


def print_mle_summary(out_estimator, error_results, hyper_exponential_flag, 
                     initial_params, bin_size, elapsed_time):
    """Print comprehensive MLE analysis summary."""
    tau_long_MLE, tau_short_MLE, ratio_MLE = out_estimator.x
    tau_long_init, tau_short_init, set_ratio = initial_params
    
    print("\n------ Summary of STEP 4 ------")
    print(f"MLE Fit Value: tau_long = {tau_long_MLE:.3f} s")
    print(f"Error on tau_long = +{error_results['tau_long_error_plus']:.3f} / -{error_results['tau_long_error_minus']:.3f} s")
    
    if hyper_exponential_flag:
        print(f"Initial Values: ratio = {set_ratio}, tau_short = {tau_short_init}")
        print(f"MLE Fit Values: tau_short = {tau_short_MLE:.3f} s, ratio = {ratio_MLE:.4f}")
        tau_short_amplitude = (1 - ratio_MLE) * ratio_MLE
        print(f"MLE Amplitude: tau_short amplitude = {tau_short_amplitude:.3f}")
        print(f"Errors on tau_short = +{error_results['tau_short_error_plus']:.3f} / -{error_results['tau_short_error_minus']:.3f} s")
        print(f"Errors on ratio = +{error_results['ratio_error_plus']:.3f} / -{error_results['ratio_error_minus']:.3f}")
    
    print(f"Histogram bin size: {bin_size}")
    print(f"Execution time of minimizer: {elapsed_time} seconds")
    print("Data analysis and optimization completed. Results and figures saved.")
