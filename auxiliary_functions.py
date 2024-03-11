"""
Created on Tuesday Novemeber 17 2021

@author: Mariano Barella

This script contains the auxiliary functions that process_picasso_data
main script uses.

"""

import os
import numpy as np
import scipy.signal as sig
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from itertools import groupby
import scipy.stats as sta
import matplotlib.pyplot as plt


# time resolution at 100 ms
R = 0.07 # resolution width, in s
R = 0.00 # resolution width, in s


# 2D peak detection algorithm
# taken from https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    # apply the local maximum filter; all pixel of maximal value 
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    # local_max is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image==0)

    # a little technicality: we must erode the background in order to 
    # successfully subtract it form local_max, otherwise a line will 
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

# distance calculation circle
def distance(x, y, xc, yc):
    d = ((x - xc)**2 + (y - yc)**2)**0.5
    return d

def gaussian_2D_angle(xy_tuple, amplitude, x0, y0, a, b, c, offset):
    (x, y) = xy_tuple
    g = offset + amplitude*np.exp( -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2 ) )
    return g.ravel()

# def N_gaussians_2D(xy_tuple, amplitude, x0, y0, a, b, c, offset):
#     (x, y) = xy_tuple
#     g1 = offset1 + amplitude*np.exp( -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2 ) )
#     g2 = offset2 + amplitude*np.exp( -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2 ) )
#     g3 = offset3 + amplitude*np.exp( -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2 ) )
#     g = g1 + g2 + g3
#     return g.ravel()

def abc_to_sxsytheta(a, b, c):
    theta_rad = 0.5*np.arctan(2*b/(a-c))
    theta_deg = 360*theta_rad/(2*np.pi)
    aux_sx = a*(np.cos(theta_rad))**2 + \
             2*b*np.cos(theta_rad)*np.sin(theta_rad) + \
             c*(np.sin(theta_rad))**2
    sx = np.sqrt(0.5/aux_sx)
    aux_sy = a*(np.sin(theta_rad))**2 - \
             2*b*np.cos(theta_rad)*np.sin(theta_rad) + \
             c*(np.cos(theta_rad))**2
    sy = np.sqrt(0.5/aux_sy)
    return theta_deg, sx, sy

# Calculate coefficient of determination
def calc_r2(observed, fitted):
    avg_y = observed.mean()
    # sum of squares of residuals
    ssres = ((observed - fitted)**2).sum()
    # total sum of squares
    sstot = ((observed - avg_y)**2).sum()
    return 1.0 - ssres/sstot

# linear fit without weights
def fit_linear(x, y):
    X = np.vstack([x, np.ones(len(x))]).T
    p, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    x_fitted = np.array(x)
    y_fitted = np.polyval(p, x_fitted)
    Rsquared = calc_r2(y, y_fitted)
    # p[0] is the slope
    # p[1] is the intercept
    slope = p[0]
    intercept = p[1]
    return x_fitted, y_fitted, slope, intercept, Rsquared

def perpendicular_distance(slope, intercept, x_point, y_point):
    # source: https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    numerator = np.abs(slope*x_point + (-1)*y_point + intercept)
    denominator = distance(slope, (-1), 0, 0)
    d = numerator/denominator
    return d

def manage_save_directory(path, new_folder_name):
    # Small function to create a new folder if not exist.
    new_folder_path = os.path.join(path, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path

def classification(value, totalbins, rango):
    # Bin the data. Classify a value into a bin.
    # totalbins = number of bins to divide rango (range)
    bin_max = totalbins - 1
    numbin = 0
    inf = rango[0]
    sup = rango[1]
    if value > sup:
        print('Value higher than max')
        return bin_max
    if value < inf:
        print('Value lower than min')
        return 0
    step = (sup - inf)/totalbins
    # tiene longitud totalbins + 1
    # pero en total son totalbins "cajitas" (bines)
    binned_range = np.arange(inf, sup + step, step)
    while numbin < bin_max:
        if (value >= binned_range[numbin] and value < binned_range[numbin+1]):
            break
        numbin += 1
        if numbin > bin_max:
            break
    return numbin


def mask(number_of_dips = 1):
    # initialize mask array
    mask_array = [0, 1]
    if number_of_dips > 0:
        for i in range(number_of_dips-1):
            mask_array.append(0)
        mask_array.append(-1)
        mask_array.append(0)
    elif number_of_dips == -1:
        mask_array = [0,-1,1,0]
    elif number_of_dips == -99:
        mask_array = [0, 1, 1]
    else:
        mask_array = [0,0]
    mask_array = 0.5*np.array(mask_array)
    return mask_array

def calculate_tau_on_times(trace, threshold, bkg, exposure_time, mask_level, mask_singles, verbose_flag, index):
    # exposure_time in ms
    # threshold in number of photons (integer)
    number_of_frames = int(trace.shape[0])
    # while the trace is below the threshold leave 0, while is above replace by 1
    zero_trace = np.zeros(number_of_frames, dtype = int)
    binary_trace = np.where(trace < threshold, zero_trace, 1)
    photons_trace = np.where(trace > threshold, trace, 0)
    # calculate array of consecutive differences
    diff_binary = np.diff(binary_trace)
    
    if mask_level == 1:
        # mask 1 step dips using convolution
        if verbose_flag:
            print('Using convolution to mask single dips...')
        conv_one_dip = sig.convolve(diff_binary, mask(1))
        localization_index_dips = np.where(conv_one_dip == 1)[0] - 1
        binary_trace[localization_index_dips] = 1
    elif mask_level == 2:
        # mask 2 step dips using convolution
        if verbose_flag:
            print('Using convolution to mask double dips...')
        conv_two_dip = sig.convolve(diff_binary, mask(2))
        localization_index_dips = np.where(conv_two_dip == 1)[0] - 1
        binary_trace[localization_index_dips] = 1
        localization_index_dips = np.where(conv_two_dip == 1)[0] - 2
        binary_trace[localization_index_dips] = 1
    elif mask_level > 2:
        # several steps mask
        if verbose_flag:
            print('Using convolution to mask %d dips...' % mask_level)
        # conv = sig.convolve(diff_binary, mask(mask_level))
        # localization_index_dips = np.where(conv == 1)[0] - 1
        # binary_trace[localization_index_dips] = 1
        # localization_index_dips = np.where(conv_two_dip == 1)[0] - 2
        # binary_trace[localization_index_dips] = 1
    else:
        # no mask defined for convolution
        if verbose_flag:
            print('No convolution is going to be applied.')

    # remove 1 step blips using convolution
    if mask_singles:
        if verbose_flag:
            print('Using convolution to mask single blips...')

        conv_one_blip = sig.convolve(diff_binary, mask(-1))
        localization_index_blips = np.where(np.abs(conv_one_blip) == 1)[0] - 1
        binary_trace[localization_index_blips] = 0
        
    # now, with the trace "restored" we can estimate tau_on...
    
    # estimate number of frames the fluorophore was ON
    # keep indexes where localizations have been found (> 1)
    if verbose_flag:
        print('Calculating binding times...')
    localization_index = np.where(binary_trace > 0)[0]
    localization_index_diff = np.diff(localization_index)
    keep_steps = np.where(localization_index_diff == 1)[0]
    localization_index_steps = localization_index[keep_steps]
    binary_trace[localization_index_steps] = 1   



    # Starting time of the tau ons
    try:
        localization_index_start = []
        localization_index_start.append(localization_index[0]-1)
        localization_index_start_remaining = [localization_index[i+1]-1 for i, k in enumerate(localization_index_diff) if (k > 1)]
        localization_index_start.extend(localization_index_start_remaining)
    except:
        return np.array([False]), [False], [False], [False], [False], [False], [False]


    # conv_start_time = sig.convolve(diff_binary, mask(-99))
    # localization_index_start = np.where(conv_start_time == 1)[0] - 1

    # ### uncomment plot to check filters and binary trace
    # plt.figure()
    # plt.plot(trace/max(trace),'-')
    # plt.plot(photons_trace/max(photons_trace),'ok')
    # # plt.plot(binary_trace)
    # # plt.plot(conv_one_blip)
    # # plt.plot(conv_one_dip)
    # # plt.plot(conv_two_dip)
    # plt.xlim([0,10000])
    # plt.show()
    
    # calculate tau on and off,
    t_on = [len(l) for l in [list(g) for k, g in groupby(list(binary_trace), key = lambda x:x!=0) if k]]
    t_off = [len(l) for l in [list(g) for k, g in groupby(list(binary_trace), key = lambda x:x==0) if k]]

    # calculate SNR
    new_photon_trace = photons_trace*binary_trace

    # Compute avg and std when a docking location is emitting light. Since we are only considering the middle values of
    # the array (array[1:-1]) since the in the first and last values we can get docking locations which are not on
    # during the entire duration of the exposure time. We must therefore consider arrays larger than 3 to get more than
    # one value.
    sum_photons = [np.sum(np.array(l)) for l in [list(g) for k, g in groupby(list(new_photon_trace), key = lambda x:x!=0) if k]]
    avg_photons = [np.mean(np.array(l[1:-1])) for l in [list(g) for k, g in groupby(list(new_photon_trace), key = lambda x:x!=0) if k] if len(l) > 4]
    std_photons = [np.std(np.array(l[1:-1]), ddof = 1) for l in [list(g) for k, g in groupby(list(new_photon_trace), key = lambda x:x!=0) if k] if len(l) > 4]

    if binary_trace[0] == 1:
        t_on = t_on[1:]
        localization_index_start = localization_index_start[1:]
    else:
        t_off = t_off[1:]
    if binary_trace[-1] == 1:
        t_on = t_on[:-1]
        localization_index_start = localization_index_start[:-1]
    else:
        t_off = t_off[:-1]     
        
    t_on = np.asarray(t_on)
    sum_photons = np.asarray(sum_photons)
    # plt.plot(t_on), plt.show()
    t_off = np.asarray(t_off)
    start_time = np.asarray(localization_index_start)

    SNR = np.array(avg_photons)/np.array(std_photons)
    SBR = np.array(avg_photons)/bkg
    if verbose_flag:
        print('---------------------------')
    return t_on*exposure_time, t_off*exposure_time, binary_trace, start_time*exposure_time, SNR, SBR, sum_photons

# definition of hyperexponential p.d.f.
def hyperexp_func(time, real_binding_time, short_on_time, ratio):
    beta_binding_time = 1/real_binding_time
    beta_short_time = 1/short_on_time
    A = ratio/(ratio + 1)
    B = 1/(ratio + 1)
    f_binding = beta_binding_time*np.exp(-time*beta_binding_time)
    f_short = beta_short_time*np.exp(-time*beta_short_time)
    f = A*f_binding + B*f_short
    return f

# definition of monoexponential p.d.f.
def monoexp_func(time, real_binding_time, short_on_time, amplitude):
    # short on time is not used
    # neither the amplitude
    beta_binding_time = 1/real_binding_time
    f = beta_binding_time*np.exp(-time*beta_binding_time)
    return f

# definition of hyperexponential p.d.f. including instrumental error
def hyperexp_func_with_error(time, real_binding_time, short_on_time, ratio):
    beta_binding_time = 1/real_binding_time
    beta_short_time = 1/short_on_time
    A = ratio/(ratio + 1)
    B = 1/(ratio + 1)
    f_binding = beta_binding_time*np.exp(-time*beta_binding_time)
    f_short = beta_short_time*np.exp(-time*beta_short_time)
    argument_binding = time/R - beta_binding_time*R
    argument_short = time/R - beta_short_time*R
    std_norm_distro = sta.norm(loc=0, scale=1)
    G_binding = std_norm_distro.cdf(argument_binding)
    G_short = std_norm_distro.cdf(argument_short)
    f_binding_new = np.exp(0.5*(beta_binding_time*R)**2)*G_binding*f_binding
    f_short_new = np.exp(0.5*(beta_short_time*R)**2)*G_short*f_short
    f = A*f_binding_new + B*f_short_new
    return f


# definition of monoexponential p.d.f.
def monoexp_func_with_error(time, real_binding_time, short_on_time, amplitude):
    # short on time is not used
    beta = 1/real_binding_time
    f_mono = amplitude*beta*np.exp(-time*beta)
    argument_binding = time/R - beta*R
    std_norm_distro = sta.norm(loc=0, scale=1)
    G = std_norm_distro.cdf(argument_binding)
    f_mono_new = np.exp(0.5*(beta*R)**2)*G*f_mono
    return f_mono_new

# definition of hyperlikelihood function
def log_likelihood_hyper(theta_param, data):
    real_binding_time = theta_param[0]
    short_on_time = theta_param[1]
    ratio = theta_param[2]
    pdf_data = hyperexp_func(data, real_binding_time, short_on_time, ratio)
    log_likelihood = -np.sum(np.log(pdf_data))
    # print(log_likelihood)
    return log_likelihood

# definition of hyperlikelihood function
def log_likelihood_hyper_with_error(theta_param, data):
    real_binding_time = theta_param[0]
    short_on_time = theta_param[1]
    ratio = theta_param[2]
    pdf_data = hyperexp_func_with_error(data, real_binding_time, short_on_time, ratio)
    log_pdf = np.log(pdf_data)
    log_pdf = log_pdf[~np.isinf(log_pdf)]
    log_pdf = log_pdf[~np.isnan(log_pdf)]
    log_likelihood = -np.sum(log_pdf)
    # print(log_likelihood)
    return log_likelihood

# definition of hyperlikelihood function
def log_likelihood_mono_with_error(theta_param, data):
    # no error actually
    real_binding_time = theta_param[0]
    short_on_time = theta_param[1]
    ratio = theta_param[2]
    pdf_data = monoexp_func(data, real_binding_time, short_on_time, ratio)
    log_pdf = np.log(pdf_data)
    log_pdf = log_pdf[~np.isinf(log_pdf)]
    log_pdf = log_pdf[~np.isnan(log_pdf)]
    log_likelihood = -np.sum(log_pdf)
    # print(log_likelihood)

    return log_likelihood

def plot_vs_time_with_hist(data, time, order = 3, fit_line = False):
    dict = {}
    # Start with a square Figure.
    fig = plt.figure(figsize=(6, 6))  # Set figure size

    # Create a gridspec for 1 row and 2 columns with a ratio of 4:1 between the main and marginal plots.
    # Adjust subplot parameters for optimal layout.
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[4, 1],
                          left=0.15, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05)  # `hspace` is not needed for 1 row

    # Create the main plot area.
    ax = fig.add_subplot(gs[0, 0])

    for x, y in zip(time, data):
        if x in dict:
            dict[x] = np.append(dict[x], y)
        else:
            dict[x] = np.array([y])

    for x in dict.keys():
        dict[x] = np.mean(dict[x], axis=None)

    unique_time_values = np.array(list(dict.keys()))
    summed_data = np.array(list(dict.values()))
    sorted_indices = np.argsort(unique_time_values)
    unique_time_values = unique_time_values[sorted_indices]
    summed_data = summed_data[sorted_indices]

    filtered_data = sig.savgol_filter(summed_data, window_length=int(len(summed_data)/20), polyorder=1)
    if fit_line:
        x_fitted, y_fitted, slope, intercept, Rsquared = fit_linear(unique_time_values, filtered_data)
    ax.scatter(unique_time_values, summed_data, s=0.8)
    ax.plot(unique_time_values, filtered_data, 'r--', linewidth=3, alpha = 0.8)

    bin_edges = np.histogram_bin_edges(data, 'fd')
    # Create the marginal plot on the right of the main plot, sharing the y-axis with the main plot.
    ax_histy = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_histy.hist(data, bins=bin_edges, orientation='horizontal')

    # Make sure the marginal plot's y-axis ticks don't overlap with the main plot.
    plt.setp(ax_histy.get_yticklabels(), visible=False)
    x_limit = [0, unique_time_values[-1]]
    ax.set_xlim(x_limit)
    y_limit_photons = [0, round(np.max(data, axis=None)+10**order, -order)]
    ax.set_ylim(y_limit_photons)

    if fit_line:
        return ax, slope, intercept

    return ax
