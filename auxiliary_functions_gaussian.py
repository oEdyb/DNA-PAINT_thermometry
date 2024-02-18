import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# sx, sxy, sy Gaussian
def gaussian_2d_cov(xy_tuple, x0, y0, amp, sx, sy, rho, offset):
    (x, y) = xy_tuple
    xy = np.array([x, y]).reshape(2, -1)
    xy = np.array([x, y]).reshape(2, -1)
    cov = [[sx**2, rho*sx*sy], [rho*sx*sy, sy**2]]
    cov = np.array(cov)
    det_cov = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    a = cov_inv[0, 0]
    b = cov_inv[0, 1]
    c = cov_inv[1, 1]

    #g1 = ((np.pi) ** (-1)) * (np.sqrt(det_cov)) * np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))
    g2 = 1/(2*np.pi*sx*sy*np.sqrt(1-rho**2)) * np.exp(-1/(2*(1-rho**2)) * (((x-x0)/sx)**2 - 2*rho*((x-x0)/sx)*((y-y0)/sy) + ((y-y0)/sy)**2))
    return g2


def ll_gaussian_cov(theta, data):
    x0, y0, amplitude, sx, sy, rho, offset = theta
    model = gaussian_2d_cov((data[:, 0], data[:, 1]), x0, y0, amplitude, sx, sy, rho, offset)
    #print(theta)
    f = -np.sum(np.log(model))
    #print(f)
    return f


# a, b, c Gaussian
def gaussian_2d_angle(xy_tuple, x0, y0, amplitude, a, b, c, offset):
    (x, y) = xy_tuple
    g = amplitude*np.exp( -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2 ) ) + offset
    if np.isnan(g).any():
        print('nan detected')
    return g


def ll_gaussian_abc(theta, data):
    x0, y0, amplitude, a, b, c, offset = theta
    model = gaussian_2d_angle((data[:, 0], data[:, 1]), x0, y0, amplitude, a, b, c, offset)
    f = -np.log(np.sum(model))
    #print(f)
    return f


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


def multivariate_gaussian_pdf(x, y, mean, covariance_matrix):
    n = len(mean)
    constant = 1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(covariance_matrix) ** 0.5)
    # Calculate the exponent term including covariance
    exponent = -0.5 * (np.dot(np.dot((x - mean[0]), np.linalg.inv(covariance_matrix)[0, 0]), (x - mean[0]))
                       + np.dot(np.dot((y - mean[1]), np.linalg.inv(covariance_matrix)[1, 1]), (y - mean[1]))
                       + 2 * np.linalg.inv(covariance_matrix)[0, 1] * (x - mean[0]) * (y - mean[1]))
    return constant * np.exp(exponent)


def plot_gaussian_2d(xlims, ylims, x0, y0, amp, sx, sy, rho, offset, color='k'):
    # Plots a Gaussian at the contour levels of sigma and 2 * sigma.
    x = np.linspace(xlims[0], xlims[1], num=1000)
    y = np.linspace(ylims[0], ylims[1], num=1000)

    X, Y = np.meshgrid(x, y)
    g = gaussian_2d_cov((X, Y), x0, y0, amp, sx, sy, rho, offset)
    g_sigma = gaussian_2d_cov((x0 + sx, y0 + sy), x0, y0, amp, sx, sy, rho, offset)
    g_2sigma = gaussian_2d_cov((x0 + 2 * sx, y0 + 2 * sy), x0, y0, amp, sx, sy, rho, offset)

    plt.contour(X, Y, g, levels=[g_sigma], colors=color, alpha=0.6)

def derivative_abc_gaussian(xy_tuple, x0, y0, amplitude, a, b, c, offset):

    (x, y) = xy_tuple
    model = gaussian_2d_angle(xy_tuple, x0, y0, amplitude, a, b, c, offset)

    dgda = -model * (x-x0)**2
    d2gd2a = model * (x-x0)**4

    dgdb = -model * 2 * (x-x0)*(y-y0)
    d2gd2b = model * 4 * ((x-x0)*(y-y0))**2

    dgdc = -model * (y-y0)**2
    d2gd2c = model * (y-y0)**4

    dgdx0 = model * - (-a*2*(x-x0) - 2*b*(y-y0))
    d2gd2x0 = model * (-a*2*(x-x0) - 2*b*(y-y0))**2 + model * - (a * 2)

    dgdy0 = model * - (-c*2*(y-y0) - 2*b*(x-x0))
    d2gd2y0 = model * (-c*2*(x-x0) - 2*b*(y-y0))**2 + model * - (c * 2)

    dgdamplitude = model
    d2gd2amplitude = np.zeros(shape=(len(x)))

    dgddoffset = np.ones(shape=(len(x)))
    d2gd2offset = np.zeros(shape=(len(x)))

    dmodel = dgdx0, dgdy0, dgdamplitude, dgda, dgdb, dgdc, dgddoffset
    dmodel = np.array(dmodel)
    d2model = d2gd2x0, d2gd2y0, d2gd2amplitude, d2gd2a, d2gd2b, d2gd2c, d2gd2offset
    d2model = np.array(d2model)

    return dmodel, d2model


def wolfe_conditions(func, x, y, theta, theta_index, d_model, p, rho, maxit):
    delta_1, delta_2, it = 1, 1, 0
    c1 = 0.1
    c2 = 0.9
    #mask = np.zeros(len(theta))
    #mask[theta_index] = 1
    p_indexed = p
    while delta_1 > 0 and delta_2 > 0 and it < maxit:
        alpha = rho ** it
        step = alpha * p_indexed
        theta_step = theta + step
        f = -np.sum(np.log(func((x, y), theta[0], theta[1], theta[2], theta[3], theta[4], theta[5], theta[6])))
        f_step = -np.sum(np.log(func((x, y), theta_step[0], theta_step[1], theta_step[2], theta_step[3], theta_step[4], theta_step[5], theta_step[6])))

        delta_1 = f_step - f - c1 * alpha * np.dot(d_model, p_indexed)
        d_model_d_theta, _ = derivative_abc_gaussian((x, y), theta_step[0], theta_step[1], theta_step[2], theta_step[3], theta_step[4], theta_step[5], theta_step[6])
        model = gaussian_2d_angle((x, y), theta[0], theta[1], theta[2], theta[3], theta[4], theta[5],
                                  theta[6])
        d_of_log = 1 / model - 1
        d_ll_model = (d_of_log * d_model_d_theta)
        d_ll_model = np.sum(d_ll_model, axis=1)
        delta_2 = abs(np.dot(d_ll_model, p_indexed)) - c2 * np.dot(d_model, p_indexed)
        it += 1
    theta += step
    print(f_step-f)
    return theta