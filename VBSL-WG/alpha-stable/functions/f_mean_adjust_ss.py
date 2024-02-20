import numpy as np
import math 
# import numba
# from numba import njit


# Checking types
# # get numba types e.g.,
# print('alphatype: {}'.format(numba.typeof(alpha)))
# print('betatype: {}'.format(numba.typeof(beta)))
# print('gammatype: {}'.format(numba.typeof(gamma)))
# print('deltatype: {}'.format(numba.typeof(delta)))
# print('dataset_sizetype: {}'.format(numba.typeof(dataset_size)))
# print('num_datasetstype: {}'.format(numba.typeof(num_datasets)))
# print('rtype: {}'.format(numba.typeof(r)))

# datatype: array(float64, 1d, C)
# outputtype: array(float64, 1d, C)
# alphatype: float64
# betatype: float64
# gammatype: float64
# deltatype: float64
# dataset_sizetype: int64
# num_datasetstype: int64
# rtype: array(float64, 2d, C)

# to specify eagerly would be like (float64[:,::1](float64, float64, float64, float64, int64, int64))

# Be careful with which functions you are jit ing - there is overhead so 
# you don't want to jit everything

# GENERATE DATA FROM STABLE DISTRI 
# @njit
def alpha_stable(alpha, beta, gamma, delta, dataset_size, num_datasets):
    V = np.pi / 2 * (2 * np.random.rand(num_datasets, dataset_size) - 1)
    W = - np.log(np.random.rand(num_datasets, dataset_size))
    #r = np.zeros(n)
    
    if alpha != 1:
        const = beta * np.tan(np.pi * alpha / 2)
        b = math.atan(const)
        s = (1 + const * const)**(1 / (2 * alpha))
        r = s * np.sin(alpha * V + b) / ((np.cos(V)) ** (1/alpha)) * (( np.cos( (1-alpha) * V - b ) / W )**((1-alpha)/alpha))
        r = gamma * r + delta
    else:
        piover2 = np.pi / 2
        sclshftV = piover2 + beta * V
        r = 1/piover2 * (sclshftV * np.tan(V) - beta * np.log( (piover2 * W * np.cos(V) ) / sclshftV ))
        r = gamma * r + (2 / np.pi) * beta * gamma * np.log(gamma) + delta
    return r

# CALCULATE SUMMARY STATS 
# @njit
def alpha_stable_ss(data: np.ndarray) -> np.ndarray:
    # Compute quantile statistics
    v_stability = (np.percentile(data, 95) - np.percentile(data, 5)) / (np.percentile(data, 75) - np.percentile(data, 25))
    v_skewness = (np.percentile(data, 95) + np.percentile(data, 5) - 2 * np.percentile(data, 50)) / (np.percentile(data, 95) - np.percentile(data, 5))
    v_scale = (np.percentile(data, 75) - np.percentile(data, 25)) / 1
    v_loc = np.mean(data)

    # Define interpolation matrices (see [1])
    tv_stability = np.array([2.439, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0])
    tv_skewness = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    t_stability = np.array([2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    t_skewness = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    psi1 = np.array([[2.000, 2.000, 2.000, 2.000, 2.000, 2.000, 2.000],
    [1.916, 1.924, 1.924, 1.924, 1.924, 1.924, 1.924],
    [1.808, 1.813, 1.829, 1.829, 1.829, 1.829, 1.829],
    [1.729, 1.730, 1.737, 1.745, 1.745, 1.745, 1.745],
    [1.664, 1.663, 1.663, 1.668, 1.676, 1.676, 1.676],
    [1.563, 1.560, 1.553, 1.548, 1.547, 1.547, 1.547],
    [1.484, 1.480, 1.471, 1.460, 1.448, 1.438, 1.438],
    [1.391, 1.386, 1.378, 1.364, 1.337, 1.318, 1.318],
    [1.279, 1.273, 1.266, 1.250, 1.210, 1.184, 1.150],
    [1.128, 1.121, 1.114, 1.101, 1.067, 1.027, 0.973],
    [1.029, 1.021, 1.014, 1.004, 0.974, 0.935, 0.874],
    [0.896, 0.892, 0.887, 0.883, 0.855, 0.823, 0.769],
    [0.818, 0.812, 0.806, 0.801, 0.780, 0.756, 0.691],
    [0.698, 0.695, 0.692, 0.689, 0.676, 0.656, 0.595],
    [0.593, 0.590, 0.588, 0.586, 0.579, 0.563, 0.513]])

    psi2 = np.array([[0.000, 2.160, 1.000, 1.000, 1.000, 1.000, 1.000],
    [0.000, 1.592, 3.390, 1.000, 1.000, 1.000, 1.000],
    [0.000, 0.759, 1.800, 1.000, 1.000, 1.000, 1.000],
    [0.000, 0.482, 1.048, 1.694, 1.000, 1.000, 1.000],
    [0.000, 0.360, 0.760, 1.232, 2.229, 1.000, 1.000],
    [0.000, 0.253, 0.518, 0.823, 1.575, 1.000, 1.000],
    [0.000, 0.203, 0.410, 0.632, 1.244, 1.906, 1.000],
    [0.000, 0.165, 0.332, 0.499, 0.943, 1.560, 1.000],
    [0.000, 0.136, 0.271, 0.404, 0.689, 1.230, 2.195],
    [0.000, 0.109, 0.216, 0.323, 0.539, 0.827, 1.917],
    [0.000, 0.096, 0.190, 0.284, 0.472, 0.693, 1.759],
    [0.000, 0.082, 0.163, 0.243, 0.412, 0.601, 1.596],
    [0.000, 0.074, 0.147, 0.220, 0.377, 0.546, 1.482],
    [0.000, 0.064, 0.128, 0.191, 0.330, 0.478, 1.362],
    [0.000, 0.056, 0.112, 0.167, 0.285, 0.428, 1.274]])

    psi3 = np.array([[1.908, 1.908, 1.908, 1.908, 1.908],
    [1.914, 1.915, 1.916, 1.918, 1.921],
    [1.921, 1.922, 1.927, 1.936, 1.947],
    [1.927, 1.930, 1.943, 1.961, 1.987],
    [1.933, 1.940, 1.962, 1.997, 2.043],
    [1.939, 1.952, 1.988, 2.045, 2.116],
    [1.946, 1.967, 2.022, 2.106, 2.211],
    [1.955, 1.984, 2.067, 2.188, 2.333],
    [1.965, 2.007, 2.125, 2.294, 2.491],
    [1.980, 2.040, 2.205, 2.435, 2.696],
    [2.000, 2.085, 2.311, 2.624, 2.973],
    [2.040, 2.149, 2.461, 2.886, 3.356],
    [2.098, 2.244, 2.676, 3.265, 3.912],
    [2.189, 2.392, 3.004, 3.844, 4.775],
    [2.337, 2.635, 3.542, 4.808, 6.247],
    [2.588, 3.073, 4.534, 6.636, 9.144]])


    psi4 = np.array([[0.0,    0.0,    0.0,    0.0,  0.0],  
    [0.0, -0.017, -0.032, -0.049, -0.064],
    [0.0, -0.030, -0.061, -0.092, -0.123],
    [0.0, -0.043, -0.088, -0.132, -0.179],
    [0.0, -0.056, -0.111, -0.170, -0.232],
    [0.0, -0.066, -0.134, -0.206, -0.283],
    [0.0, -0.075, -0.154, -0.241, -0.335],
    [0.0, -0.084, -0.173, -0.276, -0.390],
    [0.0, -0.090, -0.192, -0.310, -0.447],
    [0.0, -0.095, -0.208, -0.346, -0.508],
    [0.0, -0.098, -0.223, -0.383, -0.576],
    [0.0, -0.099, -0.237, -0.424, -0.652],
    [0.0, -0.096, -0.250, -0.469, -0.742],
    [0.0, -0.089, -0.262, -0.520, -0.853],
    [0.0, -0.078, -0.272, -0.581, -0.997],
    [0.0, -0.061, -0.279, -0.659, -1.198]])

    tv_stability_i1 = max(np.append(0, np.argwhere(tv_stability <= v_stability)))
    tv_stability_i2 = min(np.append(14, np.argwhere(tv_stability >= v_stability)))
    tv_skewness_i1 = max(np.append(0, np.argwhere(tv_skewness <= abs(v_skewness))))
    tv_skewness_i2 = min(np.append(6, np.argwhere(tv_skewness >= abs(v_skewness))))
    dist_stability = tv_stability[tv_stability_i2] - tv_stability[tv_stability_i1]
    if dist_stability != 0:
        dist_stability = (v_stability - tv_stability[tv_stability_i1]) / dist_stability

    dist_skewness = tv_skewness[tv_skewness_i2] - tv_skewness[tv_skewness_i1]
    if dist_skewness != 0:
        dist_skewness = (abs(v_skewness) - tv_skewness[tv_skewness_i1]) / dist_skewness

    psi1b1 = dist_stability*psi1[tv_stability_i2,tv_skewness_i1]+(1-dist_stability)*psi1[tv_stability_i1,tv_skewness_i1]
    psi1b2 = dist_stability*psi1[tv_stability_i2,tv_skewness_i2]+(1-dist_stability)*psi1[tv_stability_i1,tv_skewness_i2]
    alpha = dist_skewness*psi1b2+(1-dist_skewness)*psi1b1
    psi2b1 = dist_stability*psi2[tv_stability_i2,tv_skewness_i1]+(1-dist_stability)*psi2[tv_stability_i1,tv_skewness_i1]
    psi2b2 = dist_stability*psi2[tv_stability_i2,tv_skewness_i2]+(1-dist_stability)*psi2[tv_stability_i1,tv_skewness_i2]
    beta = np.sign(v_skewness)*(dist_skewness*psi2b2+(1-dist_skewness)*psi2b1)
    t_stability_i1 = max(np.append(0, np.argwhere(t_stability >= alpha)))
    t_stability_i2 = min(np.append(15, np.argwhere(t_stability <= alpha)))
    t_skewness_i1 = max(np.append(0, np.argwhere(t_skewness <= abs(beta))))
    t_skewness_i2 = min(np.append(4, np.argwhere(t_skewness >= abs(beta))))

    dist_stability = t_stability[t_stability_i2] - t_stability[t_stability_i1]
    if dist_stability != 0:
        dist_stability = (alpha - t_stability[t_stability_i1]) / dist_stability

    dist_skewness = t_skewness[t_skewness_i2] - t_skewness[t_skewness_i1]
    if dist_skewness != 0:
        dist_skewness = (abs(beta) - t_skewness[t_skewness_i1]) / dist_skewness

    psi3b1 = dist_stability*psi3[t_stability_i2,t_skewness_i1]+(1-dist_stability)*psi3[t_stability_i1,t_skewness_i1]
    psi3b2 = dist_stability*psi3[t_stability_i2,t_skewness_i2]+(1-dist_stability)*psi3[t_stability_i1,t_skewness_i2]
    sigma = v_scale/(dist_skewness*psi3b2+(1-dist_skewness)*psi3b1)
    psi4b1 = dist_stability*psi4[t_stability_i2,t_skewness_i1]+(1-dist_stability)*psi4[t_stability_i1,t_skewness_i1]
    psi4b2 = dist_stability*psi4[t_stability_i2,t_skewness_i2]+(1-dist_stability)*psi4[t_stability_i1,t_skewness_i2]
    zeta = np.sign(beta)*sigma*(dist_skewness*psi4b2+(1-dist_skewness)*psi4b1) + np.percentile(data, 50)

    if abs(alpha-1) < 0.05:
        mu = zeta
    else:
        mu = zeta - beta * sigma * math.tan(0.5 * math.pi *alpha)

    return np.array([alpha, beta, sigma, mu]) #stability, skewness, scale, loc

# @jit
def summary_statistics(theta, n_samples, n_datasets):
    datasets = alpha_stable(theta[0], theta[1], theta[2], theta[3], n_samples, n_datasets)
    n_summary_statistics = np.array([alpha_stable_ss(datasets[i,:]) for i in range(n_datasets)])
    sample_mean = np.mean(n_summary_statistics, axis = 0)
    sample_variance = np.cov(n_summary_statistics.T)
    return sample_mean, sample_variance

# @jit
def variance_adjustment_summary_statistics(adjusted_theta, n_samples, n_datasets, num_coeffs, num_latent):
    theta = adjusted_theta[0:num_coeffs]
    Gamma = adjusted_theta[-num_latent:]
    sample_mean, sample_variance = summary_statistics(theta, n_samples, n_datasets)
    adjusted_sample_variance = sample_variance + np.diag(np.diag(sample_variance) * (Gamma ** 2))
    return sample_mean, adjusted_sample_variance

def mean_adjustment_summary_statistics(adjusted_theta, n_samples, n_datasets, num_coeffs, num_latent):
    theta = adjusted_theta[0:num_coeffs]
    Gamma = adjusted_theta[-num_latent:]
    sample_mean, sample_variance = summary_statistics(theta, n_samples, n_datasets)
    adjusted_sample_mean = sample_mean + np.diag(np.nan_to_num(np.sqrt(sample_variance), nan=0)) * Gamma
    return adjusted_sample_mean, sample_variance