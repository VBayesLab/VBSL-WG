import os
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
#import numdifftools as nd
import pandas as pd
#import pymc3 as pm

from tqdm import tqdm
from sklearn import preprocessing
from numpy.linalg import multi_dot
import scipy
import scipy.stats as ss
from scipy.stats import invgamma
from scipy.special import gamma
from scipy.special import digamma
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import levy_stable
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
import sklearn.mixture
from scipy.linalg import sqrtm


import logging
import warnings
from functools import partial

eps = 0.01
eps_precision = 100
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

lags = [1, 2, 4, 8]
num_coeffs = 3
num_latent = 48
num_datasets = 200
n_samples = 200
## Generate actual data
def toad(alpha,
         gamma,
         p0,
         n_toads=66,
         n_days=63,
         batch_size=1,
         random_state=None):
    """Sample the movement of Fowler's toad species.
    Models foraging steps using a levy_stable distribution where individuals
    either return to a previous site or establish a new one.
    References
    ----------
    Marchand, P., Boenke, M., and Green, D. M. (2017).
    A stochastic movement model reproduces patterns of site fidelity and long-
    distance dispersal in a population of fowlers toads (anaxyrus fowleri).
    Ecological Modelling,360:63–69.
    Parameters
    ----------
    alpha : float or array_like with batch_size entries
        step length distribution stability parameter
    gamma : float or array_like with batch_size entries
        step lentgh distribution scale parameter
    p0 : float or array_like with batch_size entries
        probability of returning to a previous refuge site
    n_toads : int, optional
        number of toads
    n_days : int, optional
        number of days
    batch_size : int, optional
    random_state : RandomState, optional
    Returns
    -------
    np.ndarray in shape (n_days x n_toads x batch_size)
    """
    X = np.zeros((n_days, n_toads, batch_size))
    random_state = random_state or np.random
    step_gen = ss.levy_stable
    step_gen.random_state = random_state

    for i in range(1, n_days):
        ret = random_state.uniform(0, 1, (n_toads, batch_size)) < np.squeeze(p0)
        non_ret = np.invert(ret)

        delta_x = step_gen.rvs(alpha, beta=0, scale=gamma, size=(n_toads, batch_size))
        X[i, non_ret] = X[i-1, non_ret] + delta_x[non_ret]

        ind_refuge = random_state.choice(i, size=(n_toads, batch_size))
        X[i, ret] = X[ind_refuge[ret], ret]

    return X


def compute_summaries(X, lag, p=np.linspace(0, 1, 3), thd=10):
    """Compute summaries for toad model.
    For displacements over lag...
        Log of the differences in the p quantiles
        The number of absolute displacements less than thd
        Median of the absolute displacements greater than thd
    Parameters
    ----------
    X : np.array of shape (ndays x ntoads x batch_size)
        observed matrix of toad displacements
    lag : list of ints, optional
        the number of days behind to compute displacement with
    p : np.array, optional
        quantiles used in summary statistic calculation (default 0, 0.1, ... 1)
    thd : float
        toads are considered returned when absolute displacement does not exceed thd (default 10m)
    Returns
    -------
    np.ndarray in shape (batch_size x len(p) + 1)
    """
    disp = obs_mat_to_deltax(X, lag)  # num disp at lag x batch size
    abs_disp = np.abs(disp)
    # returned toads
    ret = abs_disp < thd
    num_ret = np.sum(ret, axis=0)
    # non-returned toads
    abs_disp[ret] = np.nan  # ignore returned toads
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN slice encountered')
        # abs_noret_median = np.nanmedian(abs_disp, axis=0)
        abs_noret_quantiles = np.nanquantile(abs_disp, p, axis=0)
    diff = np.diff(abs_noret_quantiles, axis=0)
    logdiff = np.log(np.maximum(diff, np.exp(-20)))  # substitute zeros with a small positive
    # combine
    # ssx = np.vstack((num_ret, abs_noret_median, logdiff))
    ssx = np.vstack((np.log(num_ret), logdiff))
    ssx = np.nan_to_num(ssx, nan=np.inf)  # nans are when all toads returned
    return np.transpose(ssx)


def obs_mat_to_deltax(X, lag):
    """Convert an observation matrix to a vector of displacements.
    Parameters
    ----------
    X : np.array (n_days x n_toads x batch_size)
        observed matrix of toad displacements
    lag : int
        the number of days behind to compute displacement with
    Returns
    -------
    np.ndarray in shape (n_toads * (n_days - lag) x batch_size)
    """
    batch_size = np.atleast_3d(X).shape[-1]
    return (X[lag:] - X[:-lag]).reshape(-1, batch_size)

def compute_summaries_stacked(X, lags):
    S1 = compute_summaries(X, lags[0])
    S2 = compute_summaries(X, lags[1])
    S4 = compute_summaries(X, lags[2])
    S8 = compute_summaries(X, lags[3])
    return np.hstack((S1, S2, S4, S8))

def my_inv(x):
    return np.linalg.inv(x + (np.eye(x.shape[0]) * 1e-7))

# @jit
# def summary_statistics(theta, n_samples, n_datasets, mixture_obj_seq):
#     datasets = toad(theta[0], theta[1], theta[2],batch_size=n_datasets)
#     n_summary_statistics = np.array([compute_summaries_stacked(datasets[:,:,i], lags)[0] for i in range(n_datasets)])
#     # Wasserstein transform
#     transformed_summary_statistics = wasserstein_transform(mixture_obj_seq, n_summary_statistics)

#     sample_mean = np.mean(transformed_summary_statistics, axis = 0)
#     sample_variance = np.cov(np.array(transformed_summary_statistics).T)
#     return sample_mean, sample_variance

def summary_statistics(theta, n_datasets, actual_summary_statistics):
    datasets = toad(theta[0], theta[1], theta[2],batch_size=n_datasets)
    n_summary_statistics = np.array([compute_summaries_stacked(datasets[:,:,i], lags)[0] for i in range(n_datasets)])
    sample_mean = np.mean(n_summary_statistics, axis = 0)
    sample_precision = 1/eps_precision * np.identity(actual_summary_statistics.shape[0])
    for i in range(n_datasets): ## It will have i ranging from 0 to N-1
        diff = n_summary_statistics[i] - sample_mean
        sample_precision = sample_precision - ((1 + diff.T @ sample_precision @ diff)**(-1)) * (sample_precision @ diff.reshape(-1, 1) @ diff.reshape(1, -1) @ sample_precision)
    sample_precision = sample_precision * n_datasets
    return sample_mean, sample_precision

# def mean_adjustment_summary_statistics(adjusted_theta, n_samples, n_datasets, num_coeffs, num_latent, mixture_obj_seq):
#     theta = adjusted_theta[:num_coeffs]
#     Gamma = adjusted_theta[-num_latent:]
#     sample_mean, sample_variance = summary_statistics(theta, n_samples, n_datasets, mixture_obj_seq)
#     adjusted_sample_mean = sample_mean + np.diag(sqrtm(sample_variance)) * Gamma
#     return adjusted_sample_mean, sample_variance

def mean_adjustment_summary_statistics(adjusted_theta, n_datasets, num_coeffs, num_latent, actual_summary_statistics):
    theta = adjusted_theta[0:num_coeffs]
    Gamma = adjusted_theta[-num_latent:]
    sample_mean, sample_precision = summary_statistics(theta, n_datasets, actual_summary_statistics)
    # Create a zero matrix with shape (num_latent, num_latent)
    precision_matrix = np.zeros((num_latent, num_latent))
    # Set the diagonal elements of the precision matrix using NumPy's indexing
    precision_matrix[np.diag_indices_from(precision_matrix)] = (np.diag(sample_precision))**(-1/2)
    adjusted_sample_mean = sample_mean + np.matmul(precision_matrix, Gamma)
    return adjusted_sample_mean, sample_precision