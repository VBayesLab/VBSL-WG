import numpy as np
from scipy.stats import multivariate_normal
import math 

from numba import vectorize, float64, guvectorize, jit

from tqdm import tqdm
from sklearn import preprocessing
from numpy.linalg import multi_dot
import scipy
from scipy import stats
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

from functions.f_mean_adjust_ss import mean_adjustment_summary_statistics 

def prior(theta, num_coeffs): 
    log_prior = multivariate_normal.logpdf(theta, cov= 10 * np.identity(num_coeffs))
    return log_prior

def mean_adjustment_unbiased_log_likelihood(gs_adjusted_theta, gamma_samples, n_samples, n_datasets, num_coeffs, num_latent, actual_summary_statistics):
    mean_unbiased_log_likelihood = []
    for i in range(gamma_samples):
        mean_adj_ss = mean_adjustment_summary_statistics(gs_adjusted_theta[i,:], n_samples, n_datasets, num_coeffs, num_latent)
        adjusted_sample_mean = mean_adj_ss[0]
        sample_variance = mean_adj_ss[1]

        diff_mean_s = actual_summary_statistics - adjusted_sample_mean
        part1 = diff_mean_s.T @ np.linalg.inv(sample_variance) @ diff_mean_s
        mean_u_est_log_likelihood = -1/2 * np.log(np.linalg.det(sample_variance)) - part1
        mean_unbiased_log_likelihood.append(mean_u_est_log_likelihood)
    return np.mean(mean_unbiased_log_likelihood)

def fun_log_q(theta, mu, l):
    log_q = multivariate_normal.logpdf(theta, mean = mu, cov= np.linalg.inv(l @ l.T))
    return log_q

def gradient_log_q(theta, mu, l, num_coeffs): #indep theta
    gradient_log_q_mu = np.matmul(np.matmul(l, l.T), (theta - mu))
    gradient_log_q_l = (np.diag(np.linalg.inv(l)) - np.matmul(((np.reshape(theta - mu, (num_coeffs,1))) * theta - mu), l)).T[np.triu_indices(num_coeffs)] #use * because matmul gives scalar 
    gradient_log_q = np.array([gradient_log_q_mu, gradient_log_q_l], dtype=object)
    return gradient_log_q

def fun_gradient_lb(s, theta_samples, mu_q, l_q, c, gamma_samples, n_samples, n_datasets, num_coeffs, num_latent, actual_summary_statistics):
    theta_tilde_q = theta_samples[s]
    # Calculate theta from mu, l (lambda)
    alpha_q = (2 * np.exp(theta_tilde_q[0]) + 1.1) / (1 + np.exp(theta_tilde_q[0]))
    beta_q = (np.exp(theta_tilde_q[1]) - 1) / (np.exp(theta_tilde_q[1]) + 1)
    gamma_q = np.exp(theta_tilde_q[2])
    delta_q = theta_tilde_q[3]
    theta_q = np.array([alpha_q, beta_q, gamma_q, delta_q])

    # GENERATE GAMMA
    Gamma = np.random.laplace(scale = 0.5, size = (gamma_samples, num_latent))
    adjusted_theta_q = np.concatenate((np.tile(theta_q, (gamma_samples, 1)), Gamma), axis = 1)

    # Find gradient of LB
    llh = mean_adjustment_unbiased_log_likelihood(adjusted_theta_q, gamma_samples, n_samples, n_datasets, num_coeffs, num_latent, actual_summary_statistics)
    h_lambda = prior(theta_tilde_q, num_coeffs) + llh - fun_log_q(theta_tilde_q, mu_q, l_q)
    
    # Find gradient of LB
    grad_log_q = gradient_log_q(theta_tilde_q, mu_q, l_q, num_coeffs)
    gradient_lb = gradient_log_q(theta_tilde_q, mu_q, l_q, num_coeffs) * (h_lambda - c)
    
    # Calculate control variates
    flattened_gradient_log_q = np.concatenate((grad_log_q[0], grad_log_q[1]), axis = None)
    flattened_gradient_lb = np.concatenate((gradient_lb[0], gradient_lb[1]), axis = None)
 
    return gradient_lb, h_lambda, flattened_gradient_log_q, flattened_gradient_lb