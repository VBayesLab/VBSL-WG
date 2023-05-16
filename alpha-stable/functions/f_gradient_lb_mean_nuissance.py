import numpy as np
from scipy.stats import multivariate_normal
import math 

# from numba import vectorize, float64, guvectorize, jit

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

from f_mean_adjust_ss import summary_statistics, mean_adjustment_summary_statistics 

sigma_theta = 10
sigma_latent = 1

def prior(theta, sigma_theta, num_coeffs): 
    log_prior = multivariate_normal.logpdf(theta, cov= sigma_theta * np.identity(num_coeffs))
    return log_prior

def mean_adjustment_unbiased_log_likelihood(gs_adjusted_theta, n_samples, n_datasets, num_coeffs, num_latent, actual_summary_statistics):
    mean_adj_ss = mean_adjustment_summary_statistics(gs_adjusted_theta, n_samples, n_datasets, num_coeffs, num_latent)
    adjusted_sample_mean = mean_adj_ss[0]
    sample_variance = mean_adj_ss[1]

    diff_mean_s = actual_summary_statistics - adjusted_sample_mean
    part1 = diff_mean_s.T @ np.linalg.inv(sample_variance) @ diff_mean_s
    mean_u_est_log_likelihood = -1/2 * np.log(np.linalg.det(sample_variance)) - part1
    return mean_u_est_log_likelihood

def nuissance_mean_adjustment_unbiased_log_likelihood(Gamma, mean_nuissance, variance_nuissance):
    part1 = (Gamma - mean_nuissance).T @ np.linalg.inv(variance_nuissance) @ (Gamma - mean_nuissance)
    nuissance_mean_u_est_log_likelihood = -1/2 * np.log(np.linalg.det(variance_nuissance)) - part1
    return nuissance_mean_u_est_log_likelihood

def fun_log_q(theta, mu, l):
    log_q = multivariate_normal.logpdf(theta, mean = mu, cov= np.linalg.inv(l @ l.T))
    return log_q

def gradient_log_q(theta, mu, l, num_coeffs): #indep theta
    gradient_log_q_mu = np.matmul(np.matmul(l, l.T), (theta - mu))
    gradient_log_q_l = (np.diag(np.linalg.inv(l)) - np.matmul(((np.reshape(theta - mu, (num_coeffs,1))) * theta - mu), l)).T[np.triu_indices(num_coeffs)] #use * because matmul gives scalar 
    gradient_log_q = np.array([gradient_log_q_mu, gradient_log_q_l], dtype=object)
    return gradient_log_q

def fun_gradient_lb(s, theta_samples, mu_q, l_q, c, n_samples, n_datasets, num_coeffs, num_latent, actual_summary_statistics):
    theta_tilde_q = theta_samples[s]
    # Calculate theta from mu, l (lambda)
    alpha_q = (2 * np.exp(theta_tilde_q[0]) + 1.1) / (1 + np.exp(theta_tilde_q[0]))
    beta_q = (np.exp(theta_tilde_q[1]) - 1) / (np.exp(theta_tilde_q[1]) + 1)
    gamma_q = np.exp(theta_tilde_q[2])
    delta_q = theta_tilde_q[3]
    theta_q = np.array([alpha_q, beta_q, gamma_q, delta_q])

    ss_q = summary_statistics(theta_q, n_samples, n_datasets)
    sample_mean_q = ss_q[0]
    sample_variance_q = ss_q[1]

    # Find mean and variance for p(gamma | theta, obs)

    mean_nuissance_p1 = np.linalg.inv(np.identity(num_latent) / sigma_latent + np.diag(sample_variance_q).T @ np.linalg.inv(sample_variance_q) @ np.diag(sample_variance_q))
    diag_var = np.zeros((num_coeffs, num_coeffs))
    np.fill_diagonal(diag_var, np.diag(sample_variance_q)) 
    diff_mean = actual_summary_statistics - sample_mean_q
    mean_nuissance_p2 = diag_var @ np.linalg.inv(sample_variance_q) @ diff_mean
    mean_nuissance_q = mean_nuissance_p1 @ mean_nuissance_p2

    variance_nuissance_q = mean_nuissance_p1

    # GENERATE GAMMA
    Gamma = multivariate_normal.rvs(mean = mean_nuissance_q, cov = variance_nuissance_q)
    adjusted_theta_q = np.concatenate((theta_q, Gamma))

    # Find gradient of LB
    llh = mean_adjustment_unbiased_log_likelihood(adjusted_theta_q, n_samples, n_datasets, num_coeffs, num_latent, actual_summary_statistics)
    llh_gamma = nuissance_mean_adjustment_unbiased_log_likelihood(Gamma, mean_nuissance_q, variance_nuissance_q)

    h_lambda = prior(theta_tilde_q, sigma_theta, num_coeffs) + prior(Gamma, sigma_latent, num_latent) + llh - fun_log_q(theta_tilde_q, mu_q, l_q) - llh_gamma
    
    # Find gradient of LB
    grad_log_q = gradient_log_q(theta_tilde_q, mu_q, l_q, num_coeffs)
    gradient_lb = gradient_log_q(theta_tilde_q, mu_q, l_q, num_coeffs) * (h_lambda - c)
    
    # Calculate control variates
    flattened_gradient_log_q = np.concatenate((grad_log_q[0], grad_log_q[1]), axis = None)
    flattened_gradient_lb = np.concatenate((gradient_lb[0], gradient_lb[1]), axis = None)
 
    return gradient_lb, h_lambda, flattened_gradient_log_q, flattened_gradient_lb
