import numpy as np
from scipy.stats import multivariate_normal
import math 

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
from scipy.linalg import sqrtm
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

from f_mean_adjust_ss_wasserstein_toad_without_tau import summary_statistics, mean_adjustment_summary_statistics  

sigma_theta = 10
sigma_latent = 1

def my_inv(x):
    return np.linalg.inv(x + (np.eye(x.shape[0]) * 1e-7))

def prior(theta): 
    log_prior = np.sum(np.log(np.exp(theta) / (1 + np.exp(theta))**2))
    return log_prior

def mean_adjustment_unbiased_log_likelihood(gs_adjusted_theta, n_samples, n_datasets, num_coeffs, num_latent, actual_summary_statistics):
    mean_adj_ss = mean_adjustment_summary_statistics(gs_adjusted_theta, n_samples, n_datasets, num_coeffs, num_latent, actual_summary_statistics)
    adjusted_sample_mean = mean_adj_ss[0]
    sample_precision = mean_adj_ss[1]

    # mean_u_est_log_likelihood = multivariate_normal.logpdf(actual_summary_statistics, mean = adjusted_sample_mean, cov= sample_variance)

    diff_mean_s = actual_summary_statistics - adjusted_sample_mean
    part1 = diff_mean_s.T @ sample_precision @ diff_mean_s
    mean_u_est_log_likelihood = 1/2 * np.linalg.slogdet(sample_precision)[1] - 1/2 * part1
    return mean_u_est_log_likelihood

def nuissance_mean_adjustment_unbiased_log_likelihood(Gamma, mean_nuissance, variance_nuissance):
    # part1 = (Gamma - mean_nuissance).T @ np.linalg.inv(variance_nuissance) @ (Gamma - mean_nuissance)
    # nuissance_mean_u_est_log_likelihood = -1/2 * np.log(np.linalg.det(variance_nuissance)) - part1
    nuissance_mean_u_est_log_likelihood = multivariate_normal.logpdf(Gamma, mean = mean_nuissance, cov= variance_nuissance)
    return nuissance_mean_u_est_log_likelihood

def fun_log_q(theta, mu, l):
    log_q = multivariate_normal.logpdf(theta, mean = mu, cov= my_inv(l @ l.T))
    return log_q

def gradient_log_q(theta, mu, l, num_coeffs): #indep theta
    gradient_log_q_mu = l @ l.T @ (theta - mu)
    # gradient_log_q_l = (np.diag(np.linalg.inv(l)) - np.matmul(((np.reshape(theta - mu, (num_coeffs,1))) * theta - mu), l)).T[np.triu_indices(num_coeffs)] #use * because matmul gives scalar 
    diag_inv_l = np.zeros((num_coeffs, num_coeffs))
    np.fill_diagonal(diag_inv_l, np.diag(my_inv(l)))
    gradient_log_q_l = (diag_inv_l - np.reshape(theta - mu, (num_coeffs,1)) @ np.reshape(theta - mu, (1,num_coeffs)) @ l).T[np.triu_indices(num_coeffs)] #use * because matmul gives scalar 
    gradient_log_q = np.array([gradient_log_q_mu, gradient_log_q_l], dtype=object)
    return gradient_log_q

def fun_gradient_lb(s, theta_samples, mu_q, l_q, c, n_samples, n_datasets, num_coeffs, num_latent, actual_summary_statistics):
    theta_tilde_q = theta_samples[s]

    alpha_q = (2 * np.exp(theta_tilde_q[0]) + 1) / (1 + np.exp(theta_tilde_q[0]))
    beta_q = (100 * np.exp(theta_tilde_q[1]) + 0) / (1 + np.exp(theta_tilde_q[1]))
    gamma_q = (0.9 * np.exp(theta_tilde_q[2]) + 0) / (1 + np.exp(theta_tilde_q[2]))
    theta_q = np.array([alpha_q, beta_q, gamma_q])

    ss_q = summary_statistics(theta_q, n_datasets, actual_summary_statistics)
    sample_mean_q = ss_q[0]
    sample_precision_q = ss_q[1]

    # Find mean and variance for p(gamma | theta, obs)
 
    diag_precision = np.zeros((num_latent, num_latent))
    np.fill_diagonal(diag_precision, np.diag(sample_precision_q)**(-1/2))
    mean_nuissance_p1 = my_inv(np.identity(num_latent) / sigma_latent + diag_precision.T @ sample_precision_q @ diag_precision)
    diff_mean = actual_summary_statistics - sample_mean_q
    mean_nuissance_p2 = diag_precision @ sample_precision_q @ diff_mean
    mean_nuissance_q = mean_nuissance_p1 @ mean_nuissance_p2

    variance_nuissance_q = mean_nuissance_p1

    # GENERATE GAMMA
    Gamma = multivariate_normal.rvs(mean = mean_nuissance_q, cov = variance_nuissance_q)
    # Gamma = Gamma[0]
    adjusted_theta_q = np.concatenate((theta_q, Gamma))

    # Find gradient of LB
    llh = mean_adjustment_unbiased_log_likelihood(adjusted_theta_q, n_samples, n_datasets, num_coeffs, num_latent, actual_summary_statistics)
    llh_gamma = nuissance_mean_adjustment_unbiased_log_likelihood(Gamma, mean_nuissance_q, variance_nuissance_q)

    h_lambda = prior(theta_tilde_q) + prior(Gamma) + llh - fun_log_q(theta_tilde_q, mu_q, l_q) - llh_gamma
    
    # Find gradient of LB
    grad_log_q = gradient_log_q(theta_tilde_q, mu_q, l_q, num_coeffs)
    gradient_lb = grad_log_q * (h_lambda - c)
    
    # Calculate control variates
    flattened_gradient_log_q = np.concatenate((grad_log_q[0], grad_log_q[1]), axis = None)
    flattened_gradient_lb = np.concatenate((gradient_lb[0], gradient_lb[1]), axis = None)
 
    return gradient_lb, h_lambda, flattened_gradient_log_q, flattened_gradient_lb