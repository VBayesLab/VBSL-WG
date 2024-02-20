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

from functions_wasserstein.f_ss_wasserstein import summary_statistics 

def prior(theta, num_coeffs): 
    log_prior = multivariate_normal.logpdf(theta, cov= 10 * np.identity(num_coeffs))
    return log_prior

def unbiased_log_likelihood(theta, n_samples, n_datasets, actual_summary_statistics, mixture_obj_seq):
    ss = summary_statistics(theta, n_samples, n_datasets, mixture_obj_seq)
    sample_mean = ss[0]
    sample_variance = ss[1]

    diff_mean_s = actual_summary_statistics - sample_mean
    part1 = diff_mean_s.T @ np.linalg.inv(sample_variance) @ diff_mean_s
    u_est_log_likelihood = -1/2 * np.log(np.linalg.det(sample_variance)) - part1
    return u_est_log_likelihood

def fun_log_q(theta, mu, l):
    log_q = multivariate_normal.logpdf(theta, mean = mu, cov= np.linalg.inv(l @ l.T))
    return log_q

def gradient_log_q(theta, mu, l, num_coeffs): #indep theta
    gradient_log_q_mu = np.matmul(np.matmul(l, l.T), (theta - mu))
    gradient_log_q_l = (np.diag(np.linalg.inv(l)) - np.matmul(((np.reshape(theta - mu, (num_coeffs,1))) * theta - mu), l)).T[np.triu_indices(num_coeffs)] #use * because matmul gives scalar 
    gradient_log_q = np.array([gradient_log_q_mu, gradient_log_q_l], dtype=object)
    return gradient_log_q

def fun_gradient_lb(s, theta_samples, mu_q, l_q, c, n_samples, n_datasets, num_coeffs, actual_summary_statistics, mixture_obj_seq):
    theta_tilde_q = theta_samples[s]
    # Calculate theta from mu, l (lambda)
    alpha_q = (2 * np.exp(theta_tilde_q[0]) + 1.1) / (1 + np.exp(theta_tilde_q[0]))
    beta_q = (np.exp(theta_tilde_q[1]) - 1) / (np.exp(theta_tilde_q[1]) + 1)
    gamma_q = np.exp(theta_tilde_q[2])
    delta_q = theta_tilde_q[3]
    theta_q = np.array([alpha_q, beta_q, gamma_q, delta_q])

    # Find gradient of LB
    llh = unbiased_log_likelihood(theta_q, n_samples, n_datasets, actual_summary_statistics, mixture_obj_seq)
    h_lambda = prior(theta_tilde_q, num_coeffs) + llh - fun_log_q(theta_tilde_q, mu_q, l_q)
    
    # Find gradient of LB
    grad_log_q = gradient_log_q(theta_tilde_q, mu_q, l_q, num_coeffs)
    gradient_lb = gradient_log_q(theta_tilde_q, mu_q, l_q, num_coeffs) * (h_lambda - c)
    
    # Calculate control variates
    flattened_gradient_log_q = np.concatenate((grad_log_q[0], grad_log_q[1]), axis = None)
    flattened_gradient_lb = np.concatenate((gradient_lb[0], gradient_lb[1]), axis = None)
 
    return gradient_lb, h_lambda, flattened_gradient_log_q, flattened_gradient_lb