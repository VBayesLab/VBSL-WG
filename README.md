# VBSL-WG
Code for Bayesian Synthetic Likelihood with Wasserstein Gaussianization 

## Required Packages

Requires Python 3.7 or later with the following libraries: 

`matplotlib==3.5.3`
`scikit-learn==1.1.2`
`sklearn`

## Running Files

We have 4 different methods to run: the original VBSL, the VBSL with robust likelihood, the VBSL with Wasserstein Gaussianization and the VBSL with robust likelihood and Wasserstein Gaussianization

```{bash}
VBSL-WG (main) 

Arguments to adjust for general VBSL:

--true_theta                    True theta to simulate the models
--num_datasets                  Number of datasets that need to be simulated
--stop                          When to stop training
--num_samples                   Number of samples needed to train the model's theta
--Patience                      Maximal number of iterations after which the model no longer improves
--learning_rate                 Learning rate for training
--t_w                           Smoothing window for calculation of Lower Bound
--l_threshold                   If l2-norm of gradient of LB passes l_threshold, scale back
--adaptive_lr_1                 Adaptive learning rate 1 for gradient of LB
--adaptive_lr_2                 Adaptive learning rate 2 for gradient of LB

For Wasserstein Gaussianization:

--patience_max                  Maximal number of iterations after which the model no longer improves
--stop                          When to stop training Wasserstein Gaussianization
--eps                           Learning rate for Wasserstein Gaussianization training
--t_w                           Smoothing window for calculation of Lower Bound

In case you want to run the speedup version, please find below a brief walk-through of how to run rBSL-WG for alpha-stable example:

1. Make sure the folder `functions_robust_wasserstein` are in the same directory as the file you want to run, in this case, `vbsl_mcmc_va_speedup_wasserstein.ipynb`

2. Run `from functions_robust_wasserstein.f_var_adjust_ss_wasserstein import (alpha_stable, alpha_stable_ss, wasserstein_transform, summary_statistics, variance_adjustment_summary_statistics)` and `from functions_robust_wasserstein.f_gradient_lb_wasserstein import (fun_gradient_lb)`
```

## Data

Simulated data and summary statistics used for this project:
- Toy example
- alpha-stable distribution [here](https://github.com/megannguyen6898/VBSL/blob/master/functions/f_mean_adjust_ss.py)
- g-and-k model [here](https://github.com/megannguyen6898/VBSL/blob/master/gnk/vbsl_gnk.ipynb)
- toads' movement model [here](https://github.com/megannguyen6898/VBSL/blob/master/functions_robust_wasserstein_toad/f_mean_adjust_ss_wasserstein_toad.py). Please also find the original paper [here](https://www.researchgate.net/publication/318444943_A_stochastic_movement_model_reproduces_patterns_of_site_fidelity_and_long-distance_dispersal_in_a_population_of_Fowler's_toads_Anaxyrus_fowleri)
