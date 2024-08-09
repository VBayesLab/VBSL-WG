# VBSL-WG
Code for Bayesian Synthetic Likelihood with Wasserstein Gaussianization 

## Required Packages

Requires Python 3.9 or later with the libraries in the following file: 

`requirements.txt`

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

--num_layers                    Number of layers for the normalizing flow
--Patience                      Maximal number of iterations after which the model no longer improves
--stop                          When to stop training Wasserstein Gaussianization
--eps                           Learning rate for Wasserstein Gaussianization training
--t_w                           Smoothing window for calculation of Lower Bound
```
## Data

Simulated data and summary statistics used for this project:
- Toy example
- alpha-stable distribution [here](https://github.com/megannguyen6898/VBSL/blob/master/functions/f_mean_adjust_ss.py)
- g-and-k model [here](https://github.com/megannguyen6898/VBSL/blob/master/gnk/vbsl_gnk.ipynb)
- Toads' movement model [here](https://github.com/megannguyen6898/VBSL/blob/master/functions_robust_wasserstein_toad/f_mean_adjust_ss_wasserstein_toad.py). Please also find the original paper [here](https://www.researchgate.net/publication/318444943_A_stochastic_movement_model_reproduces_patterns_of_site_fidelity_and_long-distance_dispersal_in_a_population_of_Fowler's_toads_Anaxyrus_fowleri)

## Instructions:

```{bash}
I. Toy example

We included both the MATLAB version and Python version. Please run the Python version to reproduce the experiments.

1. For original VBSL: Run `vbsl_toy.ipynb`
2. For rBSL: Run `vbsl_ma_nuissance_toy.ipynb`
3. For VBSL-WG and rBSL-WG: Run `norm_vbsl_jax_v3_nf_nuissance_wo_tau.ipynb`

II. alpha-stable distribution

1. For original VBSL: Run `vbsl_mcmc_final.ipynb`
2. For rBSL:
- Make sure the folder `functions` are in the same directory as the file you want to run, in this case, `vbsl_mcmc_ma_speedup_nuissance.ipynb`
- Run `from f_mean_adjust_ss import (alpha_stable, alpha_stable_ss, summary_statistics, mean_adjustment_summary_statistics)` and `from f_gradient_lb_mean_nuissance import (fun_gradient_lb, prior, mean_adjustment_unbiased_log_likelihood, nuissance_mean_adjustment_unbiased_log_likelihood)`
______ For mixture models:
3. For VBSL-WG:
- Make sure the folder `functions_wasserstein` are in the same directory as the file you want to run, in this case, `vbsl_speedup_wasserstein.ipynb`
- Run `from f_ss_wasserstein import (alpha_stable, alpha_stable_ss, wasserstein_transform, summary_statistics)` and `from f_gradient_lb_wasserstein import (fun_gradient_lb)`
4. For rBSL-WG:
- Make sure the folder `functions_robust_wasserstein` are in the same directory as the file you want to run, in this case, `vbsl_mcmc_ma_speedup_nuissance_wasserstein.ipynb`
- Run `from f_mean_adjust_ss_wasserstein import (alpha_stable, alpha_stable_ss, summary_statistics, mean_adjustment_summary_statistics)` and `from f_gradient_lb_wasserstein import (fun_gradient_lb)`

!!!!! Note that we can reduce computational time significantly by running the notebooks with JAX. Please find the files for selected methods in the folder `jax version.`
______ For normalizing flow:
3. For VBSL-WG and rBSL-WG: Run `vbsl_jax_v3_nf_nuissance_prec.ipynb`

III. g-and-k model

1. For original VBSL: Run `vbsl_gnk.ipynb`
2. For rBSL: Run `vbsl_ma_nuissance.ipynb`
______ For mixture models:
3. For VBSL-WG: Run `vbsl_wasserstein_gnk.ipynb`
4. For rBSL-WG: Run `vbsl_ma_nuissance_wasserstein_with_mcmc.ipynb`
______ For normalizing flow:
3. For VBSL-WG and rBSL-WG: Run `vbsl_jax_v3_nf_nuissance_gnk_prec_wo_tau.ipynb`

!!!!! Note that we can reduce computational time significantly by running the notebooks with JAX. Please find the files for selected methods in the folder `jax version.` The JAX versions implement reparameterization of parameters differently from the non-JAX versions. We collect the results from JAX versions.

IV. Toads' movement model

1. For original VBSL: Run `vbsl_toad.ipynb`
2. For rBSL: Run `robustvbsl_robust_ma_toad.ipynb`
______ For mixture models:
3. For VBSL-WG and rBSL-WG:
- Make sure the folder `functions_robust_wasserstein_toad` are in the same directory as the file you want to run, in this case, `vbsl_ma_speedup_nuissance_wasserstein_toad.ipynb`
- Run `from f_mean_adjust_ss_wasserstein_toad import (toad, compute_summaries, compute_summaries_stacked, mean_adjustment_summary_statistics)` and `from f_gradient_lb_mean_wasserstein_nuissance_toad import (fun_gradient_lb)`
______ For mixture models:
3. For VBSL-WG:
- Make sure the folder `functions_robust_wasserstein_toad` are in the same directory as the file you want to run, in this case, `vbsl_nf_toad_without_tau.ipynb`
4. For rBSL-WG:
- Make sure the folder `functions_robust_wasserstein_toad` are in the same directory as the file you want to run, in this case, `vbsl_ma_speedup_nuissance_nf_toad_without tau.ipynb`
```

