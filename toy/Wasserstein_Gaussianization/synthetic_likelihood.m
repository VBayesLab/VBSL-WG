function llh = synthetic_likelihood(theta,param)
% Calculate synthetic likelihood after using Wasserstein Gaussianization 
d = 2;

% generate simulated data
S = zeros(param.N,d);
scale = param.scale; alpha = param.alpha; beta = param.beta; n = param.n;
for i = 1:param.N
    y = theta + scale*sqrt((beta^2/alpha))*(gamrnd(alpha,1/beta,n,1)-alpha/beta); 
    s = [mean(y);var(y)];
    S(i,:) = s';
end

% transform to Gaussian
for iter = 1:length(param.mixture_object_seq)
    stepsize = param.eps0;
    mixture_obj = param.mixture_object_seq{iter};        
    [~,grad_log_q] = grad_log_density_mixture(S,mixture_obj); 
    v = -S-grad_log_q; % velocity                
    S = S + stepsize*v;    
end

% now calculate the synthetic likelihood
s_obs = param.s_obs_transformed;
mu_theta = mean(S);
Sigma_theta = cov(S);
llh = -d/2*log(2*pi)-1/2*logdet(Sigma_theta)-1/2*(s_obs-mu_theta)*(Sigma_theta\(s_obs-mu_theta)');
end
