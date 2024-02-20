% This is the toy example described in the paper
clear all
rng(2022)

% Data generating process
n = 30;
theta_true = 0;
alpha = 1;
beta = 0.01;
scale = 2;
y_obs = theta_true + scale*sqrt((beta^2/alpha))*(gamrnd(alpha,1/beta,n,1)-alpha/beta);
s_obs = [mean(y_obs),var(y_obs)];  % observed summary statistics

param_s_obs = s_obs;
param.alpha = alpha; 
param.beta = beta;
param.scale = scale;
param.n = n;  % n is size of the observed data set
param.N = 10; % N is the number of simulated data sets

%------------------------------------------------------------------%
% -------------- Training the Wesserstein Gaussianization ---------%

% generate data for training the transform
M = 2000;
data = zeros(3*M,2);
for i = 1:3*M
    y = theta_true + scale*sqrt((beta^2/alpha))*(gamrnd(alpha,1/beta,n,1)-alpha/beta); 
    s = [mean(y);var(y)];
    data(i,:) = s';
end
data_train = data(1:M,:); % training data
data_val = data(M+1:2*M,:); % validation data to calculate the lower bound
data_test = data(2*M+1:end,:); % test data
data_train_original = data_train;
data_test_original = data_test;

% start Particle VB (see the Wesserstein Gaussianization algorithm in the paper)
eps0 = 0.01; % learning rate
param.eps0 = eps0;
patience_max = 20;
max_iter = 1000;
t_w = 20;  % smoothing window of the lower bound 
maxG = 4;  % maximum of components in the mixtures of normals

[mixture_obj,~] = mixture_model_fit(data_train,maxG); % fitting a mixture to current particles
mixture_object_seq{1} = mixture_obj;  % a sequence to store the mixture densities. This is later used 
                                      % in computation of the Wesserstein
                                      % Gaussianization transformation
iter = 0;
stop = false;
LB_bar = 0; patience = 0;
warning off
while ~stop
    iter = iter+1
    
    stepsize = eps0;
        
    [~,grad_log_q] = grad_log_density_mixture(data_train,mixture_obj); % calculate gradient of log mixture density
    v = -data_train-grad_log_q; % velocity            
    data_train = data_train + stepsize*v;        
    
    [log_q,grad_log_q] = grad_log_density_mixture(data_val,mixture_obj); % calculate log mixture density and its gradient
                                                                         % Used to compute the lower bound
    v = -data_val-grad_log_q; % velocity            
    data_val = data_val + stepsize*v;        
    h_vector = -1/2*diag(data_val*data_val')-log_q;

    [~,grad_log_q] = grad_log_density_mixture(data_test,mixture_obj); % calculate gradient of log mixture density on test data
    v = -data_test-grad_log_q; % velocity            
    data_test = data_test + stepsize*v;        
      
    [mixture_obj,bestG] = mixture_model_fit(data_train,maxG); % fitting a mixture to new particles. bestG is the number of components, selected by BIC
    % in the following, we remove a component in the mixture if its
    % weight is small. I found it helpful.
    if min(mixture_obj.PComponents)<0.005
        [mixture_obj,~] = mixture_model_fit(data_train,bestG-1); % fitting a mixture to new particles
    end  
    mixture_object_seq{iter+1} = mixture_obj;
    bestG  % As we try to move to a Gaussian distribution, bestG should be 1 eventually
   
    LB(iter) = mean(h_vector); % lower bound
   
    if iter>=t_w
        LB_bar(iter-t_w+1) = mean(LB(iter-t_w+1:iter));
        LowerBound = LB_bar(iter-t_w+1)
    end

    if iter>max(patience_max,t_w)
        if LB_bar(iter-t_w+1)>=max(LB_bar)
            patience = 0;
        else
            patience = patience+1;
        end
    end
    
    if (patience>patience_max)||(iter>max_iter) stop = true; end 
 
end
data_test_transformed = data_test;
mixture_object_seq = mixture_object_seq(1:end-1); % remove the last mixture as it isn't needed

param.mixture_object_seq = mixture_object_seq;

% plotting
figure(1)
plotmatrix(data_train_original)
title('training data: original')
figure(2)
plotmatrix(data_train)
title('training data: transformed')
figure(3)
plotmatrix(data_test_original)
title('test data: original')
figure(4)
%plotmatrix(data_test_transformed)
plotmatrix(rmoutliers(data_test_transformed)) % remove the outliers before plotting
title('test data: transformed')
figure(5)
plot(LB_bar)

%------------------------------------------------------------------------%
% -- Use the trained Wesserstein Gaussianization to transform s_obs -----%
for iter = 1:length(param.mixture_object_seq)
    stepsize = param.eps0;
    mixture_obj = param.mixture_object_seq{iter};    
    [~,grad_log_q] = grad_log_density_mixture(s_obs,mixture_obj); 
    v = -s_obs-grad_log_q; % velocity                
    s_obs = s_obs + stepsize*v;    
end
s_obs_transformed = s_obs;
param.s_obs_transformed = s_obs_transformed;

%------------------------------------------------------------------------%
% -------- run MCMC for Wasserstein Gaussianization BSL------------------%
dim = 1;
niter = 10000;
mcmc_beta = .05;
chain = zeros(niter,dim); %Markov chain
ind_accept = false(niter,1);
current = mean(y_obs);
chain(1,:) = current;

llh_current = synthetic_likelihood(current,param);
prior_current = log(normpdf(current,0,10));
mcmc_iter = 1;
while mcmc_iter<niter    
    if mcmc_iter<100
        proposal = mvnrnd(current,.1^2/dim*eye(dim));
    else
        if ~mod(mcmc_iter,50) 
            Sign = cov(chain(1:mcmc_iter,:));
        end       
        if binornd(1,mcmc_beta)
            proposal = mvnrnd(current,.1^2/dim*eye(dim));
        else
            proposal = mvnrnd(current,2.38/dim*Sign);
        end
    end
    llh_proposal = synthetic_likelihood(proposal,param);
    prior_proposal = log(normpdf(proposal,0,10));

    prob = exp(prior_proposal+llh_proposal-prior_current-llh_current);
    u = unifrnd(0,1);
    if u<=prob            
        chain(mcmc_iter+1,:) = proposal;
        current = proposal;
        ind_accept(mcmc_iter+1) = 1;
        llh_current = llh_proposal;
        prior_current = prior_proposal;
    else
        chain(mcmc_iter+1,:) = current;
    end
    mcmc_iter = mcmc_iter+1
    acceptance_rate = mean(ind_accept(1:mcmc_iter))
end
chain_wBSL = chain;
figure(6)
plot(chain_wBSL)
title('MCMC BSL-WG')
