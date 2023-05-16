function [log_f,grad_log_f] = grad_log_density_mixture(x,mixture_obj)
% calculate the gradient of log_density of a normal mixture
% INPUT
%   x                a matrix, each row is a sample/particle
%   mixture_obj      store the parameters of the mixture
M = size(x,1); % number of samples/particles

Mean = mixture_obj.mu; Mean = Mean';
Sigma = mixture_obj.Sigma;
Prob = mixture_obj.PComponents;  
 
[d,K] = size(Mean); % d: dimension of the mixture density; 
                    % K: number of components of the mixture  
lw = zeros(M,K);
grad_log_f_k = cell(1,K);
for k = 1:K
        mu_k = Mean(:,k);
        Sigma_k = Sigma(:,:,k);
        pi_k = Prob(k);
        aux = x-ones(M,1)*(mu_k');  % size Mxd
        lw(:,k) = log(pi_k)-d/2*log(2*pi)-1/2*logdet(Sigma_k)-1/2*diag((aux/Sigma_k)*aux');
        grad_log_f_k{k} = -aux/Sigma_k;          % size Mxd, i.e, each ROW is a transpose of gradient of log_f        
end
max_lw = max(lw,[],2);
aux_w = lw-max_lw*ones(1,K);
W = exp(aux_w)./sum(exp(aux_w),2);
log_f = log(sum(exp(aux_w),2))+max_lw;

grad_log_f = 0;
for k = 1:K
    grad_log_f = grad_log_f+(W(:,k)*ones(1,d)).*grad_log_f_k{k};
end

end