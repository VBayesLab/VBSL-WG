function [mixture_obj,G] = mixture_model_fit(data,maxG)
% fitting a mixture of normal density to data.
% The best number of components, in th range 1:maxG, is selected by BIC
BICvalues = zeros(maxG,1); 
for G = 1:maxG
    mixture_obj = fitgmdist(data,G,'RegularizationValue',1e-10);
    BICvalues(G) = mixture_obj.BIC;
end
[~,G] = min(BICvalues); % find the best G
mixture_obj = fitgmdist(data,G,'RegularizationValue',1e-10);

end
    