function [W,index,obj] = VCSDFS(X,XX,XXX,rho,k,NITER)
%% [1] S. Karami, F. Saberi-Movahed, P. Tiwari, P. Marttinen, S. Vahdati,
%"Unsupervised Feature Selection Based on Variance-Covariance Subspace Distance,"
% Accepted in Neural Networks,
% vol. , no. , pp. , June 2023, doi:.
%
%--------------------------- Inputs ------------------------------------------------------------------------------
%      X: Data matrix in R^(n*d), where n and d are the number of samples and features, respectively.
%      XX = (X')*X;
%      XXX = (XX)^2;
%      rho: Regularization parameter........>Please note that this parameter needs to be tuned.
%      k:   The number of selected features.
%      Niter: Maximum number of iterations.
%--------------------------- Outputs------------------------------------------------------------------------------
%      W: Feature weight matrix in in R^(d*k).
%      index: the sort of features for selection.
%      obj: the vector of objective function values.
%%-----------------------------------------------------------------------------------------------------------------
[~,d] = size(X);
W = rand(d,k);
obj = zeros(NITER,1);
iter = 1;
while (iter <= NITER)
    % Update W
    numW = XXX*W + rho*W;
    A = XX*W;
    denW = A*(W')*A + rho*ones(d,d)*W;
    fracW = numW./denW;
    fracW = nthroot(fracW,4);
    W = W.*fracW;
    % Compute the objective function
    XW = X*W;
    WW = W*(W');
    obj(iter) = 0.5 * (norm(X*(X')-XW*(XW'),'fro')^2) + rho * (trace(ones(d,d)*WW)-trace(WW));
    iter=iter+1;
 end
score=sum((W.*W),2);
[~,index]=sort(score,'descend'); % Sort the norms of the rows of the feature matrix W in a descending order
