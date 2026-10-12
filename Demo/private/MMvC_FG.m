function [S] = MMvC_FG(X,para)
lam1 = para.lam1;
lam2 = para.lam2;
M = length(X);
n = size(X{1},1);
D = zeros(n,n); A1 = D; A2 = D;
for v = 1:M
    Xv = X{v};
    XX = Xv * Xv';
    diagXX = diag(XX);
    D = diagXX + diagXX' - 2*XX;
    D(D < 0) = 0;
    A1 = A1+XX;
    A2 = A2+D;
end
E = 1/(2*lam2) *(A1-lam1*A2);
S = proj_simplex_matrix(E);
S = (S + S.')/2;
end
function S = proj_simplex_matrix(E)
% =========================
% 1. Remove main diagonal
% =========================
[n, ~] = size(E);
idx = ~eye(n);
Q = E.';
Q = Q(:);
idx_Q = idx.';
idx_Q = idx_Q(:);
E_hat = transpose(reshape(Q(idx_Q), n-1, n));
% =========================
% 2. Projecting onto the probabilistic simplex
% =========================
U    = sort(E_hat, 2, 'descend');
cssv = cumsum(U, 2) - 1.0;
j    = 1:(n-1);
cond = U > cssv ./ j;
rho = sum(cond, 2);
idx_theta = sub2ind(size(cssv), (1:n).', rho);
theta = cssv(idx_theta) ./ rho;
S_hat = max(E_hat - theta, 0.0);
% =========================
% 3. Insert the zero principal diagonal back.
% =========================
S = zeros(n);
S_hat_flat_py = S_hat.';
S_hat_flat_py = S_hat_flat_py(:);
tmp = zeros(n);
tmp(idx_Q) = S_hat_flat_py;
S = tmp.';
end
