function [U,info] = BaseSC_Anchor_QR(B,SA,K)
% Spectral embedding restricted to the anchor-induced column space.
[n,m] = size(B);
if size(SA,1) ~= m || size(SA,2) ~= m || K > m
    error('Incompatible B, SA, or cluster count.');
end
H = (SA+SA')/2;
d = B*(H*(B'*ones(n,1)));
if any(~isfinite(d)) || any(d < -1e-10)
    error('The induced graph has an invalid degree vector.');
end
C = spdiags(1./sqrt(max(d,0)+eps),0,n,n)*B;
[Q,R] = qr(full(C),0);
T = R*H*R';
T = (T+T')/2;
[Z,e] = eig(T,'vector');
[~,order] = sort(e,'descend');
chosen = order(1:K);
V = Q*Z(:,chosen);
U = V./(sqrt(sum(V.^2,2))+eps);
info = struct('eigenvalues',e(chosen),'degreeMin',min(d));
end
