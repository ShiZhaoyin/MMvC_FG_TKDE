function [U, info] = BaseSC_Dense(W, k)
% Exact unnormalized Laplacian eigenspace using dense matrix-vector products.
% The input has already been symmetrized by MMvC_FG_Blocked.
n = size(W,1);
d = sum(W,2);
Lapply = @(x) d.*x - W*x;
bins = dense_components(W);
c = max(bins);
counts = accumarray(bins(:),1,[c,1]);
Z = sparse((1:n)',bins(:),1./sqrt(counts(bins(:))),n,c);
if c == k
    V = full(Z);
    lambda = zeros(k,1);
    solver = 'dense-component-indicators';
elseif c < k
    % W is symmetric, nonnegative, and has zero diagonal. Thus
    % norm(diag(d)-W,1) = 2*max(d), matching the sparse solver's shift.
    shift = 4*max(d) + 1;
    operator = @(x) Lapply(x) + shift*Z*(Z'*x);
    opts.issym = true;
    opts.isreal = true;
    opts.tol = 1e-8;
    opts.maxit = 1000;
    opts.p = min(n,max(2*(k-c)+10,30));
    [Vpositive,D,flag] = eigs(operator,n,k-c,'smallestreal',opts);
    solver = 'dense-shifted-eigs';
    if flag ~= 0
        retry = opts;
        retry.tol = 1e-6;
        retry.maxit = 5000;
        retry.p = min(n,max(4*(k-c)+40,80));
        [Vpositive,D,flag] = eigs(operator,n,k-c,'smallestreal',retry);
        solver = 'dense-shifted-eigs-retry';
        if flag ~= 0
            error('Full-model dense eigs did not converge after retry (flag=%d)',flag);
        end
    end
    [lambdaPositive,idx] = sort(diag(D),'ascend');
    V = [full(Z),Vpositive(:,idx)];
    lambda = [zeros(c,1);lambdaPositive];
else
    opts.issym = true;
    opts.isreal = true;
    opts.tol = 1e-8;
    opts.maxit = 1000;
    opts.p = min(n,max(2*k+10,30));
    [V,D,flag] = eigs(Lapply,n,k,'smallestreal',opts);
    if flag ~= 0
        stream = RandStream('mt19937ar','Seed',20260926);
        [Q,~] = qr(randn(stream,c,k),0);
        V = full(Z*Q);
        lambda = zeros(k,1);
        solver = 'dense-component-nullspace-fallback';
    else
        [lambda,idx] = sort(diag(D),'ascend');
        V = V(:,idx);
        solver = 'dense-eigs';
    end
end
U = V ./ (sqrt(sum(V.^2,2)) + eps);
info = struct('eigenvalues',lambda,'components',c, ...
    'solver',solver, ...
    'residual',norm(Lapply(V)-V*diag(lambda),'fro') / ...
    max(1,norm(V*diag(lambda),'fro')));
end

function bins = dense_components(W)
n = size(W,1);
bins = zeros(1,n);
c = 0;
for start = 1:n
    if bins(start) ~= 0, continue; end
    c = c + 1;
    bins(start) = c;
    queue = zeros(1,n);
    queue(1) = start;
    head = 1;
    tail = 1;
    while head <= tail
        u = queue(head);
        head = head + 1;
        unseen = find(W(u,:)>0 & bins==0);
        if isempty(unseen), continue; end
        bins(unseen) = c;
        queue(tail+(1:numel(unseen))) = unseen;
        tail = tail + numel(unseen);
        if tail == n, return; end
    end
end
end