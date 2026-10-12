function [U, info] = BaseSC_Fast(S, k)
% Original MMvC-FG unnormalized Laplacian L = diag(sum(W,2)) - W.
% Use exact zero-space component indicators and sparse eigs for remaining K-c.
if ~issparse(S)
    [U,info] = BaseSC_Dense(S,k);
    return
end
W = sparse((S + S') / 2);
n = size(W,1);
d = full(sum(W,2));
L = spdiags(d,0,n,n) - W;
L = (L + L') / 2;
bins = conncomp(graph(W));
c = max(bins);
counts = accumarray(bins(:),1,[c,1]);
Z = sparse((1:n)',bins(:),1./sqrt(counts(bins(:))),n,c);
if c == k
    V = full(Z);
    lambda = zeros(k,1);
    solver = 'component-indicators';
elseif c < k
    shift = 2*norm(L,1) + 1;
    operator = @(x) L*x + shift*Z*(Z'*x);
    opts.issym = true;
    opts.isreal = true;
    opts.tol = 1e-8;
    opts.maxit = 1000;
    opts.p = min(n,max(2*(k-c)+10,30));
    [Vpositive,D,flag] = eigs(operator,n,k-c,'smallestreal',opts);
    solver = 'shifted-eigs';
    useShiftInvert = false;
    if flag ~= 0
        % A nearly disconnected graph can have many clustered positive
        % eigenvalues. Expand the Krylov space and relax only the numerical
        % tolerance; the Laplacian and target eigenspace are unchanged.
        retry = opts;
        retry.tol = 1e-6;
        retry.maxit = 5000;
        retry.p = min(n,max(4*(k-c)+40,80));
        [Vpositive,D,flag] = eigs(operator,n,k-c,'smallestreal',retry);
        solver = 'shifted-eigs-retry';
        if flag ~= 0
            % The sparse positive regularization shifts all eigenvalues by
            % the same amount. Shift-invert finds the same k smallest
            % eigenspace when Krylov iteration cannot resolve the gap.
            regularization = 1e-5*max(1,max(d));
            invertOpts = rmfield(retry,{'issym','isreal'});
            invertOpts.p = min(n,max(4*k+40,80));
            [V,D,flag] = eigs(L+regularization*speye(n),k,'smallestabs',invertOpts);
            if flag ~= 0
                error('Full-model shift-invert eigs did not converge (flag=%d)',flag);
            end
            [lambda,idx] = sort(diag(D)-regularization,'ascend');
            V = V(:,idx);
            solver = 'regularized-shift-invert';
            useShiftInvert = true;
        end
    end
    if ~useShiftInvert
        [lambdaPositive,idx] = sort(diag(D),'ascend');
        V = [full(Z),Vpositive(:,idx)];
        lambda = [zeros(c,1);lambdaPositive];
    end
else
    % Original dense eig also has a nonunique choice inside this nullspace.
    opts.tol = 1e-8;
    opts.maxit = 1000;
    opts.p = min(n,max(2*k+10,30));
    [V,D,flag] = eigs(L,k,'smallestreal',opts);
    if flag ~= 0
        % With more connected components than requested clusters, the
        % zero eigenspace has dimension c > k. Any k orthonormal vectors
        % in that space are exact minimizers. ARPACK can fail because all
        % requested eigenvalues are identical; use a reproducible random
        % basis without changing the caller's RNG (used by kmeans seeds).
        stream = RandStream('mt19937ar','Seed',20260926);
        [Q,~] = qr(randn(stream,c,k),0);
        V = full(Z*Q);
        lambda = zeros(k,1);
        solver = 'component-nullspace-fallback';
    else
        [lambda,idx] = sort(diag(D),'ascend');
        V = V(:,idx);
        solver = 'eigs';
    end
end
U = V ./ (sqrt(sum(V.^2,2)) + eps);
info = struct('eigenvalues',lambda,'components',c, ...
    'solver',solver, ...
    'residual',norm(L*V-V*diag(lambda),'fro') / ...
    max(1,norm(V*diag(lambda),'fro')));
end



