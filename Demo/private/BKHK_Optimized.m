function [B,A] = BKHK_Optimized(X,numAnchor,numNearestAnchor)
% BKHK Generates anchors and constructs the bipartite anchor graph.
%
% Inputs:
%   X                - Input data matrix of size N x d.
%                      [IMPORTANT]: Each ROW represents a sample instance.
%   numAnchor        - The exponent (p) for determining the number of anchors.
%                      Note: This is NOT the absolute count. The actual number
%                      of anchors m is calculated as m = 2^numAnchor.
%   numNearestAnchor - The number of nearest anchors (k) used to construct
%                      the sparse graph B.
%
% Outputs:
%   B                - The bipartite anchor graph matrix of size N x m.
%                      Represents the relationship between N samples and m anchors.
%   A                - The generated anchor matrix of size m x d.
%                      [IMPORTANT]: Each ROW represents an anchor centroid.
%
% Usage Example:
%   % Generate 2^7 = 128 anchors using 5 nearest neighbors
%   [B, A] = BKHK(DataMatrix, 7, 5);


[num,~] = size(X);
sampleSq = sum(X.*X,2);

locAnchor = hKM_centers(X',1:num,numAnchor,sampleSq);

A=locAnchor';

B = ConstructA_NP(X',(locAnchor),numNearestAnchor);
end

function centers = hKM_centers(X,idx0,k,sampleSq)
X0 = X(:,idx0);
if k == 1
    [centers,~,~] = BalancedKM(X0,0.0,sampleSq(idx0));
else
    [~,~,y] = BalancedKM(X0,0.5,sampleSq(idx0));
    idx1 = idx0(y==1);
    idx2 = idx0(y==2);
    centers1 = hKM_centers(X,idx1,k-1,sampleSq);
    centers2 = hKM_centers(X,idx2,k-1,sampleSq);
    centers = [centers1,centers2];
end
end

function [C, Q, y] = BalancedKM(X, ratio, aa)
class_num = 2;
n = size(X,2);

if nargin < 2
    ratio = 0.5;
end
StartInd = randsrc(n,1,1:class_num);
InitF = TransformL(StartInd, class_num);

if ratio > 0.5
    error('ratio should not larger than 0.5');
end
if ratio < 0
    ratio = 0;
end

a = floor(n*ratio);
b = floor(n*(1-ratio));

F = InitF;
fea_a = X';
for iter = 1:10
    Fprev = F;
    C = X*F./sum(F+eps);            % calculate the centers for two categories
    F = sparse(n,class_num);
    fea_b = C';
    bb = sum(fea_b.*fea_b,2);
    ab = fea_a*fea_b';
    Q = bsxfun(@plus,aa,bb') - 2*ab;
    Q(Q<0) = 0;
    q = Q(:,1)-Q(:,2);
    [temp, idx] = sort(q);
    nn = length(find(temp<0));      % find the nearest cluster center for points 
    if nn>=a && nn<=b
        cp = nn;
    elseif nn<a
        cp = a;
    else
        cp = b;
    end;
    
    if cp < 1
        cp = 1;
    elseif cp > n-1
        cp = n-1;
    end;
%     n
%     size(F)
%     size(idx)
%     cp
    F(idx(1:cp),1) = 1;
    F(:,2) = 1-F(:,1);
    if isequal(F,Fprev), break; end
    
%     XC = X-C*F';
%     obj(iter) = trace(XC*XC');
end;

[~, y] = max(F,[],2);
end

function InitF = TransformL(F, c)
F = F-1;
[n,~] = size(F);
InitF = zeros(n,c);
index = find(F==0);
InitF(index) = 1;
index = find(F==1);
InitF(index) = 1;
end

function D = EuDist2(fea_a,fea_b,bSqrt)
%EUDIST2 Efficiently Compute the Euclidean Distance Matrix by Exploring the
%Matlab matrix operations.
%
%   D = EuDist(fea_a,fea_b)
%   fea_a:    nSample_a * nFeature
%   fea_b:    nSample_b * nFeature
%   D:      nSample_a * nSample_a
%       or  nSample_a * nSample_b
%
%    Examples:
%
%       a = rand(500,10);
%       b = rand(1000,10);
%
%       A = EuDist2(a); % A: 500*500
%       D = EuDist2(a,b); % D: 500*1000
%
%   version 2.1 --November/2011
%   version 2.0 --May/2009
%   version 1.0 --November/2005
%
%   Written by Deng Cai (dengcai AT gmail.com)


if ~exist('bSqrt','var')
    bSqrt = 1;
end

if (~exist('fea_b','var')) || isempty(fea_b)
    aa = sum(fea_a.*fea_a,2);
    ab = fea_a*fea_a';
    
    if issparse(aa)
        aa = full(aa);
    end
    
    D = bsxfun(@plus,aa,aa') - 2*ab;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
    D = max(D,D');
else
    aa = sum(fea_a.*fea_a,2);
    bb = sum(fea_b.*fea_b,2);
    ab = fea_a*fea_b';

    if issparse(aa)
        aa = full(aa);
        bb = full(bb);
    end

    D = bsxfun(@plus,aa,bb') - 2*ab;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
end
end

function A = ConstructA_NP(TrainData,Anchor,k)
Dis = EuDist2(TrainData',Anchor',0);
[di,idx] = mink(Dis,k+1,2);
num = size(TrainData,2);
anchor_num = size(Anchor,2);
den = k*di(:,k+1)-sum(di(:,1:k),2)+eps;
weight = (di(:,k+1)-di)./den;
rows = repmat((1:num)',1,k+1);
A = sparse(rows(:),idx(:),weight(:),num,anchor_num);
end
