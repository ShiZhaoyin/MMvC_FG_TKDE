function report = run_demo()
%RUN_DEMO Reproduce the two Hdigit results at their selected parameters.
% Open this file in MATLAB and click Run.
here = fileparts(mfilename('fullpath'));
cfg = reference_config();
[X, y, K] = load_hdigit(fullfile(here, 'data', 'Hdigit.mat'));
nRuns = cfg.runs;
assert(nRuns == 20 && numel(y) == 10000 && K == 10);
assert(numel(cfg.full.seeds) == nRuns);
assert(numel(cfg.anchor.anchorSeeds) == nRuns);
assert(numel(cfg.anchor.clusterSeeds) == nRuns);

% Rebuild the graph and embedding in every run to measure complete runs.
fullScores = zeros(nRuns, 4);
fullTimes = zeros(nRuns, 4);
for r = 1:nRuns
    rng(cfg.full.seeds(r), 'twister');
    timer = tic;
    S = MMvC_FG(X, cfg.full.parameters);
    fullTimes(r, 1) = toc(timer);
    timer = tic;
    U = BaseSC_Fast(S, K);
    fullTimes(r, 2) = toc(timer);
    timer = tic;
    labels = kmeans(U, K, 'Replicates', 10, 'Start', 'plus', ...
        'MaxIter', 500, 'Display', 'off');
    fullTimes(r, 3) = toc(timer);
    fullTimes(r, 4) = sum(fullTimes(r, 1:3));
    score = ClusteringMeasure(y, labels);
    fullScores(r, :) = score([1 3 4 2]);
    fprintf('Full   %2d/%d  ACC %.6f  %.3f s\n', ...
        r, nRuns, fullScores(r, 1), fullTimes(r, 4));
    clear S U labels
end

% Anchor selection is randomized, so each run rebuilds B and S_A using
% its saved anchor seed before applying its saved clustering seed.
anchorScores = zeros(nRuns, 4);
anchorTimes = zeros(nRuns, 5);
Xcat = [X{:}];
dims = cellfun(@(view) size(view, 2), X);
p = cfg.anchor.parameters;
for r = 1:nRuns
    rng(cfg.anchor.anchorSeeds(r), 'twister');
    timer = tic;
    [B, A] = BKHK_Optimized(Xcat, p.p, p.knn);
    anchorTimes(r, 1) = toc(timer);
    anchorViews = mat2cell(A, size(A, 1), dims);
    timer = tic;
    SA = MMvC_FG(anchorViews, p);
    anchorTimes(r, 2) = toc(timer);
    timer = tic;
    U = BaseSC_Anchor_QR(B, SA, K);
    anchorTimes(r, 3) = toc(timer);
    rng(cfg.anchor.clusterSeeds(r), 'twister');
    timer = tic;
    labels = kmeans(U, K, 'Replicates', 10, 'Start', 'plus', ...
        'MaxIter', 500, 'Display', 'off');
    anchorTimes(r, 4) = toc(timer);
    anchorTimes(r, 5) = sum(anchorTimes(r, 1:4));
    score = ClusteringMeasure(y, labels);
    anchorScores(r, :) = score([1 3 4 2]);
    fprintf('Anchor %2d/%d  ACC %.6f  %.3f s\n', ...
        r, nRuns, anchorScores(r, 1), anchorTimes(r, 5));
    clear B A anchorViews SA U labels
end

names = {'ACC', 'NMI', 'ARI', 'Purity'};
report = struct();
report.dataset = cfg.dataset;
report.runs = nRuns;
report.scoreOrder = names;
report.matlabVersion = version;
report.platform = computer;
report.full = summarize(fullScores, fullTimes, cfg.full.expectedMean);
report.anchor = summarize(anchorScores, anchorTimes, cfg.anchor.expectedMean);
report.referenceTolerance = 1e-6;
report.referenceMatched = report.full.referenceMatched && ...
    report.anchor.referenceMatched;
fprintf('\n20-run means [ACC NMI ARI Purity]\n');
fprintf('Full:   %.6f %.6f %.6f %.6f\n', report.full.mean);
fprintf('Anchor: %.6f %.6f %.6f %.6f\n', report.anchor.mean);
fprintf('Mean complete-run time: Full %.3f s, Anchor %.3f s\n', ...
    report.full.timeMeanSeconds, report.anchor.timeMeanSeconds);
assert(report.referenceMatched, ...
    'The 20-run means differ from the saved reference.');
fprintf('Both methods match their saved 20-run reference means.\n');
end

function cfg = reference_config()
% Selected Hdigit parameters and the independent evaluation seeds.
cfg.dataset = 'Hdigit';
cfg.runs = 20;
cfg.full.parameters = struct('lam1', 10, 'lam2', 10000);
cfg.full.seeds = [ ...
    1356462843 1079037740 371326405 409511578 1257700974 ...
    332880247 1474407047 156153882 1132650338 499861486 ...
    1414696038 1497256400 556541777 1628821148 391888995 ...
    1841771874 709240582 1259450406 1591123653 1025161414];
cfg.full.expectedMean = [ ...
    0.969700000000000 0.929616166505694 ...
    0.934507223611267 0.969700000000000];
cfg.anchor.parameters = struct('lam1', 1, 'lam2', 100, ...
    'p', 11, 'knn', 15);
cfg.anchor.anchorSeeds = [ ...
    1915048201 1661378411 982368357 1005906798 1879793364 ...
    1361369394 850895780 905604975 2020188038 881279655 ...
    1846094604 2069775019 982414063 2039934332 408913337 ...
    519473906 1514649373 397600843 1185977036 944104566];
cfg.anchor.clusterSeeds = [ ...
    1277922320 921233692 1895204235 899404829 976899252 ...
    896762927 1261502789 1819766040 1857113586 1262511012 ...
    606987919 57914356 1861197641 1928069133 57504568 ...
    1585612577 179325199 184471808 816060816 1485407697];
cfg.anchor.expectedMean = [ ...
    0.971100000000000 0.932151324540624 ...
    0.937684986359887 0.971100000000000];
end

function [X, y, K] = load_hdigit(path)
raw = load(path);
X = raw.X;
if isfield(raw, 'Y'), y = raw.Y; else, y = raw.y; end
y = y(:);
n = numel(y);
K = numel(unique(y));
for v = 1:numel(X)
    if size(X{v}, 1) ~= n
        X{v} = full(X{v}');
    else
        X{v} = full(X{v});
    end
    lo = min(X{v}, [], 1);
    hi = max(X{v}, [], 1);
    X{v} = (X{v} - lo) ./ max(hi - lo, 1e-12);
end
end

function summary = summarize(scores, times, expected)
summary = struct();
summary.mean = mean(scores, 1);
summary.std = std(scores, 0, 1);
summary.referenceMean = reshape(expected, 1, []);
summary.maxAbsDifference = max(abs(summary.mean - summary.referenceMean));
summary.referenceMatched = summary.maxAbsDifference <= 1e-6;
summary.timeMeanSeconds = mean(times(:, end));
summary.timeStdSeconds = std(times(:, end));
summary.timeMedianSeconds = median(times(:, end));
summary.phaseMeanSeconds = mean(times, 1);
end
