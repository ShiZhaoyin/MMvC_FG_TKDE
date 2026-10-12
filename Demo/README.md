# MMvC-FG demo on Hdigit

Open `run_demo.m` in MATLAB and click **Run**. The script runs MMvC-FG and
ACC-MMvC-FG for 20 saved random seeds each, prints their mean ACC, NMI, ARI,
Purity, and elapsed time, and checks the means against the saved results.
It needs no input arguments, configuration file, or other repository folders.

The selected parameters are written in `run_demo.m`:

| Method | `lambda1` | `lambda2` | Other parameters | Expected mean ACC |
| --- | ---: | ---: | --- | ---: |
| MMvC-FG | 10 | 10000 | full sample graph | 0.969700 |
| ACC-MMvC-FG | 1 | 100 | 2048 anchors (`p=11`), 15 nearest anchors | 0.971100 |

`data/Hdigit.mat` is the only dataset. The six MATLAB files in `private/` are
the algorithm and clustering functions. MATLAB needs `kmeans` and `randsrc`.
The demo was verified with MATLAB R2025b. Its 20-run mean results are:

| Method | ACC | NMI | ARI | Purity |
| --- | ---: | ---: | ---: | ---: |
| MMvC-FG | 0.969700 | 0.929616 | 0.934507 | 0.969700 |
| ACC-MMvC-FG | 0.971100 | 0.932151 | 0.937685 | 0.971100 |

The script also returns a `report` struct. Runtime depends on the computer;
the reference check applies to clustering scores only.
