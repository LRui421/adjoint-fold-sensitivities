# Online Resource 2 — code and data

**Environment.** Python 3.12 with numpy, scipy, networkx, pandas and matplotlib. LaTeX (pdflatex with bibtex) is needed only to rebuild the manuscript.

**Reproduce the six figures.** From the repository root, run
`python code/make_figures.py`. The script reads `data/results2`,
`data/results_exh`, `data/results_sweep`, and `data/results_zhang`; it creates
all required intermediate inputs and writes PNG/PDF figures to `figures/`.
Set `RC_DATA` and `RC_OUTPUT` only when using a non-standard data or output
location.

**Files.**
- core2.py: models (mutualistic, regulatory), Moore–Spence fold certification, fold diagnostics, guarded fold tracking along intervention paths, sensitivities, node-resolved closure and its exact adjoint, and power iteration.
- exp2.py: per-network analysis and task list (112 networks). Run `python exp2.py`; several copies can run in parallel because they coordinate through lock files. Results go to results2/*.pkl.
- run_kevan.py: kevan1970 without the greedy, local-search and misspecification steps.
- exh.py: exhaustive benchmark on small networks (results_exh/).
- sweep.py: structural and parameter sweeps (results_sweep/).
- zhang.py: Zhang et al. (2020) comparator. Run `python zhang.py 0 1` (results_zhang/).
- analysis2.py: aggregation into tables2/*.csv. numbers2.py computes every number quoted in the text (tokens.json).
- figs2.py: the six figure functions. make_figures.py is the standalone reproduction entry point; manuscript.mplstyle supplies the compact plotting defaults. build_ms.py fills the manuscript tokens and compiles it.
- webs/: the 11 plant–pollinator incidence matrices from the R package bipartite.
- examples.json: symmetric control (six-regular graph), star and heterogeneity examples, and the periodic power-iteration example.

**Reproducibility.** Synthetic networks are generated deterministically from the seeds in exp2.build_tasks, exh.small_tasks and sweep.jobs. All random baselines use fixed seeds.
