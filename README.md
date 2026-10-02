# Adjoint fold sensitivities for cooperative networks

Reproducible code and archived data accompanying the manuscript *Adjoint fold sensitivities reveal intervention errors in strength-based reductions of cooperative networks*, prepared for **Nonlinear Dynamics**. The corresponding institution is Zhejiang Sci-Tech University.

The study tests whether dimension-reduction methods that predict a tipping threshold accurately also identify effective node and link interventions. It uses certified saddle-node folds, adjoint fold sensitivities (AFS), synthetic and empirical cooperative networks, and deterministic random seeds.

## Repository layout

```
.
├── code/
│   ├── core2.py             # Models, fold certification, sensitivities, and utilities
│   ├── exp2.py              # 112-network main experiment
│   ├── exh.py               # Exhaustive benchmark on small networks
│   ├── sweep.py             # Structural and parameter sweeps
│   ├── zhang.py             # Zhang et al. (2020) comparator
│   ├── analysis2.py         # Result aggregation and CSV-table generation
│   ├── figs2.py             # Figure 1--6 plotting functions
│   ├── make_figures.py      # Reproducible Figure 1--6 entry point
│   ├── numbers2.py          # Manuscript-number calculation helpers
│   ├── build_ms.py          # Optional LaTeX manuscript-token builder
│   ├── run_kevan.py         # Lightweight Kevan-network run
│   ├── webs/                # Plant-pollinator incidence matrices
│   └── examples.json        # Deterministic illustrative examples
├── data/
│   ├── results2/            # 112 archived main-experiment result pickles
│   ├── results_exh/         # Small-network exhaustive-search results
│   ├── results_sweep/       # Structural and parameter-sweep results
│   ├── results_zhang/       # Comparator results
│   └── tables2/             # Derived CSV tables and key-number JSON output
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Environment

The workflow was prepared for Python 3.12. Create an isolated environment and install the tested dependency ranges:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

The core analysis requires NumPy, SciPy, pandas, matplotlib, and NetworkX. LaTeX (`pdflatex` plus `bibtex`) is optional and only needed for `code/build_ms.py`.

## Reproducing the analyses

The `data/` directory already contains the archived outputs used for the manuscript, so the fast path is to regenerate the tables and figures from those files:

```bash
python code/make_figures.py
```

`make_figures.py` writes both PNG and PDF versions of Figures 1--6 into the untracked `figures/` directory. The six figures are produced by the following functions in `code/figs2.py`: `fig1` (fold certification and finite differences), `fig2` (thresholds and sensitivity rankings), `fig3` (node protection), `fig4` (link interventions), `fig5` (restoration), and `fig6` (lever priorities and robustness).

To recompute the archived simulations rather than use the supplied results, run these commands from `code/`. They are deterministic; `exp2.py` uses lock files, so separate copies may be run in parallel.

```bash
cd code
python exp2.py                 # Main 112-network experiment -> results2/
python exh.py                  # Exhaustive benchmark -> results_exh/
python sweep.py                # Structural/parameter sweeps -> results_sweep/
python zhang.py 0 1            # Comparator -> results_zhang/
python analysis2.py            # Aggregate CSV tables -> tables2/
```

`make_figures.py` configures `analysis2.py` to read the archived main results in `data/results2/` and write the derived tables to `data/tables2/`; use that entry point from the repository root. Full recomputation can take a substantial amount of time. To split the comparator workload across `N` workers, use `python zhang.py SHARD N` for each shard from 0 through `N-1`.

## Figures and tables

`python code/make_figures.py` regenerates Figures 1--6. The tables are machine-readable CSV files in `data/tables2/`; `table_certification.csv` contains the certification/validation summary used for Table 1, while the remaining `table_*.csv` files provide the derived numerical summaries, including the inputs used for Table 2. `numbers2.py` derives the manuscript numbers from these aggregation outputs.

## Data description

The package includes:

- Eleven empirical plant-pollinator incidence matrices in `code/webs/`, used as projected and bipartite cooperative networks.
- Deterministically generated Erdos-Renyi, scale-free, directed scale-free, heterogeneous, mutualistic, and regulatory networks defined in `exp2.py`.
- Serialized result records for the main experiment, exhaustive benchmark, structural/parameter sweeps, and Zhang-comparator analysis.
- CSV tables, including finite-difference checks, intervention outcomes, restoration results, link results, and supplementary numerical summaries.

## Citation

Please cite the accompanying manuscript while it is under review:

> Author(s). *Adjoint fold sensitivities reveal intervention errors in strength-based reductions of cooperative networks*. Manuscript submitted to **Nonlinear Dynamics**; Zhejiang Sci-Tech University.

The PDF supplied with this package contains placeholder author and affiliation fields. Please replace the citation above with the final published bibliographic record (authors, volume, pages, year, and DOI) once available.

## License

This repository is released under the [MIT License](LICENSE).
