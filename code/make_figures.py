"""Regenerate Figures 1--6 from the bundled result files.

Run from the repository root with ``python code/make_figures.py``.  The script
uses the archived results in ``data/`` and writes PNG and PDF figure files to
``figures/`` without changing the archived result pickles.
"""
import glob
import json
import os
import pickle

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "figures")

# analysis2 was originally run beside its result directories.  Point it to
# the repository's archival layout before importing it.
os.environ.setdefault("RC_RESULTS", os.path.join("..", "data", "results2"))
os.environ.setdefault("RC_TABLES", os.path.join("..", "data", "tables2"))

import analysis2
import figs2


def load_pickles(directory):
    return [pickle.load(open(path, "rb")) for path in sorted(glob.glob(os.path.join(directory, "*.pkl")))]


def exhaustive_frame(records):
    """Convert the compact exhaustive-search pickles to the plotting table."""
    rows = []
    for record in records:
        for (dose, budget), result in record["res"].items():
            for strategy, value in result["vals"].items():
                rows.append(dict(label=record["label"], group=record["group"], dyn=record["dyn"],
                                 d=dose, k=budget, strat=strategy,
                                 eff=float(value) / result["opt"] if result["opt"] else float("nan")))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = analysis2.main()
    sweeps = load_pickles(os.path.join(DATA, "results_sweep"))
    exhaustive = exhaustive_frame(load_pickles(os.path.join(DATA, "results_exh")))
    # For the star reference in Figure 4, <s^2>/<s>^2 of a 100-node star.
    n = 100
    examples = {"star_het": ((n - 1) ** 2 + (n - 1)) / n / (2 * (n - 1) / n) ** 2}
    with open(os.path.join(HERE, "examples.json"), encoding="utf-8") as handle:
        examples.update(json.load(handle))

    figs2.fig1(data, outdir=OUT)
    figs2.fig2(data, sweeps, outdir=OUT)
    figs2.fig3(data, exhaustive, outdir=OUT)
    figs2.fig4(data, sweeps, examples, outdir=OUT)
    figs2.fig5(data, outdir=OUT)
    figs2.fig6(data, sweeps, outdir=OUT)
    print(f"Wrote Figures 1--6 to {OUT}")


if __name__ == "__main__":
    main()
