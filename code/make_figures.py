"""Regenerate manuscript Figures 1--6 from the archived result files.

Run from the repository root with ``python code/make_figures.py``. The script
reads archived results under ``data/`` and writes PNG/PDF figures to
``figures/`` without changing the archived result pickles.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path(os.environ.get("RC_DATA", ROOT / "data")).resolve()
OUT = Path(os.environ.get("RC_OUTPUT", ROOT / "figures")).resolve()
TABLES = Path(os.environ.get("RC_TABLES", ROOT / "output_tables")).resolve()


def load_pickles(directory: Path) -> list[dict]:
    """Load all serialized result records from an archived-result directory."""
    files = sorted(directory.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No result files found in {directory}")
    records = []
    for filename in files:
        with filename.open("rb") as handle:
            records.append(pickle.load(handle))
    return records


def exhaustive_frame(records: list[dict]) -> pd.DataFrame:
    """Convert exhaustive-search records to the table consumed by ``fig3``."""
    rows = []
    for record in records:
        for (dose, budget), result in record.get("res", {}).items():
            optimum = result["opt"]
            for strategy, value in result["vals"].items():
                rows.append({
                    "label": record["label"], "group": record["group"],
                    "dyn": record["dyn"], "d": dose, "k": budget,
                    "strat": strategy, "eff": value / optimum if optimum else np.nan,
                    "n_sets": result["n_sets"], "opt_lost": result["opt_lost"],
                })
    if not rows:
        raise ValueError("The exhaustive-result files contain no benchmark records.")
    return pd.DataFrame(rows)


def zhang_frame(records: list[dict]) -> pd.DataFrame:
    """Return Zhang-comparator correlations consumed by ``fig2``.

    The figure function expects this table as ``D[\"ZH\"]``; keeping the
    transformation here makes the Zhang et al. row data-driven and complete.
    """
    rows = [
        {"label": record["label"], "group": record["group"], "rho": record["rho"]}
        for record in records if "rho" in record
    ]
    if not rows:
        raise ValueError("The Zhang-comparator files contain no correlation records.")
    return pd.DataFrame(rows)


def star_heterogeneity(leaves: int) -> float:
    """Return mean(degree**2) / mean(degree)**2 for ``K_1,leaves``."""
    degrees = np.array([leaves] + [1] * leaves, dtype=float)
    return float(np.mean(degrees**2) / np.mean(degrees) ** 2)


def main() -> None:
    for name in ("results2", "results_exh", "results_sweep", "results_zhang"):
        directory = DATA / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Required data directory is missing: {directory}")

    plt.style.use(HERE / "manuscript.mplstyle")
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    # analysis2 resolves paths at import time; configure archival inputs and
    # disposable derived-table output before importing it.
    os.environ["RC_RESULTS"] = str(DATA / "results2")
    os.environ["RC_TABLES"] = str(TABLES)
    import analysis2
    import figs2

    data = analysis2.main()
    # Figure 4 needs adjacency matrices, which analysis2 does not return.
    data["A"] = {
        label: analysis2.rebuild_model(record).A
        for label, record in data["R"].items()
    }
    sweeps = load_pickles(DATA / "results_sweep")
    exhaustive = exhaustive_frame(load_pickles(DATA / "results_exh"))
    data["ZH"] = zhang_frame(load_pickles(DATA / "results_zhang"))

    examples = json.loads((HERE / "examples.json").read_text(encoding="utf-8"))
    star = examples["star_control"]
    if not np.isclose(star["star_het"], star_heterogeneity(star["leaves"])):
        raise ValueError("examples.json has an inconsistent star_het value.")

    figs2.fig1(data, outdir=OUT)
    figs2.fig2(data, sweeps, outdir=OUT)
    figs2.fig3(data, exhaustive, outdir=OUT)
    figs2.fig4(data, sweeps, {"star_het": star["star_het"]}, outdir=OUT)
    figs2.fig5(data, outdir=OUT)
    figs2.fig6(data, sweeps, outdir=OUT)

    expected = (
        "fig1_certification", "fig2_threshold_vs_sensitivity",
        "fig3_node_protection", "fig4_links", "fig5_restoration",
        "fig6_levers_robustness",
    )
    missing = [
        name for name in expected
        if not all((OUT / f"{name}.{extension}").is_file() for extension in ("png", "pdf"))
    ]
    if missing:
        raise RuntimeError(f"Figure export failed: {', '.join(missing)}")
    print(f"Wrote Figures 1--6 to {OUT}")


if __name__ == "__main__":
    main()
