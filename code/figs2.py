"""figs2 -- figures for the revised manuscript (run inside a kernel where apply_figure_style/panel_letter exist,
after analysis2.main() has produced D)."""
import os, pickle, json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import core2 as C
import exp2 as X

GCOL = {"Mut-ER": "#9ecae1", "Mut-SF": "#4292c6", "Webs-proj": "#08519c", "Webs-bip": "#6baed6", "Mut-SF-het": "#2171b5",
        "Reg-ER": "#fdae6b", "Reg-SF": "#f16913", "Reg-dir": "#a63603", "Reg-SF-het": "#d94801",
        "RegB-ER": "#bcbddc", "RegB-SF": "#756bb1"}
GLAB = {"Mut-ER": "Mut ER", "Mut-SF": "Mut SF", "Webs-proj": "Webs (proj.)", "Webs-bip": "Webs (bip.)",
        "Mut-SF-het": "Mut SF het.", "Reg-ER": "Reg ER", "Reg-SF": "Reg SF", "Reg-dir": "Reg dir. SF",
        "Reg-SF-het": "Reg SF het.", "RegB-ER": "RegB ER", "RegB-SF": "RegB SF"}
GORDER = ["Mut-ER", "Mut-SF", "Mut-SF-het", "Webs-proj", "Webs-bip", "Reg-ER", "Reg-SF", "Reg-SF-het", "Reg-dir", "RegB-ER", "RegB-SF"]
FOCAL = "#6a1b9a"


def panel_letter(ax, letter, **kw):
    """fallback panel label (the manuscript figures were drawn with the figure-style helper of the same name)"""
    ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
GREY = "#8c8c8c"
SLAB = {"adjoint": "AFS", "adaptive_adjoint": "adaptive AFS", "closure_exact": "closure-exact", "RC_red": "closure plug-in",
        "RC1": "PI1", "RC2": "PI2", "GBB_gradient": "out-strength (GBB)", "spectral_gradient": "eigenvector (spectral)",
        "in_strength": "in-strength", "kcore": "k-core", "betweenness": "betweenness", "slow_mode": "slow mode",
        "zhang": "Zhang et al."}


def strip(ax, groups, vals, color=None, x0=0, w=0.28, s=9, med=True, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    for k, (g, v) in enumerate(zip(groups, vals)):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        col = GCOL.get(g, GREY) if color is None else color
        ax.scatter(k + x0 + rng.uniform(-w / 2, w / 2, len(v)), v, s=s, color=col, alpha=0.75, lw=0, zorder=3)
        if med and len(v):
            ax.plot([k + x0 - w / 1.6, k + x0 + w / 1.6], [np.median(v)] * 2, color="k", lw=1.2, zorder=4)


def save(fig, name):
    fig.savefig(name + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(name + ".pdf", bbox_inches="tight")


def fig1(D, net="SF_mut_0", outdir="."):
    R = D["R"]
    r = R[net]
    t = {tt["label"]: tt for tt in X.build_tasks()}[net]
    m = X.make_model(t)
    fold, b0 = X.base_collapse(m)
    # branches
    bs_hi = np.linspace(fold["beta"], 3.0 * fold["beta"], 60)
    hi = []
    x = fold["x"] + 1e-3 * fold["v"]
    for b in bs_hi[::-1]:
        x2, ok = C.high_state(m, b) if not hi else C.newton(m, hi[-1][1], b)
        hi.append((b, x2))
    hi = sorted(hi, key=lambda z: z[0])
    fr = X.base_recovery(m, fold["beta"])
    lo = []
    for b in np.linspace(0.3 * fold["beta"], fr["beta"], 60):
        xl, ok = C.low_state(m, b) if not lo else C.newton(m, lo[-1][1], b)
        lo.append((b, xl))
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.05), gridspec_kw=dict(wspace=0.78))
    ax = axs[0]
    ax.plot([fold["beta"]] + [b for b, _ in hi], [fold["x"].mean()] + [xx.mean() for _, xx in hi], color=FOCAL, lw=1.4)
    ax.plot([b for b, _ in lo], [xx.mean() for _, xx in lo], color=GREY, lw=1.4)
    ax.axvline(fold["beta"], color=FOCAL, ls=":", lw=0.8)
    ax.axvline(fr["beta"], color=GREY, ls=":", lw=0.8)
    ax.text(fold["beta"], ax.get_ylim()[1] * 0.98, r" $\beta_c$", va="top", fontsize=7, color=FOCAL)
    ax.text(fr["beta"], ax.get_ylim()[1] * 0.98, r" $\beta_r$", va="top", fontsize=7, color=GREY)
    ax.set_xlabel(r"interaction strength $\beta$")
    ax.set_ylabel(r"mean abundance $\langle x\rangle$")
    ax.set_title("Bistable range", loc="left")
    ax = axs[1]
    a = r["arrays"]
    sc = ax.scatter(a["s_out"], a["R"] / a["R"].sum(), c=a["x"], s=10, cmap="viridis", lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("out-strength")
    ax.set_ylabel(r"AFS $R_i$ (normalized)")
    ax.set_title("Priority vs strength", loc="left")
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cb.ax.set_title(r"$x_{c,i}$", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    ax = axs[2]
    best = D["FD"].sort_values("relerr").drop_duplicates(["label", "kind", "pred"])
    for kind, col, lab in (("node", FOCAL, "input"), ("lever", "#e6550d", "mortality"), ("link", "#31a354", "link weight")):
        b_ = best[best.kind == kind]
        b_ = b_[np.abs(b_.pred) > 1e-12]
        ax.scatter(np.abs(b_.pred), np.abs(b_.fd), s=5, color=col, lw=0, alpha=0.7, label=lab)
    lims = [1e-12, 1e2]
    ax.plot(lims, lims, color="k", lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-8, 1e1); ax.set_ylim(1e-8, 1e1)
    ax.set_xlabel("|adjoint sensitivity|")
    ax.set_ylabel("|central difference|")
    ax.set_title("Adjoint vs finite diff.", loc="left")
    ax.legend(frameon=False, fontsize=6, loc="upper left", handletextpad=0.1, markerscale=1.5)
    ax = axs[3]
    FD = D["FD"]
    FD = FD[np.abs(FD.pred) > 1e-10]
    for dyn, col in (("mutualistic", "#3182bd"), ("regulatory", "#e6550d")):
        g = FD[FD.dyn == dyn].groupby("h").relerr
        med = g.median(); q9 = g.quantile(0.9)
        ax.plot(med.index, med.values, "o-", color=col, ms=3, lw=1, label=dyn)
        ax.fill_between(med.index, med.values, q9.values, color=col, alpha=0.2, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("step size $h$")
    ax.set_ylabel("relative error")
    ax.set_title("Step-size test", loc="left")
    ax.legend(frameon=False, fontsize=6)
    for k, ax in enumerate(axs):
        panel_letter(ax, "abcd"[k])
    save(fig, os.path.join(outdir, "fig1_certification"))
    return fig


def fig2(D, SW, outdir="."):
    T1 = D["T1"]
    groups = [g for g in GORDER if g in T1.group.unique()]
    fig = plt.figure(figsize=(7.2, 4.3))
    gs = fig.add_gridspec(2, 3, hspace=0.8, wspace=0.95, width_ratios=[1.25, 1.25, 1])
    ax = fig.add_subplot(gs[0, 0])
    for j, (col, lab, c) in enumerate((("err_gbb", "GBB", GREY), ("err_spec", "spectral", "#bdbdbd"), ("err_closure", "closure", "#1b9e77"))):
        strip(ax, groups, [100 * T1[T1.group == g][col].values for g in groups], color=c, x0=(j - 1) * 0.28, w=0.22, s=5)
        ax.scatter([], [], color=c, s=10, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([GLAB[g] for g in groups], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("threshold error (%)")
    ax.set_title("Closures predict thresholds", loc="left")
    ax.legend(frameon=False, fontsize=5.5, ncol=1, loc="lower right", handletextpad=0.1)
    ax = fig.add_subplot(gs[0, 1:])
    scores = ["GBB_gradient", "zhang", "spectral_gradient", "closure_exact", "RC_red", "slow_mode", "RC1", "RC2"]
    Z = D.get("ZH")
    M = np.full((len(scores), len(groups)), np.nan)
    for i, s_ in enumerate(scores):
        for j, g in enumerate(groups):
            if s_ == "zhang":
                if Z is not None and len(Z):
                    M[i, j] = Z[Z.group == g].rho.mean()
            else:
                col = "rho_" + s_
                if col in T1:
                    M[i, j] = T1[T1.group == g][col].mean()
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.2, color="w" if abs(M[i, j]) > 0.6 else "k")
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels([SLAB[s_] for s_ in scores], fontsize=6)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([GLAB[g] for g in groups], rotation=60, ha="right", fontsize=6)
    ax.set_title(r"Rank correlation with the AFS ranking (Spearman $\rho$)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[1, 0])
    ratios = [1.001, 1.01, 1.05, 1.1, 1.3, 2.0]
    for g in groups:
        sub = T1[T1.group == g]
        ys = [sub[f"rho_slow_left_{q}"].mean() for q in ratios]
        ax.plot(np.array(ratios) - 1, ys, "o-", ms=2.5, lw=1, color=GCOL[g])
    ax.set_xscale("log")
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1]); ax.set_xticklabels(["0.001", "0.01", "0.1", "1"])
    ax.set_xlabel(r"distance to fold $\beta/\beta_c-1$")
    ax.set_ylabel(r"$\rho$(slow mode, AFS)")
    ax.set_title("Slow mode aligns only near the fold", loc="left")
    ax = fig.add_subplot(gs[1, 1])
    for dyn, mk in (("mutualistic", "o"), ("regulatory", "s")):
        rows = [r_[dyn] for r_ in SW if r_["kind"] == "struct" and dyn in r_ and r_[dyn].get("ok")]
        het = [q["heterogeneity"] for q in rows]
        rho = [q["rho_GBB_gradient"] for q in rows]
        ax.scatter(het, rho, s=6, marker=mk, color="#3182bd" if dyn == "mutualistic" else "#e6550d", lw=0, alpha=0.7, label=dyn)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["1", "2", "3", "4"]); ax.minorticks_off()
    ax.set_xlabel(r"strength heterogeneity $\langle s^2\rangle/\langle s\rangle^2$")
    ax.set_ylabel(r"$\rho$(out-strength, AFS)")
    ax.set_title("Structural sweep", loc="left")
    ax.legend(frameon=False, fontsize=6, loc="lower right", handletextpad=0.1)
    ax = fig.add_subplot(gs[1, 2])
    for g in groups:
        sub = T1[T1.group == g]
        ax.scatter(sub["rho_GBB_gradient"], sub["eff1_GBB_gradient"], s=7, color=GCOL[g], lw=0)
    ax.set_xlabel(r"$\rho$(out-strength, AFS)")
    ax.set_ylabel("first-order efficiency")
    ax.set_title("Rank error costs gain", loc="left")
    for k, ax in enumerate(fig.axes[:1] + fig.axes[1:2] + fig.axes[3:]):
        panel_letter(ax, "abcde"[k])
    hs = [mpl.lines.Line2D([], [], color=GCOL[g], marker="o", ms=3, lw=1) for g in groups]
    fig.legend(hs, [GLAB[g] for g in groups], loc="lower center", ncol=6, frameon=False, fontsize=5.5,
               bbox_to_anchor=(0.5, -0.06), handletextpad=0.3, columnspacing=1.0)
    save(fig, os.path.join(outdir, "fig2_threshold_vs_sensitivity"))
    return fig


def fig3(D, EXH, outdir="."):
    NP, GL = D["NP"], D["GL"]
    groups = [g for g in ["Mut-ER", "Mut-SF", "Webs-proj", "Webs-bip", "Reg-ER", "Reg-SF", "Reg-dir", "RegB-SF"] if g in NP.group.unique()]
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.05], hspace=0.55)
    top = gs[0].subgridspec(1, len(groups), wspace=0.08)
    vmax = 1.0
    for j, g in enumerate(groups):
        ax = fig.add_subplot(top[j])
        sub = NP[NP.group == g]
        piv = sub.pivot_table(index=["label", "q", "d"], columns="strat", values="comp")
        both = (piv["GBB_gradient"] < 1) & (piv["adjoint"] < 1)
        lr = np.log2((piv["GBB_gradient"] / piv["adjoint"]).where(both))
        lr = lr.groupby(level=["q", "d"]).median().unstack("d")
        el = (piv["GBB_gradient"] >= 1).groupby(level=["q", "d"]).mean().unstack("d")
        ela = (piv["adjoint"] >= 1).groupby(level=["q", "d"]).mean().unstack("d")
        im = ax.imshow(lr.values, cmap="PuOr_r", norm=TwoSlopeNorm(0, -vmax, vmax), origin="lower", aspect="auto")
        for a_ in range(lr.shape[0]):
            for b_ in range(lr.shape[1]):
                v = lr.values[a_, b_]
                txt = f"{2 ** v:.2f}" if np.isfinite(v) else "n.d."
                e1, e2 = el.values[a_, b_], ela.values[a_, b_]
                if e1 > 0 or e2 > 0:
                    txt += f"\n{int(round(100 * e1))}|{int(round(100 * e2))}"
                ax.text(b_, a_, txt, ha="center", va="center", fontsize=4.6, color="w" if np.isfinite(v) and abs(v) > 0.6 else "k", linespacing=0.9)
        ax.set_xticks(range(lr.shape[1]))
        ax.set_xticklabels([f"{d:g}" for d in lr.columns], fontsize=5.5, rotation=90)
        ax.set_yticks(range(lr.shape[0]))
        ax.set_yticklabels([f"{int(100 * q)}%" for q in lr.index] if j == 0 else [], fontsize=6)
        ax.set_title(GLAB[g], fontsize=6.5)
        if j == 0:
            ax.set_ylabel("budget")
            panel_letter(ax, "a")
        ax.set_xlabel(r"dose $\delta$", fontsize=6)
    cax = fig.add_axes([0.93, 0.60, 0.01, 0.28])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks(np.log2([0.5, 1, 2])); cb.set_ticklabels(["0.5", "1", "2"])
    cb.ax.tick_params(labelsize=6)
    cb.set_label("out-strength / AFS", fontsize=6)
    bot = gs[1].subgridspec(1, 3, wspace=0.45)
    ax = fig.add_subplot(bot[0])
    strats = ["adjoint", "adaptive_adjoint", "greedy", "GBB_gradient", "spectral_gradient", "RC1"]
    cols = [FOCAL, "#9e9ac8", "k", GREY, "#bdbdbd", "#1b9e77"]
    E_ = EXH
    for i, (s_, c) in enumerate(zip(strats, cols)):
        v = E_[E_.strat == s_].eff.clip(lower=0)
        ax.scatter(i + np.random.default_rng(1).uniform(-0.25, 0.25, len(v)), v, s=3, color=c, lw=0, alpha=0.5)
        ax.plot([i - 0.3, i + 0.3], [v.median()] * 2, color="k", lw=1.2)
    ax.set_xticks(range(len(strats)))
    ax.set_xticklabels([SLAB.get(s_, s_) for s_ in strats], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("fraction of exhaustive optimum")
    ax.set_title("Small networks: exact optimum", loc="left")
    panel_letter(ax, "b")
    ax = fig.add_subplot(bot[1])
    gg = [g for g in GORDER if g in GL.group.unique()]
    for i, (c_, col) in enumerate((("eff_adjoint", FOCAL), ("eff_screen", "#9e9ac8"), ("eff_gbb", GREY), ("eff_rnd_med", "#d9d9d9"))):
        strip(ax, gg, [GL[GL.group == g][c_].values for g in gg], color=col, x0=(i - 1.5) * 0.2, w=0.16, s=5)
    for lab, col in (("AFS", FOCAL), ("screen+rerank", "#9e9ac8"), ("out-strength", GREY), ("random (median)", "#d9d9d9")):
        ax.scatter([], [], color=col, s=10, label=lab)
    ax.set_xticks(range(len(gg)))
    ax.set_xticklabels([GLAB[g] for g in gg], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("fraction of best set found")
    ax.set_title("Greedy + local search benchmark", loc="left")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left", handletextpad=0.1)
    panel_letter(ax, "c")
    ax = fig.add_subplot(bot[2])
    ost = NP[NP.q == 0.10].groupby(["dyn", "d", "strat"]).apply(lambda s: (s.status == "lost").mean()).reset_index(name="lost")
    for s_, col, mk in (("adjoint", FOCAL, "o"), ("adaptive_adjoint", "#9e9ac8", "^"), ("GBB_gradient", GREY, "s"), ("spectral_gradient", "#bdbdbd", "D")):
        sub = ost[(ost.strat == s_) & (ost.dyn == "mutualistic")]
        ax.plot(sub.d, 100 * sub.lost, marker=mk, ls="-", ms=3, lw=1, color=col, label=SLAB[s_], mfc="none" if mk == "D" else col)
    ax.set_xscale("log")
    ax.set_xlabel(r"dose $\delta$ (mutualistic, 10% budget)")
    ax.set_ylabel("folds eliminated (%)")
    ax.set_title("Large doses remove bistability", loc="left")
    ax.legend(frameon=False, fontsize=5.5, handletextpad=0.1)
    panel_letter(ax, "d")
    save(fig, os.path.join(outdir, "fig3_node_protection"))
    return fig


def fig4(D, SW, EX, outdir="."):
    LR, ST, T1 = D["LR"], D["ST"], D["T1"]
    groups = [g for g in GORDER if g in LR.group.unique()]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=0.5, width_ratios=[1.5, 1, 1]))
    ax = axs[0]
    strats = ["adjoint", "RC1", "GBB_gradient", "strength_product", "spectral_edge", "edge_betweenness"]
    cols = [FOCAL, "#1b9e77", GREY, "#fdae6b", "#bdbdbd", "#d9d9d9"]
    sub = LR[(LR.budget == "count") & (LR.q == 0.05)]
    for i, (s_, c) in enumerate(zip(strats, cols)):
        vals = [sub[(sub.group == g) & (sub.strat == s_)].rel_best.values for g in groups]
        strip(ax, groups, vals, color=c, x0=(i - 2.5) * 0.13, w=0.1, s=3)
        ax.scatter([], [], color=c, s=10, label={"adjoint": "AFS", "RC1": "PI1", "GBB_gradient": "GBB gradient",
                                                 "strength_product": "strength product", "spectral_edge": "spectral edge",
                                                 "edge_betweenness": "edge betweenness"}[s_])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([GLAB[g] for g in groups], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("damage / best rule")
    ax.set_title("Removing 5% of links", loc="left")
    ax.legend(frameon=False, fontsize=5.5, ncol=6, loc="upper left", bbox_to_anchor=(-0.05, -0.42), handletextpad=0.1, columnspacing=0.8)
    ax = axs[1]
    gp = ST.copy()
    beff = {l: C.gbb_quantities(D["A"][l])["beta_eff"] for l in gp.label.unique()}
    gp["gbb_rel"] = gp.gbb_damage / gp.label.map(beff)
    ax.scatter(100 * gp.gbb_rel, 100 * gp.full_damage, s=4, color=FOCAL, lw=0, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"GBB predicted $\Delta\beta_c$ (%)")
    ax.set_ylabel(r"full-model $\Delta\beta_c$ (%)")
    ax.set_title("Sign test: deleting 'harmful' links", loc="left")
    ax = axs[2]
    rows = [r_["mutualistic"] for r_ in SW if r_["kind"] == "struct" and r_["mutualistic"].get("ok")]
    ax.scatter([q["heterogeneity"] for q in rows], [100 * q["frac_neg_gbb"] for q in rows], s=6, color=GREY, lw=0, label="synthetic sweep")
    for g in groups:
        s2 = T1[T1.group == g]
        q_ = [C.gbb_quantities(D["A"][l]) for l in s2.label]
        het = [(qq["s_out"] ** 2).mean() / qq["s_out"].mean() ** 2 for qq in q_]
        ax.scatter(het, 100 * s2.frac_neg_gbb, s=7, color=GCOL[g], lw=0)
    ax.scatter([EX["star_het"]], [0], marker="*", s=40, color="k", label="star")
    ax.scatter([1.0], [0], marker="D", s=14, color="k", label="regular")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"]); ax.minorticks_off()
    ax.set_xlabel(r"strength heterogeneity $\langle s^2\rangle/\langle s\rangle^2$")
    ax.set_ylabel("links with wrong GBB sign (%)")
    ax.set_title("Wrong-sign links vs heterogeneity", loc="left")
    ax.legend(frameon=False, fontsize=5.5, handletextpad=0.1, loc="upper left")
    for k, ax in enumerate(axs):
        panel_letter(ax, "abc"[k])
    save(fig, os.path.join(outdir, "fig4_links"))
    return fig


def fig5(D, net="SF_mut_0", outdir="."):
    RC, RS, TD, R = D["RC"], D["RS"], D["TD"], D["R"]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0), gridspec_kw=dict(wspace=0.6))
    ax = axs[0]
    a = R[net]["arrays"]
    sc = ax.scatter(a["R"] / a["R"].sum(), a["R_rest"] / a["R_rest"].sum(), c=np.log10(a["s_out"]), s=9, cmap="cividis", lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("collapse AFS")
    ax.set_ylabel("restoration AFS")
    ax.set_title("Two folds, two priorities", loc="left")
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.ax.set_title(r"log$_{10}s^\mathrm{out}$", fontsize=5.5); cb.ax.tick_params(labelsize=5.5)
    ax = axs[1]
    groups = [g for g in GORDER if g in RC.group.unique()]
    for j, (c_, col) in enumerate((("rho_u_ulow", FOCAL), ("rho_GBB_gradient", GREY))):
        strip(ax, groups, [RC[RC.group == g][c_].values for g in groups], color=col, x0=(j - 0.5) * 0.3, w=0.25, s=5)
    ax.scatter([], [], color=FOCAL, s=10, label="collapse AFS"); ax.scatter([], [], color=GREY, s=10, label="out-strength")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([GLAB[g] for g in groups], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel(r"$\rho$ with restoration AFS")
    ax.set_title("Restoration favours hubs", loc="left")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left", handletextpad=0.1)
    ax = axs[2]
    sub = RS[(RS.q == 0.05)]
    d0 = sub.groupby("group").d.transform(lambda x: sorted(set(x))[1])
    sub = sub[sub.d == d0]
    strats = ["restoration_adjoint", "GBB_gradient", "collapse_adjoint"]
    cols = [FOCAL, GREY, "#fdae6b"]
    for i, (s_, c) in enumerate(zip(strats, cols)):
        strip(ax, groups, [100 * sub[(sub.group == g) & (sub.strat == s_)].comp.clip(upper=1).values for g in groups], color=c, x0=(i - 1) * 0.25, w=0.2, s=4)
        ax.scatter([], [], color=c, s=10, label={"restoration_adjoint": "restoration AFS", "GBB_gradient": "out-strength", "collapse_adjoint": "collapse AFS"}[s_])
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([GLAB[g] for g in groups], rotation=60, ha="right", fontsize=6)
    ax.set_ylabel(r"lowering of $\beta_r$ (%)")
    ax.set_title("5% budget, default dose", loc="left")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", handletextpad=0.1)
    ax = axs[3]
    td = TD[TD.recovered]
    ax.scatter(td.comp.clip(upper=1) * 100, td.t_rec, s=5, color=FOCAL, lw=0, alpha=0.6, label="recovered")
    ax.set_xlabel(r"predicted lowering of $\beta_r$ (%)")
    ax.set_ylabel("time to recovery")
    ax.set_yscale("log")
    ax.set_title("Threshold shift predicts recovery", loc="left")
    for k, ax in enumerate(axs):
        panel_letter(ax, "abcd"[k])
    save(fig, os.path.join(outdir, "fig5_restoration"))
    return fig


def fig6(D, SW, outdir="."):
    CC, MS, T1 = D["CC"], D["MS"], D["T1"]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0), gridspec_kw=dict(wspace=0.62))
    ax = axs[0]
    groups = [g for g in ["Mut-ER", "Mut-SF", "Webs-proj", "Webs-bip", "Reg-ER", "Reg-SF", "Reg-dir", "RegB-SF"] if g in CC.group.unique()]
    for j, (c_, col) in enumerate((("lev_lever", FOCAL), ("lev_input", "#fdae6b"), ("lev_gbb", GREY))):
        strip(ax, groups, [(CC[CC.group == g][c_] / CC[CC.group == g]["lev_lever"]).values for g in groups], color=col, x0=(j - 1) * 0.25, w=0.2, s=4)
        ax.scatter([], [], color=col, s=10, label={"lev_lever": r"$R^\mu$ (lever AFS)", "lev_input": r"$R$ (input AFS)", "lev_gbb": "out-strength"}[c_])
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([GLAB[g] for g in groups], rotation=60, ha="right", fontsize=5.5)
    ax.set_ylabel("gain / lever AFS")
    ax.set_title("Priorities are lever-specific", loc="left")
    ax.legend(frameon=False, fontsize=5, loc="lower left", bbox_to_anchor=(0.0, 0.25), handletextpad=0.1)
    ax = axs[1]
    for g in ["Mut-ER", "Mut-SF", "Webs-proj", "Reg-ER", "Reg-SF", "Reg-dir"]:
        curves = []
        for lab, r in D["R"].items():
            if r["group"] != g:
                continue
            k5 = X.kof(0.05, r["N"])
            ms_ = [1, 2, max(1, k5 // 2), k5, 2 * k5, 4 * k5]
            con = r["concentration"]
            curves.append([con[("adjoint", mm)]["comp"] / con[("adjoint", k5)]["comp"] for mm in ms_])
        ax.plot(range(6), np.median(np.array(curves), axis=0), "o-", ms=2.5, lw=1, color=GCOL[g], label=GLAB[g])
    ax.set_xticks(range(6)); ax.set_xticklabels(["1", "2", "k/2", "k", "2k", "4k"], fontsize=6)
    ax.set_xlabel("nodes sharing the total dose")
    ax.set_ylabel("gain / (k nodes)")
    ax.set_title("Fixed total dose", loc="left")
    ax.set_ylim(top=1.95)
    ax.legend(frameon=False, fontsize=4.8, handletextpad=0.1, ncol=2, loc="upper right", columnspacing=0.5)
    ax = axs[2]
    if len(MS):
        MS = MS.copy()
        MS["eff_adj"] = MS.adj_hat / MS.adj_true
        MS["eff_gbb"] = MS.gbb_hat / MS.adj_true
        gg = [g for g in GORDER if g in MS.group.unique()]
        for j, sig in enumerate((0.1, 0.3)):
            s_ = MS[MS.sigma == sig]
            strip(ax, gg, [s_[s_.group == g].groupby("label").eff_adj.median().values for g in gg], color=FOCAL if sig == 0.1 else "#9e9ac8", x0=(j - 1) * 0.22, w=0.18, s=4)
        s_ = MS[MS.sigma == 0.3]
        strip(ax, gg, [s_[s_.group == g].groupby("label").eff_gbb.median().values for g in gg], color=GREY, x0=0.22, w=0.18, s=4)
        for lab, col in ((r"AFS, $\sigma=0.1$", FOCAL), (r"AFS, $\sigma=0.3$", "#9e9ac8"), ("out-strength", GREY)):
            ax.scatter([], [], color=col, s=10, label=lab)
        ax.set_xticks(range(len(gg))); ax.set_xticklabels([GLAB[g] for g in gg], rotation=60, ha="right", fontsize=5.5)
        ax.set_ylabel("gain / oracle AFS")
        ax.set_title("Misspecified model", loc="left")
        ax.legend(frameon=False, fontsize=5, loc="lower right", handletextpad=0.1)
    ax = axs[3]
    rows = [r_["mutualistic"] for r_ in SW if r_["kind"] == "lhs" and r_["mutualistic"].get("ok")]
    ax.scatter([q["rho_GBB_gradient"] for q in rows], [q["rho_RC1"] for q in rows], s=6, color="#3182bd", lw=0, alpha=0.7)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\rho$(out-strength, AFS)")
    ax.set_ylabel(r"$\rho$(PI1, AFS)")
    ax.set_title("Parameter sweep (mutualistic)", loc="left")
    for k, ax in enumerate(axs):
        panel_letter(ax, "abcd"[k])
    save(fig, os.path.join(outdir, "fig6_levers_robustness"))
    return fig
