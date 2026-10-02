"""analysis2 -- aggregate revised results into tables (CSV) and key numbers (JSON)."""
import os, sys, glob, json, pickle
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr
import core2 as C
import exp2 as X

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.environ.get("RC_TABLES", "tables2"))
os.makedirs(OUT, exist_ok=True)
GROUPS = ["Mut-ER", "Mut-SF", "Webs-proj", "Webs-bip", "Mut-SF-het", "Reg-ER", "Reg-SF", "Reg-dir", "Reg-SF-het",
          "RegB-ER", "RegB-SF"]
TIE = 1e-3   # relative tie tolerance for paired comparisons


def load():
    R = {}
    for f in sorted(glob.glob(os.path.join(HERE, os.environ.get("RC_RESULTS", "results2"), "*.pkl"))):
        r = pickle.load(open(f, "rb"))
        if r.get("fold_ok") and "error" not in r:
            R[r["label"]] = r
    return R


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0
    for rank, i in enumerate(o):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def boot_ci(x, stat=np.median, B=1000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = stat(x[rng.integers(0, len(x), size=(B, len(x)))], axis=1)
    return tuple(np.percentile(bs, [2.5, 97.5]))


def paired(a, b):
    """a, b: composite outcomes (larger = better). Returns wins/ties/losses of a over b, median ratio (kept pairs),
    bootstrap CI, Wilcoxon p (on log-ratio for kept pairs; on differences of comp otherwise)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rel = (a - b) / np.maximum(np.abs(b), 1e-12)
    w = int(np.sum(rel > TIE)); l = int(np.sum(rel < -TIE)); t = len(a) - w - l
    kept = (a < 1) & (b < 1) & (a > 0) & (b > 0)
    ratio = a[kept] / b[kept]
    med = float(np.median(ratio)) if kept.any() else np.nan
    ci = boot_ci(ratio)
    d = a - b
    try:
        p = float(wilcoxon(d[np.abs(d) > 0]).pvalue) if np.sum(np.abs(d) > 0) >= 5 else np.nan
    except ValueError:
        p = np.nan
    return dict(n=len(a), wins=w, ties=t, losses=l, median_ratio=med, ci_lo=ci[0], ci_hi=ci[1], p=p,
                n_kept=int(kept.sum()))


def main():
    R = load()
    KN = {}
    rows = []
    # ---------------- table 1: certification
    for lab, r in R.items():
        c = r["collapse"]
        tc = r["tol_conv"]
        tcerr = max(abs(v[1] - c["beta"]) / c["beta"] for v in tc.values() if np.isfinite(v[1]))
        ncerr = max(abs(v[0] - c["beta"]) / c["beta"] for k, v in tc.items() if k[0] == 1e-4)
        row = dict(label=lab, group=r["group"], N=r["N"], E=r["E"], beta_c=c["beta"], res_f=c["res_f"], res_Jv=c["res_Jv"],
                   res_uJ=c["res_uJ"], a=c["a"], b=c["b"], gap=c["gap"], kappa=c["kappa_eig"], PR_u=c["PR_u"],
                   PR_v=c["PR_v"], tol_spread=tcerr, nc_err_1e4=ncerr, natcont_err=abs(c["beta_nc"] - c["beta"]) / c["beta"],
                   frac_drop=r["post_tip"]["frac_drop"], mean_ratio_post=r["post_tip"]["mean_ratio"],
                   u_min=c["u_min"], T_primitive=r["T_primitive"], max_link_dbeta=r["max_link_dbeta"],
                   rec_ok=r.get("rec_ok", np.nan), loss_flux_median=r["loss_flux_median"])
        if r.get("rec_ok"):
            rc = r["recovery"]
            row.update(beta_r=rc["beta"], hyst=rc["beta"] / c["beta"], rec_res_f=rc["res_f"], rec_a=rc["a"], rec_gap=rc["gap"],
                       rec_kappa=rc["kappa_eig"], rec_PR=rc["PR_u"])
        # reductions (recomputed GBB with the robust 1-D solver)
        m = rebuild_model(r)
        kap = np.diag(m.jac(r["arrays"]["x"], c["beta"]))
        row["kappa_diag_max"] = float(kap.max())
        bg = X.gbb_threshold(m)
        row.update(err_gbb=(bg - c["beta"]) / c["beta"], err_spec=(X.spectral_threshold(m) - c["beta"]) / c["beta"],
                   err_closure=(r["beta_closure"] - c["beta"]) / c["beta"], frac_neg_gbb=r["frac_neg_gbb"])
        if r.get("rec_ok") and np.isfinite(r.get("beta_closure_rec", np.nan)):
            row["err_closure_rec"] = (r["beta_closure_rec"] - r["recovery"]["beta"]) / r["recovery"]["beta"]
        for nm, v in r["corr"].items():
            row["rho_" + nm] = v["rho"]
            row["ov10_" + nm] = v["ov10"]
            row["eff1_" + nm] = v["eff1"]
        for k, v in r["rc_lazy_rho"].items():
            row[f"rho_lazy{k}"] = v
        rows.append(row)
    T1 = pd.DataFrame(rows)
    T1.to_csv(os.path.join(OUT, "per_network.csv"), index=False)

    def gsum(df, cols, fn=("mean", "std")):
        return df.groupby("group")[cols].agg(list(fn)).reindex([g for g in GROUPS if g in df.group.unique()])

    cert = T1.groupby("group").agg(n=("label", "size"), N=("N", "mean"), beta_c=("beta_c", "mean"),
                                   res_f_max=("res_f", "max"), res_Jv_max=("res_Jv", "max"), res_uJ_max=("res_uJ", "max"),
                                   a_med=("a", "median"), a_maxabs_min=("a", lambda x: np.min(np.abs(x))), b_min=("b", "min"),
                                   gap_min=("gap", "min"), kappa_med=("kappa", "median"), kappa_max=("kappa", "max"),
                                   PR_u=("PR_u", "mean"), tol_spread_max=("tol_spread", "max"),
                                   natcont_err_med=("natcont_err", "median"), frac_drop_min=("frac_drop", "min"),
                                   max_link_dbeta=("max_link_dbeta", "max"), u_min=("u_min", "min"),
                                   primitive=("T_primitive", "mean"), rec_n=("rec_ok", "sum"), hyst=("hyst", "mean"),
                                   rec_gap_min=("rec_gap", "min"), rec_PR=("rec_PR", "mean")).reindex(GROUPS).dropna(how="all")
    cert.to_csv(os.path.join(OUT, "table_certification.csv"))
    KN["n_networks"] = int(len(T1))
    KN["res_f_max"] = float(T1.res_f.max()); KN["res_Jv_max"] = float(T1.res_Jv.max()); KN["res_uJ_max"] = float(T1.res_uJ.max())
    KN["gap_min"] = float(T1.gap.min()); KN["kappa_max"] = float(T1.kappa.max()); KN["kappa_med"] = float(T1.kappa.median())
    KN["tol_spread_max"] = float(T1.tol_spread.max()); KN["natcont_err_max"] = float(T1.natcont_err.max())
    KN["frac_drop_min"] = float(T1.frac_drop.min()); KN["max_link_dbeta"] = float(T1.max_link_dbeta.max())
    KN["abs_a_min"] = float(np.abs(T1.a).min()); KN["b_min"] = float(T1.b.min())
    red_cols = ["err_gbb", "err_spec", "err_closure", "err_closure_rec", "frac_neg_gbb"] + \
               [c for c in T1.columns if c.startswith("rho_") or c.startswith("eff1_") or c.startswith("ov10_")]
    red = T1.groupby("group")[red_cols].agg(["mean", "std", "min", "max"]).reindex(GROUPS).dropna(how="all")
    red.to_csv(os.path.join(OUT, "table_reductions.csv"))
    # ---------------- finite differences
    fd = []
    for lab, r in R.items():
        for row in r["fd"]:
            hs = [k for k in row if isinstance(k, float)]
            for h in hs:
                fd.append(dict(label=lab, group=r["group"], dyn=r["dyn"], kind=row["kind"], strat=row["strat"], h=h,
                               pred=row["pred"], fd=row[h], relerr=abs(row[h] - row["pred"]) / max(abs(row["pred"]), 1e-300),
                               fwd_relerr=abs(row.get("fwd", np.nan) - row["pred"]) / max(abs(row["pred"]), 1e-300)))
    FD = pd.DataFrame(fd)
    FD.to_csv(os.path.join(OUT, "fd_all.csv"), index=False)
    fds = FD.groupby(["dyn", "kind", "h"]).relerr.agg(["size", "median", lambda x: np.percentile(x, 95), "max"])
    fds.columns = ["n", "median", "p95", "max"]
    fds.to_csv(os.path.join(OUT, "table_fd.csv"))
    best = FD.groupby(["label", "kind", "strat", "pred"]).relerr.min().reset_index()
    KN["fd_n"] = int(len(best)); KN["fd_best_median"] = float(best.relerr.median()); KN["fd_best_max"] = float(best.relerr.max())
    KN["fd_best_p99"] = float(np.percentile(best.relerr, 99))
    KN["fd_by_kind"] = {k: dict(n=int(len(g)), median=float(g.relerr.median()), max=float(g.relerr.max())) for k, g in best.groupby("kind")}
    fwd = FD[FD.kind == "node"].dropna(subset=["fwd_relerr"]).drop_duplicates(["label", "pred"])
    KN["fd_forward_median"] = float(fwd.fwd_relerr.median())
    # ---------------- node protection grid
    nrows = []
    for lab, r in R.items():
        for (q, d), res in r["node"].items():
            rnd = np.array([c for s, c in res["random"]])
            nlost_r = np.mean([s == "lost" for s, c in res["random"]])
            for nm, o in res.items():
                if nm == "random":
                    continue
                nrows.append(dict(label=lab, group=r["group"], dyn=r["dyn"], q=q, d=d, strat=nm, status=o["status"],
                                  comp=o["comp"], shift=o["shift"], s_star=o["s_star"],
                                  pct_random=float(np.mean(rnd < o["comp"]) + 0.5 * np.mean(rnd == o["comp"])),
                                  rnd_med=float(np.median(rnd)), rnd_p95=float(np.percentile(rnd, 95)), rnd_lost=nlost_r))
    NP = pd.DataFrame(nrows)
    NP.to_csv(os.path.join(OUT, "node_grid_all.csv"), index=False)
    # paired comparisons at each (group, q, d)
    comps = []
    for (g, q, d), sub in NP.groupby(["group", "q", "d"]):
        piv = sub.pivot_table(index="label", columns="strat", values="comp")
        for other in ("GBB_gradient", "spectral_gradient", "closure_exact", "RC1", "slow_mode", "adaptive_adjoint", "betweenness", "kcore"):
            if other in piv:
                pr = paired(piv["adjoint"].values, piv[other].values)
                pr.update(group=g, q=q, d=d, other=other)
                comps.append(pr)
        pr = dict(group=g, q=q, d=d, other="random_median")
        rnd = sub[sub.strat == "adjoint"].set_index("label")
        pr.update(paired(rnd["comp"].values, rnd["rnd_med"].values))
        pr["pct_random_mean"] = float(rnd.pct_random.mean())
        comps.append(pr)
    CP = pd.DataFrame(comps)
    CP["p_holm"] = np.nan
    for _, sub in CP.groupby(["other", "q", "d"]):
        m_ = sub.p.notna()
        if m_.any():
            CP.loc[sub.index[m_], "p_holm"] = holm(sub.p[m_].values)
    CP.to_csv(os.path.join(OUT, "table_node_paired.csv"), index=False)
    # outcome-state table
    ost = NP.groupby(["group", "q", "d", "strat"]).apply(lambda s: pd.Series(dict(n=len(s), lost=int((s.status == "lost").sum()),
                                                                                  rnd_lost=float(s.rnd_lost.mean())))).reset_index()
    ost.to_csv(os.path.join(OUT, "table_outcome_states.csv"), index=False)
    # crossover dose per network (5% budget): smallest grid dose at which out-strength >= adjoint
    cr = []
    for lab, sub in NP[(NP.q == 0.05) & NP.strat.isin(["adjoint", "GBB_gradient", "adaptive_adjoint"])].groupby("label"):
        piv = sub.pivot_table(index="d", columns="strat", values="comp").sort_index()
        dd = piv.index.values
        ratio = piv["GBB_gradient"] / piv["adjoint"]
        over = dd[ratio.values >= 1.0]
        cr.append(dict(label=lab, group=sub.group.iloc[0], dstar=over.min() if len(over) else np.inf,
                       **{f"ratio_{x}": v for x, v in zip(dd, ratio.values)},
                       **{f"adapt_{x}": v for x, v in zip(dd, (piv["adaptive_adjoint"] / piv["adjoint"]).values)}))
    CR = pd.DataFrame(cr)
    CR.to_csv(os.path.join(OUT, "table_crossover.csv"), index=False)
    # first-order accuracy of the adjoint prediction (5% budget) vs dose
    fo = []
    for lab, r in R.items():
        Rv = r["arrays"]["R"]
        k = X.kof(0.05, r["N"])
        top = C.topk(Rv, k)
        for d in X.DOSES[r["dyn"]]:
            o = r["node"][(0.05, d)]["adjoint"]
            pred = d * Rv[top].sum() / r["collapse"]["beta"]
            fo.append(dict(label=lab, group=r["group"], d=d, pred=pred, real=o["shift"], ratio=pred / o["shift"] if o["status"] == "kept" else np.nan,
                           dtilde=d / r["loss_flux_median"]))
    FO = pd.DataFrame(fo)
    FO.to_csv(os.path.join(OUT, "table_first_order.csv"), index=False)
    # ---------------- greedy / local search
    gl = []
    for lab, r in R.items():
        if "greedy" not in r:
            continue
        d0 = X.D0[r["dyn"]]
        res = r["node"][(0.05, d0)]
        ls_best = max(r["local_search"]["from_greedy"][1], r["local_search"]["from_adjoint"][1], r["greedy"]["seq"][-1])
        gl.append(dict(label=lab, group=r["group"], greedy=r["greedy"]["seq"][-1], ls_best=ls_best,
                       ls_gain=ls_best / r["greedy"]["seq"][-1] - 1, adjoint=res["adjoint"]["comp"],
                       adaptive=res["adaptive_adjoint"]["comp"], gbb=res["GBB_gradient"]["comp"],
                       spectral=res["spectral_gradient"]["comp"], screen=r["screen_rerank"]["comp"],
                       rnd_med=float(np.median([c for s, c in res["random"]])),
                       any_lost=any(o["status"] == "lost" for k_, o in res.items() if k_ != "random"),
                       best_lost=ls_best >= 1.0))
    GL = pd.DataFrame(gl)
    for c in ("adjoint", "adaptive", "gbb", "spectral", "screen", "rnd_med", "greedy"):
        GL["eff_" + c] = GL[c] / GL["ls_best"]
    GL.to_csv(os.path.join(OUT, "table_greedy.csv"), index=False)
    # ---------------- concentration, knapsack, lever
    cc = []
    for lab, r in R.items():
        con = r["concentration"]
        row = dict(label=lab, group=r["group"], alloc_nodes=con["alloc_nodes"], adaptive_alloc=con[("adaptive_alloc", 0)]["comp"])
        for key, o in con.items():
            if isinstance(key, tuple) and key[0] != "adaptive_alloc":
                row[f"{key[0]}_{key[1]}"] = o["comp"]
        kn = r["knapsack"]
        row.update(kn_ratio=kn["ratio"]["comp"], kn_adjoint=kn["adjoint"]["comp"], kn_gbb=kn["GBB_gradient"]["comp"],
                   kn_cheap=kn["cheapest"]["comp"], kn_rnd=float(np.median(kn["random"])))
        lv = r["lever"]
        row.update(lev_lever=lv["lever_adjoint"]["comp"], lev_input=lv["input_adjoint"]["comp"], lev_gbb=lv["GBB_gradient"]["comp"],
                   lev_rnd=float(np.median(lv["random"])), rho_Rmu_R=lv["rho_Rmu_R"])
        cc.append(row)
    CC = pd.DataFrame(cc)
    CC.to_csv(os.path.join(OUT, "table_concentration_knapsack_lever.csv"), index=False)
    # ---------------- links
    lr = []
    for lab, r in R.items():
        L = r["links"]
        for key, res in L.items():
            if not (isinstance(key, tuple) and key[0] in ("count", "weight")):
                continue
            dm = {nm: o["damage"] if o["status"] == "kept" else np.nan for nm, o in res.items() if nm != "random"}
            bestv = np.nanmax(list(dm.values()))
            rnd = np.array(res.get("random", [np.nan]), float)
            for nm, v in dm.items():
                lr.append(dict(label=lab, group=r["group"], budget=key[0], q=key[1], strat=nm, damage=v, rel_best=v / bestv,
                               lcc=res[nm].get("lcc", np.nan), rnd_med=np.nanmedian(rnd) if rnd.size else np.nan,
                               pct_random=float(np.nanmean(rnd < v)) if rnd.size else np.nan,
                               lost=res[nm]["status"] == "lost"))
    LR = pd.DataFrame(lr)
    LR.to_csv(os.path.join(OUT, "links_all.csv"), index=False)
    lsum = LR.groupby(["group", "budget", "q", "strat"]).agg(damage=("damage", "mean"), rel_best=("rel_best", "mean"),
                                                             rel_best_min=("rel_best", "min"), pct_random=("pct_random", "mean"),
                                                             lcc=("lcc", "mean")).reset_index()
    lsum.to_csv(os.path.join(OUT, "table_links.csv"), index=False)
    # exact single-link check and greedy
    sl = []
    for lab, r in R.items():
        L = r["links"]
        if "single_all" in L:
            dm, pr = L["single_all"]["damage"], L["single_all"]["pred"]
            ok = np.isfinite(dm)
            sl.append(dict(label=lab, group=r["group"], rho=spearmanr(dm[ok], pr[ok])[0], best_is_top=int(np.nanargmax(dm) == np.argmax(pr)),
                           best_rel=float(dm[np.argmax(pr)] / np.nanmax(dm))))
        if "greedy1" in L:
            ad = L[("count", 0.01)]["adjoint"]
            sl[-1 if sl and sl[-1]["label"] == lab else len(sl):] if False else None
    SL = pd.DataFrame(sl)
    SL.to_csv(os.path.join(OUT, "table_single_link.csv"), index=False)
    gr = []
    for lab, r in R.items():
        L = r["links"]
        if "greedy1" in L and L["greedy1"]["seq"]:
            g_ = L["greedy1"]["seq"][-1]
            a_ = L[("count", 0.01)]["adjoint"]["damage"]
            gr.append(dict(label=lab, group=r["group"], greedy=g_, adjoint=a_, eff=a_ / g_))
    pd.DataFrame(gr).to_csv(os.path.join(OUT, "table_link_greedy.csv"), index=False)
    # sign test and strengthening
    st = []
    for lab, r in R.items():
        for gpred, dmg, lin in r["sign_test"]:
            st.append(dict(label=lab, group=r["group"], gbb_damage=gpred, full_damage=dmg, first_order=lin))
    ST = pd.DataFrame(st)
    ST.to_csv(os.path.join(OUT, "table_sign_test.csv"), index=False)
    KN["sign_n"] = int(len(ST)); KN["sign_all_positive"] = bool((ST.full_damage > 0).all()) if len(ST) else None
    KN["sign_n_nonpositive"] = int((ST.full_damage <= 0).sum()) if len(ST) else 0
    sg = []
    for lab, r in R.items():
        s_ = r["links"]["strengthen"]
        row = dict(label=lab, group=r["group"])
        for nm, o in s_.items():
            row[nm] = (-np.median(o) if isinstance(o, list) else (-o["damage"] if o["status"] == "kept" else np.nan))
        sg.append(row)
    SG = pd.DataFrame(sg)
    SG.to_csv(os.path.join(OUT, "table_strengthen.csv"), index=False)
    # ---------------- restoration
    rs = []
    for lab, r in R.items():
        if not r.get("rec_ok"):
            continue
        for (q, d), res in r["restoration"].items():
            rnd = np.array(res["random"])
            for nm, o in res.items():
                if nm in ("random", "_collapse"):
                    continue
                rs.append(dict(label=lab, group=r["group"], q=q, d=d, strat=nm, comp=o["comp"], status=o["status"],
                               collapse_comp=res["_collapse"][nm], rnd_med=float(np.median(rnd)),
                               pct_random=float(np.mean(rnd < o["comp"]))))
    RS = pd.DataFrame(rs)
    RS.to_csv(os.path.join(OUT, "restoration_all.csv"), index=False)
    rp = []
    for (g, q, d), sub in RS.groupby(["group", "q", "d"]):
        piv = sub.pivot_table(index="label", columns="strat", values="comp")
        for other in ("collapse_adjoint", "GBB_gradient", "spectral_gradient", "RC1_low", "closure_low"):
            if other in piv:
                pr = paired(piv["restoration_adjoint"].values, piv[other].values)
                pr.update(group=g, q=q, d=d, other=other)
                rp.append(pr)
    RP = pd.DataFrame(rp)
    if len(RP):
        RP["p_holm"] = np.nan
        for _, sub in RP.groupby(["other", "q", "d"]):
            m_ = sub.p.notna()
            if m_.any():
                RP.loc[sub.index[m_], "p_holm"] = holm(sub.p[m_].values)
    RP.to_csv(os.path.join(OUT, "table_restoration_paired.csv"), index=False)
    rc = []
    for lab, r in R.items():
        if r.get("rec_ok"):
            rc.append(dict(label=lab, group=r["group"], **{"rho_" + k: v for k, v in r["rest_corr"].items()},
                           rho_u_ulow=spearmanr(r["arrays"]["R"], r["arrays"]["R_rest"])[0],
                           PR_low=r["recovery"]["PR_u"], hyst=r["recovery"]["beta"] / r["collapse"]["beta"]))
    RC = pd.DataFrame(rc)
    RC.to_csv(os.path.join(OUT, "table_restoration_corr.csv"), index=False)
    td = []
    for lab, r in R.items():
        for row in r.get("time_domain", []):
            td.append(dict(label=lab, group=r["group"], **row))
    TD = pd.DataFrame(td)
    TD.to_csv(os.path.join(OUT, "time_domain_all.csv"), index=False)
    if len(TD):
        KN["td_n"] = int(len(TD)); KN["td_agree"] = float((TD.predicted == TD.recovered).mean())
        KN["td_n_recovered"] = int(TD.recovered.sum()); KN["td_persist"] = float(TD.persist.dropna().astype(float).mean())
        KN["td_disagree"] = int((TD.predicted != TD.recovered).sum())
    # ---------------- misspecification
    ms = []
    for lab, r in R.items():
        for row in r.get("misspec", []):
            ms.append(dict(label=lab, group=r["group"], **row))
    MS = pd.DataFrame(ms)
    MS.to_csv(os.path.join(OUT, "misspec_all.csv"), index=False)
    KN["kappa_diag_max"] = float(T1.kappa_diag_max.max())
    json.dump(KN, open(os.path.join(OUT, "key_numbers_partial.json"), "w"), indent=1, default=float)
    return dict(T1=T1, FD=FD, NP=NP, CP=CP, CR=CR, FO=FO, GL=GL, CC=CC, LR=LR, SL=SL, ST=ST, SG=SG, RS=RS, RP=RP, RC=RC,
                TD=TD, MS=MS, KN=KN, R=R)


_TASKS = None


def rebuild_model(r):
    global _TASKS
    if _TASKS is None:
        _TASKS = {t["label"]: t for t in X.build_tasks()}
    return X.make_model(_TASKS[r["label"]])


if __name__ == "__main__":
    D = main()
    print(json.dumps(D["KN"], indent=1, default=float))
