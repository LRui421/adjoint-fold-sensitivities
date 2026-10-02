"""numbers2 -- every number quoted in the manuscript, computed from the aggregated results (D from analysis2.main(),
EXH, SWd, ZH, CCn, examples.json).  Produces TOK: token -> formatted string."""
import json
import numpy as np
import pandas as pd


def f2(x, nd=2):
    t = f"{x:.{nd}f}"
    return f"${t}$" if t.startswith("-") else t


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def sci(x, nd=1):
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"${m:.{nd}f}\\times10^{{{e}}}$"


def rng(a, b, nd=2, fn=f2):
    sa, sb = fn(a, nd), fn(b, nd)
    return f"{sa} to {sb}" if ("-" in sa or "-" in sb) else f"{sa}--{sb}"


def build(D, EXH, SWd, ZH, CCn, EXJ):
    T1, CP, FD, NP, GL, LR, ST, SG, RS, RP, RC, TD, MS, CC = (D[k] for k in
        ["T1", "CP", "FD", "NP", "GL", "LR", "ST", "SG", "RS", "RP", "RC", "TD", "MS", "CC"])
    TOK = {}
    g = T1.groupby("group")
    TOK["n_networks"] = str(len(T1))
    TOK["n_groups"] = str(T1.group.nunique())
    # certification
    TOK["res_f_max"] = sci(T1.res_f.max()); TOK["res_Jv_max"] = sci(T1.res_Jv.max()); TOK["res_uJ_max"] = sci(T1.res_uJ.max())
    TOK["abs_a_min"] = sci(np.abs(T1.a).min()); TOK["b_min"] = f2(T1.b.min(), 1)
    TOK["gap_min"] = f2(T1.gap.min()); TOK["kappa_med"] = f2(T1.kappa.median()); TOK["kappa_max"] = f2(T1.kappa.max())
    TOK["kappa_nondir_max"] = f2(T1[T1.group != "Reg-dir"].kappa.max())
    TOK["kappa_dir_range"] = rng(T1[T1.group == "Reg-dir"].kappa.min(), T1[T1.group == "Reg-dir"].kappa.max())
    TOK["natcont_err_max"] = sci(T1.natcont_err.max()); TOK["natcont_err_med"] = sci(T1.natcont_err.median())
    tc = []
    for r in D["R"].values():
        b = r["collapse"]["beta"]
        for (rt, tol), (bnc, bms, it) in r["tol_conv"].items():
            tc.append(dict(rt=rt, tol=tol, err=abs(bms - b) / b))
    TC = pd.DataFrame(tc)
    TOK["tol_fine_max"] = sci(TC[(TC.rt <= 1e-3) & (TC.tol <= 1e-9)].err.max())
    TOK["tol_coarse_nbad"] = str(int((TC[(TC.rt == 1e-2) & (TC.tol == 1e-12)].err > 1e-6).sum()))
    TOK["kappa_diag_max"] = f2(T1.kappa_diag_max.max())
    TOK["post_mut_range"] = rng(T1[T1.group.str.startswith(("Mut", "Webs"))].mean_ratio_post.min() * 100,
                                T1[T1.group.str.startswith(("Mut", "Webs"))].mean_ratio_post.max() * 100, 1)
    TOK["post_regB_range"] = rng(T1[T1.group.str.startswith("RegB")].mean_ratio_post.min() * 100,
                                 T1[T1.group.str.startswith("RegB")].mean_ratio_post.max() * 100, 1)
    TOK["PR_u_range"] = rng(g.PR_u.mean().min(), g.PR_u.mean().max())
    TOK["n_bip_primitive"] = str(int(T1[T1.group == "Webs-bip"].T_primitive.sum()))
    TOK["n_nonbip_primitive"] = str(int(T1[T1.group != "Webs-bip"].T_primitive.sum()))
    TOK["n_nonbip"] = str(int((T1.group != "Webs-bip").sum()))
    TOK["n_rec"] = str(int(T1.rec_ok.fillna(0).sum()))
    TOK["hyst_range"] = rng(g.hyst.mean().min(), g.hyst.mean().max())
    TOK["rec_gap_min"] = f2(T1.rec_gap.min())
    # verification of first fold
    V = pd.DataFrame([(r["label"], nm, d, bool(v["match"])) for r in D["R"].values() for (nm, d, v) in r["verify"]],
                     columns=["label", "strat", "d", "match"])
    TOK["verify_n"] = str(len(V)); TOK["verify_match"] = str(int(V.match.sum())); TOK["verify_bad"] = str(int((~V.match).sum()))
    # finite differences
    best = FD.sort_values("relerr").drop_duplicates(["label", "kind", "strat", "pred"])
    big = best[np.abs(best.pred) > 1e-10]
    TOK["fd_n"] = str(len(best)); TOK["fd_n_big"] = str(len(big))
    TOK["fd_med"] = sci(big.relerr.median()); TOK["fd_max"] = sci(big.relerr.max()); TOK["fd_p99"] = sci(np.percentile(big.relerr, 99))
    for k in ("node", "lever", "link"):
        TOK[f"fd_n_{k}"] = str(int((big.kind == k).sum()))
    TOK["fd_n_tiny"] = str(int((np.abs(best.pred) <= 1e-10).sum()))
    fwd = FD[(FD.kind == "node")].dropna(subset=["fwd_relerr"]).drop_duplicates(["label", "pred"])
    TOK["fd_fwd_med"] = sci(fwd.fwd_relerr.median())
    TOK["max_link_dbeta"] = sci(T1.max_link_dbeta.max())
    # reductions
    m = g[["err_gbb", "err_spec", "err_closure", "frac_neg_gbb", "rho_GBB_gradient", "rho_spectral_gradient", "rho_closure_exact",
           "rho_RC_red", "rho_RC1", "rho_RC2", "rho_slow_mode", "eff1_GBB_gradient", "eff1_closure_exact", "eff1_RC1",
           "ov10_GBB_gradient", "ov10_RC1", "err_closure_rec"]].mean()
    s = g[["rho_GBB_gradient", "err_gbb", "err_closure"]].std()
    for gg in m.index:
        key = gg.replace("-", "")
        TOK[f"gbb_{key}"] = f2(100 * m.loc[gg, "err_gbb"], 1)
        TOK[f"spec_{key}"] = f2(100 * m.loc[gg, "err_spec"], 1)
        TOK[f"cl_{key}"] = f2(100 * m.loc[gg, "err_closure"], 1)
        TOK[f"rhoout_{key}"] = f2(m.loc[gg, "rho_GBB_gradient"])
        TOK[f"rhoout_sd_{key}"] = f2(s.loc[gg, "rho_GBB_gradient"])
        TOK[f"rhoeig_{key}"] = f2(m.loc[gg, "rho_spectral_gradient"])
        TOK[f"rhocl_{key}"] = f2(m.loc[gg, "rho_closure_exact"])
        TOK[f"rhopl_{key}"] = f2(m.loc[gg, "rho_RC_red"])
        TOK[f"rhopi1_{key}"] = f2(m.loc[gg, "rho_RC1"])
        TOK[f"rhopi2_{key}"] = f2(m.loc[gg, "rho_RC2"])
        TOK[f"eff1out_{key}"] = f2(m.loc[gg, "eff1_GBB_gradient"])
        TOK[f"eff1cl_{key}"] = f2(m.loc[gg, "eff1_closure_exact"])
        TOK[f"neg_{key}"] = f2(100 * m.loc[gg, "frac_neg_gbb"], 1)
    nonbip = m.drop(index=["Webs-bip"], errors="ignore")
    TOK["cl_abs_max_nonbip"] = f2(100 * np.abs(nonbip.err_closure).max(), 1)
    TOK["gbb_range"] = rng(100 * m.err_gbb.max(), 100 * m.err_gbb.min(), 1)
    TOK["pi1_range"] = rng(m.rho_RC1.min(), m.rho_RC1.max()); TOK["pi2_range"] = rng(m.rho_RC2.min(), m.rho_RC2.max())
    TOK["eff1pi1_range"] = rng(m.eff1_RC1.min(), m.eff1_RC1.max())
    TOK["slow_range"] = rng(m.rho_slow_mode.min(), m.rho_slow_mode.max())
    sl = {q: g[f"rho_slow_left_{q}"].mean() for q in (1.001, 1.01, 1.3, 2.0)}
    TOK["slow1001_range"] = rng(sl[1.001].min(), sl[1.001].max()); TOK["slow13_range"] = rng(sl[1.3].min(), sl[1.3].max())
    TOK["slowR13_range"] = rng(g["rho_slow_right_1.3"].mean().min(), g["rho_slow_right_1.3"].mean().max())
    TOK["neg_max_web"] = f2(100 * T1[T1.group.str.startswith("Webs")].frac_neg_gbb.max(), 0)
    # h factor correlation (computed from arrays)
    hh = []
    for r in D["R"].values():
        a = r["arrays"]
        ok = a["s_out"] > 0
        h = a["RC1"][ok] / a["s_out"][ok]
        ls, lh = np.log(a["s_out"][ok]), np.log(h)
        slope = np.polyfit(ls, lh, 1)[0] if np.ptp(ls) > 0 else np.nan
        from scipy.stats import spearmanr as _sp
        hh.append(dict(g=r["group"], rho_sh=_sp(a["s_out"][ok], h)[0], rho_su1=_sp(a["s_out"][ok], a["RC1"][ok])[0],
                       rho_su=_sp(a["s_out"], a["R"])[0], slope=slope))
    HH = pd.DataFrame(hh).dropna()
    TOK["h_corr"] = f2(np.corrcoef(HH.slope, HH.rho_su)[0, 1])
    TOK["h_slope_agree"] = pct(np.mean((HH.slope > -1) == (HH.rho_su1 > 0)), 0)
    TOK["h_n"] = str(len(HH))
    TOK["h_neg_but_pos"] = str(int(((HH.rho_sh < 0) & (HH.rho_su > 0)).sum()))
    # Zhang
    zg = ZH.groupby("group")
    TOK["zh_rho_range"] = rng(zg.rho.mean().min(), zg.rho.mean().max())
    TOK["zh_rsmall_range"] = rng(zg.r_small.median().min(), zg.r_small.median().max())
    TOK["zh_r5_range"] = rng(zg.r5.median().min(), zg.r5.median().max())
    TOK["zh_rho_sout_range"] = rng(zg.rho_sout.mean().min(), zg.rho_sout.mean().max())
    # node protection
    g5 = CP[(CP.q == 0.05) & (CP.other == "GBB_gradient")]
    dmin = {"mutualistic": 0.1, "regulatory": 0.01}
    small = g5[((g5.d == 0.1) & g5.group.str.startswith(("Mut", "Webs"))) | ((g5.d == 0.01) & g5.group.str.startswith("Reg"))]
    TOK["small_wins"] = str(int(small.wins.sum())); TOK["small_n"] = str(int(small.n.sum())); TOK["small_losses"] = str(int(small.losses.sum()))
    TOK["small_ratio_range"] = rng(small.median_ratio.min(), small.median_ratio.max())
    for gg in g5.group.unique():
        key = gg.replace("-", "")
        for d in sorted(g5[g5.group == gg].d.unique()):
            row = g5[(g5.group == gg) & (g5.d == d)].iloc[0]
            TOK[f"r_{key}_{d:g}"] = f2(row.median_ratio) if np.isfinite(row.median_ratio) else "n.d."
            TOK[f"ci_{key}_{d:g}"] = f"{f2(row.ci_lo)}--{f2(row.ci_hi)}" if np.isfinite(row.ci_lo) else "n.d."
            TOK[f"wl_{key}_{d:g}"] = f"{int(row.wins)}/{int(row.ties)}/{int(row.losses)}"
            TOK[f"p_{key}_{d:g}"] = sci(row.p_holm) if np.isfinite(row.p_holm) else "n.d."
    rnd = CP[(CP.q == 0.05) & (CP.other == "random_median")]
    dd = rnd[((rnd.d == 1.0) & rnd.group.str.startswith(("Mut", "Webs"))) | ((rnd.d == 0.1) & rnd.group.str.startswith("Reg"))]
    TOK["pctrand_range"] = rng(100 * dd.pct_random_mean.min(), 100 * dd.pct_random_mean.max(), 0)
    TOK["rand_ratio_range"] = rng(dd.median_ratio.min(), dd.median_ratio.max())
    ad = CP[(CP.q == 0.05) & (CP.other == "adaptive_adjoint")]
    TOK["adapt_small_min"] = f2(ad[ad.d.isin([0.1, 0.01])].median_ratio.min(), 3)
    TOK["adapt_large_range"] = rng(ad[ad.d.isin([3.0, 0.3])].median_ratio.min(), ad[ad.d.isin([3.0, 0.3])].median_ratio.max())
    FO = D["FO"]
    fo = FO.groupby(["group", "d"]).ratio.median()
    TOK["fo_mut_small"] = rng(fo[[k for k in fo.index if k[1] == 0.1 and k[0].startswith(("Mut", "Webs"))]].min(),
                              fo[[k for k in fo.index if k[1] == 0.1 and k[0].startswith(("Mut", "Webs"))]].max())
    TOK["fo_mut_d1"] = rng(fo[[k for k in fo.index if k[1] == 1.0]].min(), fo[[k for k in fo.index if k[1] == 1.0]].max())
    TOK["fo_mut_d3"] = rng(fo[[k for k in fo.index if k[1] == 3.0]].min(), fo[[k for k in fo.index if k[1] == 3.0]].max())
    TOK["fo_reg_range"] = rng(fo[[k for k in fo.index if k[0].startswith("Reg") and k[1] <= 0.1]].min(),
                              fo[[k for k in fo.index if k[0].startswith("Reg") and k[1] <= 0.1]].max())
    dt = FO.groupby(["group", "d"]).dtilde.median()
    TOK["dt_mut_d1"] = rng(dt[[k for k in dt.index if k[1] == 1.0]].min(), dt[[k for k in dt.index if k[1] == 1.0]].max())
    TOK["dt_reg_d01"] = rng(dt[[k for k in dt.index if k[1] == 0.1 and k[0].startswith("Reg")]].min(),
                            dt[[k for k in dt.index if k[1] == 0.1 and k[0].startswith("Reg")]].max())
    CR = D["CR"]
    for gg in ["Mut-ER", "Mut-SF", "Webs-proj", "Webs-bip", "Reg-SF", "Reg-ER", "Reg-dir", "RegB-SF"]:
        c = CR[CR.group == gg].dstar
        TOK[f"dstar_{gg.replace('-', '')}"] = f"{np.median(c):g}" if np.isfinite(np.median(c)) else "none"
        TOK[f"dstar_n_{gg.replace('-', '')}"] = str(int(np.isfinite(c).sum()))
    # outcome states
    ost = NP[NP.q == 0.10].groupby(["dyn", "d", "strat"]).apply(lambda s_: (s_.status == "lost").mean())
    TOK["elim_out_mut3"] = pct(ost[("mutualistic", 3.0, "GBB_gradient")], 0)
    TOK["elim_adj_mut3"] = pct(ost[("mutualistic", 3.0, "adjoint")], 0)
    TOK["elim_eig_mut3"] = pct(ost[("mutualistic", 3.0, "spectral_gradient")], 0)
    TOK["elim_adapt_mut3"] = pct(ost[("mutualistic", 3.0, "adaptive_adjoint")], 0)
    TOK["elim_any_mut1"] = pct(NP[(NP.dyn == "mutualistic") & (NP.d <= 1.0)].status.eq("lost").mean(), 1)
    TOK["elim_any_reg"] = pct(NP[(NP.dyn == "regulatory") & (NP.d <= 0.1)].status.eq("lost").mean(), 1)
    TOK["elim_rnd_mut3"] = pct(NP[(NP.q == 0.10) & (NP.d == 3.0)].rnd_lost.mean(), 0)
    # exhaustive
    ex = EXH.groupby(["dyn", "d", "strat"]).eff.median()
    for dyn, ds in (("mutualistic", (0.3, 1.0, 3.0)), ("regulatory", (0.03, 0.1, 0.3))):
        for d in ds:
            for s_ in ("adjoint", "adaptive_adjoint", "GBB_gradient", "spectral_gradient", "RC1", "greedy"):
                TOK[f"ex_{dyn[:3]}_{d:g}_{s_}"] = f2(ex[(dyn, d, s_)], 3 if s_ == "adjoint" else 2)
    TOK["ex_n_nets"] = str(EXH.label.nunique()); TOK["ex_n_sets"] = str(int(EXH.drop_duplicates(["label", "d", "k"]).n_sets.sum()))
    TOK["ex_greedy_min"] = f2(EXH[EXH.strat == "greedy"].eff.clip(lower=0).min())
    TOK["ex_greedy_frac_opt"] = pct((EXH[EXH.strat == "greedy"].eff > 1 - 1e-9).mean(), 0)
    TOK["ex_opt_lost"] = pct(EXH.drop_duplicates(["label", "d", "k"]).opt_lost.mean(), 1)
    # greedy / LS
    gl = GL.groupby("group")
    TOK["gl_n"] = str(len(GL)); TOK["ls_gain_max"] = sci(GL.ls_gain.max()) if GL.ls_gain.max() > 0 else "0"
    for gg in gl.groups:
        key = gg.replace("-", "")
        TOK[f"gl_adj_{key}"] = f2(gl.eff_adjoint.median()[gg]); TOK[f"gl_out_{key}"] = f2(gl.eff_gbb.median()[gg])
        TOK[f"gl_scr_{key}"] = f2(gl.eff_screen.median()[gg]); TOK[f"gl_rnd_{key}"] = f2(gl.eff_rnd_med.median()[gg])
    TOK["gl_scr_min"] = f2(GL.eff_screen.min())
    # levers, knapsack, concentration
    C2 = CC.copy()
    C2["li"] = C2.lev_input / C2.lev_lever; C2["lg"] = C2.lev_gbb / C2.lev_lever
    C2["ka"] = C2.kn_adjoint / C2.kn_ratio; C2["kg"] = C2.kn_gbb / C2.kn_ratio; C2["kc"] = C2.kn_cheap / C2.kn_ratio
    cg = C2.groupby("group").median(numeric_only=True)
    mutg = [x for x in cg.index if x.startswith(("Mut", "Webs"))]; regg = [x for x in cg.index if x.startswith("Reg")]
    TOK["lev_rho_range"] = rng(cg.rho_Rmu_R.min(), cg.rho_Rmu_R.max())
    TOK["lev_rho_MutER"] = f2(cg.rho_Rmu_R["Mut-ER"])
    TOK["lev_in_mut"] = rng(cg.li[mutg].min(), cg.li[mutg].max()); TOK["lev_in_reg"] = rng(cg.li[regg].min(), cg.li[regg].max())
    TOK["lev_out_range"] = rng(cg.lg.min(), cg.lg.max())
    TOK["kn_adj_range"] = rng(cg.ka.min(), cg.ka.max()); TOK["kn_out_range"] = rng(cg.kg.min(), cg.kg.max())
    TOK["kn_cheap_range"] = rng(cg.kc.min(), cg.kc.max())
    TOK["kn_ratio_wins"] = str(int((C2[["ka", "kg", "kc"]].max(axis=1) <= 1 + 1e-9).sum())); TOK["kn_n"] = str(len(C2))
    cn = CCn.groupby("group").median(numeric_only=True)
    TOK["alloc_mut"] = rng(cn.alloc[mutg].min(), cn.alloc[mutg].max()); TOK["alloc_reg"] = rng(cn.alloc[[x for x in regg if x != "Reg-dir"]].min(), cn.alloc[[x for x in regg if x != "Reg-dir"]].max())
    TOK["alloc_dir"] = f2(cn.alloc["Reg-dir"]); TOK["conc_dir_1"] = f2(cn.adjoint_1["Reg-dir"])
    TOK["spread_mut_4k"] = rng(cn.adjoint_4k[mutg].min(), cn.adjoint_4k[mutg].max())
    TOK["conc_mut_1"] = rng(cn.adjoint_1[mutg].min(), cn.adjoint_1[mutg].max())
    # links
    l5 = LR[(LR.budget == "count") & (LR.q == 0.05)].groupby(["group", "strat"]).rel_best.mean().unstack()
    l1 = LR[(LR.budget == "count") & (LR.q == 0.01)].groupby(["group", "strat"]).rel_best.mean().unstack()
    lw = LR[(LR.budget == "weight")].groupby(["group", "strat"]).rel_best.mean().unstack()
    nd = [x for x in l5.index if x != "Reg-dir"]
    TOK["l5_adj_nd"] = rng(l5.adjoint[nd].min(), l5.adjoint[nd].max()); TOK["l5_adj_dir"] = f2(l5.adjoint["Reg-dir"])
    TOK["l5_pi1_dir"] = f2(l5.RC1["Reg-dir"]); TOK["l1_adj"] = rng(l1.adjoint.min(), l1.adjoint.max())
    TOK["l5_sp_nd"] = rng(l5.strength_product[nd].min(), l5.strength_product[nd].max()); TOK["l5_sp_dir"] = f2(l5.strength_product["Reg-dir"], 3)
    TOK["l5_gbb_nd"] = rng(l5.GBB_gradient[nd].min(), l5.GBB_gradient[nd].max()); TOK["l5_gbb_dir"] = f2(l5.GBB_gradient["Reg-dir"])
    TOK["l5_spec_dir"] = f2(l5.spectral_edge["Reg-dir"]); TOK["l5_eb_range"] = rng(l5.edge_betweenness.min(), l5.edge_betweenness.max())
    TOK["l5_pl_range"] = rng(l5.RC_red.min(), l5.RC_red.max())
    TOK["lw_adj"] = rng(lw.adjoint[nd].min(), lw.adjoint[nd].max()); TOK["lw_adj_dir"] = f2(lw.adjoint["Reg-dir"])
    TOK["lw_weight_range"] = rng(lw.weight.min(), lw.weight.max())
    dm = LR[(LR.budget == "count") & (LR.q == 0.05) & (LR.strat == "adjoint")].groupby("group").agg(d=("damage", "mean"), r=("rnd_med", "mean"), p=("pct_random", "mean"), l=("lcc", "min"))
    TOK["l5_dmg_range"] = rng(100 * dm.d.min(), 100 * dm.d.max(), 1); TOK["l5_rnd_range"] = rng(100 * dm.r.min(), 100 * dm.r.max(), 1)
    TOK["l5_lcc_min"] = pct(dm.l.min(), 0)
    SL = D["SL"]
    TOK["sl_rho_min"] = f2(SL.groupby("group").rho.mean().min()); TOK["sl_n"] = str(len(SL))
    TOK["sl_top_frac"] = pct(SL.best_is_top.mean(), 0); TOK["sl_rel_min"] = f2(SL.best_rel.min())
    LG = pd.read_csv(D["_tables"] + "/table_link_greedy.csv")
    lgg = LG.groupby("group").eff.median()
    TOK["lg_nd"] = rng(lgg.drop(index="Reg-dir").min(), lgg.drop(index="Reg-dir").max()); TOK["lg_dir"] = f2(lgg["Reg-dir"]); TOK["lg_n"] = str(len(LG))
    TOK["sign_n"] = str(len(ST)); TOK["sign_pos"] = str(int((ST.full_damage > 1e-13).sum()))
    TOK["sign_min"] = sci(ST.full_damage.min()); TOK["sign_med_max"] = f2(100 * ST.groupby("group").full_damage.median().max(), 3)
    TOK["sign_max"] = f2(100 * ST.full_damage.max(), 2)
    sg = SG.groupby("group").mean(numeric_only=True)
    TOK["str_gbbneg_range"] = rng(100 * sg.GBB_most_negative.min(), 100 * sg.GBB_most_negative.max(), 2)
    TOK["str_adj_range"] = rng(100 * sg.adjoint.min(), 100 * sg.adjoint.max(), 1)
    TOK["str_nonneg"] = str(int((SG.GBB_most_negative > 0).sum())); TOK["str_n"] = str(int(SG.GBB_most_negative.notna().sum()))
    # restoration
    rc = RC.groupby("group").mean(numeric_only=True)
    TOK["rho_uul_range"] = rng(rc.rho_u_ulow.min(), rc.rho_u_ulow.max()); TOK["rho_uul_MutER"] = f2(rc.rho_u_ulow["Mut-ER"])
    TOK["rho_ulout_range"] = rng(rc.rho_GBB_gradient.min(), rc.rho_GBB_gradient.max())
    TOK["rho_uleig_range"] = rng(rc.rho_spectral_gradient.min(), rc.rho_spectral_gradient.max())
    TOK["PR_low_range"] = rng(rc.PR_low.min(), rc.PR_low.max())
    rp = RP[(RP.q == 0.05)]
    rpd = rp[((rp.d == 0.1) & rp.group.str.startswith(("Mut", "Webs"))) | ((rp.d == 0.03) & rp.group.str.startswith("RegB"))]
    TOK["rest_coll_range"] = rng(rpd[rpd.other == "collapse_adjoint"].median_ratio.min(), rpd[rpd.other == "collapse_adjoint"].median_ratio.max(), 1)
    TOK["rest_out_range"] = rng(rpd[rpd.other == "GBB_gradient"].median_ratio.min(), rpd[rpd.other == "GBB_gradient"].median_ratio.max())
    w_ = rpd[rpd.other == "GBB_gradient"]
    TOK["rest_out_wtl"] = f"{int(w_.wins.sum())}/{int(w_.ties.sum())}/{int(w_.losses.sum())}"
    w2 = rpd[rpd.other == "collapse_adjoint"]
    TOK["rest_coll_wtl"] = f"{int(w2.wins.sum())}/{int(w2.ties.sum())}/{int(w2.losses.sum())}"
    rs = RS[(RS.q == 0.05)]
    rsd = rs[((rs.d == 0.1) & rs.group.str.startswith(("Mut", "Webs"))) | ((rs.d == 0.03) & rs.group.str.startswith("RegB"))]
    a_ = rsd[rsd.strat == "restoration_adjoint"].groupby("group").comp.median()
    TOK["rest_adj_range"] = rng(100 * a_.min(), 100 * a_.max(), 1)
    c_ = rsd[rsd.strat == "collapse_adjoint"].groupby("group").comp.median()
    TOK["rest_colladj_range"] = rng(100 * c_.min(), 100 * c_.max(), 1)
    r_ = rsd[rsd.strat == "restoration_adjoint"].groupby("group").rnd_med.median()
    TOK["rest_rnd_range"] = rng(100 * r_.min(), 100 * r_.max(), 1)
    cc_ = rsd.groupby(["group", "strat"]).collapse_comp.median().unstack()
    TOK["rest_on_collapse"] = rng(100 * cc_.restoration_adjoint.min(), 100 * cc_.restoration_adjoint.max(), 2)
    TOK["coll_on_collapse"] = rng(100 * cc_.collapse_adjoint.min(), 100 * cc_.collapse_adjoint.max(), 2)
    TOK["td_n"] = str(len(TD)); TOK["td_agree"] = str(int((TD.predicted == TD.recovered).sum())); TOK["td_rec"] = str(int(TD.recovered.sum()))
    TOK["td_persist"] = pct(TD.persist.dropna().astype(float).mean(), 0)
    tdr = TD[TD.recovered].copy(); tdr["s"] = tdr.strat.str.replace(r"_\d+", "", regex=True)
    tp = tdr.groupby("s").t_pulse_min.median()
    TOK["tp_rest"] = f"{tp['restoration_adjoint']:g}"; TOK["tp_rnd"] = f"{tp['random']:g}"; TOK["tp_out"] = f"{tp['GBB_gradient']:g}"
    TOK["tp_coll"] = f"{tp['collapse_adjoint']:g}"
    fr = TD.assign(s=TD.strat.str.replace(r"_\d+", "", regex=True)).groupby(["frac", "s"]).recovered.mean().unstack()
    TOK["td_half_rest"] = pct(fr.loc[0.5, "restoration_adjoint"], 0); TOK["td_half_coll"] = pct(fr.loc[0.5, "collapse_adjoint"], 0)
    TOK["td_half_rnd"] = pct(fr.loc[0.5, "random"], 0); TOK["td_half_out"] = pct(fr.loc[0.5, "GBB_gradient"], 0)
    TOK["cl_rec_range"] = rng(100 * m.err_closure_rec.dropna().min(), 100 * m.err_closure_rec.dropna().max(), 1)
    # misspecification
    M2 = MS.copy(); M2["ea"] = M2.adj_hat / M2.adj_true; M2["eg"] = M2.gbb_hat / M2.adj_true
    mg = M2.groupby(["group", "sigma"])[["ea", "eg", "rho"]].median()
    TOK["ms_ea_01"] = rng(mg.xs(0.1, level="sigma").ea.min(), mg.xs(0.1, level="sigma").ea.max())
    TOK["ms_ea_03"] = rng(mg.xs(0.3, level="sigma").ea.min(), mg.xs(0.3, level="sigma").ea.max())
    TOK["ms_rho_03"] = rng(mg.xs(0.3, level="sigma").rho.min(), mg.xs(0.3, level="sigma").rho.max())
    TOK["ms_eg_ER"] = f2(mg.loc[("Mut-ER", 0.3), "eg"]); TOK["ms_eg_SF"] = f2(mg.loc[("Mut-SF", 0.3), "eg"])
    TOK["ms_n"] = str(len(M2))
    # sweeps
    st_ = SWd[(SWd.kind == "struct") & SWd.ok].copy()
    st_["r"] = st_.comp_GBB_gradient / st_.comp_adjoint
    TOK["sw_n"] = str(len(st_)); TOK["sw_n_all"] = str(int((SWd.kind == "struct").sum()))
    for dyn in ("mutualistic", "regulatory"):
        q = st_[st_.dyn == dyn]
        TOK[f"sw_pi1_{dyn[:3]}"] = rng(q.rho_RC1.min(), q.rho_RC1.max())
        TOK[f"sw_pi1med_{dyn[:3]}"] = f2(q.rho_RC1.median())
        TOK[f"sw_out_{dyn[:3]}"] = rng(q.rho_GBB_gradient.min(), q.rho_GBB_gradient.max())
        loc = q[q.fam.isin(["assort+", "mod0.05"])]
        TOK[f"sw_loc_PR_{dyn[:3]}"] = rng(loc.PR_u.min(), loc.PR_u.max()); TOK[f"sw_loc_eff_{dyn[:3]}"] = rng(loc.eff1_GBB_gradient.min(), loc.eff1_GBB_gradient.max())
        TOK[f"sw_loc_gbb_{dyn[:3]}"] = rng(100 * loc.err_gbb.max(), 100 * loc.err_gbb.min(), 0)
    sfq = st_[(st_.fam == "SF")]
    TOK["sw_neg_by_gamma"] = ", ".join(f"{100 * sfq[(sfq.gamma == gm) & (sfq.dyn == 'mutualistic')].frac_neg_gbb.mean():.0f}\\%" for gm in (3.0, 2.5, 2.1))
    lh = SWd[(SWd.kind == "lhs")]
    lok = lh[lh.ok].copy(); lok["r"] = lok.comp_GBB_gradient / lok.comp_adjoint
    TOK["lhs_n"] = str(len(lh)); TOK["lhs_ok"] = str(len(lok)); TOK["lhs_nofold"] = str(len(lh) - len(lok))
    TOK["lhs_pi1"] = rng(lok.rho_RC1.min(), lok.rho_RC1.max()); TOK["lhs_pi2_min"] = f2(lok.rho_RC2.min())
    TOK["lhs_out_q"] = f"{f2(lok.rho_GBB_gradient.quantile(0.25))}--{f2(lok.rho_GBB_gradient.quantile(0.75))}"
    TOK["lhs_eff1out_med"] = f2(lok.eff1_GBB_gradient.median()); TOK["lhs_eff1pi1_med"] = f2(lok.eff1_RC1.median())
    TOK["lhs_eff1out_lt09"] = pct((lok.eff1_GBB_gradient < 0.9).mean(), 0)
    TOK["lhs_cl_q"] = rng(100 * lok.err_closure.quantile(0.25), 100 * lok.err_closure.quantile(0.75), 0)
    TOK["lhs_r_med"] = f2(lok.r.median())
    # symmetric control and examples
    TOK["reg_xc"] = f"{EXJ['regular']['x_c']:.10f}"; TOK["reg_bc"] = f"{EXJ['regular']['beta_c']:.10f}"
    TOK["reg_xr"] = f"{EXJ['regular']['x_r']:.10f}"; TOK["reg_br"] = f"{EXJ['regular']['beta_r']:.10f}"
    TOK["reg_rmin"] = f"{100 * EXJ['regular']['rand5_min']:.3f}\\%"; TOK["reg_rmax"] = f"{100 * EXJ['regular']['rand5_max']:.3f}\\%"
    TOK["neg_gamma4"] = f"{100 * EXJ['sf_neg_by_gamma']['4.0']:.0f}"; TOK["neg_gamma22"] = f"{100 * EXJ['sf_neg_by_gamma']['2.2']:.0f}"
    TOK["neg_gamma205"] = f"{100 * EXJ['sf_neg_by_gamma']['2.05']:.0f}"
    return TOK
