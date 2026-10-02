"""sweep -- structural sweep (N, <k>, gamma, modularity, assortativity) and mutualistic parameter LHS."""
import os, sys, time, pickle, traceback
import numpy as np
import networkx as nx
from scipy.stats import spearmanr, qmc
import core2 as C
import exp2 as X

HERE = os.path.dirname(os.path.abspath(__file__))


def lcc(G):
    cc = max(nx.connected_components(G), key=len)
    return nx.to_numpy_array(nx.convert_node_labels_to_integers(G.subgraph(cc).copy()))


def rewire_assort(A, target, seed, maxit=200000):
    rng = np.random.default_rng(seed)
    G = nx.from_numpy_array(A)
    deg = dict(G.degree())
    edges = list(G.edges())
    r = nx.degree_assortativity_coefficient(G)
    it = 0
    while (r - target) * np.sign(target) < 0 and it < maxit:
        it += 1
        i1, i2 = rng.choice(len(edges), 2, replace=False)
        a, b = edges[i1]
        c, d = edges[i2]
        nodes = [a, b, c, d]
        if len(set(nodes)) < 4:
            continue
        srt = sorted(nodes, key=lambda z: deg[z])
        if target > 0:
            e1, e2 = (srt[3], srt[2]), (srt[1], srt[0])
        else:
            e1, e2 = (srt[3], srt[0]), (srt[2], srt[1])
        if G.has_edge(*e1) or G.has_edge(*e2):
            continue
        G.remove_edge(a, b); G.remove_edge(c, d)
        G.add_edge(*e1); G.add_edge(*e2)
        edges[i1], edges[i2] = e1, e2
        if it % 200 == 0:
            r = nx.degree_assortativity_coefficient(G)
    return lcc(G), nx.degree_assortativity_coefficient(G)


def sbm(N, k, mix, seed, B=4):
    sizes = [N // B] * B
    pin = k * (1 - mix) / (N / B - 1)
    pout = k * mix / (N - N / B)
    P = np.full((B, B), pout)
    np.fill_diagonal(P, pin)
    return lcc(nx.stochastic_block_model(sizes, P.tolist(), seed=seed))


def metrics(m, directed=False):
    fold, _ = X.base_collapse(m)
    if not fold.get("ok"):
        return dict(ok=False, reason=fold.get("reason"))
    N = m.N
    bc = fold["beta"]
    R = C.node_sens(m, fold)
    st = C.structural_scores(m.A)
    its, _ = C.power_iterates(m, fold["x"], bc, ks=(1, 2))
    cf = C.closure_fold(m, -1, beta_hint=bc * 2)
    k5 = X.kof(0.05, N)
    top = C.topk(R, k5)
    d0 = X.D0[m.name]
    ev = {nm: X.eval_theta(m, fold, C.topk(sc, k5), d0) for nm, sc in
          (("adjoint", R), ("GBB_gradient", st["out_strength"]), ("spectral_gradient", st["eigenvector"]))}
    I, J = X.link_pairs(m.A, directed)
    q = C.gbb_quantities(m.A)
    g = (q["s_out"][I] + q["s_in"][J] - q["beta_eff"]) / q["S"]
    if not directed:
        g = g + (q["s_out"][J] + q["s_in"][I] - q["beta_eff"]) / q["S"]
    out = dict(ok=True, N=N, E=len(I), beta_c=bc, a=fold["a"], gap=fold["gap"], kappa=fold["kappa_eig"],
               err_gbb=(X.gbb_threshold(m) - bc) / bc, err_spec=(X.spectral_threshold(m) - bc) / bc,
               err_closure=((cf["beta"] - bc) / bc) if cf.get("ok") else np.nan,
               PR_u=C.participation(fold["u"]), frac_neg_gbb=float(np.mean(g < 0)),
               heterogeneity=float((q["s_out"] ** 2).mean() / q["s_out"].mean() ** 2),
               assort=float(nx.degree_assortativity_coefficient(nx.from_numpy_array((m.A > 0).astype(int)))) if not directed else np.nan)
    for nm, sc in (("GBB_gradient", st["out_strength"]), ("spectral_gradient", st["eigenvector"]), ("RC1", its[1]),
                   ("RC2", its[2]), ("closure_exact", cf.get("sens_closure", np.full(N, np.nan)))):
        if np.all(np.isfinite(sc)):
            out["rho_" + nm] = spearmanr(sc, R)[0]
            out["eff1_" + nm] = float(R[C.topk(sc, k5)].sum() / R[top].sum())
    for nm, o in ev.items():
        out["comp_" + nm] = o["comp"]
        out["status_" + nm] = o["status"]
    return out


def jobs():
    J = []
    for fam in ("SF", "ER"):
        for N in (50, 100, 200):
            for k in (4, 6, 12):
                for gam in ((2.1, 2.5, 3.0) if fam == "SF" else (None,)):
                    for r in range(3):
                        J.append(dict(kind="struct", fam=fam, N=N, k=k, gamma=gam, r=r))
    for var in ("mod0.05", "mod0.2", "assort+", "assort-"):
        for r in range(3):
            J.append(dict(kind="struct", fam=var, N=100, k=6, gamma=2.5, r=r))
    sampler = qmc.LatinHypercube(d=6, seed=7)
    U = sampler.random(60)
    lo = np.array([0.02, 0.5, 2.0, 1.0, 0.1, 0.05])
    hi = np.array([0.3, 1.5, 10.0, 10.0, 2.0, 1.0])
    P = lo + U * (hi - lo)
    for i, p in enumerate(P):
        for r in range(3):
            J.append(dict(kind="lhs", i=i, par=p, r=r))
    return J


def job_label(j):
    if j["kind"] == "struct":
        return f"st_{j['fam']}_N{j['N']}_k{j['k']}_g{j['gamma']}_r{j['r']}"
    return f"lhs_{j['i']}_r{j['r']}"


def run(j):
    out = dict(j)
    if j["kind"] == "struct":
        seed = 3000 + j["r"] + 10 * j["N"] + 1000 * j["k"]
        if j["fam"] == "SF":
            A = X.sf_undirected(j["N"], j["k"], seed, gamma=j["gamma"])
        elif j["fam"] == "ER":
            A = X.er_undirected(j["N"], j["k"], seed)
        elif j["fam"].startswith("mod"):
            A = sbm(j["N"], j["k"], float(j["fam"][3:]), seed)
        else:
            A0 = X.sf_undirected(j["N"], j["k"], seed)
            A, _ = rewire_assort(A0, 0.2 if j["fam"] == "assort+" else -0.3, seed)
        for dyn in ("mutualistic", "regulatory"):
            m = C.Mutualistic(A) if dyn == "mutualistic" else C.Regulatory(A)
            try:
                out[dyn] = metrics(m)
            except Exception:
                out[dyn] = dict(ok=False, reason=traceback.format_exc()[-300:])
    else:
        A = X.sf_undirected(100, 6, 1000 + j["r"])
        B, mu, K, D, E, H = j["par"]
        m = C.Mutualistic(A, theta=B, mu=mu, K=K, D=D, E=E, H=H)
        try:
            out["mutualistic"] = metrics(m)
        except Exception:
            out["mutualistic"] = dict(ok=False, reason=traceback.format_exc()[-300:])
    return out


if __name__ == "__main__":
    od = os.path.join(HERE, "results_sweep")
    os.makedirs(od, exist_ok=True)
    for j in jobs():
        fn = os.path.join(od, job_label(j) + ".pkl")
        if os.path.exists(fn):
            continue
        try:
            fd = os.open(fn + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
        except FileExistsError:
            continue
        t0 = time.time()
        r = run(j)
        r["runtime_s"] = time.time() - t0
        pickle.dump(r, open(fn, "wb"))
        print(job_label(j), round(r["runtime_s"], 1), flush=True)
