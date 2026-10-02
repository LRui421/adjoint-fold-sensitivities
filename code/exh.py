"""exh -- exhaustive optimum of k-node input protection on small networks (N <= 34, k <= 3)."""
import os, sys, time, pickle, itertools, traceback
import numpy as np
import networkx as nx
import core2 as C
import exp2 as X

HERE = os.path.dirname(os.path.abspath(__file__))


def small_tasks():
    T = []
    for r in range(5):
        seed = 2000 + r
        Aer = X.er_undirected(24, 4, seed)
        Asf = X.sf_undirected(24, 4, seed)
        for dyn in ("mutualistic", "regulatory"):
            T.append(dict(label=f"exh_ER_{dyn[:3]}_{r}", group=("Mut" if dyn == "mutualistic" else "Reg") + "-ER24", dyn=dyn, A=Aer, directed=False))
            T.append(dict(label=f"exh_SF_{dyn[:3]}_{r}", group=("Mut" if dyn == "mutualistic" else "Reg") + "-SF24", dyn=dyn, A=Asf, directed=False))
    for w in ("Safariland", "schemske1978", "small1976", "motten1982", "olito2015"):
        A = X.web_projection(X.web_matrix(w))
        if len(A) <= 34:
            T.append(dict(label=f"exh_EMP_{w}", group="Webs-proj", dyn="mutualistic", A=A, directed=False))
    return T


def run(t):
    m = X.make_model(t)
    fold, _ = X.base_collapse(m)
    if not fold.get("ok"):
        return dict(label=t["label"], error="nofold")
    N = m.N
    R = C.node_sens(m, fold)
    st = C.structural_scores(m.A)
    its, _ = C.power_iterates(m, fold["x"], fold["beta"], ks=(1,))
    doses = {"mutualistic": (0.3, 1.0, 3.0), "regulatory": (0.03, 0.1, 0.3)}[m.name]
    out = dict(label=t["label"], group=t["group"], dyn=t["dyn"], N=N, beta_c=fold["beta"], res={})
    for d in doses:
        allv = {}
        for k in (1, 2, 3):
            for S in itertools.combinations(range(N), k):
                allv[S] = X.eval_theta(m, fold, list(S), d)
        for k in (1, 2, 3):
            vals = {S: v for S, v in allv.items() if len(S) == k}
            comp = np.array([v["comp"] for v in vals.values()])
            keys = list(vals.keys())
            best = keys[int(np.argmax(comp))]

            def val(S):
                return allv[tuple(sorted(int(i) for i in S))]["comp"]

            strat = dict(adjoint=C.topk(R, k), GBB_gradient=C.topk(st["out_strength"], k),
                         spectral_gradient=C.topk(st["eigenvector"], k), RC1=C.topk(its[1], k))
            # greedy on exact finite marginals
            g = []
            for _ in range(k):
                cand = [i for i in range(N) if i not in g]
                g.append(max(cand, key=lambda i: val(g + [i])))
            strat["greedy"] = g
            # adaptive adjoint
            th = m.theta.copy(); cur = fold; ch = []
            for _ in range(k):
                Rc = C.node_sens(m.copy(theta=th), cur); Rc[ch] = -np.inf
                i = int(np.argmax(Rc)); ch.append(i)
                th_old = th.copy(); th = th.copy(); th[i] += d
                tr = C.track_fold(lambda s: m.copy(theta=th_old + s * (th - th_old)), cur)
                if tr["status"] != "kept":
                    rem = [j for j in np.argsort(-R) if j not in ch]
                    ch += [int(j) for j in rem[: k - len(ch)]]
                    break
                cur = tr
            strat["adaptive_adjoint"] = ch
            sv = {nm: val(S) for nm, S in strat.items()}
            pct = {nm: float(np.mean(comp <= v + 1e-15)) for nm, v in sv.items()}
            nlost = int(np.sum([v["status"] == "lost" for v in vals.values()]))
            out["res"][(d, k)] = dict(opt=float(comp.max()), opt_set=best, mean=float(comp.mean()), vals=sv, pct=pct,
                                      n_sets=len(comp), n_lost=nlost, opt_lost=vals[best]["status"] == "lost")
    return out


if __name__ == "__main__":
    od = os.path.join(HERE, "results_exh")
    os.makedirs(od, exist_ok=True)
    for t in sorted(small_tasks(), key=lambda t: -len(t["A"])):
        fn = os.path.join(od, t["label"] + ".pkl")
        if os.path.exists(fn):
            continue
        try:
            fd = os.open(fn + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.close(fd)
        except FileExistsError:
            continue
        t0 = time.time()
        try:
            r = run(t)
        except Exception:
            r = dict(label=t["label"], error=traceback.format_exc())
        r["runtime_s"] = time.time() - t0
        pickle.dump(r, open(fn, "wb"))
        print(t["label"], round(r["runtime_s"], 1), r.get("error", "")[-200:], flush=True)
