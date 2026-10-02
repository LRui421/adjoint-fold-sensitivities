"""exp2 -- revised per-network analysis (certified folds, two-part outcomes, dose x budget grids,
Monte-Carlo random baselines, greedy/local search, links, levers, restoration, misspecification)."""
import os, sys, time, pickle, traceback
import numpy as np
import networkx as nx
from scipy.stats import spearmanr, kendalltau
import core2 as C

HERE = os.path.dirname(os.path.abspath(__file__))
NRAND = 100
BUDGETS = (0.01, 0.02, 0.05, 0.10)
DOSES = {"mutualistic": (0.1, 0.3, 1.0, 3.0), "regulatory": (0.01, 0.03, 0.1, 0.3)}
D0 = {"mutualistic": 1.0, "regulatory": 0.1}
REST_DOSES = {"mutualistic": (0.03, 0.1, 0.3), "regulatory": (0.01, 0.03, 0.1)}
REST_D0 = {"mutualistic": 0.1, "regulatory": 0.03}
LEVER_EPS = 0.1          # relative reduction of the per-capita loss rate


def kof(q, N):
    return max(1, int(round(q * N)))


# ------------------------------------------------------------------ networks
def er_undirected(N, k, seed):
    G = nx.gnp_random_graph(N, k / (N - 1), seed=seed)
    cc = max(nx.connected_components(G), key=len)
    return nx.to_numpy_array(nx.convert_node_labels_to_integers(G.subgraph(cc).copy()))


def sf_undirected(N, k, seed, gamma=2.5):
    rng = np.random.default_rng(seed)
    i = np.arange(1, N + 1)
    wts = i ** (-1.0 / (gamma - 1))
    wts = wts / wts.mean() * k
    rng.shuffle(wts)
    G = nx.Graph(nx.expected_degree_graph(list(wts), seed=int(seed), selfloops=False))
    cc = max(nx.connected_components(G), key=len)
    return nx.to_numpy_array(nx.convert_node_labels_to_integers(G.subgraph(cc).copy()))


def directed_chunglu(N, k, seed, gamma=2.5):
    rng = np.random.default_rng(seed)
    i = np.arange(1, N + 1)

    def wv(g):
        w = i ** (-1.0 / (g - 1))
        w = w / w.mean() * k
        return rng.permutation(w)

    win, wout = wv(gamma), wv(gamma)
    P = np.minimum(1.0, np.outer(win, wout) / (k * N))
    np.fill_diagonal(P, 0)
    A = (rng.random((N, N)) < P).astype(float)
    D = nx.from_numpy_array(A.T, create_using=nx.DiGraph)
    cc = max(nx.strongly_connected_components(D), key=len)
    idx = sorted(cc)
    return A[np.ix_(idx, idx)]


def web_matrix(w):
    return (np.loadtxt(os.path.join(HERE, "webs", w + ".csv"), delimiter=",") > 0).astype(float)


def web_projection(M):
    P = M / M.sum(1, keepdims=True)
    A = M.T @ P
    np.fill_diagonal(A, 0.0)
    G = nx.from_numpy_array(A)
    idx = sorted(max(nx.connected_components(G), key=len))
    return A[np.ix_(idx, idx)]


def web_bipartite(M):
    M = M[M.sum(1) > 0][:, M.sum(0) > 0]
    n1, n2 = M.shape
    A = np.zeros((n1 + n2, n1 + n2))
    A[:n1, n1:] = M
    A[n1:, :n1] = M.T
    G = nx.from_numpy_array(A)
    idx = sorted(max(nx.connected_components(G), key=len))
    return A[np.ix_(idx, idx)], np.array([i < n1 for i in idx])


WEBS = ["Safariland", "barrett1987", "elberling1999", "inouye1988", "kevan1970", "memmott1999",
        "motten1982", "olito2015", "ollerton2003", "schemske1978", "small1976"]


def build_tasks():
    tasks = []
    for r in range(10):
        seed = 1000 + r
        Aer, Asf = er_undirected(100, 6, seed), sf_undirected(100, 6, seed)
        Adi = directed_chunglu(130, 6, seed)
        sub = r < 5
        tasks += [dict(label=f"ER_mut_{r}", group="Mut-ER", dyn="mutualistic", A=Aer, directed=False, sub=sub, rest=True),
                  dict(label=f"SF_mut_{r}", group="Mut-SF", dyn="mutualistic", A=Asf, directed=False, sub=sub, rest=True),
                  dict(label=f"ER_reg_{r}", group="Reg-ER", dyn="regulatory", A=Aer, directed=False, sub=sub, rest=False),
                  dict(label=f"SF_reg_{r}", group="Reg-SF", dyn="regulatory", A=Asf, directed=False, sub=sub, rest=False),
                  dict(label=f"DSF_reg_{r}", group="Reg-dir", dyn="regulatory", A=Adi, directed=True, sub=sub, rest=False)]
        rng = np.random.default_rng(5000 + r)
        cv = 0.3
        sig = np.sqrt(np.log(1 + cv ** 2))
        ln = lambda n: np.exp(rng.normal(-sig ** 2 / 2, sig, n))
        tasks.append(dict(label=f"SF_muthet_{r}", group="Mut-SF-het", dyn="mutualistic", A=Asf, directed=False, sub=False,
                          rest=True, theta=0.1 * ln(len(Asf)), mu=ln(len(Asf))))
        tasks.append(dict(label=f"SF_reghet_{r}", group="Reg-SF-het", dyn="regulatory", A=Asf, directed=False, sub=False,
                          rest=False, mu=ln(len(Asf))))
        tasks.append(dict(label=f"ER_regb_{r}", group="RegB-ER", dyn="regulatory", A=Aer, directed=False, sub=False,
                          rest=True, theta=0.05))
        tasks.append(dict(label=f"SF_regb_{r}", group="RegB-SF", dyn="regulatory", A=Asf, directed=False, sub=False,
                          rest=True, theta=0.05))
    for w in WEBS:
        M = web_matrix(w)
        tasks.append(dict(label=f"EMP_{w}", group="Webs-proj", dyn="mutualistic", A=web_projection(M), directed=False,
                          sub=True, rest=True))
        Ab, isplant = web_bipartite(M)
        tasks.append(dict(label=f"BIP_{w}", group="Webs-bip", dyn="mutualistic", A=Ab, directed=False, sub=False,
                          rest=True, isplant=isplant))
    return tasks


def make_model(t):
    if t["dyn"] == "mutualistic":
        return C.Mutualistic(t["A"], theta=t.get("theta", 0.1), mu=t.get("mu", 1.0))
    return C.Regulatory(t["A"], theta=t.get("theta", 0.0), B=t.get("mu", 1.0))


def gbb_threshold(m):
    q = C.gbb_quantities(m.A)
    w = q["w"]
    A1 = np.array([[q["beta_eff"]]])
    if m.name == "mutualistic":
        m1 = C.Mutualistic(A1, theta=float(w @ m.theta), mu=float(w @ m.mu), K=m.K, D=m.D, E=m.E, H=m.H)
    else:
        m1 = C.Regulatory(A1, theta=float(w @ m.theta), B=float(w @ m.B))
    # one-dimensional equilibrium curve beta(x) is explicit; the collapse fold is the local minimum
    # of beta(x) on the upper branch, refined by Moore-Spence
    xs = np.geomspace(1e-3, 200.0, 20000)
    if m.name == "mutualistic":
        num = -(m1.theta[0] - m1.mu[0] * xs - xs ** 2 / m1.K)
        den = q["beta_eff"] * xs * xs / (m1.D + (m1.E + m1.H) * xs)
    else:
        num = m1.B[0] * xs - m1.theta[0]
        den = q["beta_eff"] * xs ** 2 / (1 + xs ** 2)
    bx = num / den
    ok = (bx > 0) & np.isfinite(bx)
    loc = ok[1:-1] & ok[:-2] & ok[2:] & (bx[1:-1] <= bx[:-2]) & (bx[1:-1] <= bx[2:])
    idx = list(np.nonzero(loc)[0] + 1)
    if not idx:
        return np.nan
    i = idx[-1]
    ms = C.moore_spence(m1, np.array([xs[i]]), bx[i], np.array([1.0]))
    return float(ms["beta"]) if ms.get("ok") else float(bx[i])


def spectral_threshold(m):
    """Spectral (Laurence et al.) reduction with left Perron weights: beta_c^sp = Lambda_c/lambda_1 (homogeneous params)."""
    lam = float(np.max(np.linalg.eigvals(m.A).real))
    q = C.gbb_quantities(m.A)
    return gbb_threshold(m) * q["beta_eff"] / lam


def base_collapse(m):
    q = C.gbb_quantities(m.A)
    b0 = 2.0 * C.LAMBDA_C[m.name] / q["beta_eff"]
    x, ok = C.high_state(m, b0)
    for _ in range(25):
        if ok and x.min() > 0.3 * (1 if m.name == "mutualistic" else 1):
            break
        b0 *= 1.5
        x, ok = C.high_state(m, b0)
    return C.locate_fold(m, x, b0, -1), b0


def base_recovery(m, beta_c):
    b = beta_c * 1.02
    x, ok = C.low_state(m, b)
    r = C.locate_fold(m, x, b, +1)
    return r


# ------------------------------------------------------------------ evaluation of interventions
def outcome(tr, beta0, sign=+1):
    """sign=+1 for collapse (shift = decrease of beta_c), recovery also decreases beta_r."""
    if tr["status"] == "kept":
        sh = (beta0 - tr["beta"]) / beta0
        return dict(status="kept", beta=tr["beta"], shift=sh, comp=sh, s_star=1.0)
    return dict(status="lost", beta=tr["beta"], shift=np.nan, comp=1.0 + (1.0 - tr["s_star"]), s_star=tr["s_star"])


def eval_theta(m, fold, S, d):
    S = np.asarray(S, int)

    def mk(s):
        th = m.theta.copy()
        th[S] += s * d
        return m.copy(theta=th)

    return outcome(C.track_fold(mk, fold), fold["beta"])


def eval_theta_vec(m, fold, dvec):
    def mk(s):
        return m.copy(theta=m.theta + s * dvec)

    return outcome(C.track_fold(mk, fold), fold["beta"])


def eval_lever(m, fold, S, eps):
    S = np.asarray(S, int)

    def mk(s):
        mu = m.mu.copy()
        mu[S] *= (1 - s * eps)
        return m.copy(mu=mu)

    return outcome(C.track_fold(mk, fold), fold["beta"])


def eval_links(m, fold, dA):
    def mk(s):
        return m.copy(A=m.A - s * dA)

    tr = C.track_fold(mk, fold)
    if tr["status"] == "kept":
        return dict(status="kept", beta=tr["beta"], damage=(tr["beta"] - fold["beta"]) / fold["beta"])
    return dict(status="lost", beta=tr["beta"], damage=np.nan, s_star=tr["s_star"])


def verify_first_fold(mnew, out, beta_ref):
    """Independent check: continue the high branch of the modified model downward from well above the
    tracked fold and compare the first fold encountered with the tracked one."""
    b0 = max(beta_ref, out["beta"]) * 1.5
    x, ok = C.high_state(mnew, b0)
    r = C.locate_fold(mnew, x, b0, -1)
    if out["status"] == "lost":
        return dict(match=(not r.get("ok")) or r.get("reason") == "no_fold", beta_first=r.get("beta", np.nan))
    if not r.get("ok"):
        return dict(match=False, beta_first=np.nan)
    return dict(match=abs(r["beta"] - out["beta"]) < 1e-6 * out["beta"], beta_first=r["beta"])


# ------------------------------------------------------------------ link helpers
def link_pairs(A, directed):
    if directed:
        I, J = np.nonzero(A)
    else:
        I, J = np.nonzero(np.triu(A, 1))
    return I, J


def pair_scores(m, fold, I, J, directed, u=None, x=None, beta=None):
    """first-order increase of beta_c when the link (pair) is deleted = A_ij R_ij (+ A_ji R_ji)."""
    u = fold["u_raw"] if u is None else u
    x = fold["x"] if x is None else x
    beta = fold["beta"] if beta is None else beta
    den = u @ m.fbeta(x)
    s = m.A[I, J] * beta * u[I] * C.edge_G(m, x, I, J) / den
    if not directed:
        s = s + m.A[J, I] * beta * u[J] * C.edge_G(m, x, J, I) / den
    return s


def dA_of(A, I, J, sel, directed, frac=1.0):
    dA = np.zeros_like(A)
    dA[I[sel], J[sel]] = frac * A[I[sel], J[sel]]
    if not directed:
        dA[J[sel], I[sel]] = frac * A[J[sel], I[sel]]
    return dA


def connectivity(A, directed):
    if directed:
        D = nx.from_numpy_array((A.T > 0).astype(int), create_using=nx.DiGraph)
        return max(len(c) for c in nx.strongly_connected_components(D)) / len(A)
    G = nx.from_numpy_array((A > 0).astype(int))
    return max(len(c) for c in nx.connected_components(G)) / len(A)


# ------------------------------------------------------------------ main analysis
def analyze(t):
    t0 = time.time()
    out = dict(label=t["label"], group=t["group"], dyn=t["dyn"], directed=t["directed"])
    m = make_model(t)
    N = m.N
    out["N"] = N
    out["E"] = int((m.A > 0).sum() if t["directed"] else (np.triu(m.A, 1) > 0).sum())
    fold, bstart = base_collapse(m)
    out["fold_ok"] = fold.get("ok", False)
    if not fold.get("ok"):
        out["error"] = "no collapse fold: " + str(fold.get("reason"))
        return out
    bc = fold["beta"]
    diag_keys = ["beta", "beta_nc", "res_f", "res_Jv", "res_uJ", "a", "b", "gap", "kappa_eig", "lam1", "smin_rel", "it",
                 "xmin", "u_min"]
    out["collapse"] = {k: fold[k] for k in diag_keys}
    out["collapse"]["PR_u"] = C.participation(fold["u"])
    out["collapse"]["PR_v"] = C.participation(fold["v"])
    # tolerance convergence (natural continuation stopped early, then Moore-Spence at several tolerances)
    tc = {}
    for rt in (1e-2, 1e-3, 1e-4):
        nc = C.natural_continuation(m, C.high_state(m, bstart)[0], bstart, -1, rtol=rt)
        _, v0, _, _ = C.crit_vectors(m.jac(nc["x"], nc["beta"]))
        for tol in (1e-6, 1e-9, 1e-12):
            ms = C.moore_spence(m, nc["x"], nc["beta"], v0, tol=tol)
            tc[(rt, tol)] = (nc["beta"], ms["beta"] if ms.get("ok") else np.nan, ms.get("it", -1))
    out["tol_conv"] = tc
    # post-tipping state (where does the jump land?)
    xs = fold["x"] - 0.05 * np.abs(fold["x"]).mean() * fold["v"] / np.abs(fold["v"]).max()
    xpost = C.integrate(m, xs, bc * (1 - 1e-4), T=5000.0)
    out["post_tip"] = dict(mean_ratio=float(xpost.mean() / fold["x"].mean()),
                           frac_drop=float(np.mean(xpost < 0.5 * fold["x"])))
    # thresholds of reductions
    out["beta_gbb"] = gbb_threshold(m)
    out["beta_spec"] = spectral_threshold(m)
    cf = C.closure_fold(m, -1, beta_hint=bc * 2)
    out["beta_closure"] = cf["beta"] if cf.get("ok") else np.nan
    R = C.node_sens(m, fold)
    u = fold["u"]
    # ---------------- centralities
    st = C.structural_scores(m.A)
    J = m.jac(fold["x"], bc)
    its, T = C.power_iterates(m, fold["x"], bc, ks=(1, 2, 3, 5, 10))
    itl, _ = C.power_iterates(m, fold["x"], bc, ks=(1, 2, 5), lazy=0.5)
    cent = dict(adjoint=R, closure_exact=cf["sens_closure"] if cf.get("ok") else np.full(N, np.nan),
                GBB_gradient=st["out_strength"], spectral_gradient=st["eigenvector"],
                in_strength=st["in_strength"], kcore=st["kcore"], betweenness=st["betweenness"],
                RC1=its[1], RC2=its[2])
    if cf.get("ok"):
        itc, _ = C.power_iterates(m, cf["phi"], cf["beta"], ks=(1,))
        cent["RC_red"] = itc[1]
    # slow-mode scan
    slow = {}
    xprev = None
    for ratio in (1.001, 1.01, 1.05, 1.1, 1.3, 2.0):
        b = bc * ratio
        xi = fold["x"] + np.sqrt(fold["b"] * (b - bc) / abs(fold["a"])) * fold["v"] if xprev is None else xprev
        xr, ok = C.newton(m, xi, b)
        if not ok or C.lam_sorted(m.jac(xr, b))[0].real >= 0 or xr.mean() < fold["x"].mean() * 0.999:
            xr, ok = C.high_state(m, b)
        xprev = xr
        Jr = m.jac(xr, b)
        ev, VL = np.linalg.eig(Jr.T)
        k = int(np.argmax(ev.real))
        ll = np.real(VL[:, k]); ll *= np.sign(ll.sum())
        ev2, VR = np.linalg.eig(Jr)
        k2 = int(np.argmax(ev2.real))
        rr = np.real(VR[:, k2]); rr *= np.sign(rr.sum())
        slow[ratio] = dict(left=ll / np.abs(ll).sum(), right=rr / np.abs(rr).sum(), lam=float(ev[k].real))
        cent[f"slow_left_{ratio}"] = slow[ratio]["left"]
        cent[f"slow_right_{ratio}"] = slow[ratio]["right"]
    cent["slow_mode"] = slow[1.3]["left"]
    k5 = kof(0.05, N)
    k10 = kof(0.10, N)
    top_R = set(C.topk(R, k5))
    corr = {}
    for nm, sc in cent.items():
        if np.any(~np.isfinite(sc)):
            continue
        rho = spearmanr(sc, R)[0]
        tau = kendalltau(sc, R)[0]
        ov5 = len(set(C.topk(sc, k5)) & top_R) / k5
        ov10 = len(set(C.topk(sc, k10)) & set(C.topk(R, k10))) / k10
        eff1 = float(R[C.topk(sc, k5)].sum() / R[list(top_R)].sum())
        corr[nm] = dict(rho=rho, tau=tau, ov5=ov5, ov10=ov10, eff1=eff1)
    out["corr"] = corr
    out["rc_resid"] = {k: float(np.linalg.norm(v @ T - v) / np.linalg.norm(v)) for k, v in its.items()}
    out["rc_lazy_rho"] = {k: spearmanr(v, R)[0] for k, v in itl.items()}
    Tb = (T > 0).astype(float)
    out["T_primitive"] = bool(np.all(np.linalg.matrix_power(Tb + 0 * np.eye(N), min(3 * N, 60)) > 0)) if N <= 400 else None
    out["arrays"] = dict(R=R, u=u, v=fold["v"], x=fold["x"], s_out=st["out_strength"], s_in=st["in_strength"],
                         eig=st["eigenvector"], RC1=its[1], closure=cent["closure_exact"], slow=slow[1.3]["left"],
                         loss=m.loss_flux(fold["x"]), theta=m.theta.copy(), mu=m.mu.copy())
    out["loss_flux_median"] = float(np.median(m.loss_flux(fold["x"])))
    # ---------------- finite-difference validation
    rng = np.random.default_rng(12345)
    order = np.argsort(R)
    qn = [order[int(round(q * (N - 1)))] for q in np.linspace(0, 1, 7)]
    rn = list(rng.choice(np.setdiff1d(np.arange(N), qn), 3, replace=False))
    fd = []
    hs = (1e-2, 1e-3, 1e-4, 1e-5) if m.name == "mutualistic" else (1e-3, 1e-4, 1e-5, 1e-6)

    def ms_beta(mm):
        r = C.moore_spence(mm, fold["x"], bc, fold["v"])
        return r["beta"] if r.get("ok") else np.nan

    for i in qn + rn:
        row = dict(kind="node", i=int(i), pred=float(R[i]), strat="quantile" if i in qn else "random")
        for h in hs:
            th1, th2 = m.theta.copy(), m.theta.copy()
            th1[i] += h
            th2[i] -= h
            row[h] = -(ms_beta(m.copy(theta=th1)) - ms_beta(m.copy(theta=th2))) / (2 * h)
            if h == hs[1]:
                row["fwd"] = -(ms_beta(m.copy(theta=th1)) - bc) / h
        fd.append(row)
    Rmu = C.lever_sens_mortality(m, fold)
    for i in list(rng.choice(N, 3, replace=False)):
        row = dict(kind="lever", i=int(i), pred=float(Rmu[i]), strat="random")
        for h in hs:
            m1, m2 = m.mu.copy(), m.mu.copy()
            m1[i] -= h
            m2[i] += h
            row[h] = -(ms_beta(m.copy(mu=m1)) - ms_beta(m.copy(mu=m2))) / (2 * h)
        fd.append(row)
    I, Jj = link_pairs(m.A, t["directed"])
    E = len(I)
    ps = pair_scores(m, fold, I, Jj, t["directed"])   # first-order damage of deleting the pair
    # per-unit-weight derivative of beta_c w.r.t. the pair weight (both directions for undirected)
    lo = np.argsort(ps)
    lsel = [lo[int(round(q * (E - 1)))] for q in np.linspace(0, 1, 5)]
    for e in lsel:
        pred = -ps[e] / m.A[I[e], Jj[e]]
        row = dict(kind="link", i=int(I[e]), j=int(Jj[e]), pred=float(pred), strat="quantile")
        for h in hs:
            dA = np.zeros_like(m.A)
            dA[I[e], Jj[e]] = h
            if not t["directed"]:
                dA[Jj[e], I[e]] = h
            row[h] = (ms_beta(m.copy(A=m.A + dA)) - ms_beta(m.copy(A=m.A - dA))) / (2 * h)
        fd.append(row)
    out["fd"] = fd
    out["max_link_dbeta"] = float(np.max(-ps / m.A[I, Jj]))  # should be <= 0 (links never raise beta_c)
    # ---------------- node protection: budget x dose grid
    D = DOSES[m.name]
    strat_scores = dict(adjoint=R, closure_exact=cent["closure_exact"], RC1=its[1], RC2=its[2],
                        RC_red=cent.get("RC_red", np.full(N, np.nan)), GBB_gradient=st["out_strength"],
                        spectral_gradient=st["eigenvector"], in_strength=st["in_strength"], kcore=st["kcore"],
                        betweenness=st["betweenness"], slow_mode=slow[1.3]["left"])
    node = {}
    verify = []
    for q in BUDGETS:
        k = kof(q, N)
        for d in D:
            res = {}
            for nm, sc in strat_scores.items():
                if np.any(~np.isfinite(sc)):
                    continue
                S = C.topk(sc, k)
                res[nm] = eval_theta(m, fold, S, d)
                if q == 0.05 and nm in ("adjoint", "GBB_gradient", "spectral_gradient", "RC1", "closure_exact"):
                    th = m.theta.copy(); th[S] += d
                    verify.append((nm, d, verify_first_fold(m.copy(theta=th), res[nm], bc)))
            # adaptive adjoint
            th = m.theta.copy()
            cur = fold
            chosen = []
            lost = None
            for step in range(k):
                if lost is None:
                    Rc = C.node_sens(m.copy(theta=th), cur)
                    Rc[chosen] = -np.inf
                    i = int(np.argmax(Rc))
                else:
                    rem = [j for j in np.argsort(-R) if j not in chosen]
                    i = int(rem[0])
                chosen.append(i)
                th_new = th.copy(); th_new[i] += d
                if lost is None:
                    th_old = th.copy()
                    tr = C.track_fold(lambda s: m.copy(theta=th_old + s * (th_new - th_old)), cur)
                    if tr["status"] == "kept":
                        cur = dict(tr)
                        cur["u_raw"] = tr["u_raw"]
                    else:
                        lost = tr
                th = th_new
            res["adaptive_adjoint"] = eval_theta(m, fold, chosen, d)
            # Monte-Carlo random baseline
            rr = []
            for rep in range(NRAND):
                S = rng.choice(N, k, replace=False)
                o = eval_theta(m, fold, S, d)
                rr.append((o["status"], o["comp"]))
                if rep < 5 and q == 0.05 and d == D0[m.name]:
                    th = m.theta.copy(); th[S] += d
                    verify.append(("random", d, verify_first_fold(m.copy(theta=th), o, bc)))
            res["random"] = rr
            node[(q, d)] = res
    out["node"] = node
    out["verify"] = verify
    # ---------------- greedy (finite marginal) + 1-swap local search, default dose, <=5% budget
    if t.get("sub"):
        d = D0[m.name]
        k = k5
        chosen, gseq = [], []
        for step in range(k):
            best, bi = -np.inf, None
            for i in range(N):
                if i in chosen:
                    continue
                o = eval_theta(m, fold, chosen + [i], d)
                if o["comp"] > best:
                    best, bi = o["comp"], i
            chosen.append(bi)
            gseq.append(best)
        out["greedy"] = dict(set=chosen, seq=gseq)

        def local_search(S0):
            S = list(S0)
            val = eval_theta(m, fold, S, d)["comp"]
            nev, improved = 0, True
            while improved and nev < 4000:
                improved = False
                for a in list(S):
                    for b in range(N):
                        if b in S:
                            continue
                        S2 = [b if z == a else z for z in S]
                        o = eval_theta(m, fold, S2, d)["comp"]
                        nev += 1
                        if o > val + 1e-12:
                            S, val, improved = S2, o, True
                            break
                    if improved:
                        break
            return S, val, nev

        Sg, vg, ng = local_search(chosen)
        Sa, va, na = local_search(list(C.topk(R, k)))
        out["local_search"] = dict(from_greedy=(Sg, vg, ng), from_adjoint=(Sa, va, na))
        # screen-and-rerank: greedy restricted to the 3k highest-R candidates
        cand = list(C.topk(R, min(N, 3 * k)))
        ch2 = []
        for step in range(k):
            best, bi = -np.inf, None
            for i in cand:
                if i in ch2:
                    continue
                o = eval_theta(m, fold, ch2 + [i], d)["comp"]
                if o > best:
                    best, bi = o, i
            ch2.append(bi)
        out["screen_rerank"] = eval_theta(m, fold, ch2, d)
    # ---------------- fixed total dose: concentration vs spreading
    d0 = D0[m.name]
    Q = k5 * d0
    conc = {}
    for mm_ in sorted(set([1, 2, max(1, k5 // 2), k5, 2 * k5, 4 * k5])):
        for nm, sc in (("adjoint", R), ("GBB_gradient", st["out_strength"])):
            conc[(nm, mm_)] = eval_theta(m, fold, C.topk(sc, mm_), Q / mm_)
    # adaptive continuous allocation in 20 increments (repeat allocations allowed)
    dvec = np.zeros(N)
    cur = fold
    for step in range(20):
        Rc = C.node_sens(m.copy(theta=m.theta + dvec), cur)
        i = int(np.argmax(Rc))
        dnew = dvec.copy(); dnew[i] += Q / 20
        dold = dvec.copy()
        tr = C.track_fold(lambda s: m.copy(theta=m.theta + dold + s * (dnew - dold)), cur)
        dvec = dnew
        if tr["status"] != "kept":
            break
        cur = tr
    conc[("adaptive_alloc", 0)] = eval_theta_vec(m, fold, dvec)
    conc["alloc_nodes"] = int((dvec > 0).sum())
    out["concentration"] = conc
    # ---------------- heterogeneous costs (0-1 knapsack, greedy by ratio)
    cost = st["out_strength"] / st["out_strength"].mean()
    Cb = float(k5)

    def fill(order):
        S, c = [], 0.0
        for i in order:
            if c + cost[i] <= Cb + 1e-12:
                S.append(int(i)); c += cost[i]
        return S

    kn = dict(ratio=eval_theta(m, fold, fill(np.argsort(-R / cost)), d0),
              adjoint=eval_theta(m, fold, fill(np.argsort(-R)), d0),
              GBB_gradient=eval_theta(m, fold, fill(np.argsort(-st["out_strength"])), d0),
              cheapest=eval_theta(m, fold, fill(np.argsort(cost + 1e-9 * rng.random(N))), d0))
    kn["random"] = [eval_theta(m, fold, fill(rng.permutation(N)), d0)["comp"] for _ in range(30)]
    out["knapsack"] = kn
    # ---------------- per-capita loss-rate lever
    lev = dict(lever_adjoint=eval_lever(m, fold, C.topk(Rmu, k5), LEVER_EPS),
               input_adjoint=eval_lever(m, fold, C.topk(R, k5), LEVER_EPS),
               GBB_gradient=eval_lever(m, fold, C.topk(st["out_strength"], k5), LEVER_EPS))
    lev["random"] = [eval_lever(m, fold, rng.choice(N, k5, replace=False), LEVER_EPS)["comp"] for _ in range(30)]
    lev["rho_Rmu_R"] = spearmanr(Rmu, R)[0]
    out["lever"] = lev
    # ---------------- links
    q = C.gbb_quantities(m.A)
    g_gbb = (q["s_out"][I] + q["s_in"][Jj] - q["beta_eff"]) / q["S"]
    if not t["directed"]:
        g_gbb = g_gbb + (q["s_out"][Jj] + q["s_in"][I] - q["beta_eff"]) / q["S"]
    gbb_dmg = m.A[I, Jj] * g_gbb
    uA, _ = C.left_perron(m.A)
    vA = C.right_perron(m.A)
    Du = nx.from_numpy_array((m.A.T > 0).astype(int), create_using=nx.DiGraph) if t["directed"] else nx.from_numpy_array((m.A > 0).astype(int))
    eb = nx.edge_betweenness_centrality(Du)
    ebv = np.array([eb.get((int(Jj[e]), int(I[e])), eb.get((int(I[e]), int(Jj[e])), 0.0)) for e in range(E)])
    lsc = dict(adjoint=ps,
               RC1=pair_scores(m, fold, I, Jj, t["directed"], u=its[1] * fold["u_raw"].sum() / its[1].sum()),
               GBB_gradient=gbb_dmg,
               strength_product=(q["s_out"][Jj] * q["s_in"][I]) if t["directed"] else q["s_out"][I] * q["s_out"][Jj],
               spectral_edge=m.A[I, Jj] * np.abs(uA[I]) * np.abs(vA[Jj]) + (0 if t["directed"] else m.A[Jj, I] * np.abs(uA[Jj]) * np.abs(vA[I])),
               edge_betweenness=ebv, weight=m.A[I, Jj] + 1e-9 * rng.random(E))
    if cf.get("ok"):
        lsc["RC_red"] = pair_scores(m, fold, I, Jj, t["directed"], u=cent["RC_red"], x=cf["phi"], beta=cf["beta"])
    links = {}
    Wtot = m.A[I, Jj].sum()
    for qq in (0.01, 0.02, 0.05):
        ne = max(1, int(round(qq * E)))
        res = {}
        for nm, sc in lsc.items():
            sel = C.topk(sc, ne)
            o = eval_links(m, fold, dA_of(m.A, I, Jj, sel, t["directed"]))
            o["lcc"] = connectivity(m.A - dA_of(m.A, I, Jj, sel, t["directed"]), t["directed"])
            res[nm] = o
        rr = []
        for rep in range(NRAND):
            sel = rng.choice(E, ne, replace=False)
            rr.append(eval_links(m, fold, dA_of(m.A, I, Jj, sel, t["directed"]))["damage"])
        res["random"] = rr
        links[("count", qq)] = res
    # fixed removed-weight budget (5% of total weight)
    res = {}
    for nm, sc in lsc.items():
        order_ = np.argsort(-sc)
        cw = np.cumsum(m.A[I[order_], Jj[order_]])
        sel = order_[: int(np.searchsorted(cw, 0.05 * Wtot)) + 1]
        o = eval_links(m, fold, dA_of(m.A, I, Jj, sel, t["directed"]))
        o["n_removed"] = len(sel)
        res[nm] = o
    links[("weight", 0.05)] = res
    # greedy single-link exact search (1% budget), synthetic and small webs only
    ne1 = max(1, int(round(0.01 * E)))
    if E <= 800:
        chosen, seq = [], []
        for step in range(ne1):
            best, be = -np.inf, None
            cand = C.topk(ps, min(E, 60))  # exact marginals over the 60 highest first-order candidates
            for e in cand:
                if e in chosen:
                    continue
                o = eval_links(m, fold, dA_of(m.A, I, Jj, np.array(chosen + [e]), t["directed"]))
                val = o["damage"] if o["status"] == "kept" else -np.inf
                if val > best:
                    best, be = val, int(e)
            chosen.append(be); seq.append(best)
        # exhaustive single-link check at step one (all links)
        if E <= 400:
            dm = np.array([eval_links(m, fold, dA_of(m.A, I, Jj, np.array([e]), t["directed"]))["damage"] for e in range(E)])
            links["single_all"] = dict(damage=dm, pred=ps / bc)
        links["greedy1"] = dict(set=chosen, seq=seq)
    # sign test: links whose GBB gradient says deletion helps
    neg = np.where(gbb_dmg < 0)[0]
    out["frac_neg_gbb"] = float(len(neg) / E)
    sgn = []
    for e in neg[np.argsort(gbb_dmg[neg])][:20]:
        o = eval_links(m, fold, dA_of(m.A, I, Jj, np.array([e]), t["directed"]))
        sgn.append((float(gbb_dmg[e]), o["damage"], float(ps[e] / bc)))
    out["sign_test"] = sgn
    # strengthening +25% of top 5% links selected by adjoint gain vs GBB gradient vs strength product
    ne5 = max(1, int(round(0.05 * E)))
    strg = {}
    for nm in ("adjoint", "GBB_gradient", "strength_product", "weight"):
        sel = C.topk(lsc[nm], ne5)
        strg[nm] = eval_links(m, fold, -dA_of(m.A, I, Jj, sel, t["directed"], frac=0.25))
    strg["GBB_most_negative"] = eval_links(m, fold, -dA_of(m.A, I, Jj, C.topk(-gbb_dmg, ne5), t["directed"], frac=0.25))
    strg["random"] = [eval_links(m, fold, -dA_of(m.A, I, Jj, rng.choice(E, ne5, replace=False), t["directed"], frac=0.25))["damage"] for _ in range(30)]
    links["strengthen"] = strg
    out["links"] = links
    out["link_meta"] = dict(E=E, Wtot=float(Wtot))
    # ---------------- restoration (recovery fold)
    if t.get("rest"):
        fr = base_recovery(m, bc)
        out["rec_ok"] = fr.get("ok", False)
        if fr.get("ok"):
            br = fr["beta"]
            out["recovery"] = {k: fr[k] for k in diag_keys}
            out["recovery"]["PR_u"] = C.participation(fr["u"])
            Rr = C.node_sens(m, fr)
            cfr = C.closure_fold(m, +1, beta_hint=br * 0.9)
            its_r, _ = C.power_iterates(m, fr["x"], br, ks=(1,))
            rsc = dict(restoration_adjoint=Rr, collapse_adjoint=R, GBB_gradient=st["out_strength"],
                       spectral_gradient=st["eigenvector"], RC1_low=its_r[1])
            if cfr.get("ok"):
                rsc["closure_low"] = cfr["sens_closure"]
                out["beta_closure_rec"] = cfr["beta"]
            out["rest_corr"] = {nm: spearmanr(sc, Rr)[0] for nm, sc in rsc.items()}
            out["arrays"]["R_rest"] = Rr
            out["arrays"]["x_low"] = fr["x"]
            rest = {}
            for qq in (0.02, 0.05, 0.10):
                k = kof(qq, N)
                for d in REST_DOSES[m.name]:
                    res = {nm: eval_theta(m, fr, C.topk(sc, k), d) for nm, sc in rsc.items()}
                    # effect of each set on the collapse threshold too
                    res["_collapse"] = {nm: eval_theta(m, fold, C.topk(sc, k), d)["comp"] for nm, sc in rsc.items()}
                    res["random"] = [eval_theta(m, fr, rng.choice(N, k, replace=False), d)["comp"] for _ in range(NRAND)]
                    rest[(qq, d)] = res
            out["restoration"] = rest
            # time-domain recovery test (sustained and pulsed inputs) at 5% budget, default restoration dose
            d = REST_D0[m.name]
            k = k5
            sets = {nm: C.topk(sc, k) for nm, sc in rsc.items() if nm in ("restoration_adjoint", "collapse_adjoint", "GBB_gradient")}
            for rep in range(10):
                sets[f"random_{rep}"] = rng.choice(N, k, replace=False)
            pred = {nm: eval_theta(m, fr, S, d) for nm, S in sets.items()}
            sh_ad = pred["restoration_adjoint"]["comp"]
            xlow_ref = fr["x"]
            td = []
            for frac in (0.25, 0.5, 0.75, 1.25):
                b0 = br * (1 - frac * min(sh_ad, 0.5))
                if b0 <= bc * 1.001:
                    continue
                xl, ok = C.low_state(m, b0)
                xh, okh = C.high_state(m, b0)
                thr = 0.5 * (xl.mean() + xh.mean())
                for nm, S in sets.items():
                    th = m.theta.copy(); th[S] += d
                    mi = m.copy(theta=th)
                    sol = C.solve_ivp(lambda tt, x: mi.f(x, b0), (0, 3000), xl, method="LSODA",
                                      jac=lambda tt, x: mi.jac(x, b0), rtol=1e-8, atol=1e-10, dense_output=True)
                    tg = np.linspace(0, 3000, 3001)
                    mx = sol.sol(tg).mean(0)
                    rec = bool(mx[-1] > thr)
                    trec = float(tg[np.argmax(mx > thr)]) if rec else np.nan
                    persist = np.nan
                    tmin = np.nan
                    if rec:
                        xe = C.integrate(m, sol.y[:, -1], b0, T=3000.0)
                        persist = bool(xe.mean() > thr)
                        for Tp in (5, 10, 20, 50, 100, 200, 500, 1000):
                            s2 = C.solve_ivp(lambda tt, x: mi.f(x, b0), (0, Tp), xl, method="LSODA",
                                             jac=lambda tt, x: mi.jac(x, b0), rtol=1e-8, atol=1e-10)
                            xe2 = C.integrate(m, s2.y[:, -1], b0, T=3000.0)
                            if xe2.mean() > thr:
                                tmin = Tp
                                break
                    pr = pred[nm]
                    predicted = (pr["status"] == "lost") or (pr["beta"] < b0)
                    td.append(dict(frac=frac, beta0=b0, strat=nm, predicted=bool(predicted), recovered=rec,
                                   t_rec=trec, persist=persist, t_pulse_min=tmin, comp=pr["comp"]))
            out["time_domain"] = td
    # ---------------- model misspecification (subset)
    if t.get("sub"):
        mis = []
        d = D0[m.name]
        o_true = eval_theta(m, fold, C.topk(R, k5), d)["comp"]
        for sig in (0.1, 0.3):
            for rep in range(20):
                A2 = m.A * np.exp(rng.normal(0, sig, m.A.shape))
                if not t["directed"]:
                    A2 = np.triu(A2, 1); A2 = A2 + A2.T
                # 5% of links missing
                I2, J2 = link_pairs(A2, t["directed"])
                drop = rng.random(len(I2)) < 0.05
                A2[I2[drop], J2[drop]] = 0
                if not t["directed"]:
                    A2[J2[drop], I2[drop]] = 0
                m2 = make_model(t).copy(A=A2, mu=m.mu * (1 + 0.1 * rng.normal(size=N)).clip(0.5, 1.5),
                                        theta=m.theta * (1 + 0.1 * rng.normal(size=N)).clip(0.5, 1.5) if m.theta.any() else m.theta)
                f2, _ = base_collapse(m2)
                if not f2.get("ok"):
                    continue
                R2 = C.node_sens(m2, f2)
                oa = eval_theta(m, fold, C.topk(R2, k5), d)["comp"]
                og = eval_theta(m, fold, C.topk(A2.sum(0), k5), d)["comp"]
                mis.append(dict(sigma=sig, adj_hat=oa, gbb_hat=og, adj_true=o_true, rho=spearmanr(R2, R)[0]))
        out["misspec"] = mis
    out["runtime_s"] = time.time() - t0
    return out


if __name__ == "__main__":
    outdir = os.path.join(HERE, "results2")
    os.makedirs(outdir, exist_ok=True)
    tasks = build_tasks()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    todo = [t for t in tasks if (only is None or t["label"].startswith(only))]
    todo.sort(key=lambda t: (-len(t["A"]) * (3 if t.get("sub") else 1), t["label"]))
    for t in todo:
        fn = os.path.join(outdir, t["label"] + ".pkl")
        lock = fn + ".lock"
        if os.path.exists(fn):
            continue
        try:
            fdl = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fdl)
        except FileExistsError:
            continue
        try:
            res = analyze(t)
        except Exception:
            res = dict(label=t["label"], group=t["group"], error=traceback.format_exc())
        with open(fn, "wb") as fh:
            pickle.dump(res, fh)
        print(t["label"], res.get("fold_ok"), round(res.get("runtime_s", -1), 1), res.get("error", "")[-300:], flush=True)
