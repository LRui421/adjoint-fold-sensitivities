"""core2 -- models, certified saddle-node folds, fold tracking along intervention paths,
adjoint fold sensitivities and closures (revision code for the adjoint-fold-sensitivity paper).

Conventions
-----------
A[i, j] >= 0 is the weight with which node j acts on node i;  s_in = A.sum(1), s_out = A.sum(0).
Stress parameter: global coupling scale beta.  theta_i: additive input rate on node i.
A fold is certified by solving the Moore-Spence (minimally extended) system
    f(x, beta) = 0,   J(x, beta) v = 0,   c^T v = 1
with Newton's method and analytic second derivatives.
"""
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp


class Mutualistic:
    """Obligate mutualism: dx_i/dt = theta_i - mu_i x_i - x_i^2/K + beta sum_j A_ij x_i x_j/(D+E x_i+H x_j)."""
    name = "mutualistic"
    xhigh0 = 5.0
    positive = True

    def __init__(self, A, theta=0.1, mu=1.0, K=5.0, D=5.0, E=0.9, H=0.1):
        self.A = np.asarray(A, float)
        self.N = self.A.shape[0]
        self.theta = np.full(self.N, float(theta)) if np.isscalar(theta) else np.asarray(theta, float).copy()
        self.mu = np.full(self.N, float(mu)) if np.isscalar(mu) else np.asarray(mu, float).copy()
        self.K, self.D, self.E, self.H = K, D, E, H
        I, J = np.nonzero(self.A)
        self.I, self.J, self.w = I, J, self.A[I, J]

    def copy(self, A=None, theta=None, mu=None):
        return Mutualistic(self.A if A is None else A, self.theta if theta is None else theta,
                           self.mu if mu is None else mu, self.K, self.D, self.E, self.H)

    def _g(self, xi, xj):
        den = self.D + self.E * xi + self.H * xj
        G = xi * xj / den
        G1 = xj * (self.D + self.H * xj) / den ** 2
        G2 = xi * (self.D + self.E * xi) / den ** 2
        return den, G, G1, G2

    def _g2(self, xi, xj, den):
        G11 = -2 * self.E * xj * (self.D + self.H * xj) / den ** 3
        G22 = -2 * self.H * xi * (self.D + self.E * xi) / den ** 3
        G12 = (self.D ** 2 + self.D * self.E * xi + self.D * self.H * xj + 2 * self.E * self.H * xi * xj) / den ** 3
        return G11, G12, G22

    def Gedge(self, x):
        return self._g(x[self.I], x[self.J])[1]

    def fbeta(self, x):
        return np.bincount(self.I, weights=self.w * self.Gedge(x), minlength=self.N)

    def f(self, x, beta):
        return self.theta - self.mu * x - x ** 2 / self.K + beta * self.fbeta(x)

    def jac(self, x, beta):
        xi, xj = x[self.I], x[self.J]
        den, G, G1, G2 = self._g(xi, xj)
        M = np.zeros((self.N, self.N))
        M[self.I, self.J] = beta * self.w * G2
        M[np.diag_indices(self.N)] += -self.mu - 2 * x / self.K + beta * np.bincount(self.I, weights=self.w * G1, minlength=self.N)
        return M

    def jbeta_v(self, x, v):
        den, G, G1, G2 = self._g(x[self.I], x[self.J])
        return np.bincount(self.I, weights=self.w * (G1 * v[self.I] + G2 * v[self.J]), minlength=self.N)

    def hess_v(self, x, beta, v):
        """H[i,k] = d (J(x) v)_i / d x_k."""
        xi, xj = x[self.I], x[self.J]
        den, G, G1, G2 = self._g(xi, xj)
        G11, G12, G22 = self._g2(xi, xj, den)
        vi, vj = v[self.I], v[self.J]
        M = np.zeros((self.N, self.N))
        M[self.I, self.J] += beta * self.w * (G12 * vi + G22 * vj)
        M[np.diag_indices(self.N)] += -2.0 / self.K * v + beta * np.bincount(self.I, weights=self.w * (G11 * vi + G12 * vj), minlength=self.N)
        return M

    def loss_flux(self, x):
        return self.mu * x + x ** 2 / self.K


class Regulatory:
    """Hill regulation: dx_i/dt = theta_i - B_i x_i + beta sum_j A_ij H(x_j), H = x^2/(1+x^2)."""
    name = "regulatory"
    xhigh0 = 20.0
    positive = True

    def __init__(self, A, theta=0.0, B=1.0, h=2.0):
        self.A = np.asarray(A, float)
        self.N = self.A.shape[0]
        self.theta = np.full(self.N, float(theta)) if np.isscalar(theta) else np.asarray(theta, float).copy()
        self.B = np.full(self.N, float(B)) if np.isscalar(B) else np.asarray(B, float).copy()
        self.h = h
        self.mu = self.B

    def copy(self, A=None, theta=None, mu=None):
        return Regulatory(self.A if A is None else A, self.theta if theta is None else theta,
                          self.B if mu is None else mu, self.h)

    def Hf(self, x):
        xp = np.abs(x) ** self.h
        return xp / (1 + xp)

    def dH(self, x):
        ax = np.abs(x)
        return self.h * ax ** (self.h - 1) / (1 + ax ** self.h) ** 2

    def d2H(self, x):
        return (2 - 6 * x ** 2) / (1 + x ** 2) ** 3

    def fbeta(self, x):
        return self.A @ self.Hf(x)

    def f(self, x, beta):
        return self.theta - self.B * x + beta * self.fbeta(x)

    def jac(self, x, beta):
        J = beta * self.A * self.dH(x)[None, :]
        J[np.diag_indices(self.N)] -= self.B
        return J

    def jbeta_v(self, x, v):
        return self.A @ (self.dH(x) * v)

    def hess_v(self, x, beta, v):
        return beta * self.A * (self.d2H(x) * v)[None, :]

    def loss_flux(self, x):
        return self.B * x


def edge_G(model, x, I, J):
    if model.name == "mutualistic":
        return model._g(x[I], x[J])[1]
    return model.Hf(x[J])


# ---------------------------------------------------------------- equilibria and folds
def newton(model, x0, beta, tol=1e-12, maxit=30, nls=10):
    x = np.array(x0, float)
    r = model.f(x, beta)
    for _ in range(maxit):
        nr = np.linalg.norm(r)
        if not np.isfinite(nr):
            return x, False
        if nr < tol * (1 + np.linalg.norm(x)):
            return x, True
        try:
            dx = np.linalg.solve(model.jac(x, beta), -r)
        except np.linalg.LinAlgError:
            return x, False
        lam = 1.0
        for _ in range(nls):
            xn = x + lam * dx
            rn = model.f(xn, beta)
            if np.linalg.norm(rn) < (1 - 1e-4 * lam) * nr:
                break
            lam *= 0.5
        x, r = xn, rn
    return x, bool(np.linalg.norm(r) < 1e-8 * (1 + np.linalg.norm(x)))


def integrate(model, x0, beta, T=3000.0):
    sol = solve_ivp(lambda t, x: model.f(x, beta), (0, T), np.array(x0, float), method="LSODA",
                    jac=lambda t, x: model.jac(x, beta), rtol=1e-9, atol=1e-11)
    return sol.y[:, -1]


def lam_sorted(J):
    ev = np.linalg.eigvals(J)
    return ev[np.argsort(-ev.real)]


def natural_continuation(model, x0, beta0, direction=-1, h0=0.02, rtol=1e-6, max_steps=4000, jump_tol=0.25):
    x, ok = newton(model, x0, beta0)
    if not ok or lam_sorted(model.jac(x, beta0))[0].real >= 0:
        return None
    beta, h, n = beta0, h0 * beta0, 0
    while h > rtol * beta0 and n < max_steps:
        n += 1
        J = model.jac(x, beta)
        try:
            dxdb = -np.linalg.solve(J, model.fbeta(x))
        except np.linalg.LinAlgError:
            dxdb = np.zeros_like(x)
        bn = beta + direction * h
        if bn <= 1e-9 * beta0:
            h *= 0.5
            continue
        xp = x + dxdb * direction * h
        xn, ok = newton(model, xp, bn, maxit=8, nls=2)
        acc = False
        if ok and np.all(np.isfinite(xn)) and (not model.positive or xn.min() > -1e-10):
            if lam_sorted(model.jac(xn, bn))[0].real < 0:
                if np.linalg.norm(xn - xp) < jump_tol * (np.linalg.norm(xn - x) + 1e-3 * np.linalg.norm(x) + 1e-12) + 1e-9:
                    acc = True
        if acc:
            x, beta = xn, bn
            h = min(h * 1.3, h0 * beta0)
        else:
            h *= 0.5
    return dict(x=x, beta=beta, steps=n, hit_zero=beta <= 1e-6 * beta0)


def crit_vectors(J):
    try:
        U, s, Vt = np.linalg.svd(J)
    except np.linalg.LinAlgError:
        import scipy.linalg as sla
        U, s, Vt = sla.svd(J, lapack_driver="gesvd")
    v, u = Vt[-1], U[:, -1]
    if v.sum() < 0:
        v = -v
    if u.sum() < 0:
        u = -u
    return u, v, s[-1], s[0]


def moore_spence(model, x0, beta0, v0, tol=1e-11, maxit=25):
    """Newton on the minimally extended system (f=0, Jv=0, c^T v=1)."""
    N = model.N
    c = v0 / np.linalg.norm(v0)
    x, beta, v = np.array(x0, float), float(beta0), v0 / (c @ v0)
    for it in range(maxit):
        f = model.f(x, beta)
        J = model.jac(x, beta)
        R = np.concatenate([f, J @ v, [c @ v - 1.0]])
        if it > 0 and np.linalg.norm(R) < tol * (1 + np.linalg.norm(x)):
            return dict(ok=True, x=x, beta=beta, v=v, it=it, res=float(np.linalg.norm(R)))
        M = np.zeros((2 * N + 1, 2 * N + 1))
        M[:N, :N] = J
        M[:N, N] = model.fbeta(x)
        M[N:2 * N, :N] = model.hess_v(x, beta, v)
        M[N:2 * N, N] = model.jbeta_v(x, v)
        M[N:2 * N, N + 1:] = J
        M[2 * N, N + 1:] = c
        try:
            d = np.linalg.solve(M, -R)
        except np.linalg.LinAlgError:
            return dict(ok=False, reason="singular")
        x, beta, v = x + d[:N], beta + d[N], v + d[N + 1:]
        if not np.all(np.isfinite(x)) or beta <= 0:
            return dict(ok=False, reason="diverged")
    R = np.concatenate([model.f(x, beta), model.jac(x, beta) @ v, [c @ v - 1.0]])
    ok = np.linalg.norm(R) < 1e-8 * (1 + np.linalg.norm(x))
    return dict(ok=bool(ok), x=x, beta=beta, v=v, it=maxit, res=float(np.linalg.norm(R)), reason="noconv")


def fold_diagnostics(model, x, beta, v=None):
    J = model.jac(x, beta)
    u, vs, smin, smax = crit_vectors(J)
    v = vs if v is None else v
    v = v / np.linalg.norm(v)
    if v.sum() < 0:
        v = -v
    u = u / (u @ v)
    fb = model.fbeta(x)
    Bvv = model.hess_v(x, beta, v) @ v
    ev = lam_sorted(J)
    i1 = int(np.argmin(np.abs(ev)))
    others = np.delete(ev, i1)
    un = u / np.abs(u).sum()
    return dict(u=un, u_raw=u, v=v, a=0.5 * float(u @ Bvv), b=float(u @ fb),
                res_f=float(np.max(np.abs(model.f(x, beta)))), res_Jv=float(np.max(np.abs(J @ v))),
                res_uJ=float(np.max(np.abs(u @ J)) / np.max(np.abs(u))), smin_rel=float(smin / smax),
                lam1=float(ev[i1].real), gap=float(-np.max(others.real)) if len(others) else np.nan,
                kappa_eig=float(np.linalg.norm(u) * np.linalg.norm(v) / abs(u @ v)),
                xmin=float(x.min()), u_min=float(un.min()))


def locate_fold(model, x_start, beta_start, direction=-1):
    nc = natural_continuation(model, x_start, beta_start, direction=direction, rtol=1e-6)
    if nc is None:
        return dict(ok=False, reason="start")
    if nc["hit_zero"]:
        return dict(ok=False, reason="no_fold")
    _, v0, _, _ = crit_vectors(model.jac(nc["x"], nc["beta"]))
    ms = moore_spence(model, nc["x"], nc["beta"], v0)
    if not ms.get("ok"):
        return dict(ok=False, reason="ms_" + ms.get("reason", "noconv"), beta_nc=nc["beta"])
    d = fold_diagnostics(model, ms["x"], ms["beta"], ms["v"])
    return dict(ok=True, x=ms["x"], beta=ms["beta"], beta_nc=nc["beta"], it=ms["it"], **d)


def high_state(model, beta, x0=None):
    x0 = np.full(model.N, model.xhigh0) if x0 is None else x0
    return newton(model, integrate(model, x0, beta), beta)


def low_state(model, beta):
    return newton(model, integrate(model, np.zeros(model.N), beta), beta)


def track_fold(model_of_s, fold0, s_end=1.0, ds0=1.0, ds_min=2e-4, window=(0.25, 4.0)):
    """Continue a certified fold along an intervention path s in [0, s_end].
    Each accepted step must (i) solve the Moore-Spence system, (ii) keep the branch free of unstable
    eigenvalues, and (iii) move beta_c consistently with the adjoint first-order prediction
    dbeta_lin = -u.[f(x,beta;s+ds)-f(x,beta;s)]/(u.f_beta) (ratio within `window`), which rejects jumps
    to other folds.  status 'kept' (fold certified at s_end) or 'lost' (the fold ceases to exist:
    the step size collapses before s_end, with the normal-form coefficient a -> 0)."""
    s, x, beta, v = 0.0, fold0["x"], fold0["beta"], fold0["v"]
    u = fold0["u_raw"]
    m_cur = model_of_s(0.0)
    hist = [(0.0, beta)]
    ds, prev = min(ds0, s_end), None
    while s < s_end - 1e-14:
        sn = min(s + ds, s_end)
        m = model_of_s(sn)
        df = m.f(x, beta) - m_cur.f(x, beta)
        dlin = -float(u @ df) / float(u @ m_cur.fbeta(x))
        if prev is not None:
            t = (sn - s) / (s - prev[0])
            xp, bp, vp = x + t * (x - prev[1]), beta + t * (beta - prev[2]), v + t * (v - prev[3])
        else:
            xp, bp, vp = x, beta + dlin, v
        ms = moore_spence(m, xp, bp, vp, maxit=15)
        good = bool(ms.get("ok")) and np.all(np.isfinite(ms["x"])) and ms["x"].min() > -1e-9 and ms["x"].max() < 1e6
        if good:
            db = ms["beta"] - beta
            tol_abs = 1e-9 * beta
            if abs(dlin) > tol_abs:
                r_ = db / dlin
                good = window[0] <= r_ <= window[1]
            else:
                good = abs(db) <= 10 * tol_abs + 1e-3 * abs(dlin)
        if good:
            ev = lam_sorted(m.jac(ms["x"], ms["beta"]))
            good = np.sum(ev.real > 1e-7 * max(1.0, abs(ev[-1].real))) == 0
        if good:
            prev = (s, x, beta, v)
            s, x, beta, v = sn, ms["x"], ms["beta"], ms["v"]
            m_cur = m
            Jn = m.jac(x, beta)
            u_, _, _, _ = crit_vectors(Jn)
            u = u_ / (u_ @ (v / np.linalg.norm(v)))
            hist.append((s, beta))
            ds = min(ds * 1.5, s_end)
        else:
            ds *= 0.5
            if ds < ds_min:
                dg = fold_diagnostics(m_cur, x, beta, v)
                return dict(status="lost", s_star=s, beta=beta, x=x, v=v, a_last=dg["a"], hist=hist, u=dg["u"])
    dg = fold_diagnostics(m_cur, x, beta, v)
    return dict(status="kept", beta=beta, x=x, hist=hist, **dg)


# ---------------------------------------------------------------- sensitivities
def node_sens(model, fold):
    u = fold["u_raw"]
    return u / (u @ model.fbeta(fold["x"]))


def lever_sens_mortality(model, fold):
    u = fold["u_raw"]
    return u * fold["x"] / (u @ model.fbeta(fold["x"]))


def link_sens(model, fold, I, J):
    u, x, b = fold["u_raw"], fold["x"], fold["beta"]
    return b * u[I] * edge_G(model, x, I, J) / (u @ model.fbeta(x))


# ---------------------------------------------------------------- node-resolved closure
class Closure:
    """mutualistic: g_i = theta_i - mu_i p_i - p_i^2/K + beta s_in_i G(p_i, m);  g_m = sum_k w_k p_k - m
    regulatory : g_i = theta_i - B_i p_i + beta s_in_i m;                    g_m = sum_k w_k H(p_k) - m"""
    positive = False
    name = "closure"

    def __init__(self, full):
        self.full = full
        self.s_in = full.A.sum(1)
        s_out = full.A.sum(0)
        self.w = s_out / s_out.sum()
        self.N = full.N + 1
        self.xhigh0 = full.xhigh0
        self.kind = full.name

    def f(self, z, beta):
        p, m, F = z[:-1], z[-1], self.full
        if self.kind == "mutualistic":
            g = F.theta - F.mu * p - p ** 2 / F.K + beta * self.s_in * F._g(p, m)[1]
            gm = self.w @ p - m
        else:
            g = F.theta - F.B * p + beta * self.s_in * m
            gm = self.w @ F.Hf(p) - m
        return np.append(g, gm)

    def fbeta(self, z):
        p, m, F = z[:-1], z[-1], self.full
        if self.kind == "mutualistic":
            return np.append(self.s_in * F._g(p, m)[1], 0.0)
        return np.append(self.s_in * m, 0.0)

    def jac(self, z, beta):
        p, m, F = z[:-1], z[-1], self.full
        n = F.N
        M = np.zeros((n + 1, n + 1))
        if self.kind == "mutualistic":
            den, G, G1, G2 = F._g(p, m)
            M[np.arange(n), np.arange(n)] = -F.mu - 2 * p / F.K + beta * self.s_in * G1
            M[:n, n] = beta * self.s_in * G2
            M[n, :n] = self.w
        else:
            M[np.arange(n), np.arange(n)] = -F.B
            M[:n, n] = beta * self.s_in
            M[n, :n] = self.w * F.dH(p)
        M[n, n] = -1.0
        return M

    def jbeta_v(self, z, v):
        p, m, F = z[:-1], z[-1], self.full
        n = F.N
        out = np.zeros(n + 1)
        if self.kind == "mutualistic":
            den, G, G1, G2 = F._g(p, m)
            out[:n] = self.s_in * (G1 * v[:n] + G2 * v[n])
        else:
            out[:n] = self.s_in * v[n]
        return out

    def hess_v(self, z, beta, v):
        p, m, F = z[:-1], z[-1], self.full
        n = F.N
        M = np.zeros((n + 1, n + 1))
        if self.kind == "mutualistic":
            den, G, G1, G2 = F._g(p, m)
            G11, G12, G22 = F._g2(p, m, den)
            vi, vm = v[:n], v[n]
            M[np.arange(n), np.arange(n)] = -2.0 / F.K * vi + beta * self.s_in * (G11 * vi + G12 * vm)
            M[:n, n] = beta * self.s_in * (G12 * vi + G22 * vm)
        else:
            M[n, :n] = self.w * F.d2H(p) * v[:n]
        return M


def closure_fold(full, direction=-1, beta_hint=1.0):
    cl = Closure(full)
    b0 = beta_hint
    if direction < 0:
        for _ in range(40):
            z, ok = newton(cl, integrate(cl, np.full(cl.N, cl.xhigh0), b0, T=500.0), b0)
            if ok and z[:-1].mean() > 0.5 and lam_sorted(cl.jac(z, b0))[0].real < 0:
                break
            b0 *= 1.5
    else:
        z, ok = newton(cl, integrate(cl, np.zeros(cl.N), b0, T=500.0), b0)
    r = locate_fold(cl, z, b0, direction=direction)
    if r.get("ok"):
        r["phi"], r["m"] = r["x"][:-1], r["x"][-1]
        y = r["u_raw"]
        r["sens_closure"] = y[:-1] / (y @ cl.fbeta(r["x"]))
    return r


def gbb_quantities(A):
    s_in, s_out, S = A.sum(1), A.sum(0), A.sum()
    return dict(s_in=s_in, s_out=s_out, S=S, w=s_out / S, beta_eff=float(s_out @ s_in / S))


LAMBDA_C = {"mutualistic": 3.9590517, "regulatory": 2.0}


# ---------------------------------------------------------------- centralities
def power_iterates(model, x, beta, ks=(1, 2), lazy=None):
    J = model.jac(x, beta)
    d = -np.diag(J).copy()
    Joff = J.copy()
    np.fill_diagonal(Joff, 0.0)
    T = Joff / d[None, :]
    if lazy is not None:
        T = (1 - lazy) * np.eye(len(d)) + lazy * T
    out, u = {}, np.ones(len(d)) / len(d)
    for k in range(1, max(ks) + 1):
        u = u @ T
        u = u / u.sum()
        if k in ks:
            out[k] = u.copy()
    return out, T


def left_perron(M):
    w, V = np.linalg.eig(M.T)
    k = int(np.argmax(w.real))
    u = np.real(V[:, k])
    u = u * np.sign(u.sum())
    return u / np.abs(u).sum(), w


def right_perron(M):
    w, V = np.linalg.eig(M)
    k = int(np.argmax(w.real))
    v = np.real(V[:, k])
    v = v * np.sign(v.sum())
    return v / np.abs(v).sum()


def structural_scores(A):
    D = nx.from_numpy_array((A.T > 0).astype(int), create_using=nx.DiGraph)
    Gu = D.to_undirected()
    s_out, s_in = A.sum(0), A.sum(1)
    core = nx.core_number(Gu)
    core = np.array([core[i] for i in range(len(A))], float)
    btw = nx.betweenness_centrality(D)
    btw = np.array([btw[i] for i in range(len(A))])
    uA, _ = left_perron(A)
    vA = right_perron(A)
    return dict(out_strength=s_out, in_strength=s_in, eigenvector=np.abs(uA), eig_right=np.abs(vA),
                kcore=core + 1e-6 * s_out / (s_out.max() + 1e-12), betweenness=btw)


def participation(u):
    u = np.abs(u)
    return float(u.sum() ** 2 / (len(u) * (u ** 2).sum()))


def topk(v, k, seed=0):
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(v)) * 1e-12 * (np.abs(v).max() + 1e-300)
    return np.argsort(-(v + jitter))[:k]
