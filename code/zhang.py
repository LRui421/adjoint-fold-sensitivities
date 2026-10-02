"""zhang -- comparator: node 'resilience centrality' of Zhang et al. (PRE 2020), defined from the change of
beta_eff when a node (and its links) is removed; evaluated as a targeting rule for input protection."""
import os, sys, pickle, time
import numpy as np
from scipy.stats import spearmanr
import core2 as C
import exp2 as X

HERE = os.path.dirname(os.path.abspath(__file__))


def zhang_rc(A):
    b0 = C.gbb_quantities(A)["beta_eff"]
    out = np.zeros(len(A))
    for i in range(len(A)):
        keep = np.r_[0:i, i + 1:len(A)]
        out[i] = b0 - C.gbb_quantities(A[np.ix_(keep, keep)])["beta_eff"]
    return out


if __name__ == "__main__":
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    od = os.path.join(HERE, "results_zhang")
    os.makedirs(od, exist_ok=True)
    tasks = X.build_tasks()[shard::nsh]
    for t in tasks:
        fn = os.path.join(od, t["label"] + ".pkl")
        if os.path.exists(fn):
            continue
        m = X.make_model(t)
        fold, _ = X.base_collapse(m)
        if not fold.get("ok"):
            continue
        R = C.node_sens(m, fold)
        z = zhang_rc(m.A)
        k = X.kof(0.05, m.N)
        res = dict(label=t["label"], group=t["group"], rho=spearmanr(z, R)[0], rho_sout=spearmanr(z, m.A.sum(0))[0],
                   eff1=float(R[C.topk(z, k)].sum() / R[C.topk(R, k)].sum()), ev={})
        for d in X.DOSES[m.name]:
            res["ev"][d] = dict(zhang=X.eval_theta(m, fold, C.topk(z, k), d)["comp"],
                                adjoint=X.eval_theta(m, fold, C.topk(R, k), d)["comp"],
                                GBB_gradient=X.eval_theta(m, fold, C.topk(m.A.sum(0), k), d)["comp"])
        pickle.dump(res, open(fn, "wb"))
        print(t["label"], flush=True)
