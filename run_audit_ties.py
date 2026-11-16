"""Why first-round and released values differ: (a) tie sensitivity of the exact robust anchor,
(b) sampling variability between two independent 24-geometry sets."""
import json
import numpy as np
from multiprocessing import Pool
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
from experiment import run_geometry
from paths import RESULTS


def anchor_tie(ctx, b, w):
    """Exact robust anchor with token tie preference w (smaller = preferred) for both the solver and the fill."""
    g = ctx.g
    E = g.edges; nE, T = len(E), g.T
    c = np.concatenate([-ctx.p[E[:, 0]], 1e-9 * w / (w.max() + 1)])
    A = lil_matrix((g.K + T + 1, nE + T))
    for i, (k, r) in enumerate(E):
        A[k, i] = 1; A[g.K + r, i] = 1
    for r in range(T):
        A[g.K + r, nE + r] = -1; A[g.K + T, nE + r] = 1
    ub = np.concatenate([np.ones(g.K), np.zeros(T), [b]])
    res = milp(c, constraints=LinearConstraint(csr_matrix(A), -np.inf, ub), bounds=Bounds(0, 1),
               integrality=np.concatenate([np.zeros(nE), np.ones(T)]), options={"mip_rel_gap": 0})
    x = res.x[:nE]
    used = sorted(set(E[x > 0.5, 1].tolist()))
    for r in np.argsort(w, kind="stable"):
        if len(used) >= b: break
        if r not in used: used.append(int(r))
    return used[:b]


def job(gi):
    res, (geo, reg, ctx, reserv, F_train, F_test) = run_geometry("het", gi, budgets=[24], keep_frames=True)
    rng = np.random.default_rng(np.random.SeedSequence([20261001, 31, gi]))
    T = geo.T
    rules = {"low_index": np.arange(T, dtype=float), "high_index": np.arange(T, dtype=float)[::-1].copy()}
    for i in range(20):
        rules["random%d" % i] = rng.permutation(T).astype(float)
    out = {}
    J0 = ctx.J(reserv[24]["anchor"])
    for name, w in rules.items():
        R = anchor_tie(ctx, 24, w)
        out[name] = {"deliv": float(ctx.delivery(F_test, R).mean()), "cert": ctx.J(R)}
        assert abs(out[name]["cert"] - J0) < 1e-12
    return gi, out


if __name__ == "__main__":
    with Pool(2) as p:
        res = dict(p.map(job, range(24)))
    names = list(res[0].keys())
    means = {n: 100 * np.mean([res[g][n]["deliv"] for g in range(24)]) for n in names}
    het = json.load(open(RESULTS / "het.json")); ext = json.load(open(RESULTS / "extra.json"))
    samp = {}
    for pol in ("portfolio", "anchor", "indep"):
        a = np.array([r["deliv"]["%s|24" % pol] for r in het]) * 100
        b = np.array([r["deliv"]["%s|24" % pol] for r in ext]) * 100
        samp[pol] = {"set1": a.mean(), "set2": b.mean(), "diff": b.mean() - a.mean(), "sd_geom": a.std(ddof=1)}
    summary = {"anchor_rule_means": means, "min": min(means.values()), "max": max(means.values()),
               "low_index": means["low_index"], "sampling": samp}
    json.dump({"per_geometry": res, "summary": summary}, open(RESULTS / "audit_ties.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
