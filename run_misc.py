"""Radius study, six-flow exhaustive references, hybrid examples, runtime."""
import itertools, json, time
import numpy as np
from core import csr_from_edges, batch_matching
from policies import Ctx, greedy_W, greedy_hybrid, exact_anchor
from model import Geometry, Regime
from experiment import rngs, run_geometry
from smallgraph import G

OUT = "/home/claude/rfrescue/results"
out = {}
rng = np.random.default_rng(np.random.SeedSequence([20261001, 88]))
SETS6 = np.array([[(m >> k) & 1 for k in range(6)] for m in range(64)], bool)


def exact_nominal(g, p, R):
    pr = np.prod(np.where(SETS6, p[None, :], 1 - p[None, :]), axis=1)
    return float(pr @ batch_matching(SETS6, np.asarray(R, np.int64), g.tptr, g.tflw, g.T))


# ---- six-flow exhaustive references
six = {}
for regime in ("hom", "het"):
    binding, cost, gain = 0, [], []
    for i in range(100):
        while True:
            E = [(k, r) for k in range(6) for r in range(8) if rng.uniform() < 0.35]
            g = G(6, 8, E)
            if (g.deg > 0).all():
                break
        p = np.full(6, 0.2) if regime == "hom" else rng.uniform(0.05, 0.6, 6)
        ctx = Ctx(g, p)
        subs = [list(c) for c in itertools.combinations(range(8), 3)]
        nom = np.array([exact_nominal(g, p, R) for R in subs])
        J = np.array([ctx.J(R) for R in subs])
        floor = J.max() - 0.01
        io = int(np.lexsort((-J, -nom))[0])  # nominal optimum, ties -> higher certificate
        if J[io] < floor - 1e-12:
            binding += 1
            ok = np.where(J >= floor - 1e-12)[0]
            ic = ok[np.argmax(nom[ok])]
            cost.append((nom[io] - nom[ic]) / 6 * 100)
            gain.append((J[ic] - J[io]) * 100)
    six[regime] = {"binding": binding, "mean_cost": float(np.mean(cost)) if cost else 0.0,
                   "mean_gain": float(np.mean(gain)) if gain else 0.0, "max_gain": float(np.max(gain)) if gain else 0.0}
out["six"] = six
print("six", six)

# ---- hybrid tie example
g = G(6, 3, [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (4, 1), (2, 2), (3, 2), (5, 2)])
p = np.full(6, 0.2)
ctx = Ctx(g, p)
out["tie"] = {"nom_12": exact_nominal(g, p, [0, 1]), "nom_23": exact_nominal(g, p, [1, 2]),
              "W_12": ctx.W([0, 1]), "W_23": ctx.W([1, 2])}
print("tie", out["tie"])


# ---- search for a non-monotone greedy certificate in mu (exact nominal objective)
def hybrid_exact(g, p, b, mu):
    ctx = Ctx(g, p)
    sel = []
    for _ in range(b):
        best, br = -1, None
        for r in range(g.T):
            if r in sel:
                continue
            v = ctx.W(sel + [r]) + mu * exact_nominal(g, p, sel + [r])
            if v > best + 1e-12:
                best, br = v, r
        sel.append(br)
    return sel


found = None
mus = np.round(np.arange(0, 2.0001, 0.02), 2)
for it in range(4000):
    T = int(rng.integers(4, 8)); b = int(rng.integers(2, 4))
    E = [(k, r) for k in range(6) for r in range(T) if rng.uniform() < 0.4]
    if not E:
        continue
    g = G(6, T, E)
    p = np.round(rng.uniform(0.05, 0.7, 6), 2)
    ctx = Ctx(g, p)
    Js = [ctx.J(hybrid_exact(g, p, b, mu)) for mu in mus]
    for i in range(len(Js)):
        for j in range(i + 1, len(Js)):
            if Js[j] < Js[i] - 1e-9:
                for k in range(j + 1, len(Js)):
                    if Js[k] > Js[j] + 1e-9:
                        found = {"T": T, "b": b, "p": p.tolist(), "edges": [list(map(int, e)) for e in E],
                                 "mu": [float(mus[i]), float(mus[j]), float(mus[k])], "J": [Js[i], Js[j], Js[k]]}
                        break
                if found: break
        if found: break
    if found:
        break
out["nonmono"] = found
print("nonmono", found)

# ---- radius study (12 heterogeneous geometries, b = 24 portfolio reservation)
het = json.load(open(f"{OUT}/het.json"))
scales = [0.5, 1.0, 2.0, 4.0]
dl = {s: [] for s in scales}
mdiff = 0.0
for gi in range(12):
    geo = Geometry(rngs("het", gi)[0])
    R = het[gi]["reserv"]["24"]["portfolio"]
    base = Regime(geo, True).p
    for s in scales:
        reg = Regime(geo, True, radius_scale=s)
        mdiff = max(mdiff, float(np.abs(reg.p - base).max()))
        ctx = Ctx(geo, reg.p)
        F = reg.field(np.random.default_rng(np.random.SeedSequence([20261001, 66, gi])), 2048)
        dl[s].append(ctx.delivery(F, R).mean() * 100)
from scipy import stats
hw = lambda x: stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x))
out["radius"] = {str(s): (float(np.mean(v)), float(hw(v))) for s, v in dl.items()}
out["radius_mdiff"] = mdiff
print("radius", out["radius"], mdiff)

# ---- runtime (single process, warmed)
run_geometry("het", 0, budgets=[6, 12])
tp = {"het": [], "hom": []}
for name, hetero in (("het", True), ("hom", False)):
    for gi in range(24):
        r = run_geometry(name, gi, hetero=hetero)
        tp[name].append(r["t_port"])
geo = Geometry(rngs("het", 0)[0]); reg = Regime(geo, True); ctx = Ctx(geo, reg.p)
R = het[0]["reserv"]["18"]["portfolio"]
F = reg.field(np.random.default_rng(1), 2048)
ctx.match(F, R)
ts = []
for _ in range(20):
    t0 = time.perf_counter(); ctx.match(F, R); ts.append((time.perf_counter() - t0) / 2048)
out["runtime"] = {"port_het": float(np.median(tp["het"])), "port_hom": float(np.median(tp["hom"])), "match_us": float(np.median(ts) * 1e6)}
print("runtime", out["runtime"])
json.dump(out, open(f"{OUT}/misc.json", "w"), indent=1)
