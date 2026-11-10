"""Independent mathematical and probabilistic checks (Section 6.3 and supplementary material)."""
import itertools, json, time
import numpy as np
from scipy.optimize import linprog
from scipy.integrate import dblquad
from core import csr_from_edges, batch_matching
from policies import Ctx, exact_anchor, exact_floor, greedy_W, greedy_hybrid, MU_GRID
from model import phi, Geometry, Regime
from experiment import rngs, run_geometry

OUT = "/home/claude/rfrescue/results"
rng = np.random.default_rng(np.random.SeedSequence([20261001, 77]))


from smallgraph import G


def rand_graph(K, T, pe=0.45):
    while True:
        E = [(k, r) for k in range(K) for r in range(T) if rng.uniform() < pe]
        if E:
            return G(K, T, E)


def nu(g, Fmask_list, R):
    F = np.array(Fmask_list, bool).reshape(-1, g.K)
    return batch_matching(F, np.asarray(R, np.int64), g.tptr, g.tflw, g.T)


def all_sets(K):
    return np.array([[(m >> k) & 1 for k in range(K)] for m in range(2 ** K)], bool)


def deficit_all(g, R):
    S = all_sets(g.K)
    return S, S.sum(1) - nu(g, S, R)


out = {}
# ---- 1. envelope vs LP, nested chain, Hall, supermodularity
t0 = time.time()
md_lp = md_chain = 0.0
n_hall = n_sup = 0
hall_ok = sup_ok = True
for i in range(80):
    K = 2 + i % 6
    T = int(rng.integers(1, K + 1))
    g = rand_graph(K, T)
    p = rng.uniform(0.02, 0.98, K)
    ctx = Ctx(g, p)
    R = list(range(T))
    S, d = deficit_all(g, R)
    env = p.sum() - ctx.W(R)
    # nested chain Eq. (12)
    o = np.argsort(-p)
    pp = np.append(p[o], 0.0)
    chain = 0.0
    for j in range(1, K + 1):
        Sj = np.zeros((1, K), bool); Sj[0, o[:j]] = True
        chain += (pp[j - 1] - pp[j]) * (j - nu(g, Sj, R)[0])
    # LP over all 2^K laws
    A_eq = np.vstack([S.T.astype(float), np.ones(2 ** K)])
    b_eq = np.append(p, 1.0)
    lp = linprog(-d.astype(float), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    md_lp = max(md_lp, abs(-lp.fun - env))
    md_chain = max(md_chain, abs(chain - env))
    # Hall: d(F) = max_{U subset F} (|U| - |N(U)|)^+
    for m in range(2 ** K):
        F = [k for k in range(K) if (m >> k) & 1]
        best = 0
        for sub in range(m + 1):
            if sub & ~m:
                continue
            U = [k for k in range(K) if (sub >> k) & 1]
            NU = set()
            for k in U:
                NU |= set(g.ftok[g.fptr[k]:g.fptr[k + 1]].tolist())
            best = max(best, len(U) - len(NU))
        hall_ok &= (best == d[m]); n_hall += 1
    # supermodularity: d(F+i)+d(F+j) <= d(F+i+j)+d(F)
    for m in range(2 ** K):
        for a in range(K):
            for b in range(a + 1, K):
                if (m >> a) & 1 or (m >> b) & 1:
                    continue
                lhs = d[m | 1 << a] + d[m | 1 << b]
                rhs = d[m | 1 << a | 1 << b] + d[m]
                sup_ok &= (lhs <= rhs); n_sup += 1
out["lp"] = {"max_lp": md_lp, "max_chain": md_chain, "n_hall": n_hall, "hall_ok": bool(hall_ok), "n_sup": n_sup, "sup_ok": bool(sup_ok)}
print("lp", out["lp"], time.time() - t0)

# ---- 2. exact reservations vs exhaustive
md_res = 0.0
for i in range(80):
    K = int(rng.integers(3, 8)); T = int(rng.integers(3, 9)); b = int(rng.integers(1, T))
    g = rand_graph(K, T, 0.4)
    ctx = Ctx(g, rng.uniform(0.02, 0.98, K))
    Ra = exact_anchor(ctx, b)
    best = max(ctx.W(list(c)) for c in itertools.combinations(range(T), b))
    md_res = max(md_res, abs(best - ctx.W(Ra)))
out["res"] = {"n": 80, "max_diff": md_res}
print("res", out["res"])

# ---- 3. floor problems vs exhaustive
md_floor = 0.0
nfl = 0
for i in range(200):
    if nfl >= 24:
        break
    K = int(rng.integers(4, 8)); T = int(rng.integers(4, 9)); b = int(rng.integers(1, T))
    g = rand_graph(K, T, 0.4)
    p = rng.uniform(0.05, 0.6, K)
    ctx = Ctx(g, p)
    F = rng.uniform(size=(24, K)) < p
    Ja = ctx.J(exact_anchor(ctx, b))
    jmin = Ja - 0.01
    Re, sol = exact_floor(ctx, b, F, jmin)
    if Re is None or sol.status != 0:
        continue
    vals = [ctx.match(F, list(c)).mean() for c in itertools.combinations(range(T), b) if ctx.J(list(c)) >= jmin - 1e-12]
    md_floor = max(md_floor, abs(max(vals) - ctx.match(F, Re).mean()))
    nfl += 1
out["floor"] = {"n": nfl, "max_diff": md_floor}
print("floor", out["floor"])

# ---- 4. multistate greedy ratio
ratios = []
for i in range(80):
    K = int(rng.integers(3, 7)); T = int(rng.integers(4, 9)); b = int(rng.integers(1, T))
    gs = [rand_graph(K, T, 0.35) for _ in range(3)]
    w = rng.dirichlet(np.ones(3))
    ps = [rng.uniform(0.02, 0.98, K) for _ in range(3)]
    # multistate objective with state-dependent graphs: sum_s w_s W_{G_s}(R;p_s)
    ctxs = [Ctx(g_, p_) for g_, p_ in zip(gs, ps)]
    Fv = lambda R: sum(wi * c.W(R) for wi, c in zip(w, ctxs))
    sel = []
    for _ in range(b):
        cand = [r for r in range(T) if r not in sel]
        gains = [Fv(sel + [r]) for r in cand]
        sel.append(cand[int(np.argmax(gains))])
    opt = max(Fv(list(c)) for c in itertools.combinations(range(T), b))
    if opt > 1e-12:
        ratios.append(Fv(sel) / opt)
out["multistate"] = {"n": len(ratios), "min_ratio": float(min(ratios))}
print("multistate", out["multistate"])

# ---- 5. four-flow benchmark
g4 = G(4, 2, [(0, 0), (1, 0), (2, 1), (3, 1)])
ctx4 = Ctx(g4, np.full(4, 0.2))
N4 = 250_000
r4 = np.random.default_rng(np.random.SeedSequence([20261001, 44]))
laws = {}
U = r4.uniform(size=(N4, 4)) < 0.2
laws["Independent"] = U
c = r4.uniform(size=(N4, 2)) < 0.2
laws["Aligned groups"] = np.stack([c[:, 0], c[:, 0], c[:, 1], c[:, 1]], 1)
c = r4.uniform(size=(N4, 2)) < 0.2
laws["Interleaved groups"] = np.stack([c[:, 0], c[:, 1], c[:, 0], c[:, 1]], 1)
c = r4.uniform(size=N4) < 0.2
laws["Common cause"] = np.stack([c] * 4, 1)
exact = {"Independent": (2.0, 7.84), "Aligned groups": (10.0, 36.0), "Interleaved groups": (2.0, 4.0), "Common cause": (10.0, 20.0)}
out["four"] = {}
for k, F in laws.items():
    loss = 1 - ctx4.delivery(F, [0, 1])
    m = loss.mean() * 100
    h = 1.96 * loss.std(ddof=1) / np.sqrt(N4) * 100
    out["four"][k] = {"exact": exact[k][0], "sim": m, "hw": h, "anymiss": exact[k][1]}
print("four", out["four"])

# ---- 6. quadrature check of Eq. (6)
md_q = 0.0
for ka in [0.0, 0.05, 0.2, 0.5, 1.0, 2.0, -0.3]:
    a = 1.0
    val, _ = dblquad(lambda r, th: r * np.exp(ka * r * np.cos(th)), 0, 2 * np.pi, 0, a, epsabs=1e-14, epsrel=1e-14)
    md_q = max(md_q, abs(val / (np.pi * a * a) - phi(np.array([ka]))[0]))
out["quad"] = md_q
print("quad", md_q)

# ---- 7. field sampler check (100 000 frames, heterogeneous geometry 0)
r_geo = rngs("het", 0)[0]
geo = Geometry(r_geo)
reg = Regime(geo, True)
rs = np.random.default_rng(np.random.SeedSequence([20261001, 55]))
NF = 100_000
cnt = np.zeros(geo.K)
for i in range(NF // 5000):
    cnt += reg.field(rs, 5000).sum(0)
emp = cnt / NF
margin = np.sqrt(np.log(2 * 60 / 0.05) / (2 * NF))
out["sampler"] = {"max_disc": float(np.abs(emp - reg.p).max()), "margin": float(margin)}
print("sampler", out["sampler"])

json.dump(out, open(f"{OUT}/checks.json", "w"), indent=1)
