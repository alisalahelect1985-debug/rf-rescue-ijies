"""Per-geometry experiment used by the main study, the sensitivity study and the extra geometries."""
import time
import numpy as np
from model import Geometry, Regime, WEATHER_P, PMIN
from policies import (Ctx, BUDGETS, MU_GRID, exact_anchor, greedy_W, greedy_hybrid, degree_first, need_based,
                      portfolio)

MASTER = 20261001
N_TRAIN, N_VAL, N_CLOUD, N_TEST = 128, 512, 128, 2048
POLICIES = ["anchor", "condrob", "indep", "hybrid1", "portfolio", "cloud", "degree", "need"]


def stream_seed(stream):
    # stable (python hash is salted) -> explicit table
    return {"het": 1, "hom": 2, "extra": 3, "radius": 4, "inter": 5}[stream]


def rngs(stream, gi):
    ss = np.random.SeedSequence([MASTER, stream_seed(stream), gi])
    return [np.random.default_rng(k) for k in ss.spawn(6)]


def run_geometry(stream, gi, hetero=True, pmin=PMIN, gamma_req=8.0, allowances=(0.005,), K=60, budgets=BUDGETS,
                 n_test=N_TEST, ablation=False, test_variants=None, keep_frames=False, geo_stream=None):
    t_all = time.time()
    r_geo, r_train, r_cloud, r_val, r_test, r_abl = rngs(geo_stream or stream, gi)
    geo = Geometry(r_geo, K=K, gamma_req=gamma_req)
    reg = Regime(geo, hetero=hetero, pmin=pmin)
    ctx = Ctx(geo, reg.p)
    F_train = reg.indep(r_train, N_TRAIN)
    F_val = reg.indep(r_val, N_VAL)
    F_cloud = reg.field(r_cloud, N_CLOUD)
    bmax = max(budgets)

    t0 = time.time()
    order_indep = greedy_hybrid(ctx, bmax, F_train, np.inf)
    order_hyb = {mu: (greedy_W(ctx, bmax) if mu == 0.0 else greedy_hybrid(ctx, bmax, F_train, mu)) for mu in MU_GRID}
    anchors = {b: exact_anchor(ctx, b) for b in budgets}
    port = {}
    for a in allowances:
        for b in budgets:
            fam = {"indep": order_indep[:b]}
            for mu in MU_GRID:
                fam["hyb%g" % mu] = order_hyb[mu][:b]
            port[(a, b)] = portfolio(ctx, b, anchors[b], fam, F_val, allowance=a)
    t_port = time.time() - t0

    order_cond = greedy_W(ctx, bmax, list(reg.ps), list(WEATHER_P))
    order_cloud = greedy_hybrid(ctx, bmax, F_cloud, np.inf)

    res = {"gi": gi, "p": reg.p.tolist(), "n_edges": int(len(geo.edges)), "t_port": t_port, "budgets": list(budgets),
           "cand_snr_edge_min": float(np.nanmin(geo.cand_snr_edge)) if np.isfinite(geo.cand_snr_edge).any() else None,
           "n_qual_tokens": int((geo.cand_sat >= 0).sum())}
    a0 = allowances[0]
    reserv = {}
    for b in budgets:
        reserv[b] = {"anchor": anchors[b], "condrob": order_cond[:b], "indep": order_indep[:b],
                     "hybrid1": order_hyb[1.0][:b], "portfolio": port[(a0, b)][0], "cloud": order_cloud[:b],
                     "degree": degree_first(ctx, b), "need": need_based(ctx, b)}
    # certificates
    cert, certc = {}, {}
    for b in budgets:
        for pol, R in reserv[b].items():
            cert[(pol, b)] = ctx.J(R)
            certc[(pol, b)] = float(sum(w * ctx.J(R, ps) for w, ps in zip(WEATHER_P, reg.ps)))
    # test frames (complete field)
    F_test = reg.field(r_test, n_test)
    t1 = time.time()
    deliv = {}
    for b in budgets:
        for pol, R in reserv[b].items():
            deliv[(pol, b)] = float(ctx.delivery(F_test, R).mean())
    t_match = (time.time() - t1) / (n_test * len(budgets) * len(POLICIES))
    res["t_match_frame"] = t_match
    res["deliv"] = {"%s|%d" % k: v for k, v in deliv.items()}
    res["cert"] = {"%s|%d" % k: v for k, v in cert.items()}
    res["certc"] = {"%s|%d" % k: v for k, v in certc.items()}
    res["jmin"] = {str(b): port[(a0, b)][2]["jmin"] for b in budgets}
    res["n_distinct"] = {str(b): port[(a0, b)][2]["n_distinct"] for b in budgets}
    res["n_feasible"] = {str(b): port[(a0, b)][2]["n_feasible"] for b in budgets}
    res["port_choice"] = {str(b): port[(a0, b)][1] for b in budgets}
    res["reserv"] = {str(b): {k: [int(x) for x in v] for k, v in reserv[b].items()} for b in budgets}
    res["replicas"] = {str(b): int(sum(geo.deg[r] for r in reserv[b]["portfolio"])) for b in budgets}
    # other allowances (selection only)
    res["allow"] = {}
    for a in allowances:
        for b in budgets:
            R, name, info = port[(a, b)]
            res["allow"]["%g|%d" % (a, b)] = {
                "deliv": float(ctx.delivery(F_test, R).mean()), "cert": ctx.J(R), "n_feasible": info["n_feasible"],
                "jmin": info["jmin"], "indep_viol": bool(cert[("indep", b)] < info["jmin"] - 1e-12),
                "jacc": len(set(R) & set(reserv[b]["portfolio"])) / len(set(R) | set(reserv[b]["portfolio"]))}
    if ablation:
        F_ind = reg.indep(r_abl, n_test)
        F_cin = reg.cond_indep(r_abl, n_test)
        res["abl"] = {}
        for b in budgets:
            R = reserv[b]["portfolio"]
            res["abl"][str(b)] = {"indep": float(1 - ctx.delivery(F_ind, R).mean()),
                                  "cond": float(1 - ctx.delivery(F_cin, R).mean()),
                                  "field": float(1 - deliv[("portfolio", b)])}
    if test_variants:
        res["variants"] = {}
        for name, kw in test_variants.items():
            rv = np.random.default_rng(np.random.SeedSequence([MASTER, 99, gi, len(name)]))
            Fv = reg.field(rv, n_test, **kw)
            res["variants"][name] = {str(b): {pol: float(ctx.delivery(Fv, reserv[b][pol]).mean()) for pol in ("portfolio", "indep")}
                                     for b in budgets}
    res["t_total"] = time.time() - t_all
    if keep_frames:
        return res, (geo, reg, ctx, reserv, F_train, F_test)
    return res
