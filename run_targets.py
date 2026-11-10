import json, sys, time
import numpy as np
from multiprocessing import Pool
from experiment import run_geometry, rngs, N_TRAIN, N_VAL
from policies import portfolio, exact_floor, exact_anchor, greedy_hybrid, greedy_W, MU_GRID
from model import Geometry, Regime

OUT = "/home/claude/rfrescue/results"
TARGETS = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.86, 0.87, 0.88, 0.89, 0.90]


def targets_job(gi):
    res, (geo, reg, ctx, reserv, F_train, F_test) = run_geometry("het", gi, budgets=[24], keep_frames=True)
    r_geo, r_train, r_cloud, r_val, r_test, r_abl = rngs("het", gi)
    # regenerate validation set identically
    Geometry(r_geo); reg.indep(r_train, N_TRAIN)
    F_val = reg.indep(r_val, N_VAL)
    b = 24
    anchor = reserv[b]["anchor"]
    Ja = ctx.J(anchor)
    R_ind = reserv[b]["indep"]
    J_ind = ctx.J(R_ind)
    d_ind = ctx.delivery(F_test, R_ind).mean()
    order_indep = greedy_hybrid(ctx, b, F_train, np.inf)
    fam = {"indep": order_indep[:b]}
    for mu in MU_GRID:
        fam["hyb%g" % mu] = (greedy_W(ctx, b) if mu == 0 else greedy_hybrid(ctx, b, F_train, mu))[:b]
    out = {"gi": gi, "Ja": Ja, "J_ind": J_ind, "t": {}}
    for tg in TARGETS:
        rec = {"feasible": bool(Ja >= tg - 1e-12), "active": bool(J_ind < tg - 1e-12)}
        if rec["feasible"]:
            Rp, name, info = portfolio(ctx, b, anchor, fam, F_val, jmin=tg)
            t0 = time.time()
            Re, sol = exact_floor(ctx, b, F_train, tg)
            rec.update({"d_port": float(ctx.delivery(F_test, Rp).mean()), "d_ind": float(d_ind),
                        "d_exact": float(ctx.delivery(F_test, Re).mean()), "J_exact": ctx.J(Re),
                        "train_port": float(ctx.match(F_train, Rp).mean()), "train_exact": float(ctx.match(F_train, Re).mean()),
                        "mip_gap": float(getattr(sol, "mip_gap", np.nan) or 0.0), "mip_status": int(sol.status), "t_mip": time.time() - t0})
        out["t"][str(tg)] = rec
    return out


def inter_job(args):
    K, gi = args
    b = int(round(24 * K / 60))
    res, (geo, reg, ctx, reserv, F_train, F_test) = run_geometry("inter", gi + 1000 * K, K=K, budgets=[b], keep_frames=True)
    jmin = res["jmin"][str(b)]
    Rp = reserv[b]["portfolio"]
    t0 = time.time()
    Re, sol = exact_floor(ctx, b, F_train, jmin)
    return {"K": K, "gi": gi, "b": b, "d_port": float(ctx.delivery(F_test, Rp).mean()),
            "d_exact": float(ctx.delivery(F_test, Re).mean()), "train_port": float(ctx.match(F_train, Rp).mean()),
            "train_exact": float(ctx.match(F_train, Re).mean()), "J_exact": ctx.J(Re), "jmin": jmin,
            "status": int(sol.status), "t": time.time() - t0}


if __name__ == "__main__":
    mode = sys.argv[2] if len(sys.argv) > 2 else "targets"
    t = time.time()
    with Pool(2) as pool:
        if mode == "targets":
            out = list(pool.imap_unordered(targets_job, range(24)))
            json.dump(out, open(f"{OUT}/targets.json", "w"))
        else:
            out = list(pool.imap_unordered(inter_job, [(K, gi) for K in (12, 20, 30) for gi in range(20)]))
            json.dump(out, open(f"{OUT}/inter.json", "w"))
    print("done", time.time() - t)
