"""Explicit schedule verification for every geometry, budget, policy and test frame, and the worked example."""
import json
import numpy as np
from multiprocessing import Pool
from experiment import run_geometry, POLICIES
from schedule import edge_table, check_frames, replicas, rf_interval, access_interval
from core import matching_assign
from paths import RESULTS


def job(args):
    stream, gi, hetero = args
    res, (geo, reg, ctx, reserv, F_train, F_test) = run_geometry(stream, gi, hetero=hetero, keep_frames=True)
    E = edge_table(geo)
    tot = {}
    worst = {"snr_margin": np.inf, "arrival_slack": np.inf, "completion": 0.0}
    npairs = 0
    for b, pols in reserv.items():
        for pol, R in pols.items():
            v, w, n = check_frames(geo, R, F_test, etab=E)
            npairs += n
            for k, x in v.items():
                tot[k] = tot.get(k, 0) + x
            worst["snr_margin"] = min(worst["snr_margin"], w["snr_margin"])
            worst["arrival_slack"] = min(worst["arrival_slack"], w["arrival_slack"])
            worst["completion"] = max(worst["completion"], w["completion"])
    return {"stream": stream, "gi": gi, "viol": tot, "worst": worst, "pairs": npairs,
            "edges_all": {"min_snr_act": float(E[:, 5].min()), "min_act_minus_aged": float((E[:, 5] - E[:, 6]).min()),
                          "max_arrival": float(E[:, 7].max()), "max_completion": float(E[:, 9].max())},
            "replicas24": sum(len(s) for s in replicas(geo, reserv[24]["portfolio"]).values()),
            "sats24": len(replicas(geo, reserv[24]["portfolio"]))}


def demo(gi=0, b=24):
    res, (geo, reg, ctx, reserv, F_train, F_test) = run_geometry("het", gi, keep_frames=True)
    R = reserv[b]["portfolio"]
    E = {(int(e[0]), int(e[1])): e for e in edge_table(geo)}
    rep = replicas(geo, R)
    nf = F_test.sum(1)
    m = np.array([ctx.match(F_test[i:i + 1], R)[0] for i in range(len(F_test))])
    # first frame with 6-9 failures and at least one unrescued flow (shows a binding neighbourhood)
    cand = [i for i in range(len(F_test)) if 6 <= nf[i] <= 9 and m[i] < nf[i]]
    fi = cand[0]
    f = F_test[fi]
    mt = matching_assign(f, np.asarray(R, np.int64), geo.tptr, geo.tflw, geo.T)
    rows = []
    for r in sorted(R):
        k = int(mt[r])
        if k < 0:
            continue
        e = E[(k, r)]
        rows.append({"flow": k + 1, "token": int(r) + 1, "cell": int(r // 6) + 1, "rank": int((r % 6) // 2) + 1,
                     "slot": int(r % 2) + 1, "sat": int(geo.cand_sat[r]), "rf": [round(e[3], 2), round(e[4], 2)],
                     "snr_act": round(e[5], 2), "snr_aged": round(e[6], 2), "arrival": round(e[7], 2),
                     "access": [round(e[8], 2), round(e[9], 2)], "replica_on_sat": k in rep[int(geo.cand_sat[r])]})
    failed = [int(k) + 1 for k in np.where(f)[0]]
    served = [x["flow"] for x in rows]
    return {"gi": gi + 1, "b": b, "frame": fi, "R": [int(r) + 1 for r in sorted(R)], "n_sats": len(rep),
            "replicas": {str(j): sorted(int(k) + 1 for k in s) for j, s in rep.items()},
            "n_replicas": sum(len(s) for s in rep.values()), "failed": failed, "served": served,
            "unserved": [k for k in failed if k not in served], "rows": rows,
            "token_cells": {str(int(r) + 1): int(r // 6) + 1 for r in R}}


if __name__ == "__main__":
    tasks = [("het", gi, True) for gi in range(24)] + [("hom", gi, False) for gi in range(24)]
    with Pool(2) as p:
        out = list(p.imap_unordered(job, tasks))
    d = demo()
    json.dump({"geometries": out, "demo": d}, open(RESULTS / "schedule.json", "w"), indent=1)
    V = {}
    for o in out:
        for k, x in o["viol"].items():
            V[k] = V.get(k, 0) + x
    print("violations", V, "pairs", sum(o["pairs"] for o in out))
    print("worst snr margin", min(o["worst"]["snr_margin"] for o in out), "min arrival slack",
          min(o["worst"]["arrival_slack"] for o in out), "max completion", max(o["worst"]["completion"] for o in out))
    print("all-edge min snr", min(o["edges_all"]["min_snr_act"] for o in out), "min act-aged",
          min(o["edges_all"]["min_act_minus_aged"] for o in out), "max arrival", max(o["edges_all"]["max_arrival"] for o in out))
    het = [o for o in out if o["stream"] == "het"]
    print("replicas24 mean", np.mean([o["replicas24"] for o in het]), "sats24 mean", np.mean([o["sats24"] for o in het]))
    print(json.dumps({k: v for k, v in d.items() if k != "rows"})[:1500])
    for r in d["rows"]:
        print(r)
