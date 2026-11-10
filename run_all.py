import json, os, sys, time
from multiprocessing import Pool
from experiment import run_geometry
from model import PMIN

OUT = "/home/claude/rfrescue/results"
os.makedirs(OUT, exist_ok=True)
ALLOW = (0.005, 0.0025, 0.01, 0.02)
VARIANTS = {"resid03": {"resid_rho": 0.3}, "markov09": {"markov": 0.9}}

JOBS = {
    "het": dict(stream="het", hetero=True, allowances=ALLOW, ablation=True, test_variants=VARIANTS),
    "hom": dict(stream="hom", hetero=False),
    "extra": dict(stream="extra", hetero=True),
    "pmin05": dict(stream="het", hetero=True, pmin=0.5 * PMIN),
    "pmin2": dict(stream="het", hetero=True, pmin=2 * PMIN),
    "gam6": dict(stream="het", hetero=True, gamma_req=6.0),
    "gam10": dict(stream="het", hetero=True, gamma_req=10.0),
}


def work(args):
    name, gi = args
    return name, gi, run_geometry(gi=gi, **JOBS[name])


if __name__ == "__main__":
    names = sys.argv[1:] or list(JOBS)
    tasks = [(n, gi) for n in names for gi in range(24)]
    t = time.time()
    out = {n: [None] * 24 for n in names}
    with Pool(2) as pool:
        for n, gi, r in pool.imap_unordered(work, tasks):
            out[n][gi] = r
    for n in names:
        json.dump(out[n], open(f"{OUT}/{n}.json", "w"))
    print("done", time.time() - t)
