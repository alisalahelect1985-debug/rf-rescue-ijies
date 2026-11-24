"""Recompute the class-R first-submission values and the revised (v1.1) counterparts listed in first_submission_values.csv."""
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analyze import load, table, violations, paired  # noqa: E402
from model import Geometry, Regime  # noqa: E402
from experiment import rngs  # noqa: E402

ok = True


def check(name, got, want, tol=1e-9):
    global ok
    good = abs(got - want) <= tol
    ok &= good
    print("%-58s %-12s %-12s %s" % (name, "%.6g" % got, "%.6g" % want, "ok" if good else "MISMATCH"))


# class R
p = Regime(Geometry(rngs("hom", 0)[0]), hetero=False).p
check("R1 homogeneous marginal", p[0], 0.146489, 5e-7)
TH = table(load("hom"))
for b, w in ((12, 88.281), (18, 89.746), (24, 91.211)):
    check("R%d homogeneous certificate b=%d" % ({12: 2, 18: 3, 24: 4}[b], b), TH[("portfolio", b)][2], w, 5e-4)
ck = json.load(open(ROOT / "results/checks.json"))
for law, w in (("Independent", 2.0), ("Aligned groups", 10.0), ("Interleaved groups", 2.0), ("Common cause", 10.0)):
    check("R5 four-flow exact loss, " + law, ck["four"][law]["exact"], w)
check("R7 five-flow shared token nominal (%)", 100 * (1 - (1.6 - (1 - 0.75 ** 4)) / 5), 81.672, 5e-4)
check("R7 five-flow shared token certificate (%)", 100 * (1 - (1.6 - 0.25) / 5), 73.0)
mi = json.load(open(ROOT / "results/misc.json"))
check("R9 tie example, pair with token 1", mi["tie"]["nom_12"], 0.91456)
check("R9 tie example, pair without token 1", mi["tie"]["nom_23"], 0.976)
check("R10 conditioning example, unconditional", 1 - (1.0 - 0.5) / 2, 0.75)
check("R10 conditioning example, conditional", 1 - (1.0 - 0.9) / 2, 0.95)
check("R11 Hoeffding margin", ck["sampler"]["margin"], 0.006238, 5e-7)
# revised counterparts (class P), a representative subset
het = load("het")
T = table(het)
check("P20 independence greedy b=24 mean (revised)", T[("indep", 24)][0], 95.32, 5e-3)
check("P22 portfolio b=24 mean (revised)", T[("portfolio", 24)][0], 95.33, 5e-3)
check("P19 anchor b=24 mean (revised)", T[("anchor", 24)][0], 94.37, 5e-3)
for b, w in ((18, 1), (24, 4), (30, 6)):
    check("P33 independence-greedy violations b=%d (revised)" % b, violations(het, "indep", b), w)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
