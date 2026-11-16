"""Quick reproducibility check (about 1 minute): recompute three geometries and compare with the stored results."""
import json, sys
from experiment import run_geometry
from paths import RESULTS

ok = True
for stream, gi, hetero in (("het", 0, True), ("het", 17, True), ("hom", 5, False)):
    kw = dict(allowances=(0.005, 0.0025, 0.01, 0.02), ablation=True) if stream == "het" else {}
    r = run_geometry(stream, gi, hetero=hetero, **kw)
    s = json.load(open(RESULTS / f"{stream}.json"))[gi]
    same = r["reserv"] == s["reserv"] and all(abs(r["deliv"][k] - s["deliv"][k]) < 1e-12 for k in s["deliv"]) \
        and all(abs(r["cert"][k] - s["cert"][k]) < 1e-12 for k in s["cert"])
    print(f"{stream} geometry {gi + 1}: {'identical' if same else 'DIFFERENT'}")
    ok &= same
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
