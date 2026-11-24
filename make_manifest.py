"""Write audit/MANIFEST.json: version tag, SHA-256 of every result file and figure, and the producing script."""
import hashlib, json, platform, sys
import numpy, scipy, numba, matplotlib
from paths import ROOT, RESULTS, FIGS, AUDIT

VERSION = "v1.2"
PRODUCER = {"het": "run_all.py", "hom": "run_all.py", "extra": "run_all.py", "pmin05": "run_all.py", "pmin2": "run_all.py",
            "gam6": "run_all.py", "gam10": "run_all.py", "targets": "run_targets.py", "inter": "run_targets.py",
            "checks": "run_checks.py", "misc": "run_misc.py", "schedule": "run_schedule.py", "audit_ties": "run_audit_ties.py"}


TIMING = {"t_port", "t_match_frame", "t_total", "t", "t_mip"}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if not (k in TIMING and not isinstance(v, dict))}
    if isinstance(o, list):
        return [_strip(x) for x in o]
    return o


def content_sha(p):
    """Hash of the numerical content with wall-clock timing fields removed (stable across reruns)."""
    o = json.load(open(p))
    if isinstance(o, list) and o and isinstance(o[0], dict) and "gi" in o[0]:
        o = sorted(o, key=lambda x: (x.get("K", 0), x["gi"]))
    return hashlib.sha256(json.dumps(_strip(o), sort_keys=True).encode()).hexdigest()


files = {}
for p in sorted(RESULTS.glob("*.json")):
    files["results/" + p.name] = {"content_sha256": content_sha(p), "script": PRODUCER.get(p.stem, "?")}
for p in sorted(FIGS.glob("image*.png")):
    files["figs/" + p.name] = {"sha256": sha(p), "script": "figs.py"}
for p in sorted(ROOT.glob("*.py")):
    files[p.name] = {"sha256": sha(p)}
for p in sorted((ROOT / "historical").glob("*")):
    files["historical/" + p.name] = {"sha256": sha(p)}
man = {"version": VERSION, "master_seed": 20261001, "python": sys.version.split()[0], "platform": platform.platform(),
       "packages": {"numpy": numpy.__version__, "scipy": scipy.__version__, "numba": numba.__version__,
                    "matplotlib": matplotlib.__version__}, "files": files}
json.dump(man, open(AUDIT / "MANIFEST.json", "w"), indent=1)
print("manifest:", len(files), "files")
