"""Regenerate every result, figure and audit file (about 25 minutes on two cores)."""
import subprocess, sys
from paths import ROOT

STEPS = [["run_all.py"], ["run_targets.py", "0.86,0.87,0.88,0.89,0.90,0.91", "targets"], ["run_targets.py", "0.86", "inter"],
         ["run_checks.py"], ["run_misc.py"], ["run_schedule.py"], ["run_audit_ties.py"], ["figs.py"],
         ["paper_values.py"], ["historical/check_historical.py"], ["make_manifest.py"]]
for step in STEPS:
    print(">>", " ".join(step), flush=True)
    subprocess.run([sys.executable, str(ROOT / step[0])] + step[1:], check=True, cwd=ROOT)
print("All results regenerated. Compare audit/MANIFEST.json with the published copy.")
