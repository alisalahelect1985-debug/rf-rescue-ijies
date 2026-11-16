"""Repository-relative locations (no machine-specific paths)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"
AUDIT = ROOT / "audit"
for _d in (RESULTS, FIGS, AUDIT):
    _d.mkdir(exist_ok=True)
