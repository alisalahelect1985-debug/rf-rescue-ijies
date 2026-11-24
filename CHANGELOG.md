# Changelog

## v1.2 — third revision (current)
* **Historical record of the first submission** (`historical/`): all 46 first-submission results transcribed with their
  v1.1 counterparts and a reproducibility class; SHA-256 identification of the surviving first-submission manuscript and
  its seven figure images; `historical/check_historical.py` recomputes every sample-independent value.
* The relation to the first submission is restated below as documented facts, an inferred explanation, and comparisons
  that cannot be independently reproduced.
* No change to code that produces results; every v1.1 value is unchanged.

## v1.1 — second revision
* **Portability:** all paths are repository-relative (`paths.py`); scripts run from any working directory.
* **End-to-end schedule check** (`schedule.py`, `run_schedule.py`, `results/schedule.json`): for every geometry, budget,
  policy and test frame, the admitted matching is converted into absolute RF and access intervals and checked for replica
  availability, SNR with satellites propagated to the actual transmission time, arrival before the access slot,
  disjoint intervals and the 300 ms deadline. Includes the worked example of Section 6.2.
* **Audit of changed values** (`run_audit_ties.py`, `results/audit_ties.json`): tie sensitivity of the exact robust anchor
  and the difference between two independent 24-geometry samples.
* **Replica count definition:** replicas are now counted once per serving satellite (101.9 per frame at b = 24).
  v1.0 and the first revision of the manuscript reported 218.5, which counted a packet once per reserved token.
* New helpers `core.matching_assign` (returns the same matching as `matching_size`) and `model.constellation_at`
  (orbit propagation). They do not change any v1.0 result: rerunning the full pipeline with v1.1 code reproduces
  every v1.0 numerical value exactly; only wall-clock timing fields differ between runs, so `audit/MANIFEST.json`
  lists content hashes computed with those fields removed.
* Audit tools: `paper_values.py` (every number in Section 6 with its script and key, `audit/paper_values.csv`),
  `make_manifest.py`, `verify_reproduction.py`, `reproduce.py`.

## v1.0 — first revision (commit d8d5b14)
* First public release. Produces all values of the first revised manuscript.

## Relation to the first submission
**Documented facts.** The only surviving outputs of the first-submission computation are the values and seven figure
images printed in that manuscript (identified in `historical/README.md`). No code, configuration, derived seed, geometry,
failure set, per-geometry output or run log of that computation is available, although the first submission described
such an archive. Documented implementation changes therefore exist only from v1.0 onward (this file and the git history).

**Inferred explanation (not verifiable).** The first-to-revised differences are attributed to a different realization of
the random geometries and failures, caused by a different derivation of sub-streams from the same master seed and by
implementation rules (RF candidate selection, tie breaking, order of random draws) that v1.0 fixed explicitly. Two
diagnostics are consistent with this explanation: independent 24-geometry samples of v1.1 differ by 0.28 points in
portfolio delivery at budget 24, and tie rules move the exact robust anchor's mean by at most 0.04 points.

**Not independently reproducible.** Of the 46 first-submission results, the 11 that do not depend on random sampling are
recomputed exactly (class R). The 28 sample-dependent results (class P), the 4 machine-precision checks (class M) and the
withdrawn non-monotonicity example (class W) cannot be reproduced, so no first-to-revised comparison of these values can be
verified independently. The first-submission values are superseded by those of the revised manuscript.
