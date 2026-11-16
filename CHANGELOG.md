# Changelog

## v1.1 — second revision (current)
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
The values in the first submission were produced by an earlier implementation that was not archived together with its
derived seed streams and several implementation rules (RF candidate selection, tie breaking, order of random draws).
v1.0 re-implemented the model from its specification and fixed these rules explicitly (see README). The master seed is
the same, but the derived random streams, and therefore the realized geometries and failure sets, are not. The
first-submission values are superseded; every value in the revised manuscript is produced by this repository.
