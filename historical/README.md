# Historical record: first submission (Paper ID 20267022)

This folder documents what survives of the computation behind the **first submission** and how each of its
numerical results relates to the released code (v1.1). It separates three kinds of statement.

## 1. Documented facts
* **Surviving record.** The only surviving outputs of the first-submission computation are the values and the seven
  figure images printed in the first-submission manuscript. The authors' copy of that manuscript is identified below by
  SHA-256; it is not redistributed here (the journal holds the submitted version).
* **Not recoverable.** No code, configuration file, derived seed, geometry, failure set, per-geometry output or run log of
  the first-submission computation is available, although the first submission described such an archive
  (rows 45–46 of `first_submission_values.csv`).
* **Documented implementation changes** exist only from v1.0 onward: every change from v1.0 to v1.1 is recorded in
  `../CHANGELOG.md` and in the git history, and v1.1 reproduces every v1.0 value.

| File | SHA-256 |
|---|---|
| first-submission manuscript (.docx, authors' copy) | `462ec0642e6db8e1a5f13ab66611bc1f59defac4ecad8a8917a01aead10eb1aa` |
| figure image 1 | `5bb380b2fc59fb0072ddd9bcafe5d27a5d89ff93aa46fdb65f5c92f67e9a5344` |
| figure image 2 | `19888982ad5cf114c79c400115c865d3a37de7cd47192b503d2fb54afedbd741` |
| figure image 3 | `d75077c78d97c3332f29154e10a08a7efea2ea12563616e2d1b38d601fb23684` |
| figure image 4 | `6c240bb2befa0da820f15f1fe9bc1cf696ad55d5b037423ec5abf9ea325a4324` |
| figure image 5 | `b6cc425eb9f7f0be3fd6fada89dae3f0b09d953e143f1ba40dee2c2348102e8f` |
| figure image 6 | `4f12b0bdc0bec1c4c8efe50fe9eabf6cfa65924c164e54755f63e7908045dfa4` |
| figure image 7 | `7c846260de41cb06f15224d559965a42a90dae99aa4112a2fedd10ab79993350` |

## 2. Inferred explanation (not verifiable)
The first-to-revised differences are attributed to a different realization of the random geometries and failures,
caused by a different derivation of the sub-streams from the same master seed and by implementation rules that v1.0
fixed explicitly (RF candidate selection, tie breaking, order of random draws). This explanation is **consistent with**
two diagnostics but **cannot be verified**, because the earlier implementation and its streams are unavailable:
* two independent 24-geometry samples of v1.1 differ by 0.28 points in portfolio delivery at budget 24, more than the
  0.063-point first-to-revised change of that value;
* over 22 tie rules the exact robust anchor's mean delivery moves by at most 0.04 points, so its 0.34-point change is
  not a tie artefact (`../results/audit_ties.json`).

## 3. Comparisons that cannot be independently reproduced
`first_submission_values.csv` lists all 46 first-submission results with their revised counterparts and a class:

| Class | Meaning | Rows |
|---|---|---|
| R | Does not depend on random sampling; recomputed exactly by v1.1 | 11 |
| M | Machine-precision agreement check; the agreement is reproduced, the digits are not comparable | 4 |
| P | Depends on random geometries, failures or instances; the first-submission value cannot be reproduced, and the revised value comes from a different sample | 28 |
| W | Not reproduced; withdrawn | 1 |
| A | Archive described in the first submission; not recoverable | 2 |

No class-P, M or W comparison between the first submission and the revised manuscript can be reproduced
independently. Agreement in a class-P row (for example the floor-violation counts 1, 4, 6) does not verify the earlier
computation. Run `python historical/check_historical.py` to recompute every class-R value and the revised counterparts.
