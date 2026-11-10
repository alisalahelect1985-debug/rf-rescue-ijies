# Correlation-robust RF rescue for UAV-assisted optical satellite networks

Code and results for the IJIES paper *"Correlation-Robust RF Rescue for UAV-Assisted Optical Satellite Networks:
Capacity Cuts and Certified Resource Provisioning"* (Paper ID 20267022). Every number, table and figure in
Section 6 and in the supplementary material is produced by the scripts below. All randomness derives from the
master seed `20261001` through `numpy.random.SeedSequence`, so the results are reproduced exactly.

## Requirements
Python ≥ 3.10, `numpy`, `scipy` (≥ 1.11, HiGHS MILP), `numba`, `matplotlib`:

    pip install -r requirements.txt

## Layout
| File | Content |
|---|---|
| `core.py` | Numba kernels: maximum matchings, weighted envelope W(R;p), incremental scenario gains |
| `model.py` | Walker-delta geometry, cloud-layer intersections, RF qualification (Eq. 21), optical marginals, integrated-intensity Boolean cloud field (Eqs. 2–7) |
| `policies.py` | Certificate J(R), exact robust anchor (MILP of the flow network), greedy/hybrid/conditional/cloud-informed greedy, degree-first, need-based translation of [24], certificate-screened portfolio, exact floor-constrained MILP (Eq. 17) |
| `experiment.py` | Per-geometry experiment shared by all studies |
| `run_all.py` | Main study (24 geometries per regime), allowance and threshold sensitivity, departures, 24 extra geometries |
| `run_targets.py` | Absolute targets with exact MILP references; intermediate sizes (12/20/30 flows) |
| `run_checks.py` | Envelope vs. LP, Hall and supermodularity enumeration, exact-vs-exhaustive checks, multistate ratio, four-flow table, quadrature, sampler |
| `run_misc.py` | Radius study, six-flow exhaustive references, hybrid tie example, non-monotonicity search, runtime |
| `analyze.py` | Summary statistics (paired differences, violations, deficits, ablation) |
| `figs.py` | All figures (`figs/image1.png` … `image7.png`) |

## Reproduce

    python run_all.py                                   # ~1 min on 2 cores  -> results/{het,hom,extra,pmin05,pmin2,gam6,gam10}.json
    python run_targets.py 0.86,0.87,0.88,0.89,0.90,0.91 targets   # ~10 min -> results/targets.json
    python run_targets.py 0.86 inter                    # ~1 min  -> results/inter.json
    python run_checks.py                                # ~2 min  -> results/checks.json
    python run_misc.py                                  # ~3 min  -> results/misc.json
    python analyze.py                                   # prints the values reported in Section 6
    python figs.py

## Result-to-file map
| Paper item | Script | Output |
|---|---|---|
| Table 3, Figs. 1–2, Sections 6.4–6.6, Supplementary Tables S2–S3, Figs. S3–S5 | `run_all.py`, `analyze.py`, `figs.py` | `results/het.json`, `results/hom.json` |
| Table 4 | `run_all.py` | `results/het.json` (allowances), `pmin05.json`, `pmin2.json`, `gam6.json`, `gam10.json` |
| Table 5 and intermediate sizes | `run_targets.py` | `results/targets.json`, `results/inter.json` |
| Pooled 48 geometries | `run_all.py` | `results/extra.json` |
| Section 6.3, Table S4 | `run_checks.py` | `results/checks.json` |
| Radius study, six-flow references, runtime | `run_misc.py` | `results/misc.json` |
| Modelling departures (Section 6.7) | `run_all.py` | `results/het.json` (`variants`) |

## Implementation choices not fixed by the model description
* RF candidates: satellites ranked by elevation at the cell centre; the three highest whose aged edge SNR
  (range from the centre + 15 km·cos e + Δ, 3 dB beam-edge loss) meets γ_req are kept. Missing identities have no edges.
* Token order: (cell, candidate rank, slot); identity i uses [40 + 1.05(i−1), 41.05 + 1.05(i−1)] ms.
* Budgets {6, 12, 18, 24, 30, 36}; no fixed tokens; greedy ties and matching ties broken by lowest index.
* Hybrid objective W(R;p) + μ·ν̄(R) with ν̄ the mean matching over the 128 independent-link training scenarios;
  μ = 0 is robust greedy.
* Correlated residual errors use a Gaussian copula with a common factor (marginals preserved);
  Markov frames use P(s'|s) = 0.9·1{s'=s} + 0.1·ω_{s'}.
