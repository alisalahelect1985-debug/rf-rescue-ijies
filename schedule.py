"""End-to-end executability check of the frame schedule (Section 6.2).

Timeline (ms after admission): optical opportunity k in [0.5k, 0.5k+0.5); control [30, 40);
RF identity i (0-based, ordered by cell, rank, slot) payload [40+1.05i, 41+1.05i), guard to 41.05+1.05i;
propagation reserve [197.5, 209.5); access opportunity of flow k [209.5+1.05k, 210.5+1.05k); deadline 300 ms.
For every qualified edge (k, r) the checker recomputes the SNR with the satellite propagated to the actual
transmission time (no aging bound), the RF arrival time including propagation, and the replica requirement."""
import numpy as np
from model import (Geometry, constellation_at, rf_snr, CLIGHT, DELTA)
from core import matching_assign

T_RF0, T_TOK, T_PAY, T_ACC0, T_ACC, DEADLINE = 40.0, 1.05, 1.0, 209.5, 1.05, 300.0


def rf_interval(r):
    return T_RF0 + T_TOK * r, T_RF0 + T_TOK * r + T_PAY


def access_interval(k):
    return T_ACC0 + T_ACC * k, T_ACC0 + T_ACC * k + T_PAY


def edge_table(geo):
    """Per qualified edge: actual SNR at transmission time, aged SNR used for qualification, arrival, access start."""
    rows = []
    dist = np.linalg.norm(geo.xy[:, None, :] - geo.cells[None, :, :], axis=2)
    cache = {}
    for k, r in geo.edges:
        j = geo.cand_sat[r]
        a, b_ = rf_interval(r)
        tmid = 0.5 * (a + b_) / 1000.0
        if r not in cache:
            cache[r] = constellation_at(*geo.orbit_params, tmid)[j]
        sat_t = cache[r]
        ci = r // 6
        d_act = np.linalg.norm(sat_t - geo.U[k])
        d0 = np.linalg.norm(geo.sats[j] - geo.U[k])
        snr_act = rf_snr(d_act, dist[k, ci])
        snr_aged = rf_snr(d0 + DELTA, dist[k, ci])
        arrival = b_ + d_act / CLIGHT * 1e6  # ms (km / (km/ms))
        acc0, acc1 = access_interval(k)
        rows.append((k, r, j, a, b_, snr_act, snr_aged, arrival, acc0, acc1))
    return np.array(rows, float)


def replicas(geo, R):
    """Replica sets per serving satellite: union of eligible flows over that satellite's reserved tokens."""
    rep = {}
    for r in R:
        j = int(geo.cand_sat[r])
        flows = geo.tflw[geo.tptr[r]:geo.tptr[r + 1]]
        rep.setdefault(j, set()).update(int(k) for k in flows)
    return rep


def check_frames(geo, R, F, gamma_req=8.0, etab=None):
    """Explicit per-frame schedule check. Returns dict of counts and worst-case margins."""
    etab = edge_table(geo) if etab is None else etab
    ekey = {(int(e[0]), int(e[1])): e for e in etab}
    rep = replicas(geo, R)
    Rarr = np.asarray(R, np.int64)
    viol = {"not_edge": 0, "no_replica": 0, "snr": 0, "late_arrival": 0, "deadline": 0, "overlap": 0, "double_use": 0}
    worst = {"snr_margin": np.inf, "arrival_slack": np.inf, "completion": 0.0}
    n_pairs = 0
    for f in F:
        mt = matching_assign(f, Rarr, geo.tptr, geo.tflw, geo.T)
        used = [(r, int(mt[r])) for r in R if mt[r] >= 0]
        if len(set(k for _, k in used)) != len(used):
            viol["double_use"] += 1
        ivs = sorted(rf_interval(r) for r, _ in used)
        viol["overlap"] += sum(ivs[i + 1][0] < ivs[i][1] + 0.05 - 1e-9 for i in range(len(ivs) - 1))
        for r, k in used:
            n_pairs += 1
            e = ekey.get((k, r))
            if e is None:
                viol["not_edge"] += 1
                continue
            if k not in rep[int(geo.cand_sat[r])]:
                viol["no_replica"] += 1
            if e[5] < gamma_req:
                viol["snr"] += 1
            if e[7] > e[8]:
                viol["late_arrival"] += 1
            if e[9] > DEADLINE:
                viol["deadline"] += 1
            worst["snr_margin"] = min(worst["snr_margin"], e[5] - gamma_req)
            worst["arrival_slack"] = min(worst["arrival_slack"], e[8] - e[7])
            worst["completion"] = max(worst["completion"], e[9])
    return viol, worst, n_pairs
