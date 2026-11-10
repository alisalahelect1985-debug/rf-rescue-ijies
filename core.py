"""Numba kernels: bipartite matchings between failed flows and reserved RF tokens.

Graph storage (CSR):
  tptr, tflw : token -> eligible flows
  fptr, ftok : flow  -> eligible tokens
"""
import numpy as np
from numba import njit


def csr_from_edges(K, T, edges):
    """edges: list/array of (k, r) pairs."""
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    tl = [[] for _ in range(T)]
    fl = [[] for _ in range(K)]
    for k, r in edges:
        tl[r].append(k)
        fl[k].append(r)
    tptr = np.zeros(T + 1, np.int64)
    fptr = np.zeros(K + 1, np.int64)
    for r in range(T):
        tptr[r + 1] = tptr[r] + len(tl[r])
    for k in range(K):
        fptr[k + 1] = fptr[k] + len(fl[k])
    tflw = np.array([k for r in range(T) for k in sorted(tl[r])], np.int64)
    ftok = np.array([r for k in range(K) for r in sorted(fl[k])], np.int64)
    return tptr, tflw, fptr, ftok


@njit(cache=True)
def _augment_from_token(root, tptr, tflw, failed, match_f, match_t, prev_t, seen, queue, commit):
    """BFS for an augmenting path from token `root` to an unmatched failed flow.
    match_f[k]: token matched to flow k (-1); match_t[r]: flow matched to token r (-1).
    Returns True if found (and augments when commit)."""
    K = failed.shape[0]
    for k in range(K):
        seen[k] = False
    head = 0
    tail = 0
    queue[tail] = root
    tail += 1
    found = -1
    while head < tail and found < 0:
        t = queue[head]
        head += 1
        for i in range(tptr[t], tptr[t + 1]):
            k = tflw[i]
            if not failed[k] or seen[k]:
                continue
            seen[k] = True
            prev_t[k] = t
            if match_f[k] < 0:
                found = k
                break
            queue[tail] = match_f[k]
            tail += 1
    if found < 0:
        return False
    if commit:
        k = found
        while True:
            t = prev_t[k]
            nk = match_t[t]
            match_f[k] = t
            match_t[t] = k
            if t == root:
                break
            k = nk
    return True


@njit(cache=True)
def matching_size(failed, R, tptr, tflw, T):
    K = failed.shape[0]
    match_f = -np.ones(K, np.int64)
    match_t = -np.ones(T, np.int64)
    prev_t = np.zeros(K, np.int64)
    seen = np.zeros(K, np.bool_)
    queue = np.zeros(T + 1, np.int64)
    m = 0
    for j in range(R.shape[0]):
        if _augment_from_token(R[j], tptr, tflw, failed, match_f, match_t, prev_t, seen, queue, True):
            m += 1
    return m


@njit(cache=True)
def batch_matching(F, R, tptr, tflw, T):
    """F: bool [N, K] failed sets. Returns matched counts [N]."""
    N = F.shape[0]
    out = np.zeros(N, np.int64)
    for n in range(N):
        out[n] = matching_size(F[n], R, tptr, tflw, T)
    return out


@njit(cache=True)
def weighted_W(p, inR, fptr, ftok, tptr, tflw, order):
    """W(R;p): max-weight matching value with flow weights p over tokens with inR True.
    Matroid greedy in decreasing p (order = argsort(-p)).  Augmenting search from flow side."""
    K = p.shape[0]
    T = inR.shape[0]
    match_f = -np.ones(K, np.int64)
    match_t = -np.ones(T, np.int64)
    seen_t = np.zeros(T, np.bool_)
    prev_f = np.zeros(T, np.int64)
    queue = np.zeros(K + 1, np.int64)
    W = 0.0
    for oi in range(K):
        root = order[oi]
        if p[root] <= 0.0:
            continue
        for t in range(T):
            seen_t[t] = False
        head = 0
        tail = 0
        queue[tail] = root
        tail += 1
        found = -1
        while head < tail and found < 0:
            k = queue[head]
            head += 1
            for i in range(fptr[k], fptr[k + 1]):
                t = ftok[i]
                if not inR[t] or seen_t[t]:
                    continue
                seen_t[t] = True
                prev_f[t] = k
                if match_t[t] < 0:
                    found = t
                    break
                queue[tail] = match_t[t]
                tail += 1
        if found >= 0:
            t = found
            while True:
                k = prev_f[t]
                nt = match_f[k]
                match_t[t] = k
                match_f[k] = t
                if k == root:
                    break
                t = nt
            W += p[root]
    return W


@njit(cache=True)
def scen_gains(cands, F, MF, MT, tptr, tflw):
    """For each candidate token, number of scenarios in which adding it increases the matching."""
    N, K = F.shape
    T = MT.shape[1]
    prev_t = np.zeros(K, np.int64)
    seen = np.zeros(K, np.bool_)
    queue = np.zeros(T + 1, np.int64)
    g = np.zeros(cands.shape[0], np.int64)
    for c in range(cands.shape[0]):
        r = cands[c]
        cnt = 0
        for n in range(N):
            if _augment_from_token(r, tptr, tflw, F[n], MF[n], MT[n], prev_t, seen, queue, False):
                cnt += 1
        g[c] = cnt
    return g


@njit(cache=True)
def scen_commit(r, F, MF, MT, tptr, tflw):
    N, K = F.shape
    T = MT.shape[1]
    prev_t = np.zeros(K, np.int64)
    seen = np.zeros(K, np.bool_)
    queue = np.zeros(T + 1, np.int64)
    cnt = 0
    for n in range(N):
        if _augment_from_token(r, tptr, tflw, F[n], MF[n], MT[n], prev_t, seen, queue, True):
            cnt += 1
    return cnt
