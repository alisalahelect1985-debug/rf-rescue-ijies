"""Reservation policies and certificates."""
import heapq
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix
from core import weighted_W, scen_gains, scen_commit, batch_matching

BUDGETS = [6, 12, 18, 24, 30, 36]
MU_GRID = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]


class Ctx:
    def __init__(self, geo, p):
        self.g = geo
        self.p = np.asarray(p, float)
        self.order = np.argsort(-self.p, kind="stable")
        self.T = geo.T

    def W(self, R, p=None, order=None):
        p = self.p if p is None else p
        order = self.order if order is None else order
        inR = np.zeros(self.T, np.bool_)
        inR[np.asarray(R, np.int64)] = True
        return weighted_W(p, inR, self.g.fptr, self.g.ftok, self.g.tptr, self.g.tflw, order)

    def J(self, R, p=None):
        p = self.p if p is None else p
        return 1.0 - (p.sum() - self.W(R, p, np.argsort(-p, kind="stable"))) / len(p)

    def match(self, F, R):
        return batch_matching(F, np.asarray(R, np.int64), self.g.tptr, self.g.tflw, self.T)

    def delivery(self, F, R):
        nfail = F.sum(1)
        m = self.match(F, R)
        return 1.0 - (nfail - m) / F.shape[1]


def exact_anchor(ctx, b, R0=()):
    """max W(R;p) s.t. |R|<=b.  MILP of the min-cost-flow network (integral)."""
    g = ctx.g
    E = g.edges
    nE, T = len(E), g.T
    # variables: x_e (nE), y_r (T)
    c = np.concatenate([-ctx.p[E[:, 0]], np.zeros(T)])
    # tiny index-based tie-break: prefer lower token index (deterministic)
    c[nE:] += 1e-9 * np.arange(T) / T
    A = lil_matrix((g.K + T + 1, nE + T))
    for i, (k, r) in enumerate(E):
        A[k, i] = 1
        A[g.K + r, i] = 1
    for r in range(T):
        A[g.K + r, nE + r] = -1
        A[g.K + T, nE + r] = 1
    lb = np.full(g.K + T + 1, -np.inf)
    ub = np.concatenate([np.ones(g.K), np.zeros(T), [b]])
    lo = np.zeros(nE + T)
    hi = np.ones(nE + T)
    for r in R0:
        lo[nE + r] = 1
    integ = np.concatenate([np.zeros(nE), np.ones(T)])
    res = milp(c, constraints=LinearConstraint(csr_matrix(A), lb, ub), bounds=Bounds(lo, hi), integrality=integ,
               options={"mip_rel_gap": 0, "presolve": True})
    assert res.success, res.message
    y = res.x[nE:]
    x = res.x[:nE]
    # keep only tokens actually used by the matching
    used = np.zeros(T, bool)
    used[E[x > 0.5, 1]] = True
    R = [r for r in range(T) if y[r] > 0.5 and used[r]]
    # fill with unused tokens (lowest index) so |R| == b, which cannot reduce W
    for r in range(T):
        if len(R) >= b:
            break
        if r not in R:
            R.append(r)
    return sorted(R[:b])


def lazy_greedy(T, bmax, gain_fn, commit_fn=None, cands=None):
    """Lazy greedy for monotone submodular objectives; ties -> lower token index. Returns selection order."""
    cands = list(range(T)) if cands is None else list(cands)
    sel = []
    heap = [(-np.inf, r) for r in cands]
    heapq.heapify(heap)
    stamp = {r: -1 for r in cands}
    while len(sel) < bmax and heap:
        negg, r = heapq.heappop(heap)
        if stamp[r] == len(sel):
            sel.append(r)
            if commit_fn is not None:
                commit_fn(r)
            continue
        g = gain_fn(r, sel)
        stamp[r] = len(sel)
        heapq.heappush(heap, (-g, r))
    return sel


class ScenObj:
    """Mean matching over a fixed scenario set, maintained incrementally."""

    def __init__(self, ctx, F):
        self.ctx = ctx
        self.F = np.ascontiguousarray(F)
        N, K = F.shape
        self.MF = -np.ones((N, K), np.int64)
        self.MT = -np.ones((N, ctx.T), np.int64)
        self.val = 0.0

    def gain(self, r):
        return scen_gains(np.array([r], np.int64), self.F, self.MF, self.MT, self.ctx.g.tptr, self.ctx.g.tflw)[0] / self.F.shape[0]

    def commit(self, r):
        self.val += scen_commit(r, self.F, self.MF, self.MT, self.ctx.g.tptr, self.ctx.g.tflw) / self.F.shape[0]


def greedy_W(ctx, bmax, plist=None, weights=None):
    """Greedy on sum_s w_s W(R;p_s)."""
    plist = [ctx.p] if plist is None else plist
    weights = [1.0] if weights is None else weights
    orders = [np.argsort(-p, kind="stable") for p in plist]
    cur = [0.0 for _ in plist]

    def val(R):
        return [ctx.W(R, p, o) for p, o in zip(plist, orders)]

    def gain(r, sel):
        v = val(sel + [r])
        return sum(w * (a - b) for w, a, b in zip(weights, v, cur))

    def commit(r):
        nonlocal cur
        cur = val(sel_ref + [r])
        sel_ref.append(r)

    sel_ref = []
    return lazy_greedy(ctx.T, bmax, gain, commit)


def greedy_hybrid(ctx, bmax, F_train, mu):
    """Greedy on W(R;p) + mu * nu_bar(R).  mu = inf -> nominal only (independence greedy)."""
    so = ScenObj(ctx, F_train)
    state = {"W": 0.0, "sel": []}
    nominal_only = np.isinf(mu)

    def gain(r, sel):
        gn = so.gain(r)
        if nominal_only:
            return gn
        gw = ctx.W(state["sel"] + [r]) - state["W"]
        return gw + mu * gn

    def commit(r):
        so.commit(r)
        state["sel"].append(r)
        if not nominal_only:
            state["W"] = ctx.W(state["sel"])

    return lazy_greedy(ctx.T, bmax, gain, commit)


def degree_first(ctx, b):
    return sorted(np.argsort(-ctx.g.deg, kind="stable")[:b].tolist())


def need_based(ctx, b):
    """Translation of need-based frame allocation [24]: expected failed eligible flows per token."""
    need = np.array([ctx.p[ctx.g.tflw[ctx.g.tptr[r]:ctx.g.tptr[r + 1]]].sum() for r in range(ctx.T)])
    return sorted(np.argsort(-need, kind="stable")[:b].tolist())


def portfolio(ctx, b, anchor, family, F_val, allowance=0.005, jmin=None):
    """family: dict name -> reservation.  Returns (selected R, name, info)."""
    Ja = ctx.J(anchor)
    jmin = Ja - allowance if jmin is None else jmin
    seen = {}
    for name, R in [("anchor", anchor)] + list(family.items()):
        key = tuple(sorted(R))
        if key not in seen:
            seen[key] = name
    feas = []
    for key, name in seen.items():
        Jr = ctx.J(list(key))
        if Jr >= jmin - 1e-12:
            feas.append((key, name, Jr))
    if not feas:
        return None, None, {"n_distinct": len(seen), "n_feasible": 0, "jmin": jmin}
    best = None
    for key, name, Jr in feas:
        v = ctx.match(F_val, list(key)).mean()
        if best is None or v > best[0] + 1e-12:
            best = (v, key, name, Jr)
    return list(best[1]), best[2], {"n_distinct": len(seen), "n_feasible": len(feas), "jmin": jmin}


def exact_floor(ctx, b, F_train, jmin, time_limit=300):
    """Eq. (38): max mean nominal matching over training scenarios s.t. W(R;p) >= theta, |R|<=b."""
    g = ctx.g
    E = g.edges
    K, T = g.K, g.T
    N = F_train.shape[0]
    theta = ctx.p.sum() - K * (1 - jmin)
    # variable blocks: y (T), z (nE), x^n for edges with failed flow in scenario n
    blocks = []
    for n in range(N):
        idx = np.where(F_train[n, E[:, 0]])[0]
        blocks.append(idx)
    nE = len(E)
    nx = sum(len(b_) for b_ in blocks)
    nv = T + nE + nx
    c = np.zeros(nv)
    off = T + nE
    starts = []
    for n in range(N):
        starts.append(off)
        c[off:off + len(blocks[n])] = -1.0 / N
        off += len(blocks[n])
    rows = []
    lbs, ubs = [], []
    A = lil_matrix((1 + (K + T) * (N + 1) + 1, nv))
    row = 0
    # budget
    A[row, :T] = 1; lbs.append(-np.inf); ubs.append(b); row += 1
    # witness z: flow deg <=1, token cap <= y, weight >= theta
    for k in range(K):
        ii = np.where(E[:, 0] == k)[0]
        if len(ii):
            for i in ii: A[row, T + i] = 1
            lbs.append(-np.inf); ubs.append(1); row += 1
    for r in range(T):
        ii = np.where(E[:, 1] == r)[0]
        if len(ii):
            for i in ii: A[row, T + i] = 1
            A[row, r] = -1
            lbs.append(-np.inf); ubs.append(0); row += 1
    for i, (k, r) in enumerate(E):
        A[row, T + i] = ctx.p[k]
    lbs.append(theta); ubs.append(np.inf); row += 1
    # scenario matchings
    for n in range(N):
        idx = blocks[n]
        s0 = starts[n]
        byk, byr = {}, {}
        for j, i in enumerate(idx):
            byk.setdefault(E[i, 0], []).append(s0 + j)
            byr.setdefault(E[i, 1], []).append(s0 + j)
        for k, vs in byk.items():
            if len(vs) > 1:
                for v in vs: A[row, v] = 1
                lbs.append(-np.inf); ubs.append(1); row += 1
        for r, vs in byr.items():
            for v in vs: A[row, v] = 1
            A[row, r] = -1
            lbs.append(-np.inf); ubs.append(0); row += 1
    A = csr_matrix(A[:row])
    integ = np.zeros(nv); integ[:T] = 1
    res = milp(c, constraints=LinearConstraint(A, lbs, ubs), bounds=Bounds(np.zeros(nv), np.ones(nv)),
               integrality=integ, options={"time_limit": time_limit, "mip_rel_gap": 1e-6})
    if res.x is None:
        return None, res
    R = [r for r in range(T) if res.x[r] > 0.5]
    return R, res
