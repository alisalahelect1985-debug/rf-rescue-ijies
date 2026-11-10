"""Geometry, RF qualification, optical marginals and the integrated-intensity Boolean cloud field."""
import numpy as np
from scipy.special import i1, ndtr
from core import csr_from_edges

RE, H, HU = 6371.0, 550.0, 0.3
RS, RA = RE + H, RE + HU
LAT0, LON0 = np.radians(33.3), np.radians(44.4)
NPLANE, NPER, FPH, INC = 72, 22, 1, np.radians(53.0)
WEATHER_P = np.array([0.6, 0.3, 0.1])
COVER = np.array([[0.01, 0.005], [0.10, 0.08], [0.55, 0.30]])  # state x layer
HC = np.array([1.5, 4.5])
A0 = np.array([2.5, 10.0])
KAPPA_HET = np.array([0.020, -0.012])
# RF
EIRP, GT, LATM, BW, FC, CLIGHT = 30.0, 5.0, 3.0, 10e6, 20e9, 299792458.0
DEADLINE_S = 0.3
DELTA = 8.2 * DEADLINE_S  # km
CELL_R = 15.0
# optical
PT, ETA_O, AO, DIV, SI2, PMIN, TAU0 = 0.1, 0.5, 0.05, 20e-6, 0.25, 1e-7, 0.2

E_HAT = np.array([-np.sin(LON0), np.cos(LON0), 0.0])
N_HAT = np.array([-np.sin(LAT0) * np.cos(LON0), -np.sin(LAT0) * np.sin(LON0), np.cos(LAT0)])
U_HAT = np.array([np.cos(LAT0) * np.cos(LON0), np.cos(LAT0) * np.sin(LON0), np.sin(LAT0)])


def phi(z):
    z = np.asarray(z, float)
    out = np.ones_like(z)
    nz = np.abs(z) > 1e-12
    out[nz] = 2.0 * i1(z[nz]) / z[nz]
    return out


def local_to_ecef(x, y, r):
    """Local tangent-plane coordinates (km, east/north from the reference) projected to radius r."""
    v = RE * U_HAT[None, :] + x[:, None] * E_HAT[None, :] + y[:, None] * N_HAT[None, :]
    return r * v / np.linalg.norm(v, axis=1, keepdims=True)


def ecef_to_local(P):
    u = P / np.linalg.norm(P, axis=1, keepdims=True)
    # gnomonic-type projection at the reference point, scaled by the point radius
    rr = np.linalg.norm(P, axis=1)
    return np.stack([rr * (u @ E_HAT), rr * (u @ N_HAT)], axis=1)


def constellation(rng):
    raan0 = rng.uniform(0, 2 * np.pi)
    u0 = rng.uniform(0, 2 * np.pi)
    P = np.arange(NPLANE)
    S = np.arange(NPER)
    Om = raan0 + 2 * np.pi * P[:, None] / NPLANE + 0 * S[None, :]
    u = u0 + 2 * np.pi * S[None, :] / NPER + 2 * np.pi * FPH * P[:, None] / (NPLANE * NPER)
    Om, u = Om.ravel(), u.ravel()
    x = np.cos(u) * np.cos(Om) - np.sin(u) * np.cos(INC) * np.sin(Om)
    y = np.cos(u) * np.sin(Om) + np.sin(u) * np.cos(INC) * np.cos(Om)
    z = np.sin(u) * np.sin(INC)
    return RS * np.stack([x, y, z], axis=1)


def elev_range(Pts, Sats):
    """sin(elevation) and range (km) from points [n,3] to satellites [m,3]."""
    up = Pts / np.linalg.norm(Pts, axis=1, keepdims=True)
    V = Sats[None, :, :] - Pts[:, None, :]
    d = np.linalg.norm(V, axis=2)
    se = np.einsum("nmk,nk->nm", V, up) / d
    return se, d


def uav_positions(rng, K=60):
    nclu = int(round(0.6 * K))
    pts = []
    centres = [(-22.0, -12.0), (22.0, 12.0)]
    for c in range(2):
        n = nclu // 2 + (nclu % 2 if c == 0 else 0)
        got = 0
        while got < n:
            q = rng.normal(centres[c], 6.0)
            if abs(q[0]) <= 50 and abs(q[1]) <= 50:
                pts.append(q)
                got += 1
    for _ in range(K - nclu):
        pts.append(rng.uniform(-50, 50, 2))
    return np.array(pts)


def rf_snr(d_km, rho_km):
    lfs = 20 * np.log10(4 * np.pi * d_km * 1e3 * FC / CLIGHT)
    lb = 3.0 * (rho_km / 15.0) ** 2
    return EIRP + GT - lfs - LATM - lb + 228.599 - 10 * np.log10(BW)


class Geometry:
    """One satellite/UAV snapshot with its qualified token graph and optical marginals."""

    def __init__(self, rng, K=60, gamma_req=8.0, ncell_side=5, sigma_theta=None):
        self.K = K
        sats = constellation(rng)
        self.xy = uav_positions(rng, K)
        U = local_to_ecef(self.xy[:, 0], self.xy[:, 1], RA)
        se, d = elev_range(U, sats)
        se_m = np.where(se > np.sin(np.radians(30.0)), se, -2)
        a = np.argmax(se_m, axis=1)
        assert np.all(se_m[np.arange(K), a] > -1), "no optical satellite above 30 deg"
        self.opt_sat = a
        self.opt_d = d[np.arange(K), a]
        self.opt_e = np.arcsin(se[np.arange(K), a])
        # cloud-layer intersections (local km coordinates)
        self.c = np.zeros((2, K, 2))
        for l in range(2):
            Rc = RE + HC[l]
            V = sats[a] - U
            A = np.sum(V * V, 1)
            Bq = 2 * np.sum(U * V, 1)
            Cq = np.sum(U * U, 1) - Rc ** 2
            t = (-Bq + np.sqrt(Bq ** 2 - 4 * A * Cq)) / (2 * A)
            assert np.all((t > 0) & (t < 1))
            self.c[l] = ecef_to_local(U + t[:, None] * V)
        # pointing deviation (heterogeneous regime)
        self.sigma_theta = rng.uniform(6e-6, 12e-6, K) if sigma_theta is None else sigma_theta
        # RF candidates per cell
        g = (np.arange(ncell_side) - (ncell_side - 1) / 2) * 20.0
        cx, cy = np.meshgrid(g, g, indexing="ij")
        self.cells = np.stack([cx.ravel(), cy.ravel()], 1)
        C = local_to_ecef(self.cells[:, 0], self.cells[:, 1], RA)
        sec, dc = elev_range(C, sats)
        self.gamma_req = gamma_req
        T = len(self.cells) * 6
        self.T = T
        self.cand_sat = -np.ones(T, int)
        self.cand_snr_edge = np.full(T, np.nan)
        edges = []
        dist = np.linalg.norm(self.xy[:, None, :] - self.cells[None, :, :], axis=2)
        for ci in range(len(self.cells)):
            order = np.argsort(-sec[ci])
            chosen = []
            for j in order:
                if sec[ci, j] <= 0:
                    break
                e = np.arcsin(sec[ci, j])
                d_edge = dc[ci, j] + CELL_R * np.cos(e) + DELTA
                s_edge = rf_snr(d_edge, CELL_R)
                if s_edge >= gamma_req:
                    chosen.append((j, s_edge))
                if len(chosen) == 3:
                    break
            for rank, (j, s_edge) in enumerate(chosen):
                for slot in range(2):
                    r = ci * 6 + rank * 2 + slot
                    self.cand_sat[r] = j
                    self.cand_snr_edge[r] = s_edge
                    for k in np.where(dist[:, ci] <= CELL_R)[0]:
                        dk = np.linalg.norm(sats[j] - U[k]) + DELTA
                        if rf_snr(dk, dist[k, ci]) >= gamma_req:
                            edges.append((k, r))
        self.edges = np.array(edges, int).reshape(-1, 2)
        self.tptr, self.tflw, self.fptr, self.ftok = csr_from_edges(K, T, self.edges)
        self.deg = np.diff(self.tptr)

    # ---------------- optical marginals
    def q_resid(self, pmin=PMIN):
        th = DIV
        zeta = th ** 2 / (4 * self.sigma_theta ** 2)
        P0 = PT * ETA_O * 2 * AO ** 2 / (th * self.opt_d * 1e3) ** 2 * np.exp(-TAU0 / np.sin(self.opt_e))
        sI = np.sqrt(SI2)
        A = np.log(P0 / pmin) - SI2 / 2
        q = ndtr(A / sI) - np.exp(-zeta * A + zeta ** 2 * SI2 / 2) * ndtr((A - zeta * SI2) / sI)
        return np.clip(q, 0.0, 1.0)


class Atmosphere:
    """Integrated-intensity Boolean model; lam0[s,l] at radius a[l]."""

    def __init__(self, kappa, radius_scale=1.0):
        self.kappa = np.asarray(kappa, float)
        self.a = A0 * radius_scale
        lam_ref = -np.log(1 - COVER) / (np.pi * A0[None, :] ** 2)
        # Eq. (13): marginal-preserving rescaling
        self.lam0 = lam_ref * (A0 ** 2 * phi(self.kappa * A0)) / (self.a ** 2 * phi(self.kappa * self.a))[None, :]

    def clear_prob(self, c):
        """c: [2, K, 2] intersections -> [S, K] probability that both layers are clear."""
        S = len(WEATHER_P)
        out = np.ones((S, c.shape[1]))
        for l in range(2):
            mass = np.pi * self.a[l] ** 2 * phi(np.array([self.kappa[l] * self.a[l]]))[0] * np.exp(self.kappa[l] * c[l, :, 0])
            out *= np.exp(-self.lam0[:, l][:, None] * mass[None, :])
        return out

    def blocked(self, rng, c, s):
        """Sample one field in state s; return bool [K] blocked by any layer."""
        K = c.shape[1]
        bl = np.zeros(K, bool)
        for l in range(2):
            a = self.a[l]
            lo = c[l].min(0) - a
            hi = c[l].max(0) + a
            kap = self.kappa[l]
            xstar = hi[0] if kap > 0 else lo[0]
            lmax = self.lam0[s, l] * np.exp(kap * xstar)
            area = (hi[0] - lo[0]) * (hi[1] - lo[1])
            n = rng.poisson(lmax * area)
            if n == 0:
                continue
            pts = rng.uniform(lo, hi, (n, 2))
            keep = rng.uniform(size=n) < np.exp(kap * (pts[:, 0] - xstar))
            pts = pts[keep]
            if len(pts) == 0:
                continue
            d2 = ((c[l][:, None, :] - pts[None, :, :]) ** 2).sum(2)
            bl |= (d2 <= a * a).any(1)
        return bl


class Regime:
    """Marginals and samplers for one geometry in one regime."""

    def __init__(self, geo, hetero=True, pmin=PMIN, radius_scale=1.0):
        self.geo = geo
        self.hetero = hetero
        self.atm = Atmosphere(KAPPA_HET if hetero else [0.0, 0.0], radius_scale)
        self.q = geo.q_resid(pmin) if hetero else np.full(geo.K, 0.98)
        clear = self.atm.clear_prob(geo.c)  # [S,K]
        self.ps = 1 - self.q[None, :] * clear  # conditional marginals
        self.p = WEATHER_P @ self.ps

    # ---- scenario generators
    def indep(self, rng, n):
        return rng.uniform(size=(n, self.geo.K)) < self.p[None, :]

    def cond_indep(self, rng, n):
        s = rng.choice(3, size=n, p=WEATHER_P)
        return rng.uniform(size=(n, self.geo.K)) < self.ps[s]

    def field(self, rng, n, resid_rho=0.0, markov=None, return_states=False):
        """Complete spatial law. resid_rho: Gaussian-copula correlation of residual errors (marginals kept).
        markov: persistence of a weather Markov chain (None = independent frames)."""
        from scipy.special import ndtri
        K = self.geo.K
        F = np.zeros((n, K), bool)
        st = np.zeros(n, int)
        s = rng.choice(3, p=WEATHER_P)
        thr = ndtri(np.clip(self.q, 1e-15, 1 - 1e-15))
        for i in range(n):
            if markov is None:
                s = rng.choice(3, p=WEATHER_P)
            elif i > 0:
                if rng.uniform() >= markov:
                    s = rng.choice(3, p=WEATHER_P)
            st[i] = s
            bl = self.atm.blocked(rng, self.geo.c, s)
            if resid_rho > 0:
                z = np.sqrt(resid_rho) * rng.normal() + np.sqrt(1 - resid_rho) * rng.normal(size=K)
                fail_res = z > thr
            else:
                fail_res = rng.uniform(size=K) >= self.q
            F[i] = bl | fail_res
        return (F, st) if return_states else F
