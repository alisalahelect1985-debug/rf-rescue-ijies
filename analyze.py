import json, numpy as np
from scipy import stats
D = "/home/claude/rfrescue/results"
POL = ["anchor", "condrob", "indep", "hybrid1", "portfolio", "cloud", "degree", "need"]
B = [6, 12, 18, 24, 30, 36]


def load(n):
    return json.load(open(f"{D}/{n}.json"))


def hw(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def arr(R, key, pol, b):
    return np.array([r[key]["%s|%d" % (pol, b)] for r in R])


def table(R):
    out = {}
    for pol in POL:
        for b in B:
            d = arr(R, "deliv", pol, b) * 100
            c = arr(R, "cert", pol, b) * 100
            out[(pol, b)] = (d.mean(), hw(d), c.mean(), hw(c))
    return out


def paired(R, pol, b, key="deliv", base="indep"):
    x = (arr(R, key, pol, b) - arr(R, key, base, b)) * 100
    return x.mean(), hw(x), x


def violations(R, pol, b):
    return int(sum(r["cert"]["%s|%d" % (pol, b)] < r["jmin"][str(b)] - 1e-12 for r in R))


def deficits(R, pol="indep"):
    out = []
    for r in R:
        for b in B:
            c = r["cert"]["%s|%d" % (pol, b)]
            if c < r["jmin"][str(b)] - 1e-12:
                out.append((b, r["gi"] + 1, 100 * r["jmin"][str(b)], 100 * c, 100 * (r["jmin"][str(b)] - c)))
    return out


if __name__ == "__main__":
    het, hom = load("het"), load("hom")
    T = table(het)
    for b in (12, 18, 24, 30):
        print("b", b)
        for pol in POL:
            m, h, c, hc = T[(pol, b)]
            print("  %-10s %.3f ±%.3f  cert %.3f ±%.3f" % (pol, m, h, c, hc))
    print("max hw table", max(max(v[1], v[3]) for k, v in T.items() if k[1] in (12, 24) and k[0] != "need"))
    for b in B:
        print("viol b", b, {p: violations(het, p, b) for p in POL})
    for b in B:
        m, h, _ = paired(het, "portfolio", b)
        mc, hc, _ = paired(het, "portfolio", b, "cert")
        print("paired b", b, "deliv %.4f ± %.4f  cert %.4f ± %.4f" % (m, h, mc, hc))
    print("deficits", deficits(het))
    nd = [r["n_feasible"][str(b)] for r in het for b in B]
    print("n_feasible range", min(nd), max(nd), np.median(nd))
    ab = {b: {k: np.array([r["abl"][str(b)][k] for r in het]) * 100 for k in ("indep", "cond", "field")} for b in B}
    for b in B:
        a = ab[b]
        print("abl", b, {k: round(v.mean(), 3) for k, v in a.items()}, "ratio", a["field"].mean() / a["indep"].mean(),
              "d1 %.3f±%.3f d2 %.3f±%.3f" % ((a["cond"] - a["indep"]).mean(), hw(a["cond"] - a["indep"]),
                                             (a["field"] - a["cond"]).mean(), hw(a["field"] - a["cond"])))
    TH = table(hom)
    for b in B:
        print("hom", b, {p: (round(TH[(p, b)][0], 3), round(TH[(p, b)][2], 3)) for p in ["indep", "hybrid1", "portfolio", "condrob", "anchor"]})
    print("replicas b24", np.mean([r["replicas"]["24"] for r in het]))
    print("t_port het", np.median([r["t_port"] for r in het]), "hom", np.median([r["t_port"] for r in hom]))
