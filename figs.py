import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import load, table, paired, violations, hw, B
from model import phi

C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY = "#6b6b6b"
OUT = "/home/claude/rfrescue/figs"
import os; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"], "font.size": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.5, "legend.frameon": False,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4, "lines.markersize": 4})
SIZES = {1: (1071, 780), 2: (1071, 809), 3: (1071, 875), 4: (1075, 775), 5: (1075, 779), 6: (1071, 780), 7: (1075, 780)}
DPI = 350


def fig(n):
    w, h = SIZES[n]
    return plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI)


def save(f, n):
    f.tight_layout(pad=0.3)
    f.savefig(f"{OUT}/image{n}.png", dpi=DPI)
    plt.close(f)


het, hom = load("het"), load("hom")
T, TH = table(het), table(hom)
Bx = np.array(B)

# image3 / Suppl. Fig. S5: mean delivery, heterogeneous
f = fig(3); ax = f.add_subplot(111)
for pol, lab, c, m in [("indep", "Independence greedy", C1, "o"), ("portfolio", "Screened portfolio", C2, "s"),
                       ("cloud", "Cloud-informed greedy", C3, "^"), ("anchor", "Exact robust anchor", C4, "D"),
                       ("degree", "Degree-first", GRAY, "v")]:
    y = np.array([T[(pol, b)][0] for b in B]); e = np.array([T[(pol, b)][1] for b in B])
    ax.errorbar(Bx, y, yerr=e, color=c, marker=m, capsize=1.5, elinewidth=0.6, label=lab)
ax.set_xlabel("Reserved RF tokens"); ax.set_ylabel("Mean delivery (%)"); ax.set_xticks(B)
ax.legend(loc="lower right", fontsize=6.5)
save(f, 3)

# image4 / Fig. 1: paired differences vs independence greedy
f = fig(4); ax = f.add_subplot(111)
off = {"portfolio": -0.6, "hybrid1": 0.0, "cloud": 0.6}
for pol, lab, c, m in [("portfolio", "Screened portfolio", C2, "s"), ("hybrid1", "Hybrid (μ = 1)", C5, "o"),
                       ("cloud", "Cloud-informed greedy", C3, "^")]:
    y = np.array([paired(het, pol, b)[0] for b in B]); e = np.array([paired(het, pol, b)[1] for b in B])
    ax.errorbar(Bx + off[pol], y, yerr=e, color=c, marker=m, ls="none", capsize=1.5, elinewidth=0.8, label=lab)
ax.axhline(0, color="#999999", lw=0.8)
ax.set_xlabel("Reserved RF tokens"); ax.set_ylabel("Paired difference (pp)"); ax.set_xticks(B)
ax.legend(loc="upper left", fontsize=6.5)
save(f, 4)

# image7 / Fig. 2: violations
f = fig(7); ax = f.add_subplot(111)
vi = [violations(het, "indep", b) for b in B]
vp = [violations(het, "portfolio", b) for b in B]
vc = [violations(het, "cloud", b) for b in B]
w = 1.6
ax.bar(Bx - w, vi, width=w * 0.92, color=C1, label="Independence greedy")
ax.bar(Bx, vc, width=w * 0.92, color=C3, label="Cloud-informed greedy")
ax.bar(Bx + w, vp, width=w * 0.92, color=C2, label="Screened portfolio")
for x, v in zip(Bx - w, vi):
    if v: ax.text(x, v + 0.3, str(v), ha="center", fontsize=6.5)
for x, v in zip(Bx, vc):
    if v: ax.text(x, v + 0.3, str(v), ha="center", fontsize=6.5)
ax.set_xlabel("Reserved RF tokens"); ax.set_ylabel("Floor violations (of 24)"); ax.set_xticks(B)
ax.set_ylim(0, 24); ax.set_axisbelow(True); ax.legend(loc="upper left", fontsize=6.5)
ax.text(0.03, 0.62, "Screened portfolio: 0 at every budget", transform=ax.transAxes, fontsize=6.5, color="#444444")
save(f, 7)

# image6 / Suppl. Fig. S4: ablation
f = fig(6); ax = f.add_subplot(111)
for k, lab, c, m in [("indep", "Unconditional independence", C1, "o"), ("cond", "Shared weather, independent within state", C4, "s"),
                     ("field", "Complete spatial field", C2, "^")]:
    y = np.array([np.mean([r["abl"][str(b)][k] for r in het]) * 100 for b in B])
    e = np.array([hw([r["abl"][str(b)][k] * 100 for r in het]) for b in B])
    ax.errorbar(Bx, y, yerr=e, color=c, marker=m, capsize=1.5, elinewidth=0.6, label=lab)
ax.set_xlabel("Reserved RF tokens"); ax.set_ylabel("Mean loss (%)"); ax.set_xticks(B)
ax.legend(loc="upper right", fontsize=6.5)
save(f, 6)

# image5 / Suppl. Fig. S3: homogeneous control
f = fig(5); ax = f.add_subplot(111)
for pol, lab, c, m in [("indep", "Independence greedy", C1, "o"), ("anchor", "Exact robust anchor", C4, "D"),
                       ("condrob", "Conditional robust greedy", C3, "^")]:
    y = np.array([TH[(pol, b)][0] for b in B])
    ax.plot(Bx, y, color=c, marker=m, label=lab + " (delivery)")
y = np.array([TH[("anchor", b)][2] for b in B])
ax.plot(Bx, y, color=GRAY, ls="--", marker="x", label="Optimal robust certificate")
ax.set_xlabel("Reserved RF tokens"); ax.set_ylabel("Percent"); ax.set_xticks(B)
ax.legend(loc="lower right", fontsize=6)
save(f, 5)

# image1 / Suppl. Fig. S2: homogeneous two-point Boolean calculation
f = fig(1); ax = f.add_subplot(111)
p = 0.2
for a, c, m in [(2.5, C1, None), (10.0, C2, None)]:
    lam_pa = -np.log(1 - p)
    d = np.linspace(0, 3 * a, 300)
    x = np.clip(d / (2 * a), 0, 1)
    Acap = np.where(d < 2 * a, 2 * a * a * np.arccos(x) - d / 2 * np.sqrt(np.clip(4 * a * a - d * d, 0, None)), 0)
    both = 1 - 2 * np.exp(-lam_pa) + np.exp(-lam_pa * (2 - Acap / (np.pi * a * a)))
    ax.plot(d, both / 2 * 100, color=c, label="Disc radius %g km" % a)
ax.axhline(p / 2 * 100, color=GRAY, ls=":", lw=0.9); ax.axhline(p * p / 2 * 100, color=GRAY, ls=":", lw=0.9)
ax.text(29.5, p / 2 * 100 + 0.25, "p/2", ha="right", fontsize=6.5); ax.text(29.5, p * p / 2 * 100 + 0.25, "p²/2", ha="right", fontsize=6.5)
ax.set_xlabel("Separation of cloud-layer intersections (km)"); ax.set_ylabel("Mean loss (%)")
ax.legend(loc="center right", fontsize=6.5)
save(f, 1)

# image2 / Suppl. Fig. S1: integrated intensity vs pointwise inversion
f = fig(2); ax = f.add_subplot(111)
ka = np.linspace(0, 3, 200)
f0 = 0.4
point = 1 - np.exp(np.log(1 - f0) * phi(ka))
ax.plot(ka, point * 100, color=C2, label="Pointwise inversion at disc centre")
ax.plot(ka, np.full_like(ka, f0 * 100), color=C1, label="Integrated intensity, Eq. (6)")
ax.set_xlabel("Gradient × disc radius, κa"); ax.set_ylabel("Blockage probability (%)")
ax.legend(loc="upper left", fontsize=6.5)
save(f, 2)
print("ok")
