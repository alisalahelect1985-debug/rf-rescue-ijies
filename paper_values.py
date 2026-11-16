"""Result-to-code audit trail: recompute every number quoted in Section 6 of the manuscript from the result files
and write audit/paper_values.csv (location, quantity, value, script, result file, key)."""
import csv, json
import numpy as np
from analyze import load, table, paired, violations, deficits, hw, arr, B
from paths import RESULTS, AUDIT

rows = []


def add(loc, what, val, script, src, key, fmt="%.3f"):
    rows.append([loc, what, fmt % val if isinstance(val, (float, np.floating)) else str(val), script, src, key])


het, hom, ext = load("het"), load("hom"), load("extra")
T, TH = table(het), table(hom)
for b in (12, 24):
    for pol in ("anchor", "indep", "hybrid1", "portfolio", "condrob", "cloud", "degree"):
        add("Table 3", "mean delivery %s b=%d (%%)" % (pol, b), T[(pol, b)][0], "run_all.py", "het.json", "deliv[%s|%d]" % (pol, b), "%.2f")
        add("Table 3", "certificate %s b=%d (%%)" % (pol, b), T[(pol, b)][2], "run_all.py", "het.json", "cert[%s|%d]" % (pol, b), "%.2f")
abl = {k: np.array([r["abl"]["24"][k] for r in het]) * 100 for k in ("indep", "cond", "field")}
add("Abstract; 6.5", "loss under independence b=24 (%)", abl["indep"].mean(), "run_all.py", "het.json", "abl[24].indep")
add("Abstract; 6.5", "loss under shared weather b=24 (%)", abl["cond"].mean(), "run_all.py", "het.json", "abl[24].cond")
add("Abstract; 6.5", "loss under complete field b=24 (%)", abl["field"].mean(), "run_all.py", "het.json", "abl[24].field")
add("6.5", "paired diff shared-indep (pp)", "%.3f ± %.3f" % ((abl["cond"] - abl["indep"]).mean(), hw(abl["cond"] - abl["indep"])), "run_all.py", "het.json", "abl[24]")
add("6.5", "paired diff field-shared (pp)", "%.3f ± %.3f" % ((abl["field"] - abl["cond"]).mean(), hw(abl["field"] - abl["cond"])), "run_all.py", "het.json", "abl[24]")
for b in B:
    add("6.6, Fig. 2", "independence-greedy violations b=%d" % b, violations(het, "indep", b), "run_all.py", "het.json", "cert vs jmin")
    add("6.6, Fig. 2", "cloud-informed violations b=%d" % b, violations(het, "cloud", b), "run_all.py", "het.json", "cert vs jmin")
    add("6.6, Fig. 2", "portfolio violations b=%d" % b, violations(het, "portfolio", b), "run_all.py", "het.json", "cert vs jmin")
D = deficits(het)
add("6.6, Table S3", "deficit range (pp)", "%.3f to %.2f" % (min(d[4] for d in D), max(d[4] for d in D)), "analyze.py", "het.json", "deficits()")
for b in B:
    m, h, _ = paired(het, "portfolio", b)
    add("6.4, Fig. 1", "paired portfolio-indep b=%d (pp)" % b, "%.4f ± %.4f" % (m, h), "analyze.py", "het.json", "deliv")
m, h, _ = paired(het, "portfolio", 18, "cert")
add("6.4", "certificate gain b=18 (pp)", "%.4f ± %.4f" % (m, h), "analyze.py", "het.json", "cert")
x = paired(het, "portfolio", 18)[2]
add("6.4, Table S2", "ahead/tied/behind b=18", "%d/%d/%d" % ((x > 1e-9).sum(), (abs(x) <= 1e-9).sum(), (x < -1e-9).sum()), "analyze.py", "het.json", "deliv")
pool = np.concatenate([x, paired(ext, "portfolio", 18)[2]])
add("6.4", "pooled 48-geometry difference b=18 (pp)", "%.4f ± %.4f" % (pool.mean(), hw(pool)), "run_all.py", "het.json, extra.json", "deliv")
for b in (12, 18, 24):
    add("6.4", "homogeneous certificate b=%d (%%)" % b, TH[("portfolio", b)][2], "run_all.py", "hom.json", "cert[portfolio|%d]" % b)
nf = [r["n_feasible"][str(b)] for r in het for b in B]
add("5.4", "distinct feasible candidates (min/median/max)", "%d/%g/%d" % (min(nf), np.median(nf), max(nf)), "run_all.py", "het.json", "n_feasible")
need_d, need_c = arr(het, "deliv", "need", 24).mean() * 100, arr(het, "cert", "need", 24).mean() * 100
add("6.7", "need-based [24] delivery / certificate b=24 (%)", "%.2f / %.2f" % (need_d, need_c), "run_all.py", "het.json", "need|24")
add("6.7", "need-based violations b=24", violations(het, "need", 24), "run_all.py", "het.json", "need|24")
for v in ("resid03", "markov09"):
    L = np.array([1 - r["variants"][v]["24"]["portfolio"] for r in het]) * 100
    add("6.7", "departure %s loss b=24 (%%)" % v, L.mean(), "run_all.py", "het.json", "variants.%s" % v)
    add("6.7", "departure %s half-width (pp)" % v, hw(100 - L), "run_all.py", "het.json", "variants.%s" % v)
add("6.7", "baseline half-width b=24 (pp)", hw(arr(het, "deliv", "portfolio", 24) * 100), "run_all.py", "het.json", "deliv")
tg = json.load(open(RESULTS / "targets.json"))
for t in ("0.86", "0.87", "0.88", "0.89", "0.9", "0.91"):
    rec = [g["t"][t] for g in tg]
    fe = [r for r in rec if r["feasible"]]
    add("Table 5", "target %s feasible/active" % t, "%d/%d" % (len(fe), sum(r["active"] for r in fe)), "run_targets.py", "targets.json", "t[%s]" % t)
    if fe:
        add("Table 5", "target %s nominal cost (pp)" % t, np.mean([(r["d_ind"] - r["d_port"]) * 100 for r in fe]), "run_targets.py", "targets.json", "d_ind-d_port")
        g_ = [(r["d_exact"] - r["d_port"]) * 100 for r in fe]
        add("Table 5", "target %s gap to exact (pp)" % t, ("%.3f ± %.3f" % (np.mean(g_), hw(g_))) if len(g_) > 1 else "%.3f" % g_[0], "run_targets.py", "targets.json", "d_exact-d_port")
inter = json.load(open(RESULTS / "inter.json"))
for K in (12, 20, 30):
    g_ = [(r["d_exact"] - r["d_port"]) * 100 for r in inter if r["K"] == K]
    add("6.7", "intermediate K=%d test gap (pp)" % K, "%.3f ± %.3f" % (np.mean(g_), hw(g_)), "run_targets.py", "inter.json", "d_exact-d_port")
ck = json.load(open(RESULTS / "checks.json"))
add("6.3", "envelope vs LP max difference", "%.2e" % ck["lp"]["max_lp"], "run_checks.py", "checks.json", "lp.max_lp")
add("Suppl.", "Hall identities / supermodularity inequalities", "%d / %d" % (ck["lp"]["n_hall"], ck["lp"]["n_sup"]), "run_checks.py", "checks.json", "lp")
add("Suppl.", "sampler max discrepancy / margin", "%.4f / %.4f" % (ck["sampler"]["max_disc"], ck["sampler"]["margin"]), "run_checks.py", "checks.json", "sampler")
mi = json.load(open(RESULTS / "misc.json"))
add("6.5", "radius study delivery 0.5x / 4x (%)", "%.3f / %.3f" % (mi["radius"]["0.5"][0], mi["radius"]["4.0"][0]), "run_misc.py", "misc.json", "radius")
add("6.6", "six-flow binding (het) / mean cost / mean gain", "%d / %.2f / %.2f" % (mi["six"]["het"]["binding"], mi["six"]["het"]["mean_cost"], mi["six"]["het"]["mean_gain"]), "run_misc.py", "misc.json", "six.het")
add("6.6", "runtime portfolio het / hom (s), matching (us)", "%.2f / %.2f / %.2f" % (mi["runtime"]["port_het"], mi["runtime"]["port_hom"], mi["runtime"]["match_us"]), "run_misc.py", "misc.json", "runtime")
try:
    sc = json.load(open(RESULTS / "schedule.json"))
    G = sc["geometries"]
    V = sum(sum(o["viol"].values()) for o in G)
    add("6.2", "schedule pairs checked / violations", "%d / %d" % (sum(o["pairs"] for o in G), V), "run_schedule.py", "schedule.json", "geometries")
    add("6.2", "worst SNR margin (dB)", min(o["worst"]["snr_margin"] for o in G), "run_schedule.py", "schedule.json", "worst.snr_margin", "%.2f")
    add("6.2", "minimum arrival slack (ms)", min(o["worst"]["arrival_slack"] for o in G), "run_schedule.py", "schedule.json", "worst.arrival_slack", "%.1f")
    add("6.2", "latest completion (ms)", max(o["worst"]["completion"] for o in G), "run_schedule.py", "schedule.json", "worst.completion", "%.2f")
    add("6.2", "replicas per frame b=24 (deduplicated per satellite)", np.mean([o["replicas24"] for o in G if o["stream"] == "het"]), "run_schedule.py", "schedule.json", "replicas24", "%.1f")
except FileNotFoundError:
    pass
try:
    at = json.load(open(RESULTS / "audit_ties.json"))["summary"]
    add("6.1", "anchor delivery range over 22 tie rules b=24 (%)", "%.2f to %.2f" % (at["min"], at["max"]), "run_audit_ties.py", "audit_ties.json", "summary")
    add("6.1", "portfolio delivery, two independent 24-geometry sets b=24 (%)", "%.2f / %.2f" % (at["sampling"]["portfolio"]["set1"], at["sampling"]["portfolio"]["set2"]), "run_audit_ties.py", "het.json, extra.json", "summary.sampling")
except FileNotFoundError:
    pass

with open(AUDIT / "paper_values.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["location", "quantity", "value", "script", "result_file", "key"])
    w.writerows(rows)
print("wrote", len(rows), "values to", AUDIT / "paper_values.csv")
