# -*- coding: utf-8 -*-
"""Domain-fidelity sensitivity of all-sentence channel 1, using canonical statistical helpers."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

BASE = paths.out(paths.SILVER_RESULTS / "domain_set")
sys.path.insert(0, str(BASE.parent))
sys.path.insert(0, str(_REPO / "validation/silver_labels/domain_set"))
import _sensitivity_common as sc               # noqa: E402
cc = sc.cc

N_PERM = 1000


def mahalanobis_full(RES):
    _, scores, lead, evr = cc.pca_scores(RES)
    mu = scores.mean(0)
    Sinv = np.linalg.inv(np.cov(scores, rowvar=False))
    m = np.sqrt(np.einsum("ij,jk,ik->i", scores - mu, Sinv, scores - mu))
    return m, lead


def perm_null_between(X, labels, rng, n=N_PERM):
    g = X.mean(0); total = ((X - g) ** 2).sum()
    sizes = [int(np.sum(labels == k)) for k in np.unique(labels)]
    idx = np.arange(len(X)); out = []
    for _ in range(n):
        rng.shuffle(idx); lab = np.empty(len(X), int); s = 0
        for ki, sz in enumerate(sizes):
            lab[idx[s:s + sz]] = ki; s += sz
        b = sum(np.sum(lab == ki) * ((X[lab == ki].mean(0) - g) ** 2).sum() for ki in range(len(sizes)))
        out.append(b / total)
    return float(np.mean(out)), float(np.percentile(out, 95))


def main():
    d = cc.load()
    rid = list(d["rid"])

    variants, dom = sc.build_variants(d)
    rng = np.random.default_rng(cc.SEED)

    # baseline reference ordering
    base_m, _ = mahalanobis_full(variants["baseline_full19"]["RES"])
    base_q90 = np.percentile(base_m, 90)
    base_tail = set(np.where(base_m >= base_q90)[0])

    order = ["baseline_full19", "drop_sentence_f1_lt_0.3", "drop_report_rho_lt_0.3",
             "drop_report_rho_lt_0", "downweight_sentence_f1", "downweight_report_rho"]
    res = {"seed": cc.SEED, "n_asd": int(len(d["P"])),
           "baseline_between_fraction_ref": "See computed baseline_full19 below", "variants": {}}
    for name in order:
        v = variants[name]
        m, lead = mahalanobis_full(v["RES"])
        q90 = float(np.percentile(m, 90))
        tail = set(np.where(m >= q90)[0])
        # Layer 2: 2-way split variance decomposition
        lab2, _, _ = cc.gmm_fit(lead, 2)
        _, _, _, eta2 = cc.variance_partition(v["RES"], lab2)
        null_mean, null_p95 = perm_null_between(v["RES"], lab2, rng)
        r = dict(
            kind=v["kind"], n_domains=v["n_domains"],
            layer1_mahalanobis=dict(
                mean=round(float(m.mean()), 3), p90=round(q90, 3), max=round(float(m.max()), 3),
                n_top_decile=int((m >= q90).sum())),
            layer1_vs_baseline=dict(
                rank_spearman=round(float(spearmanr(m, base_m).statistic), 3),
                top_decile_overlap=len(tail & base_tail), top_decile_n=len(base_tail)),
            layer2_variance=dict(
                between_fraction=round(float(eta2), 4), within_fraction=round(float(1 - eta2), 4),
                perm_null_between_mean=round(null_mean, 4), perm_null_between_p95=round(null_p95, 4),
                mode_sizes=[int(np.sum(lab2 == k)) for k in np.unique(lab2)]),
        )
        r["rank_projection"] = v.get("rank_projection")
        r["weighting_interpretation"] = v.get("interpretation")
        res["variants"][name] = r
    out = BASE / "sensitivity_fingerprint_results.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=2)
    print("  ->", out)


if __name__ == "__main__":
    main()
