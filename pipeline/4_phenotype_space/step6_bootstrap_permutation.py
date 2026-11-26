# -*- coding: utf-8 -*-
"""Canonical-method analysis on unweighted all-sentence channel 1. See README for methods and observed results."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import sys, os, json
from datetime import datetime
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(_REPO / "pipeline/4_phenotype_space"))
import _common_controlled as cc

HERE = str(paths.out(paths.SPACE))

# ---- settings (option A: 1,000 bootstrap / 10,000 permutation), all k 2..8 ----
K_LIST     = [2, 3, 4, 5, 6, 7, 8]
N_BOOT     = 1000
N_PERM     = 10000
GMM_N_INIT = 10   # matches seed_stability's base fit


def gmm(k, seed):
    return GaussianMixture(n_components=k, covariance_type="full",
                           n_init=GMM_N_INIT, max_iter=500, random_state=seed)


def summarise(a, name):
    a = np.asarray(a, dtype=float)
    return {"metric": name, "n": int(a.size), "mean": float(a.mean()),
            "std": float(a.std(ddof=1)), "median": float(np.median(a)),
            "interval_definition": "2.5–97.5 percentiles of the ARI distribution; not a confidence interval for its mean",
            "ci95_lo": float(np.percentile(a, 2.5)), "ci95_hi": float(np.percentile(a, 97.5)),
            "min": float(a.min()), "max": float(a.max())}


def run_one_k(k, lead, n, rng):
    ref_labels = gmm(k, cc.SEED).fit(lead).predict(lead)

    # 1. bootstrap: resample -> refit -> predict original -> ARI vs reference
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot[b] = adjusted_rand_score(ref_labels, gmm(k, b).fit(lead[idx]).predict(lead))
    # 2. permutation null: shuffle reference labels -> ARI vs reference
    null = np.empty(N_PERM)
    for p in range(N_PERM):
        perm = ref_labels.copy()
        rng.shuffle(perm)
        null[p] = adjusted_rand_score(ref_labels, perm)

    obs = float(boot.mean())
    cohens_d = (obs - null.mean()) / np.sqrt((boot.var(ddof=1) + null.var(ddof=1)) / 2)
    n_ge = int(np.sum(null >= obs))
    p_one = (n_ge + 1) / (N_PERM + 1)   # add-one convention, never exactly 0
    return {
        "bootstrap": summarise(boot, "ARI bootstrap vs reference"),
        "null": summarise(null, "ARI permutation null"),
        "effect_size": {"calibrated_permutation_p": False,
                        "interpretation": "Descriptive comparison only: bootstrap mean ARI and single shuffled-label null ARI are different statistics; exceedance fraction is not a calibrated permutation p-value","descriptive_standardized_difference": float(cohens_d),
                        "add_one_null_exceedance_fraction": p_one,
                        "n_null_ge_bootstrap_mean": n_ge,
                        "diagnostic_resolution_floor": 1.0 / (N_PERM + 1)},
        "verdict": ("stable" if obs >= cc.STABILITY_THRESHOLD else ("moderate" if obs >= 0.5 else "unstable")),
    }


def main():
    # same controlled pipeline as seed_stability: type-residual ILR -> PCA -> PC1-3
    d = cc.load()
    _, _, lead, evr = cc.pca_scores(d["RES"])
    n = lead.shape[0]
    print(f"[data] type-residual PC1-3, n_asd={n}, PC1-3 var={evr[:3].sum():.4f}")

    rng = np.random.default_rng(cc.SEED)   # one stream, reused across k (reproducible given order)
    per_k = {}
    for k in K_LIST:
        print(f"[k={k}] bootstrap n={N_BOOT}, perm n={N_PERM}, GMM(n_init={GMM_N_INIT}) ...", flush=True)
        rk = run_one_k(k, lead, n, rng)
        per_k[str(k)] = rk
        print(f"  k={k}: bootstrap ARI {rk['bootstrap']['mean']:.3f} "
              f"ARI percentile range[{rk['bootstrap']['ci95_lo']:.3f},{rk['bootstrap']['ci95_hi']:.3f}] | "
              f"null {rk['null']['mean']:.4f} | d={rk['effect_size']['descriptive_standardized_difference']:.2f} | "
              f"null exceedance diagnostic={rk['effect_size']['add_one_null_exceedance_fraction']:.2e} | {rk['verdict']}", flush=True)

    res = {
        "settings": {"k_list": K_LIST, "n_boot": N_BOOT, "n_permutations": N_PERM,
                     "n_pcs": 3, "gmm_n_init": GMM_N_INIT, "seed": cc.SEED,
                     "control": "type-residual"},
        "data": {"n_asd": int(n), "pc13_var": round(float(evr[:3].sum()), 4)},
        "per_k": per_k,
        "note": "Canonical report-level partition bootstrap conditional on the fixed type-residual/PCA representation. Repeated reports are not resampled as independent individuals; this is not full-pipeline or person-population uncertainty. ARI percentile ranges summarize the resampling distribution. Shuffled-label exceedance is descriptive, not a calibrated permutation p-value. 0.8 is the study rule, not a universal cutoff.",
        "timestamp": datetime.now().isoformat(),
    }
    out = os.path.join(HERE, "bootstrap_permutation_results.json")
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=2)
    print("  ->", out)
    return res


if __name__ == "__main__":
    main()
