# -*- coding: utf-8 -*-
"""Canonical-method analysis on unweighted all-sentence channel 1. See README for methods and observed results."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, str(_REPO / "pipeline/4_phenotype_space"))
import _common_controlled as cc

HERE = str(paths.out(paths.SPACE))
d = cc.load()
_, scores, lead, evr = cc.pca_scores(d["RES"])

mu = scores.mean(0); S = np.cov(scores, rowvar=False); Sinv = np.linalg.inv(S)
maha_full = np.sqrt(np.einsum("ij,jk,ik->i", scores - mu, Sinv, scores - mu))
muL = lead.mean(0); SL = np.cov(lead, rowvar=False); SLinv = np.linalg.inv(SL)
maha_lead = np.sqrt(np.einsum("ij,jk,ik->i", lead - muL, SLinv, lead - muL))

CLR_RES = d["CLR_RES"]
zdom = (CLR_RES - CLR_RES.mean(0)) / CLR_RES.std(0, ddof=1)


dom, rid, rtype = d["dom"], d["rid"], d["rtype"]
prof = pd.DataFrame({"report_id": rid, "report_type": rtype,
                     "mahalanobis_full": np.round(maha_full, 3),
                     "mahalanobis_lead3": np.round(maha_lead, 3)})
for j, dd in enumerate(dom):
    prof["z_" + dd] = np.round(zdom[:, j], 3)
prof["max_dev_domain"] = [dom[i] for i in np.argmax(np.abs(zdom), 1)]
prof["max_dev_z"] = np.round(zdom[np.arange(len(zdom)), np.argmax(np.abs(zdom), 1)], 3)
prof.to_csv(HERE + "/controlled_individual_profiles.tsv", sep="\t", index=False)

q90 = float(np.percentile(maha_full, 90))
res = {"n_asd": len(prof), "n_A": d["n_A"], "n_P": d["n_P"],
       "layer1_prototypicality_mahalanobis_full": {
           "mean": round(float(maha_full.mean()), 3), "p90_tail_threshold": round(q90, 3),
           "max": round(float(maha_full.max()), 3), "n_top_decile": int((maha_full >= q90).sum())},
       "layer2_deviation": "z_<domain> in type-controlled CLR space",
       "profiles_tsv": "controlled_individual_profiles.tsv"}
json.dump(res, open(HERE + "/step7_normative_results.json", "w"), ensure_ascii=False, indent=2)
print("[step7] Mahalanobis mean=%.2f p90=%.2f max=%.2f" %
      (maha_full.mean(), q90, maha_full.max()))
print("  →", HERE + "/controlled_individual_profiles.tsv")

np.savez(paths.SPACE / "channel1_type_adjusted_coordinates.npz", report_id=rid.astype(str), report_type=rtype, domain=np.asarray(dom), P=d["P"], ILR=d["ILR"], RES=d["RES"], CLR_RES=CLR_RES, pca_scores=scores, pca_components=cc.pca_scores(d["RES"])[0].components_, explained_variance_ratio=evr)
