# -*- coding: utf-8 -*-
"""Canonical-method analysis on unweighted all-sentence channel 1. See README for methods and observed results."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import sys, os, json
import numpy as np
from sklearn.decomposition import PCA
sys.path.insert(0, str(_REPO / "pipeline/4_phenotype_space"))
import _common_controlled as cc

HERE = str(paths.out(paths.SPACE))
# group A = 7 ADOS-aligned DSM-5 ASD-core; B = 9 general child-psychiatric; C = 3 format


def share_by_group(per_domain_value, dom):
    g = {"A": 0.0, "B": 0.0, "C": 0.0}
    for i, d in enumerate(dom):
        grp = "A" if i < 7 else ("B" if i < 16 else "C")
        g[grp] += float(per_domain_value[i])
    tot = sum(g.values())
    return {k: round(v / tot, 4) for k, v in g.items()}


def main():
    d = cc.load()
    dom, P, RES = list(d["dom"]), d["P"], d["RES"]

    # proportion-space variance share
    prop = share_by_group(P.var(axis=0, ddof=1), dom)

    # CLR-space variance share on the type-residual PCA (manuscript space)
    p = PCA().fit(RES)
    V = cc.helmert(len(dom))
    load_clr = V @ (p.components_.T * np.sqrt(np.maximum(p.explained_variance_, 0)))  # 19 x 18
    clr_per_domain = np.sum(load_clr ** 2, axis=1)                                    # 19
    clr = share_by_group(clr_per_domain, dom)

    # per-domain CLR-space contribution (% of total), for Figure 4c
    codes = [f"A{i+1}" for i in range(7)] + [f"B{i+1}" for i in range(9)] + \
            [f"C{i+1}" for i in range(len(dom) - 16)]
    tot_clr = float(clr_per_domain.sum())
    per_domain = sorted(
        [dict(domain=dom[i], code=codes[i],
              group=("ADOS-core" if i < 7 else "non-ADOS"),
              clr_variance=float(clr_per_domain[i]),
              contribution_pct=round(100 * clr_per_domain[i] / tot_clr, 2))
         for i in range(len(dom))],
        key=lambda r: -r["contribution_pct"])

    res = {
        "n_asd": int(len(P)), "n_domains": len(dom),
        "group_def": {"A_ADOS_core_7": dom[:7], "B_general_9": dom[7:16], "C_format_3": dom[16:]},
        "proportion_space_share": prop,
        "proportion_space_outside_ADOS": round(prop["B"] + prop["C"], 4),
        "clr_space_share": clr,
        "clr_space_outside_ADOS": round(clr["B"] + clr["C"], 4),
        "per_domain_clr": per_domain,
        "note": "All-sentence domain profiles. Interpret fitted results together with bootstrap and seed stability; 0.8 is the study decision rule, not a universal cutoff.",
    }
    out = os.path.join(HERE, "step11_ados_variance_share_results.json")
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"[step11] proportion-space: core {prop['A']:.3f} / outside {res['proportion_space_outside_ADOS']:.3f}")
    print(f"         CLR-space (manuscript): core {clr['A']:.3f} / outside {res['clr_space_outside_ADOS']:.3f}")
    print("  ->", out)


if __name__ == "__main__":
    main()
