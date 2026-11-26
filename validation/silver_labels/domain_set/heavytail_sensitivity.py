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

BASE = paths.out(paths.SILVER_RESULTS / "domain_set")
sys.path.insert(0, str(BASE.parent))
sys.path.insert(0, str(_REPO / "validation/silver_labels/domain_set"))
import _sensitivity_common as sc               # noqa: E402
cc = sc.cc


def analyze(RES):
    _, scores, lead, evr = cc.pca_scores(RES)
    full = cc.mardia_z(RES)
    lead_m = cc.mardia_z(scores[:, :3])
    resid_m = cc.mardia_z(scores[:, 3:])
    pattern = bool(lead_m["z"] < 0 and resid_m["z"] > 0)
    return dict(
        n_ilr_dim=int(RES.shape[1]),
        full_beta2=round(full["beta2"], 1), full_expected=full["expected"], full_z=round(full["z"], 1),
        full_kurt=full["kurt"], full_excess99=round(full["excess99"], 3),
        lead3_z=round(lead_m["z"], 2), lead3_kurt=lead_m["kurt"], lead3_var_frac=round(float(evr[:3].sum()), 3),
        resid_z=round(resid_m["z"], 2), resid_kurt=resid_m["kurt"], resid_var_frac=round(float(evr[3:].sum()), 3),
        pattern_preserved=pattern,
        residual_heavy_tail=bool(resid_m["z"] > 1.96),
        leading_positive_excess_kurtosis=bool(lead_m["z"] > 1.96))


def main():
    d = cc.load()
    variants, dom = sc.build_variants(d)
    order = ["baseline_full19", "drop_sentence_f1_lt_0.3", "drop_report_rho_lt_0.3",
             "drop_report_rho_lt_0", "downweight_sentence_f1", "downweight_report_rho"]
    res = {"seed": cc.SEED, "n_asd": int(len(d["P"])), "variants": {}}
    for name in order:
        v = variants[name]
        r = analyze(v["RES"])
        r["kind"] = v["kind"]
        r["rank_projection"] = v.get("rank_projection")
        r["weighting_interpretation"] = v.get("interpretation")
        res["variants"][name] = r
        print(f"[{name}] full z={r['full_z']} ({r['full_kurt']}) | lead3 z={r['lead3_z']} ({r['lead3_kurt']}) "
              f"| resid z={r['resid_z']} ({r['resid_kurt']}) | pattern={r['pattern_preserved']}")
    out = BASE / "sensitivity_heavytail_results.json"
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=2)
    print("  ->", out)


if __name__ == "__main__":
    main()
