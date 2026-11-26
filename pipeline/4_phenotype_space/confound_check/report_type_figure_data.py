"""PC1 and report type before and after adjustment."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import json
import numpy as np
from scipy.stats import pearsonr
sys.path.insert(0,str(_REPO/'pipeline/4_phenotype_space'))
import _common_controlled as cc

def main():
 d=cc.load();t=(d['rtype']=='P').astype(float);raw=cc.pca_scores(d['ILR'])[1][:,0];adj=cc.pca_scores(d['RES'])[1][:,0]
 result=dict(n_reports=len(t),panel_b=dict(report_type=d['rtype'].tolist(),pc1_raw=np.round(raw,4).tolist(),pc1_adjusted=np.round(adj,4).tolist(),corr_pc1_type_raw=round(float(pearsonr(raw,t).statistic),3),corr_pc1_type_adjusted=round(float(pearsonr(adj,t).statistic),3)))
 (paths.out(paths.SPACE)/'t2_report_type_figure_data.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
