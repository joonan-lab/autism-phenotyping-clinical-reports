"""Report prevalence of absolute type-adjusted CLR deviations above two."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import json
import numpy as np
import pandas as pd
import _common_controlled as cc

def main():
 d=cc.load(); M=d['CLR_RES'];z=(M-M.mean(0))/M.std(0,ddof=1)
 rows=[dict(domain=dom,all_sentences_n=int((abs(z[:,j])>2).sum()),all_sentences_pct=float((abs(z[:,j])>2).mean()*100),positive_n=int((z[:,j]>2).sum()),negative_n=int((z[:,j]<-2).sum())) for j,dom in enumerate(d['dom'])]
 out=paths.out(paths.SPACE)
 pd.DataFrame(rows).to_csv(out/'domain_deviation_prevalence.tsv',sep='\t',index=False)
 (out/'domain_deviation_prevalence.json').write_text(json.dumps(dict(n_reports=len(z),threshold=2,per_domain=rows),indent=2))
if __name__=='__main__':main()
