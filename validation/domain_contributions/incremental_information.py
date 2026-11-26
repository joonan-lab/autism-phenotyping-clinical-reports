"""Incremental prediction of domain-deletion logit changes in held-out individuals."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import pandas as pd
from _common import load,incremental,js

def main():
 _,_,m,*_=load();out=paths.out(paths.E_RESULTS);df=pd.read_csv(out/'domain_deletions.tsv',sep='\t');groups=m['patient'][df.report_index.values]
 summaries,predictions,tuning=incremental(df,groups)
 pd.DataFrame(summaries).to_csv(out/'heldout_model_comparisons.tsv',sep='\t',index=False)
 pd.DataFrame(predictions).to_csv(out/'grouped_heldout_predictions.tsv',sep='\t',index=False)
 pd.DataFrame(tuning).to_csv(out/'inner_tuning.tsv',sep='\t',index=False)
 js(out/'incremental_information_summary.json',dict(comparisons=summaries,bootstrap=1000,outer_folds=5,inner_folds=4))
if __name__=='__main__':main()
