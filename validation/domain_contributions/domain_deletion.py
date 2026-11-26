"""Domain deletion with re-attention and 200 size-matched random deletions."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import numpy as np
import pandas as pd
from _common import load,bags,random_losses,corr,clustered_ci,js

def main():
 cc,a,m,Gall,Mall=load();rows=[]
 for i in range(len(a['report_id'])):
  valid=np.flatnonzero(a['valid_mask'][i]);n=len(valid);G=Gall[i][np.ix_(valid,valid)];s=a['sentence_logit'][i,valid];labels=Mall[i,valid];full=bags(G,s,np.arange(n),{'mode':'valid_queries'});cache={}
  if abs(full-a['logit'][i])>1e-4:raise ValueError('Full-report replay mismatch')
  for j,domain in enumerate(a['domain']):
   removed=np.flatnonzero(labels[:,j]>0);k=len(removed)
   if not 0<k<n:continue
   delta=full-bags(G,s,np.setdiff1d(np.arange(n),removed),{'mode':'valid_queries'})
   if k not in cache:cache[k]=random_losses(i,n,k,G,s,{'mode':'valid_queries'},None,full)
   rd=cache[k]
   rows.append(dict(report_id=a['report_id'][i],report_index=i,domain=domain,domain_index=j,fold=int(a['fold'][i]),report_type=a['report_type'][i],asd_label=int(a['label'][i]),n_sentences=n,n_removed=k,removed_fraction=k/n,P=float(a['P'][i,j]),L=full,E=float(a['E'][i,j]),L_times_P=float(full*a['P'][i,j]),delta_logit=float(delta),random_delta_mean=float(rd.mean()),random_abs_delta_mean=float(abs(rd).mean()),signed_excess=float(delta-rd.mean()),absolute_delta=float(abs(delta)),absolute_excess=float(abs(delta)-abs(rd).mean())))
 df=pd.DataFrame(rows);out=paths.out(paths.E_RESULTS)
 df.to_csv(out/'domain_deletions.tsv',sep='\t',index=False);summary=[]
 for scope,sub in [('all',df)]+[(f'fold{f}',df[df.fold==f]) for f in range(1,6)]+[('ASD',df[df.asd_label==1]),('nonASD',df[df.asd_label==0])]:
  for feature in ['E','L_times_P']:
   summary.append(dict(scope=scope,feature=feature,n=len(sub),spearman_signed_delta=corr(sub[feature],sub.delta_logit),spearman_signed_excess=corr(sub[feature],sub.signed_excess),spearman_abs_delta=corr(abs(sub[feature]),sub.absolute_delta),spearman_absolute_excess=corr(abs(sub[feature]),sub.absolute_excess),sign_agreement=float(np.mean(np.sign(sub[feature])==np.sign(sub.delta_logit)))))
 pd.DataFrame(summary).to_csv(out/'domain_associations.tsv',sep='\t',index=False)
 js(out/'domain_deletion_summary.json',dict(n_domain_deletions=len(df),random_repetitions=200,associations=summary))
if __name__=='__main__':main()
