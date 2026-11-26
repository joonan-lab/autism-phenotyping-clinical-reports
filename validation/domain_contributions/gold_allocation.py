"""Compare fixed sentence contributions allocated using gold or silver labels."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import json
import numpy as np
import pandas as pd
from _common import load,corr

def main():
 cc,a,m,_,M=load();ri={r:i for i,r in enumerate(a['report_id'])};di={d:j for j,d in enumerate(a['domain'])};silver={};gold={}
 for file,out in [(paths.SILVER,silver),(paths.GOLD,gold)]:
  for line in file.open():
   r=json.loads(line);out.setdefault(r['report_id'],[]).append(r)
 MG=np.zeros_like(M)
 for rid,gr in gold.items():
  sr=sorted(silver[rid],key=lambda x:x['sentence_idx']);gr=sorted(gr,key=lambda x:x['sentence_idx'])
  if len(sr)!=len(gr):raise ValueError('Sentence counts differ')
  for s,g in zip(sr,gr):
   if s['sentence']!=g['sentence']:raise ValueError('Sentence text mismatch')
   ds=[x['domain_id'] if isinstance(x,dict) else x for x in g['labels']]
   for d in ds:MG[ri[rid],int(s['sentence_idx']),di[d]]=1/len(ds)
 EG=np.einsum('ns,nsd->nd',a['contribution'],MG);rows=[]
 for rid in sorted(gold):
  i=ri[rid];common=(a['E'][i]!=0)|(EG[i]!=0)
  rows.append(dict(report_id=rid,fold=int(a['fold'][i]),rank_spearman=corr(a['E'][i,common],EG[i,common]),sign_agreement=float(np.mean(np.sign(a['E'][i,common])==np.sign(EG[i,common])))))
 pd.DataFrame(rows).to_csv(paths.out(paths.E_RESULTS)/'gold_E_preservation.tsv',sep='\t',index=False)
if __name__=='__main__':main()
