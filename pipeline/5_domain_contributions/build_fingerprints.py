"""Add signed domain contributions to cohort-relative phenotype fingerprints."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import numpy as np
import pandas as pd
sys.path.insert(0,str(_REPO/'pipeline/4_phenotype_space'))
import _common_controlled as cc

def main():
 d=cc.load();a=np.load(paths.EVIDENCE/'channel2_core.npz',allow_pickle=False);lookup={r:i for i,r in enumerate(a['report_id'])};ix=np.array([lookup[r] for r in d['rid']])
 R=d['RES'];x=R-R.mean(0);inv=np.linalg.inv(np.cov(R,rowvar=False));distance=np.sqrt(np.einsum('ij,jk,ik->i',x,inv,x))
 z=(d['CLR_RES']-d['CLR_RES'].mean(0))/d['CLR_RES'].std(0,ddof=1)
 table=pd.DataFrame(dict(report_id=d['rid'],report_type=d['rtype'],fold=a['fold'][ix],distance=distance,tail_decile=distance>=np.quantile(distance,.9),raw_logit=a['logit'][ix]))
 table['model_probability']=1/(1+np.exp(-a['logit'][ix]))
 for j,dom in enumerate(d['dom']): table['salience_z_'+dom]=z[:,j];table['E_'+dom]=a['E'][ix,j]
 # The filename is retained for the current R input contract.
 table.to_csv(paths.out(paths.EVIDENCE)/'two_channel_fingerprints.tsv',sep='\t',index=False)
if __name__=='__main__':main()
