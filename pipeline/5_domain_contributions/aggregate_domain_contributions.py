"""Allocate each sentence contribution equally across its assigned domains."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import numpy as np
import pandas as pd
import _common as cc

def main():
    m=cc.metadata();a=dict(np.load(paths.EVIDENCE/'sentence_contributions.npz',allow_pickle=False));M,counts=cc.label_tensor(m)
    mass=M.sum(1);P=mass/m['mask'].sum(1)[:,None];E=np.einsum('ns,nsd->nd',a['contribution'],M)
    out=paths.out(paths.EVIDENCE)
    np.save(out/'sentence_label_mass.npy',M)
    np.savez(out/'channel2_core.npz',**a,report_id=m['ids'],label=m['y'],fold=m['fold'],report_type=m['rtype'],valid_mask=m['mask'],domain=np.asarray(m['domain']),P=P,E=E,domain_mass=mass,domain_sentence_count=counts)
    if not np.allclose(E.sum(1),a['logit'],atol=1e-4,rtol=0):raise ValueError('Domain contributions do not reconstruct logits')
    rows=[]
    for f in range(1,6):
        ix=m['fold']==f; sd=float(a['logit'][ix].std())
        rows.append(dict(fold=f,n_reports=int(ix.sum()),logit_mean=float(a['logit'][ix].mean()),logit_sd=sd,sentence_logit_sd=float(np.nanstd(a['sentence_logit'][ix])),decomposition_max_error=float(abs(a['contribution'][ix].sum(1)-a['logit'][ix]).max()),near_constant_logit=bool(sd<1e-4)))
    pd.DataFrame(rows).to_csv(out/'fold_quality.tsv',sep='\t',index=False)
if __name__=='__main__':main()
