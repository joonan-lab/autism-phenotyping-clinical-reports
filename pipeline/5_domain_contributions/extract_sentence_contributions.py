"""Replay OOF classifier heads and calculate additive sentence contributions."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import argparse
import shutil
import numpy as np
from scipy.special import softmax
import _common as cc

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--from-cache',action='store_true',help='Use stored full-precision sentence scores and attention instead of replaying classifier heads');args=ap.parse_args()
    m=cc.metadata(); N,S=m['mask'].shape
    stored=paths.INTER/'sentence_contributions.npz'
    if args.from_cache:
        a=dict(np.load(stored,allow_pickle=False));c=a['attention_valid']*np.nan_to_num(a['sentence_logit'])
        if not np.allclose(c.sum(1),a['logit'],atol=1e-4,rtol=0):raise ValueError('Stored sentence quantities do not reconstruct logits')
        out=paths.out(paths.EVIDENCE);np.savez(out/'sentence_contributions.npz',**{**a,'contribution':c})
        shutil.copy2(paths.INTER/'compatibility_scores.npy',out/'compatibility_scores.npy')
        print(f'Maximum logit reconstruction error {abs(c.sum(1)-a["logit"]).max():.8g}')
        return
    cached=paths.INTER/'sentence_cls.npy'
    states=cached if cached.exists() else paths.INTER/'lhs_embs_np.npy'
    if not states.is_file() or states.stat().st_size==0:
        raise FileNotFoundError(f'Classifier-head replay requires sentence_cls.npy (reports × sentences × hidden size) or a non-empty lhs_embs_np.npy in {paths.INTER}. Use --from-cache only to reuse stored sentence quantities.')
    lhs=np.load(states,mmap_mode='r')
    savedA=np.load(paths.INTER/'attention_matrices_np.npy',mmap_mode='r')
    savedL=np.load(paths.INTER/'logits_np.npy')
    s=np.full((N,S),np.nan); attention=np.zeros((N,S)); c=np.zeros((N,S)); bias=np.zeros(N); G=np.zeros((N,S,S))
    for f in range(1,6):
        state,_=cc.head_state(f)
        for i in np.flatnonzero(m['fold']==f):
            valid=np.flatnonzero(m['mask'][i]); cls=np.asarray(lhs[i,valid] if lhs.ndim==3 else lhs[i,valid,0,:],dtype=np.float64)
            ss,score,b=cc.project(cc.encode_from_cls(cls,state),state)
            w=np.asarray(savedA[i])[np.ix_(valid,valid)].mean(0)
            if not np.allclose(softmax(score,axis=-1),np.asarray(savedA[i])[np.ix_(valid,valid)],atol=1e-5,rtol=0): raise ValueError('Attention replay mismatch')
            s[i,valid]=ss;attention[i,valid]=w;c[i,valid]=w*ss;bias[i]=b;G[i][np.ix_(valid,valid)]=score
    error=abs(c.sum(1)-savedL)
    if error.max()>1e-4: raise ValueError('Logit decomposition mismatch')
    out=paths.out(paths.EVIDENCE)
    np.savez(out/'sentence_contributions.npz',sentence_logit=s,attention_valid=attention,contribution=c,bias=bias,logit=savedL)
    np.save(out/'compatibility_scores.npy',G)
    print(f'Maximum logit reconstruction error {error.max():.8g}')
if __name__=='__main__':main()
