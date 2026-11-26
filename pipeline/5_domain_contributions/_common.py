"""Shared input, OOF replay and statistical helpers for channel 2."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,torch
from scipy.special import softmax,expit
from scipy import stats

ROOT=paths.out(paths.EVIDENCE)
INTER=paths.INTER
SILVER=paths.SILVER
SEED=42

def json_write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x))
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=convert))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()

def metadata():
    ids=np.load(INTER/'report_id_array.npy',allow_pickle=True).astype(str)
    y=np.load(INTER/'labels_np.npy');fold=np.load(INTER/'fold_assignment.npy')
    mask=np.load(INTER/'valid_sentence_mask_np.npy').astype(bool)
    patients=np.load(INTER/'patient_id_array.npy',allow_pickle=True).astype(str)
    dom=json.loads((paths.VECTORS/'domain_vectors_meta_latest.json').read_text())['domain_columns']
    rtype=np.array([r.split('-')[-1][0] for r in ids])
    return dict(ids=ids,y=y,fold=fold,mask=mask,patient=patients,domain=dom,rtype=rtype)

def nuisance(meta,include_fold=True):
    cols=[np.ones(len(meta['ids'])),(meta['rtype']=='P').astype(float)]
    if include_fold:cols.extend((meta['fold']==f).astype(float) for f in range(2,6))
    return np.column_stack(cols)

def residual(values,design):
    return values-design@np.linalg.lstsq(design,values,rcond=None)[0]

def bh(p):
    p=np.asarray(p,float);order=np.argsort(p);q=np.empty(len(p));last=1.
    for i in range(len(p)-1,-1,-1):
        j=order[i];last=min(last,float(p[j])*len(p)/(i+1));q[j]=last
    return q

def head_state(fold):
    path=paths.DATA_ROOT/'pipeline/1_classifier/trained_models'/f'asd_classifier_fold{fold}.pt'
    if path.is_file():
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        return {k:state[k].detach().cpu().numpy().astype(np.float64).copy() for k in state if 'classifier.' in k or 'sentence_attention.' in k},path
    cached=paths.HEADS/f'fold{fold}.npz'
    if not cached.is_file():raise FileNotFoundError(f'Missing trained classifier or cached head for fold {fold}')
    return dict(np.load(cached,allow_pickle=False)),cached

def encode_from_cls(cls,state):
    h=np.tanh(cls@state['base_model.classifier.dense.weight'].T+state['base_model.classifier.dense.bias'])
    return h@state['base_model.classifier.out_proj.weight'].T+state['base_model.classifier.out_proj.bias']

def project(x,state):
    W=state['sentence_attention.attn.in_proj_weight'];b=state['sentence_attention.attn.in_proj_bias'];D=x.shape[1]
    q=x@W[:D].T+b[:D];k=x@W[D:2*D].T+b[D:2*D]
    values=x@W[2*D:].T+b[2*D:]
    Wo=state['sentence_attention.attn.out_proj.weight'];bo=state['sentence_attention.attn.out_proj.bias']
    w=state['sentence_attention.linear.weight'].ravel();bias=float(w@bo+state['sentence_attention.linear.bias'][0])
    s=values@(w@Wo)+bias
    score=q@k.T/np.sqrt(D)
    return s,score,bias

def bag_logit(score,s,indices=None):
    if indices is not None:
        score=score[np.ix_(indices,indices)];s=s[indices]
    if len(s)==0:raise ValueError('Empty bag is outside the observed input support')
    return float(softmax(score,axis=-1).mean(0)@s)

def batched_bag_logit(score,s,indices):
    indices=np.asarray(indices);G=score[indices[:,:,None],indices[:,None,:]]
    return np.einsum('bj,bj->b',softmax(G,axis=-1).mean(1),s[indices])

def label_tensor(meta):
    N,S=meta['mask'].shape;D=len(meta['domain']);M=np.zeros((N,S,D));counts=np.zeros((N,D),int)
    di={d:i for i,d in enumerate(meta['domain'])};ri={r:i for i,r in enumerate(meta['ids'])};seen=set()
    with SILVER.open() as f:
        for line in f:
            row=json.loads(line);i=ri[row['report_id']];j=int(row['sentence_idx'])
            if (i,j) in seen:raise ValueError('Duplicate silver sentence')
            seen.add((i,j));domains=[x['domain_id'] for x in row['labels']]
            if not meta['mask'][i,j] or not domains or len(set(domains))!=len(domains):raise ValueError('Invalid silver/input alignment')
            for domain in domains:M[i,j,di[domain]]=1/len(domains);counts[i,di[domain]]+=1
    assert len(seen)==int(meta['mask'].sum())
    assert np.allclose(M.sum(-1),meta['mask'])
    return M,counts

def load_core():
    return np.load(ROOT/'channel2_core.npz',allow_pickle=False)

