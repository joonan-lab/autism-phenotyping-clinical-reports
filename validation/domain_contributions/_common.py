"""Domain deletion and individual-grouped prediction helpers."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,mean_absolute_error
SEED=42; B=1000; RANDOM=200
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
 if isinstance(v,(np.integer,)):return int(v)
 if isinstance(v,(np.bool_,)):return bool(v)
 if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
 if isinstance(v,Path):return str(v)
 return v


def js(p,v):Path(p).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def bags(G,s,kept,adapter,pad=None):
 kept=np.asarray(kept,int);single=kept.ndim==1
 if single:kept=kept[None]
 vals=[]
 for start in range(0,len(kept),20):
  ix=kept[start:start+20]
  if adapter['mode']=='valid_queries':scores=G[ix[:,:,None],ix[:,None,:]]
  elif adapter['mode']=='production_padded_queries':
   # Full production query mean: retained valid queries plus repeated padded query.
   scores=G[ix[:,:,None],ix[:,None,:]]
   padscore=G[int(pad),ix];att=softmax(scores,axis=-1).sum(1)+(adapter['query_length']-ix.shape[1])*softmax(padscore,axis=-1)
   vals.append(np.einsum('bj,bj->b',att/adapter['query_length'],s[ix]));continue
  else:raise ValueError(adapter)
  vals.append(np.einsum('bj,bj->b',softmax(scores,axis=-1).mean(1),s[ix]))
 result=np.concatenate(vals);return float(result[0]) if single else result


def clustered_ci(values,groups,B=B,seed=42):
 values=np.asarray(values,float);groups=np.asarray(groups);u,inv=np.unique(groups,return_inverse=True);sums=np.bincount(inv,weights=values);counts=np.bincount(inv)
 rng=np.random.default_rng(seed);ix=rng.integers(0,len(u),(B,len(u)));bs=sums[ix].sum(1)/counts[ix].sum(1)
 return dict(estimate=float(values.mean()),ci95=np.quantile(bs,[.025,.975]),n_rows=len(values),n_individuals=len(u),bootstrap=B)


def corr(x,y):
 x=np.asarray(x);y=np.asarray(y);keep=np.isfinite(x)&np.isfinite(y)
 return float(spearmanr(x[keep],y[keep]).statistic) if keep.sum()>2 and np.ptp(x[keep])>1e-10 and np.ptp(y[keep])>1e-10 else None


def random_losses(i,n,k,G,s,adapter,pad,full):
 rng=np.random.default_rng(np.random.SeedSequence([42,int(i),int(k)]));kept=np.stack([rng.permutation(n)[k:] for _ in range(RANDOM)])
 return full-bags(G,s,kept,adapter,pad)


def cv_predictions(df,groups,target,features):
 # Independent individuals are held out together across every domain/report row.
 groups=np.asarray(groups);unique=np.unique(groups);assert len(unique)>=10
 X0=pd.get_dummies(df[['report_type','fold','domain']].astype(str),drop_first=False,dtype=float).to_numpy();base=np.column_stack([X0,df[['L','n_sentences','n_removed','removed_fraction','P']].to_numpy(),abs(df.L).to_numpy()])
 X=base if not features else np.column_stack([base,df[features].to_numpy()]);y=df[target].to_numpy();pred=np.full(len(df),np.nan);tuning=[]
 for fold,(tr,te) in enumerate(GroupKFold(5).split(X,y,groups),1):
  scores=[]
  for alpha in [.1,1.,10.,100.,1000.]:
   errors=[]
   for it,iv in GroupKFold(4).split(X[tr],y[tr],groups[tr]):
    model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(X[tr[it]],y[tr[it]]);errors.append(mean_absolute_error(y[tr[iv]],model.predict(X[tr[iv]])))
   scores.append((np.mean(errors),alpha))
  error,alpha=min(scores);model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(X[tr],y[tr]);pred[te]=model.predict(X[te]);tuning.append(dict(heldout_fold=fold,n_train_rows=len(tr),n_test_rows=len(te),n_train_people=len(np.unique(groups[tr])),n_test_people=len(np.unique(groups[te])),alpha=alpha,inner_mae=error,group_disjoint=not bool(set(groups[tr])&set(groups[te]))))
 return pred,tuning


def bootstrap_metrics(y,base,pred,groups):
 u,inv=np.unique(groups,return_inverse=True);counts=np.bincount(inv);D=len(u)
 se=np.bincount(inv,weights=(y-pred)**2);seb=np.bincount(inv,weights=(y-base)**2);ae=np.bincount(inv,weights=abs(y-pred));aeb=np.bincount(inv,weights=abs(y-base));sy=np.bincount(inv,weights=y);sy2=np.bincount(inv,weights=y*y)
 rng=np.random.default_rng(42);ix=rng.integers(0,D,(B,D));nn=counts[ix].sum(1);sst=sy2[ix].sum(1)-sy[ix].sum(1)**2/nn;dR=(seb[ix].sum(1)-se[ix].sum(1))/sst;dMAE=(aeb[ix].sum(1)-ae[ix].sum(1))/nn
 result=dict(R2=float(r2_score(y,pred)),MAE=float(mean_absolute_error(y,pred)),delta_R2=float(r2_score(y,pred)-r2_score(y,base)),delta_R2_ci95=np.quantile(dR,[.025,.975]),MAE_improvement=float(mean_absolute_error(y,base)-mean_absolute_error(y,pred)),MAE_improvement_ci95=np.quantile(dMAE,[.025,.975]))
 result['target_sd']=float(np.std(y));result['R2_interpretable']=bool(np.std(y)>=1e-5)
 if not result['R2_interpretable']:
  for key in ['R2','delta_R2','delta_R2_ci95']:result[key]=None
 return result


def incremental(df,groups,scope='all'):
 methods={'nuisance':[],'nuisance_LP':['L_times_P'],'nuisance_LP_E':['L_times_P','E']};summ=[];predrows=[];tunes=[]
 for target in ['delta_logit','absolute_delta','signed_excess','absolute_excess']:
  d=df.copy()
  if target in ['absolute_excess','absolute_delta']:
   for col in ['L_times_P','E']:d[col]=abs(d[col])
  predictions={}
  for name,features in methods.items():
   pred,tune=cv_predictions(d,groups,target,features);predictions[name]=pred;tunes.extend(dict(scope=scope,target=target,method=name,**r) for r in tune)
   reference='nuisance' if name in ['nuisance','nuisance_LP'] else 'nuisance_LP';ref=predictions[reference];summ.append(dict(scope=scope,target=target,method=name,reference=reference,n_rows=len(df),n_reports=df.report_id.nunique(),n_individuals=len(np.unique(groups)),**bootstrap_metrics(d[target].values,ref,pred,groups)))
   for k in range(len(df)):predrows.append(dict(scope=scope,target=target,method=name,report_id=df.report_id.iloc[k],domain=df.domain.iloc[k],observed=float(d[target].iloc[k]),prediction=float(pred[k])))
 return summ,predrows,tunes


def load():
 import importlib.util
 p=_REPO/'pipeline/5_domain_contributions/_common.py'
 spec=importlib.util.spec_from_file_location('contribution_inputs',p);cc=importlib.util.module_from_spec(spec);spec.loader.exec_module(cc)
 a=dict(np.load(paths.EVIDENCE/'channel2_core.npz',allow_pickle=False));m=cc.metadata()
 return cc,a,m,np.load(paths.EVIDENCE/'compatibility_scores.npy',mmap_mode='r'),np.load(paths.EVIDENCE/'sentence_label_mass.npy',mmap_mode='r')
