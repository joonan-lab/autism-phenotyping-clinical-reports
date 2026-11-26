"""Shared, locally copied helpers for the three JMS analyses (no external code import)."""
from __future__ import annotations

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import csv, hashlib, json, platform, sys, warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix)

DEFAULT_PIPELINE = paths.DATA_ROOT
INTERMEDIATE = ""
SEED, N_BOOTSTRAP, LOGIT_EPS = 42, 1000, 1e-6

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def clean_json(value):
    if isinstance(value, dict): return {str(k): clean_json(v) for k,v in value.items()}
    if isinstance(value, (tuple,list,np.ndarray)): return [clean_json(v) for v in value]
    if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, Path): return str(value)
    return value

def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def write_tsv(path, rows):
    rows=[{k:v for k,v in r.items() if k!='patient_id'} for r in rows]
    if not rows: raise ValueError('Cannot write a table with no rows')
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,delimiter='\t');w.writeheader()
        for r in rows: w.writerow({k:('' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in r.items()})

def configure_plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
        'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    return plt

def save_figure(fig, out, stem):
    fig.savefig(out/(stem+'.pdf'),bbox_inches='tight')
    fig.savefig(out/(stem+'.png'),bbox_inches='tight',dpi=300)

def load_classifier(pipeline):
    p=paths.INTER
    names=['probs_np','labels_np','fold_assignment','report_id_array','patient_id_array']
    a={name:np.load(p/(name+'.npy'),allow_pickle=False) for name in names}
    y=a['labels_np'].astype(int); prob=a['probs_np']; folds=a['fold_assignment']
    rid=a['report_id_array']; pid=a['patient_id_array']
    assert all(x.shape==(489,) for x in a.values())
    assert set(np.unique(y))=={0,1} and y.sum()==346
    assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    assert len(set(rid))==489 and len(set(pid))==372 and set(folds)==set(range(1,6))
    assert all(len(set(folds[pid==g]))==1 for g in np.unique(pid))
    assert np.array_equal(np.array([r.rsplit('-',1)[0] for r in rid]),pid)
    types=np.array([r.rsplit('-',1)[1][0] for r in rid]); assert set(types)=={'A','P'}
    assert sum(types=='A')==233 and sum(types=='P')==256
    manifest_path=p/'fold_split_manifest.json'; manifest=json.loads(manifest_path.read_text())
    for row in manifest:
        te=np.flatnonzero(folds==row['fold']);tr=np.flatnonzero(folds!=row['fold'])
        assert np.array_equal(te,row['valid_indices_cv']) and np.array_equal(tr,row['train_indices_cv'])
        assert list(rid[te])==row['valid_report_ids'] and list(rid[tr])==row['train_report_ids']
        assert list(pid[te])==row['valid_patient_ids'] and list(pid[tr])==row['train_patient_ids']
        it=np.asarray(row['inner_train_indices_cv'],int);iv=np.asarray(row['inner_valid_indices_cv'],int)
        assert len(it)==len(set(it)) and len(iv)==len(set(iv))
        assert set(it).isdisjoint(iv) and set(it)|set(iv)==set(tr)
        assert set(pid[it]).isdisjoint(pid[iv]) and set(pid[tr]).isdisjoint(pid[te])
        assert list(rid[iv])==row['inner_valid_report_ids']
    input_paths=[p/(n+'.npy') for n in names]+[manifest_path]
    return {'y':y,'p':prob,'fold':folds,'report_id':rid,'patient_id':pid,'type':types,'input_paths':input_paths,'split_manifest':manifest}

def reliability_bins(y,p,strategy='quantile',n_bins=10):
    y=np.asarray(y);p=np.asarray(p)
    edges=np.unique(np.quantile(p,np.linspace(0,1,n_bins+1))) if strategy=='quantile' else np.linspace(0,1,n_bins+1)
    if len(edges)==1: edges=np.array([0.,1.])
    assignment=np.searchsorted(edges[1:-1],p,side='right')
    rows=[]
    for i in range(len(edges)-1):
        m=assignment==i
        if not m.any():continue
        rows.append({'bin':i+1,'lower':float(edges[i]),'upper':float(edges[i+1]),
            'n_reports':int(m.sum()),'n_positive':int(y[m].sum()),'mean_probability':float(p[m].mean()),
            'observed_fraction':float(y[m].mean()),'min_probability':float(p[m].min()),'max_probability':float(p[m].max())})
    assert sum(r['n_reports'] for r in rows)==len(y)
    return rows

def calibration(y,p):
    """Unpenalized binary-logistic calibration: offset intercept and free intercept+slope."""
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if len(np.unique(y))<2:return {'calibration_intercept':np.nan,'calibration_slope':np.nan,'slope_model_intercept':np.nan},'one_class'
    l=logit(np.clip(p,LOGIT_EPS,1-LOGIT_EPS))
    intercept=brentq(lambda a:float(np.sum(expit(l+a)-y)),-100,100,xtol=1e-12)
    if np.ptp(l)<1e-10:return {'calibration_intercept':intercept,'calibration_slope':np.nan,'slope_model_intercept':np.nan},'constant_probability'
    pos,neg=l[y==1],l[y==0]
    if pos.min()>=neg.max() or pos.max()<=neg.min():
        complete=pos.min()>neg.max() or pos.max()<neg.min()
        return {'calibration_intercept':intercept,'calibration_slope':np.nan,'slope_model_intercept':np.nan},'complete_separation' if complete else 'quasi_complete_separation'
    X=np.column_stack([np.ones(len(y)),l])
    def objective(b):
        eta=X@b
        return float(np.sum(np.logaddexp(0,eta)-y*eta))
    def jac(b):return X.T@(expit(X@b)-y)
    def hess(b):
        q=expit(X@b);return X.T@((q*(1-q))[:,None]*X)
    fit=minimize(objective,np.array([0.,1.]),jac=jac,hess=hess,method='trust-exact',
        options={'gtol':1e-8,'maxiter':100})
    # scipy may flag an already-converged point for an unproductive trust-region step.
    gradient=float(np.max(np.abs(jac(fit.x))))
    H=hess(fit.x);eig=np.linalg.eigvalsh(H)
    valid=np.isfinite(fit.x).all() and gradient<1e-5 and eig.min()>1e-7 and np.max(np.abs(fit.x))<100
    return {'calibration_intercept':intercept,'calibration_slope':float(fit.x[1]) if valid else np.nan,
        'slope_model_intercept':float(fit.x[0]) if valid else np.nan},'ok' if valid else 'nonfinite_mle_or_nonconvergence'

def binary_metrics(y,p,with_calibration=True):
    y=np.asarray(y);p=np.asarray(p);pred=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    prev=float(y.mean()); brier=float(np.mean((p-y)**2)); ref=prev*(1-prev)
    has_both=len(np.unique(y))==2
    r={'auroc':float(roc_auc_score(y,p)) if has_both else np.nan,
        'auprc':float(average_precision_score(y,p)) if has_both else np.nan,
        'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),'specificity':float(tn/(tn+fp)) if tn+fp else np.nan,
        'f1':float(f1_score(y,pred,zero_division=0)),'brier':brier,'brier_reference':ref,
        'scaled_brier':1-brier/ref if ref else np.nan,
        'ece_quantile10':sum(b['n_reports']/len(y)*abs(b['mean_probability']-b['observed_fraction']) for b in reliability_bins(y,p)),
        'ece_uniform10':sum(b['n_reports']/len(y)*abs(b['mean_probability']-b['observed_fraction']) for b in reliability_bins(y,p,'uniform'))}
    if with_calibration:r.update(calibration(y,p)[0])
    return r

def subgroup_masks(a):return {'All':np.ones(len(a['y']),bool),'A':a['type']=='A','P':a['type']=='P'}

def cluster_resamples(patient_ids,n_bootstrap,seed=SEED):
    groups=np.unique(patient_ids);rows=[np.flatnonzero(patient_ids==g) for g in groups]
    rng=np.random.default_rng(seed)
    for _ in range(n_bootstrap):yield np.concatenate([rows[j] for j in rng.integers(0,len(rows),size=len(rows))])

def metric_tables(a, predictions, n_bootstrap=N_BOOTSTRAP):
    """Report-weighted metrics; paired patient-cluster bootstrap, conditional on fixed OOF fits."""
    masks=subgroup_masks(a);pooled={};foldrows=[]; boot=[]
    for model,p in predictions.items():
        for group,m in masks.items():
            pooled[(model,group)]=binary_metrics(a['y'][m],p[m])
            for f in range(1,6):
                fm=m&(a['fold']==f)
                vals=binary_metrics(a['y'][fm],p[fm]);status=calibration(a['y'][fm],p[fm])[1]
                foldrows.append({'model':model,'group':group,'fold':f,'n_reports':int(fm.sum()),
                    'n_patients':len(set(a['patient_id'][fm])),'n_asd':int(a['y'][fm].sum()),'calibration_fit_status':status,**vals})
    for b,idx in enumerate(cluster_resamples(a['patient_id'],n_bootstrap)):
        for model,p in predictions.items():
            for group,m in masks.items():
                ix=idx[m[idx]]
                if not len(ix):continue
                boot.append({'replicate':b+1,'model':model,'group':group,**binary_metrics(a['y'][ix],p[ix])})
        if (b+1)%250==0:print(f'Patient bootstrap {b+1}/{n_bootstrap}',flush=True)
    rows=[]
    for (model,group),vals in pooled.items():
        m=masks[group]
        br=[r for r in boot if r['model']==model and r['group']==group]
        fr=[r for r in foldrows if r['model']==model and r['group']==group]
        for metric,val in vals.items():
            bs=np.array([r[metric] for r in br]);bs=bs[np.isfinite(bs)]
            fs=np.array([r[metric] for r in fr]);fs=fs[np.isfinite(fs)]
            ci=np.quantile(bs,[.025,.975]) if len(bs) else [np.nan,np.nan]
            rows.append({'model':model,'group':group,'metric':metric,'pooled_estimate':val,
                'ci95_low':ci[0],'ci95_high':ci[1],'bootstrap_valid':len(bs),'bootstrap_requested':n_bootstrap,
                'fold_mean':float(fs.mean()) if len(fs) else np.nan,'fold_sd':float(fs.std(ddof=1)) if len(fs)>1 else np.nan,
                'fold_sd_population':float(fs.std(ddof=0)) if len(fs) else np.nan,'fold_sd_ddof':1,
                'fold_valid':len(fs),'n_reports':int(m.sum()),'n_patients':len(set(a['patient_id'][m])),
                'n_asd':int(a['y'][m].sum()),'n_non_asd':int(m.sum()-a['y'][m].sum()),
                'prevalence':float(a['y'][m].mean())})
    return rows,foldrows,boot

