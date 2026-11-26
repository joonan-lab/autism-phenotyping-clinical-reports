"""OOF provenance, source prediction performance and fold-quality diagnostics."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,accuracy_score
from scipy.optimize import minimize
from scipy.special import expit
import importlib.util
spec=importlib.util.spec_from_file_location('contribution_inputs',_REPO/'pipeline/5_domain_contributions/_common.py');cc=importlib.util.module_from_spec(spec);spec.loader.exec_module(cc)

def main():
    m=cc.metadata();core=cc.load_core();p=np.load(cc.INTER/'probs_np.npy');L=core['logit'];y=m['y'];manifest=json.loads((cc.INTER/'fold_split_manifest.json').read_text());rows=[]
    for f in manifest:
        tr=set(f['train_patient_ids']);te=set(f['valid_patient_ids']);assert not(tr&te)
        it=set(f['inner_train_indices_cv']);iv=set(f['inner_valid_indices_cv']);ot=set(f['valid_indices_cv']);assert not(it&iv or it&ot or iv&ot)
    for f in range(1,6):
        ix=m['fold']==f
        rows.append(dict(fold=f,n_reports=int(ix.sum()),n_ASD=int(y[ix].sum()),AUROC_saved_probability=float(roc_auc_score(y[ix],p[ix])),AUROC_logit=float(roc_auc_score(y[ix],L[ix])),AUPRC=float(average_precision_score(y[ix],p[ix])),accuracy=float(accuracy_score(y[ix],p[ix]>=.5)),Brier=float(brier_score_loss(y[ix],p[ix])),logit_mean=float(L[ix].mean()),logit_sd=float(L[ix].std()),near_constant=bool(L[ix].std()<1e-4)))
    def loss(b):return float(np.mean(np.logaddexp(0,b[0]+b[1]*L)-y*(b[0]+b[1]*L)))
    fit=minimize(loss,[0,1],method='BFGS');beta=fit.x
    pred=pd.DataFrame(dict(report_id=m['ids'],report_type=m['rtype'],fold=m['fold'],asd_label=y,saved_probability=p,logit=L,near_constant_fold=np.isin(m['fold'],[r['fold'] for r in rows if r['near_constant']])))
    result=dict(n_reports=len(y),n_individuals=len(np.unique(m['patient'])),n_ASD=int(y.sum()),n_ASD_individuals=len(np.unique(m['patient'][y==1])),outer_fold_person_disjoint=True,inner_outer_index_disjoint=True,
        primary_probability_source='Stored v4.1 OOF probabilities; source float32 ties are retained. Logit-ranking metrics are a sensitivity check.',
        pooled=dict(AUROC_saved_probability=float(roc_auc_score(y,p)),AUROC_logit=float(roc_auc_score(y,L)),AUPRC=float(average_precision_score(y,p)),Brier=float(brier_score_loss(y,p)),accuracy=float(accuracy_score(y,p>=.5)),calibration_intercept=float(beta[0]),calibration_slope=float(beta[1]),calibration_optimizer_success=bool(fit.success)),
        fold_average_AUROC_saved=float(np.mean([r['AUROC_saved_probability'] for r in rows])),fold_average_AUROC_logit=float(np.mean([r['AUROC_logit'] for r in rows])),folds=rows,
        interpretive_quality_rule='Prespecified from prior source audit: logit SD<1e-4 defines a near-constant fold; raw results retained, but within-fold clinical direction/relative ranks not interpreted. Diagnostic metrics include every fold.',
        zero_points='Raw s/E use the trained decision boundary (logit0), including allocated model bias. Cohort-relative deviations use type+fold-adjusted eligible ASD means and are not preservation/deficit labels.')
    # Newly trained TF-IDF models match the saved v4.1 inner train/validation and outer test.
    baseline_path=paths.CLASSIFIER_RESULTS/'baselines/oof_predictions.tsv'
    baselines=pd.read_csv(baseline_path,sep='\t',usecols=['report_id','fold','label','char_tfidf_lr','word_tfidf_lr']).set_index('report_id').loc[m['ids']]
    assert np.array_equal(baselines.label.values,y) and np.array_equal(baselines.fold.values,m['fold'])
    unique=np.unique(m['patient']);rng=np.random.default_rng(42);boot={name:[] for name in ['v4_AUROC','v4_Brier','v4_calibration_slope','char_minus_v4_AUROC','word_minus_v4_AUROC']}
    for _ in range(1000):
        selected=rng.choice(unique,len(unique),replace=True);ix=np.concatenate([np.flatnonzero(m['patient']==person) for person in selected]);yy=y[ix];pp=p[ix];ll=L[ix]
        if len(np.unique(yy))<2:continue
        auc=roc_auc_score(yy,pp);boot['v4_AUROC'].append(auc);boot['v4_Brier'].append(brier_score_loss(yy,pp))
        fitb=minimize(lambda b:float(np.mean(np.logaddexp(0,b[0]+b[1]*ll)-yy*(b[0]+b[1]*ll))),beta,method='BFGS')
        if np.isfinite(fitb.x).all():boot['v4_calibration_slope'].append(float(fitb.x[1]))
        boot['char_minus_v4_AUROC'].append(roc_auc_score(yy,baselines.char_tfidf_lr.values[ix])-auc)
        boot['word_minus_v4_AUROC'].append(roc_auc_score(yy,baselines.word_tfidf_lr.values[ix])-auc)
    result['individual_bootstrap_fixed_predictions']={name:dict(n_valid=len(v),percentile_interval=np.percentile(v,[2.5,97.5]).tolist()) for name,v in boot.items()}
    result['baseline_comparison']=dict(source=str(baseline_path),outer_labels_and_folds_match=True,char_TFIDF_pooled_AUROC=float(roc_auc_score(y,baselines.char_tfidf_lr)),word_TFIDF_pooled_AUROC=float(roc_auc_score(y,baselines.word_tfidf_lr)),
        limits='Conditional comparison of fixed OOF predictions; no model refitting in bootstrap. TF-IDF selects C on the exact saved v4.1 inner validation and trains only on its inner train; final training report sets match neural v4.1. No refitting in bootstrap. No v4 superiority claim.')
    cc.json_write(paths.out(paths.CLASSIFIER_RESULTS)/'classifier_model_audit_results.json',result);pd.DataFrame(rows).to_csv(paths.CLASSIFIER_RESULTS/'fold_performance.tsv',sep='\t',index=False);pred.to_csv(paths.CLASSIFIER_RESULTS/'classifier_oof_predictions.tsv',sep='\t',index=False)
    print(json.dumps(result['pooled'],indent=2));print('near-constant folds',[r['fold'] for r in rows if r['near_constant']])

if __name__=='__main__':main()
