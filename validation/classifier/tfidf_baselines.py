"""B1: TF-IDF + L2 logistic regression with exact v4.1 inner splits or optional nested refit."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import argparse, json, warnings
from pathlib import Path
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning
from _common import *

C_GRID=[.01,.1,1.,10.,100.]
INNER_FOLDS=4
FEATURES={'char_tfidf_lr':{'analyzer':'char','ngram_range':(2,4)},
          'word_tfidf_lr':{'analyzer':'word','ngram_range':(1,2),'token_pattern':r'(?u)\b\w+\b'}}

def vectorizer(settings):
    return TfidfVectorizer(**settings,min_df=2,sublinear_tf=True,norm='l2',lowercase=True,dtype=np.float64)

def fit_lr(X,y,C):
    model=LogisticRegression(C=C,penalty='l2',solver='liblinear',max_iter=5000,tol=1e-6,random_state=SEED)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always',ConvergenceWarning);model.fit(X,y)
    if any(isinstance(w.message,ConvergenceWarning) for w in ws):raise RuntimeError('Logistic regression failed to converge')
    return model

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pipeline-root',type=Path,default=DEFAULT_PIPELINE)
    ap.add_argument('--bootstrap',type=int,default=N_BOOTSTRAP)
    ap.add_argument('--training-protocol',choices=['matched_v4_1','nested_refit'],default='matched_v4_1',
        help='matched_v4_1: exact neural inner train/validation, no refit; nested_refit: historical 4-fold tuning and full outer-train refit')
    args=ap.parse_args();assert args.bootstrap>=1
    out=paths.out(paths.CLASSIFIER_RESULTS/'baselines');plt=configure_plots();a=load_classifier(args.pipeline_root)
    textdir=paths.INPUT_ROOT/'step1_nlp_asd/489reports/reports_txt_no_headings'
    paths=[textdir/(r+'.txt') for r in a['report_id']]
    texts=np.array([p.read_text(encoding='utf-8') for p in paths],dtype=object)
    assert all(t.strip() for t in texts)
    predictions={'v4_1':a['p'].copy()};inner_rows=[];chosen_rows=[];split_records=[]
    modeldir=out/'models';modeldir.mkdir(exist_ok=True)
    for name,settings in FEATURES.items():
        probs=np.full(len(texts),np.nan);writes=np.zeros(len(texts),int)
        for f in range(1,6):
            tr=np.flatnonzero(a['fold']!=f);te=np.flatnonzero(a['fold']==f)
            split=next(r for r in a['split_manifest'] if r['fold']==f)
            if args.training_protocol=='matched_v4_1':
                pairs=[(np.asarray(split['inner_train_indices_cv'],int),np.asarray(split['inner_valid_indices_cv'],int))]
            else:
                sgkf=StratifiedGroupKFold(n_splits=INNER_FOLDS,shuffle=True,random_state=SEED+f)
                pairs=[(tr[itr],tr[iva]) for itr,iva in sgkf.split(texts[tr],a['y'][tr],a['patient_id'][tr])]
            scores={c:[] for c in C_GRID}
            for j,(ti,vi) in enumerate(pairs,1):
                assert not (set(a['patient_id'][ti])&set(a['patient_id'][vi]))
                assert not (set(a['patient_id'][ti])&set(a['patient_id'][te]))
                assert not (set(a['patient_id'][vi])&set(a['patient_id'][te]))
                assert len(set(a['y'][ti]))==len(set(a['y'][vi]))==2
                split_records.append({'model':name,'outer_fold':f,'inner_fold':j,
                    'train_oof_indices':ti.tolist(),'validation_oof_indices':vi.tolist(),'outer_test_oof_indices':te.tolist(),
                    'n_train_patients':len(set(a['patient_id'][ti])),'n_validation_patients':len(set(a['patient_id'][vi])),
                    'patient_overlap':0})
                vec=vectorizer(settings);Xtr=vec.fit_transform(texts[ti]);Xva=vec.transform(texts[vi])
                for c in C_GRID:
                    lr=fit_lr(Xtr,a['y'][ti],c);prob=lr.predict_proba(Xva)[:,list(lr.classes_).index(1)]
                    score=float(roc_auc_score(a['y'][vi],prob));scores[c].append(score)
                    inner_rows.append({'model':name,'outer_fold':f,'inner_fold':j,'C':c,'validation_auroc':score,
                        'n_train':len(ti),'n_validation':len(vi),'n_features':Xtr.shape[1],'iterations':int(lr.n_iter_[0])})
            # Strictly highest inner mean AUROC, tie broken by the first/smallest C.
            best=max(C_GRID,key=lambda c:np.mean(scores[c]))
            if args.training_protocol=='matched_v4_1':tr=np.asarray(split['inner_train_indices_cv'],int)
            vec=vectorizer(settings);Xtr=vec.fit_transform(texts[tr]);Xte=vec.transform(texts[te])
            lr=fit_lr(Xtr,a['y'][tr],best);probs[te]=lr.predict_proba(Xte)[:,list(lr.classes_).index(1)];writes[te]+=1
            joblib.dump({'vectorizer':vec,'classifier':lr,'outer_fold':f,'training_report_ids':a['report_id'][tr].tolist()},
                modeldir/f'{name}_fold{f}.joblib',compress=3)
            chosen_rows.append({'model':name,'outer_fold':f,'selected_C':best,'inner_mean_auroc':float(np.mean(scores[best])),
                'n_features':Xtr.shape[1],'iterations':int(lr.n_iter_[0]),'n_train':len(tr),'n_test':len(te)})
            print(f'{name} outer fold {f}: C={best}, inner AUROC={np.mean(scores[best]):.4f}, test AUROC={roc_auc_score(a["y"][te],probs[te]):.4f}, features={Xtr.shape[1]}',flush=True)
        assert np.isfinite(probs).all() and np.all(writes==1)
        predictions[name]=probs;np.save(out/f'oof_probs_{name}.npy',probs)
    np.save(out/'oof_probs_bow_lr.npy',predictions['char_tfidf_lr'])
    prevalence=np.empty(len(texts))
    for f in range(1,6):
        split=next(r for r in a['split_manifest'] if r['fold']==f)
        train=np.asarray(split['inner_train_indices_cv'],int) if args.training_protocol=='matched_v4_1' else np.flatnonzero(a['fold']!=f)
        prevalence[a['fold']==f]=a['y'][train].mean()
    predictions['training_prevalence']=prevalence
    np.save(out/'oof_probs_training_prevalence.npy',prevalence)
    write_tsv(out/'inner_cv_scores.tsv',inner_rows);write_tsv(out/'selected_hyperparameters.tsv',chosen_rows)
    write_json(out/'inner_split_manifest.json',split_records)
    write_tsv(out/'oof_predictions.tsv',[{'report_id':a['report_id'][i],'patient_id':a['patient_id'][i],
        'report_type':a['type'][i],'fold':int(a['fold'][i]),'label':int(a['y'][i]),**{n:float(p[i]) for n,p in predictions.items()}}
        for i in range(len(texts))])
    metrics,folds,boot=metric_tables(a,predictions,args.bootstrap)
    write_tsv(out/'baseline_metrics.tsv',metrics);write_tsv(out/'per_fold_metrics.tsv',folds);write_tsv(out/'bootstrap_metrics.tsv',boot)
    lookup={(r['replicate'],r['model'],r['group']):r for r in boot};point={(r['model'],r['group'],r['metric']):r['pooled_estimate'] for r in metrics}
    differences=[]
    for name in ['char_tfidf_lr','word_tfidf_lr']:
        for group in subgroup_masks(a):
            for metric in ['auroc','auprc','accuracy','f1','brier','calibration_slope']:
                diffs=np.array([lookup[(b,name,group)][metric]-lookup[(b,'v4_1',group)][metric] for b in range(1,args.bootstrap+1)])
                diffs=diffs[np.isfinite(diffs)];ci=np.quantile(diffs,[.025,.975]) if len(diffs) else [np.nan,np.nan]
                differences.append({'model':name,'reference':'v4_1','group':group,'metric':metric,
                    'difference_baseline_minus_v4_1':point[(name,group,metric)]-point[('v4_1',group,metric)],
                    'ci95_low':ci[0],'ci95_high':ci[1],'bootstrap_valid':len(diffs),
                    'interpretation':'negative favors baseline for Brier; positive favors baseline for discrimination; slope difference has no simple better/worse direction'})
    write_tsv(out/'paired_model_differences.tsv',differences)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    selected=[r for r in metrics if r['group']=='All' and r['metric'] in ['auroc','brier']]
    colors=['#bb6356','#35618f','#45927c','#999999']
    for ax,metric,title in zip(axes,['auroc','brier'],['Discrimination (higher is better)','Brier score (lower is better)']):
        rr=[r for r in selected if r['metric']==metric];x=np.arange(len(rr));vals=np.array([r['pooled_estimate'] for r in rr])
        lo=np.array([r['ci95_low'] for r in rr]);hi=np.array([r['ci95_high'] for r in rr])
        ax.bar(x,vals,color=colors)
        ax.errorbar(x,(lo+hi)/2,yerr=(hi-lo)/2,fmt='none',ecolor='black',capsize=3)
        ax.set_xticks(x,['v4_1','Char TF-IDF','Word TF-IDF','Train prevalence'],rotation=15,ha='right');ax.set(title=title,ylabel=metric.upper())
    fig.suptitle('Original outer folds; 95% patient-cluster bootstrap CIs',fontsize=10)
    fig.tight_layout();save_figure(fig,out,'baseline_comparison');plt.close(fig)
    config={'analysis':'B1','seed':SEED,'bootstrap':args.bootstrap,'fold_sd_ddof':1,'fold_sd_population_ddof':0,'outer_folds':'original fold_assignment, patient-disjoint',
        'training_protocol':args.training_protocol,'inner_cv':('exact v4.1 saved inner validation; one split; no refit' if args.training_protocol=='matched_v4_1' else '4-fold StratifiedGroupKFold, seed=42+outer_fold; full outer-train refit'),
        'C_grid':C_GRID,'selection_metric':'unweighted mean inner-fold AUROC; exact ties select smaller C',
        'primary_model':'char_tfidf_lr','secondary_model':'word_tfidf_lr','features':FEATURES,
        'tfidf':{'min_df':2,'sublinear_tf':True,'norm':'l2','lowercase':True,'max_features':None},
        'logistic_regression':{'penalty':'l2','solver':'liblinear','class_weight':None,'max_iter':5000,'tol':1e-6},
        'classification_threshold':.5,'intervals':'paired individual bootstrap, report weighted, fixed OOF outputs; no refitting',
        'prevalence_baseline':'ASD prevalence in the final TF-IDF training indices under the selected protocol; no outer-test labels used',
        'calibration_separation_handling':'complete/quasi separation: joint slope MLE undefined; slope left missing, finite folds counted explicitly',
        'limitations':['matched_v4_1 holds back identical inner-validation individuals; optional nested_refit uses more final training reports than neural v4.1',
            'TF-IDF uses full preprocessed report text, while v4_1 encodes up to 153 sentences x128 tokens',
            'training-prevalence scores vary by fold, so pooled AUROC need not equal 0.5; each fold AUROC is 0.5',
            'no lexical removal beyond the existing diagnosis-cue-removal preprocessing']}
    write_json(out/'results_summary.json',{'metrics':metrics,'paired_differences':differences,'hyperparameters':chosen_rows,'config':config})
    for r in metrics:
        if r['group']=='All' and r['metric'] in ['auroc','brier','calibration_slope']:
            print(f"{r['model']} {r['metric']}: {r['pooled_estimate']:.6f}; fold {r['fold_mean']:.6f} +/- {r['fold_sd']:.6f}",flush=True)

if __name__=='__main__':main()
