"""All-sentence silver/gold agreement and report-LOO prototype ranking."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
from pathlib import Path
from collections import Counter
import json, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0,str(_REPO / 'pipeline/4_phenotype_space'))
import _common_controlled as cc
OUT=paths.out(paths.SILVER_RESULTS)
SILVER=paths.SILVER
GOLD=paths.GOLD

def read(path):
    with path.open() as f:return [json.loads(line) for line in f if line.strip()]

def matrix(rows,ids,domains):
    index={r:i for i,r in enumerate(ids)};di={d:i for i,d in enumerate(domains)}
    X=np.zeros((len(ids),len(domains)));counts=np.zeros(len(ids),dtype=int)
    for r in rows:
        if r['report_id'] not in index:continue
        labels=[x.get('domain_id') if isinstance(x,dict) else x for x in r['labels']]
        if not labels or not set(labels)<=set(domains):raise ValueError('Unmapped/empty gold or silver labels')
        i=index[r['report_id']];counts[i]+=1
        for d in labels:X[i,di[d]]+=1/len(labels)
    assert np.all(counts>0)
    return X/X.sum(1,keepdims=True),counts

def fit_residual(M,design):
    B=np.linalg.lstsq(design,M,rcond=None)[0]
    return M-design@B,B

def ci(r,n):
    # Same approximation as the manuscript top-10 validation; cluster bootstrap is also reported.
    z=np.arctanh(np.clip(r,-1+1e-12,1-1e-12));se=np.sqrt(1.06/(n-3))
    return np.tanh([z-1.96*se,z+1.96*se]).tolist()

def main():
    d=cc.load();dom=list(d['dom']);silver=read(SILVER);gold=read(GOLD)
    ids=sorted({r['report_id'] for r in gold});S,sc=matrix(silver,ids,dom);G,gc=matrix(gold,ids,dom)
    assert np.array_equal(sc,gc),'Sentence counts differ between silver/gold'
    silver_texts=Counter((r['report_id'],r['sentence'].strip()) for r in silver if r['report_id'] in set(ids))
    gold_texts=Counter((r['report_id'],r['sentence'].strip()) for r in gold)
    assert silver_texts==gold_texts,'Per-report sentence-text multisets differ; sentence indices are not comparable across gold/silver'
    allids=np.load(paths.INTER/'report_id_array.npy',allow_pickle=True).astype(str)
    allx=np.load(paths.VECTORS/'domain_vectors_proportion.npy');ix={r:i for i,r in enumerate(allids)}
    assert np.allclose(S,allx[[ix[r] for r in ids]],atol=1e-12,rtol=0)
    design=np.column_stack([np.ones(len(d['rid'])),(d['rtype']=='P').astype(float)])
    goldtype=np.array([r.split('-')[-1][0] for r in ids]);dg=np.column_stack([np.ones(len(ids)),(goldtype=='P').astype(float)])
    pos={r:i for i,r in enumerate(d['rid'])}
    patients=np.load(paths.INTER/'patient_id_array.npy',allow_pickle=True).astype(str)
    patient_lookup=dict(zip(allids,patients))
    person=lambda r:patient_lookup[str(r)]
    si,gi=cc.ilr(S),cc.ilr(G)
    def distances(holdout):
        ds=[];dgg=[];conditions=[]
        for i,rid in enumerate(ids):
            keep=np.ones(len(d['rid']),dtype=bool)
            if holdout=='report' and rid in pos:keep[pos[rid]]=False
            if holdout=='person':keep=np.array([person(r)!=person(rid) for r in d['rid']])
            R,B=fit_residual(d['ILR'][keep],design[keep]);cov=np.cov(R,rowvar=False);inv=np.linalg.inv(cov);mu=R.mean(0)
            conditions.append(float(np.linalg.cond(cov)))
            a=si[i]-dg[i]@B-mu;b=gi[i]-dg[i]@B-mu
            ds.append(float(np.sqrt(a@inv@a)));dgg.append(float(np.sqrt(b@inv@b)))
        return np.array(ds),np.array(dgg),conditions
    a,b,conds=distances('report');aa,bb,_=distances('person');ins,ing,_=distances('none')
    rho=float(spearmanr(a,b).statistic);rhoper=float(spearmanr(aa,bb).statistic)
    cosine=np.einsum('ij,ij->i',S,G)/(np.linalg.norm(S,axis=1)*np.linalg.norm(G,axis=1))
    per=[]
    for j,domain in enumerate(dom):
        constant=np.std(S[:,j])<1e-10 or np.std(G[:,j])<1e-10
        per.append(dict(domain=domain,group='Core' if j<7 else ('Associated' if j<16 else 'Report elements'),
            spearman=None if constant else float(spearmanr(S[:,j],G[:,j]).statistic),
            pearson=None if constant else float(pearsonr(S[:,j],G[:,j]).statistic),constant_column=bool(constant),
            silver_mean=float(S[:,j].mean()),gold_mean=float(G[:,j].mean())))
    groups={group:float(np.median([r['spearman'] for r in per if r['group']==group and r['spearman'] is not None])) for group in ['Core','Associated','Report elements']}
    people=np.array([person(r) for r in ids]);unique=np.unique(people);rng=np.random.default_rng(42);boot=[]
    for _ in range(1000):
        chosen=rng.choice(unique,len(unique),replace=True);sel=np.concatenate([np.flatnonzero(people==p) for p in chosen])
        r=float(spearmanr(a[sel],b[sel]).statistic)
        if np.isfinite(r):boot.append(r)
    result=dict(n_gold_reports=len(ids),n_gold_individuals=len(unique),n_domains=len(dom),n_sentences=int(sc.sum()),
        vectors='all sentences, 1/m mass allocation; full-precision gamma0 ASD cohort',
        sentence_pairing='Exact per-report sentence-text multiset equality; gold/silver sentence indices are not required to match',
        cosine=dict(mean=float(cosine.mean()),median=float(np.median(cosine)),min=float(cosine.min()),max=float(cosine.max())),
        pooled_vector_spearman=float(spearmanr(S.ravel(),G.ravel()).statistic),per_domain={r['domain']:r for r in per},
        per_group_median_spearman=groups,
        mahalanobis_rank_spearman=dict(type_adjusted_leave_one_out=rho,type_adjusted_leave_one_out_ci95=ci(rho,len(ids)),
            type_adjusted_leave_individual_out=rhoper,type_adjusted_in_sample=float(spearmanr(ins,ing).statistic),
            individual_bootstrap_ci95=np.percentile(boot,[2.5,97.5]).tolist(),individual_bootstrap_n_valid=len(boot),
            n_gold_reports_inside_cohort=sum(r in pos for r in ids)),
        max_LOO_covariance_condition=float(max(conds)),
        limitations=['Gold n=26; rank uncertainty is wide. Report-LOO is canonical; person-LOO and individual bootstrap are sensitivity analyses.',
                      'Bootstrap resamples fixed distance pairs; it does not refit the entire reference cohort.',
                      'High profile agreement does not imply validated individual extremity ranking.'])
    write_profile_baselines(S,G,ids,goldtype,people,dom)
    (OUT/'gold_all_sentences_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    table=pd.DataFrame(dict(report_id=ids,report_type=goldtype,n_sentences=sc,cosine=cosine,
        silver_distance_report_LOO=a,gold_distance_report_LOO=b,silver_distance_person_LOO=aa,gold_distance_person_LOO=bb))
    table.to_csv(OUT/'per_report_all_sentences.tsv',sep='\t',index=False)
    pd.DataFrame(per).to_csv(OUT/'per_domain_all_sentences.tsv',sep='\t',index=False)
    print(json.dumps({k:result[k] for k in ['n_gold_reports','n_gold_individuals','cosine','mahalanobis_rank_spearman']},ensure_ascii=False,indent=2))

def write_profile_baselines(S,G,ids,types,people,domains):
    metrics=[]; paired=[]; rhos=[]
    allids=np.load(paths.INTER/'report_id_array.npy',allow_pickle=False).astype(str)
    allP=np.load(paths.VECTORS/'domain_vectors_proportion.npy'); alltypes=np.array([r.rsplit('-',1)[1][0] for r in allids])
    cohort=np.array([allP[allids!=rid].mean(0) for rid in ids])
    typed=np.array([allP[(allids!=rid)&(alltypes==t)].mean(0) for rid,t in zip(ids,types)])
    profiles={'silver_report':S,'cohort_mean_loo_report':cohort,'type_mean_loo_report':typed}
    rng=np.random.default_rng(42);u=np.unique(people)
    draws=[np.concatenate([np.flatnonzero(people==p) for p in rng.choice(u,len(u),replace=True)]) for _ in range(1000)]
    for group,keep in [('All',np.ones(len(S),bool)),('A',types=='A'),('P',types=='P')]:
        ix=np.flatnonzero(keep); local=[j[keep[j]] for j in draws];local=[j for j in local if len(j)>=2];values={}
        for name,M in profiles.items():
            cos=np.einsum('ij,ij->i',M,G)/(np.linalg.norm(M,axis=1)*np.linalg.norm(G,axis=1));values[name]=cos
            bs=[float(cos[j].mean()) for j in local];lo,hi=np.quantile(bs,[.025,.975])
            metrics.append(dict(group=group,model=name,metric='mean_cosine',n_reports=len(ix),estimate=float(cos[ix].mean()),ci95_low=float(lo),ci95_high=float(hi)))
            rho=lambda j:float(spearmanr(np.round(M[j],12).ravel(),np.round(G[j],12).ravel()).statistic)
            rhos.append(dict(group=group,model=name,new_estimate=rho(ix),new_ci=np.quantile([rho(j) for j in local],[.025,.975]).tolist()))
        for name in ['cohort_mean_loo_report','type_mean_loo_report']:
            dif=values['silver_report']-values[name];lo,hi=np.quantile([dif[j].mean() for j in local],[.025,.975])
            paired.append(dict(group=group,baseline=name,mean_cosine_difference_silver_minus_baseline=float(dif[ix].mean()),ci95_low=float(lo),ci95_high=float(hi)))
    (OUT/'profile_baselines.json').write_text(json.dumps(dict(metrics=metrics,paired_differences=paired),indent=2))
    (OUT/'profile_spearman_round12.json').write_text(json.dumps(rhos,indent=2))

if __name__=='__main__':main()
