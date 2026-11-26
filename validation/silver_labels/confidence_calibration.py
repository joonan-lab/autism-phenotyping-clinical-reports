"""A2: expert agreement versus LLM verbal confidence on adopted labels and raw votes."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths_config
import argparse
from pathlib import Path
import numpy as np
from _confidence_common import *

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pipeline-root',type=Path,default=DEFAULT_PIPELINE)
 ap.add_argument('--bootstrap',type=int,default=1000);args=ap.parse_args();assert args.bootstrap>=1
 out=paths_config.out(paths_config.SILVER_RESULTS/'confidence');plt=plots()
 data,rids,types,dom,dgroup,alignment,audit,paths=load_data(args.pipeline_root)
 observations=[];exclusions=[];di={d:i for i,d in enumerate(dom)}
 for row in data:
  base={k:row[k] for k in ['report_index','report_id','patient_id','report_type','ordinal']}
  if row['fallback']:
   for d,c in row['five'].items():exclusions.append({**base,'domain':d,'stored_confidence':c,'reason':'no domain passed reconstructed 3/5 and mean confidence>=0.5; fallback is not a calibrated label confidence'})
  else:
   for d,c in row['five'].items():observations.append({**base,'source':'adopted','vote':0,'domain':d,'domain_group':dgroup[di[d]],'confidence':c,'expert_agreement':int(d in row['gold'])})
  for vi,vote in enumerate(row['raw'],1):
   for d,c in labels(vote.get('labels',[]),dom).items():
    observations.append({**base,'source':'raw_votes','vote':vi,'domain':d,'domain_group':dgroup[di[d]],'confidence':c,'expert_agreement':int(d in row['gold'])})
 write_tsv(out/'label_observations.tsv',observations);write_tsv(out/'excluded_fallback_labels.tsv',exclusions)
 hist={};values={};point={};bins=[];groupnames=['All','Core','Associated','Report']
 for source in ['adopted','raw_votes']:
  rows=[r for r in observations if r['source']==source];vv=np.unique([r['confidence'] for r in rows]);vi={v:i for i,v in enumerate(vv)}
  values[source]=vv
  for group in groupnames:
   h=np.zeros((26,len(vv),2),dtype=np.int64)
   for r in rows:
    if group=='All' or r['domain_group']==group:h[r['report_index'],vi[r['confidence']],r['expert_agreement']]+=1
   hist[(source,group)]=h;assert h.sum()>0
   metrics,bf,bq=confidence_metrics(vv,h.sum(0),source);point[(source,group)]=metrics
   for strategy,bb in [('fixed',bf),('quantile5',bq)]:bins.extend({'source':source,'group':group,'strategy':strategy,**b} for b in bb)
 write_tsv(out/'reliability_bins.tsv',bins)
 boots=[]
 for b,ix in enumerate(resamples(26,args.bootstrap)):
  for source in ['adopted','raw_votes']:
   for group in groupnames:
    h=hist[(source,group)][ix].sum(0)
    if not h.sum():continue
    m,_,_=confidence_metrics(values[source],h,source)
    boots.append({'replicate':b+1,'source':source,'group':group,'n_pairs':int(h.sum()),**m})
  if (b+1)%250==0:print(f'Gold-individual bootstrap {b+1}/{args.bootstrap}',flush=True)
 write_tsv(out/'bootstrap_metrics.tsv',boots);metrics=[]
 for (source,group),pm in point.items():
  h=hist[(source,group)];rr=[r for r in boots if r['source']==source and r['group']==group]
  for metric,val in pm.items():
   ci,n=metric_ci([r[metric] for r in rr]);metrics.append({'source':source,'group':group,'metric':metric,'estimate':val,
      'ci95_low':ci[0],'ci95_high':ci[1],'bootstrap_valid':n,'bootstrap_requested':args.bootstrap,
      'n_pairs':int(h.sum()),'n_agree':int(h[:,:,1].sum()),'n_disagree':int(h[:,:,0].sum()),
      'n_reports_with_pairs':int(np.count_nonzero(h.sum((1,2))))})
 write_tsv(out/'confidence_calibration.tsv',metrics)
 fig,axes=plt.subplots(2,4,figsize=(12,6),sharex='row',sharey='row')
 for i,source in enumerate(['adopted','raw_votes']):
  for j,group in enumerate(groupnames):
   ax=axes[i,j]
   for strategy,color,fmt in [('fixed','#35618f','o-'),('quantile5','#cc796c','s--')]:
    bb=[r for r in bins if r['source']==source and r['group']==group and r['strategy']==strategy]
    ax.plot([r['mean_confidence'] for r in bb],[r['expert_agreement'] for r in bb],fmt,color=color,label=strategy,alpha=.85)
   ax.plot([0,1],[0,1],':',color='gray');ax.set(xlim=((.5,1) if source=='adopted' else (0,1)),ylim=(0,1),
       title=f'{source}: {group}\nn={int(hist[(source,group)].sum())}',xlabel='LLM confidence',ylabel='Expert agreement' if j==0 else '')
 axes[0,0].legend(fontsize=7);fig.tight_layout();savefig(fig,out,'confidence_reliability');plt.close(fig)
 # Distribution plot makes the conditional confidence range and concentration visible.
 fig,axes=plt.subplots(1,2,figsize=(8,3))
 for ax,source in zip(axes,['adopted','raw_votes']):
  h=hist[(source,'All')].sum(0);ax.bar(values[source],h.sum(1),width=.008,color='#35618f')
  ax.set(title=source,xlabel='Confidence',ylabel='Predicted label pairs',xlim=(0,1))
 fig.tight_layout();savefig(fig,out,'confidence_distribution');plt.close(fig)
 audit.update({'adopted_label_pairs_including_fallback':sum(len(r['five']) for r in data),
   'fallback_pairs_excluded':len(exclusions),'adopted_pairs_evaluated':int(hist[('adopted','All')].sum()),
   'raw_vote_pairs_evaluated':int(hist[('raw_votes','All')].sum()),
   'empty_raw_votes':sum(not v.get('labels') for r in data for v in r['raw']),
   'raw_votes_over_three_labels':sum(len(v.get('labels',[]))>3 for r in data for v in r['raw'])})
 config={'analysis':'A2','seed':SEED,'bootstrap':args.bootstrap,'primary':'consensus-adopted sentence-domain pairs excluding reconstructed fallback',
   'secondary':'all deduplicated returned domain-confidence pairs in five raw responses; empty votes have no pairs',
   'fixed_bins_adopted':[.5,.6,.7,.8,.9,1.],'fixed_bins_raw':'0-1 by 0.1','quantile_bins':5,
   'fixed_bin_boundaries':'decimal edges rounded to12 places; left inclusive/right exclusive, final bin includes1',
   'confidence_target':'membership in one expert gold annotation; agreement, not clinical truth',
   'weighting':'each predicted label pair equal weight; repeated labels/votes stay together within resampled individual',
   'fallback_exclusion':'reconstruct actual majority and mean-confidence adoption; exclude no-eligible-domain fallback regardless of stored confidence',
   'interval':'fixed-label individual percentile bootstrap; bins recomputed; no multiplicity adjustment',
   'limitations':['primary selection is truncated by consensus/confidence>=0.5; cannot validate choice of 0.5 threshold or unpredicted labels',
     'raw responses are repeated calls to one model, not independent annotators',
     'confidence discrimination/monotonicity does not establish probability calibration',
     '26 individuals, one expert, single institution; label generation uncertainty not resampled']}
 write_json(out/'results_summary.json',{'metrics':metrics,'input_audit':audit,'config':config})
 for r in metrics:
  if r['group']=='All' and r['metric'] in ['mean_confidence','expert_agreement','brier','ece_fixed','confidence_auroc']:
   print(f"{r['source']} {r['metric']}: {r['estimate']:.6f} [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}]",flush=True)

if __name__=='__main__':main()
