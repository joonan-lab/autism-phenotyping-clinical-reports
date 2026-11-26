"""Self-contained data alignment, metrics, resampling and provenance for A2/B4."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv,hashlib,json,sys,platform
import numpy as np
import yaml
from scipy.stats import spearmanr
DEFAULT_PIPELINE=paths.DATA_ROOT
SEED=42

def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()

def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
 if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
 if isinstance(v,np.integer):return int(v)
 if isinstance(v,np.bool_):return bool(v)
 if isinstance(v,Path):return str(v)
 return v

def write_json(p,v):Path(p).write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_tsv(p,rows):
 rows=[{k:v for k,v in r.items() if k!='patient_id'} for r in rows];assert rows
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader()
  for r in rows:w.writerow({k:('' if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in r.items()})



def plots():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'savefig.dpi':300,
                     'axes.spines.top':False,'axes.spines.right':False})
 return plt
def savefig(fig,out,name):
 fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),bbox_inches='tight',dpi=300)

def ontology(p):
 domains=[];groups=[]
 for ci,c in enumerate(yaml.safe_load(p.read_text())['categories']):
  for s in c['subcategories']:
   for d in s['domains']:domains.append(d['id']);groups.append(['Core','Associated','Report'][ci])
 assert len(domains)==len(set(domains))==19
 return domains,np.array(groups)

def labels(items,domains,require_conf=True):
 result={}
 for x in items:
  d=x['domain_id'] if isinstance(x,dict) else x
  assert d in domains,('unknown domain',d)
  c=float(x['confidence']) if isinstance(x,dict) else 1.
  assert np.isfinite(c) and 0<=c<=1,('invalid confidence',d,c)
  if d not in result:result[d]=c
 return result

def consensus(raw,domains):
 assert len(raw)==5
 by={}
 for v in raw:
  for d,c in labels(v.get('labels',[]),domains).items():by.setdefault(d,[]).append(c)
 eligible={d:round(float(np.mean(cs)),3) for d,cs in by.items() if len(cs)>=3 and np.mean(cs)>=.5}
 accepted=dict(sorted(eligible.items(),key=lambda x:-x[1])[:3])
 fallback=not accepted
 if fallback:
  allc=[c for cs in by.values() for c in cs]
  accepted={'other_general':round(float(np.mean(allc)),3) if allc else 0.}
 return accepted,fallback

def single_policy(raw,domains):
 original=labels(raw.get('labels',[]),domains)
 cs=dict(sorted(((d,round(c,3)) for d,c in original.items() if c>=.5),key=lambda x:-x[1])[:3])
 return cs or {'other_general':round(float(np.mean(list(original.values()))),3) if original else 0.}

def read_rows(p,keep=None):
 by={};total=0;seen=set();models=set();dates=set()
 for line in p.open():
  if not line.strip():continue
  r=json.loads(line);total+=1;models.add(r.get('model',''));dates.add(str(r.get('timestamp',''))[:10])
  key=(r['report_id'],r['sentence_idx']);assert key not in seen;seen.add(key)
  if keep is None or r['report_id'] in keep:by.setdefault(r['report_id'],[]).append(r)
 for v in by.values():v.sort(key=lambda r:r['sentence_idx'])
 return by,{'file_rows':total,'file_reports':len({k[0] for k in seen}),'models':sorted(models),'stored_dates':sorted(dates)}

def load_data(root,historical=False):
 base=paths.INPUT_ROOT/'step2_silver_labeling'
 gp=paths.GOLD;sp=paths.SILVER;dp=paths.ONTOLOGY
 dom,dgroup=ontology(dp);gold,_=read_rows(gp);assert len(gold)==26
 silver,sa=read_rows(sp,set(gold));assert set(silver)==set(gold) and sa['file_rows']==32317 and sa['file_reports']==489
 hp=base/'data/silver_label/silver_labels_26reports_singlevote.jsonl'
 hist,ha=read_rows(hp,set(gold)) if historical else ({},None)
 rids=sorted(gold);pids=[r.rsplit('-',1)[0] for r in rids];assert len(set(pids))==26
 types=np.array([r.rsplit('-',1)[1][0] for r in rids]);assert sum(types=='A')==12 and sum(types=='P')==14
 alignment=[];matched=[];fallback=0;idxdiff=0
 for ri,rid in enumerate(rids):
  assert len(gold[rid])==len(silver[rid])
  if historical:assert len(hist[rid])==len(gold[rid])
  for ordinal,(g,s) in enumerate(zip(gold[rid],silver[rid])):
   assert g['sentence']==s['sentence'],('text mismatch',rid,ordinal)
   gc=labels(g['labels'],dom);sc=labels(s['labels'],dom);reb,fb=consensus(s['raw_responses'],dom)
   assert set(sc)==set(reb) and all(np.isclose(sc[d],reb[d],atol=1e-12,rtol=0) for d in sc),('consensus reconstruction mismatch',rid,ordinal)
   assert 1<=len(gc)<=3 and 1<=len(sc)<=3
   h=hist[rid][ordinal] if historical else None
   if h is not None:assert h['sentence']==g['sentence']
   row={'report_index':ri,'report_id':rid,'patient_id':pids[ri],'report_type':types[ri],
     'ordinal':ordinal,'gold':gc,'five':sc,'raw':s['raw_responses'],'fallback':fb,'historical':labels(h['labels'],dom) if h else None}
   matched.append(row);fallback+=fb;idxdiff+=g['sentence_idx']!=s['sentence_idx']
   alignment.append({'report_id':rid,'patient_id':pids[ri],'ordinal':ordinal,'gold_sentence_idx':g['sentence_idx'],
      'five_sentence_idx':s['sentence_idx'],'historical_sentence_idx':h['sentence_idx'] if h else '',
      'sentence_sha256':hashlib.sha256(g['sentence'].encode()).hexdigest(),'five_fallback':fb})
 assert len(matched)==1653
 audit={'n_gold_reports':26,'n_gold_individuals':26,'n_gold_sentences':1653,'gold_types':{'A':12,'P':14},
    'five_file':sa,'historical_file':ha,'five_fallback_sentences':fallback,
    'gold_five_different_indices':idxdiff,'ordered_sentence_hash_alignment':True,'canonical_reconstruction':True}
 return matched,rids,types,dom,dgroup,alignment,audit,[gp,sp,dp]+([hp] if historical else [])

def resamples(n,B,seed=SEED):
 rng=np.random.default_rng(seed)
 for _ in range(B):yield rng.integers(0,n,size=n)

def weighted_quantiles(values,counts,q):
 N=int(counts.sum());assert N>0
 cdf=np.cumsum(counts);a=(N-1)*np.asarray(q);lo=np.floor(a).astype(int);hi=np.ceil(a).astype(int)
 vl=values[np.searchsorted(cdf,lo,side='right')];vh=values[np.searchsorted(cdf,hi,side='right')]
 return vl+(a-lo)*(vh-vl)

def weighted_spearman_binary(h):
 n=h.sum();nx=h.sum(1);ny=h.sum(0)
 if n==0 or np.count_nonzero(nx)<2 or np.count_nonzero(ny)<2:return np.nan
 rx=np.cumsum(nx)-nx+(nx+1)/2;ry=np.cumsum(ny)-ny+(ny+1)/2;mean=(n+1)/2
 cov=np.sum(h*(rx[:,None]-mean)*(ry[None,:]-mean))
 return float(cov/np.sqrt(np.sum(nx*(rx-mean)**2)*np.sum(ny*(ry-mean)**2)))

def weighted_auroc(h):
 n0,n1=h.sum(0)
 if not n0 or not n1:return np.nan
 return float(np.sum(h[:,1]*(np.cumsum(h[:,0])-h[:,0]+.5*h[:,0]))/(n0*n1))

def reliability(values,h,edges):
 edges=np.unique(edges)
 if len(edges)<2:edges=np.array([0.,1.])
 assignment=np.searchsorted(edges[1:-1],values,side='right');rows=[]
 for b in range(len(edges)-1):
  mask=assignment==b;n=h[mask].sum()
  if not n:continue
  rows.append({'bin':b+1,'lower':float(edges[b]),'upper':float(edges[b+1]),'n_pairs':int(n),
    'mean_confidence':float(np.sum(values[mask]*h[mask].sum(1))/n),'expert_agreement':float(h[mask,1].sum()/n)})
 return rows

def confidence_metrics(values,h,source):
 n=h.sum();assert n>0
 # Stabilize mathematical decimal boundaries: e.g. confidence 0.6 belongs to [0.6,0.7).
 fixed=np.round(np.linspace(.5,1,6) if source=='adopted' else np.linspace(0,1,11),12)
 quant=weighted_quantiles(values,h.sum(1),np.linspace(0,1,6))
 bf=reliability(values,h,fixed);bq=reliability(values,h,quant)
 def ece(bins):return sum(r['n_pairs']/n*abs(r['mean_confidence']-r['expert_agreement']) for r in bins)
 def mono(bins):
  if len(bins)<2 or np.ptp([b['expert_agreement'] for b in bins])==0:return np.nan
  return float(spearmanr([b['bin'] for b in bins],[b['expert_agreement'] for b in bins]).statistic)
 c=float(np.sum(values*h.sum(1))/n);a=float(h[:,1].sum()/n)
 return {'mean_confidence':c,'expert_agreement':a,'confidence_minus_agreement':c-a,
   'brier':float(np.sum(h[:,0]*values**2+h[:,1]*(values-1)**2)/n),'ece_fixed':ece(bf),'ece_quantile5':ece(bq),
   'confidence_auroc':weighted_auroc(h),'confidence_spearman':weighted_spearman_binary(h),
   'bin_order_spearman_fixed':mono(bf),'bin_order_spearman_quantile':mono(bq)},bf,bq

def metric_ci(values):
 a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
 return (np.quantile(a,[.025,.975]) if len(a) else np.array([np.nan,np.nan])),len(a)
