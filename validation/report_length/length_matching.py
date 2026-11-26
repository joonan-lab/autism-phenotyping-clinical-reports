"""Shorten A-type reports to lengths sampled from P-type reports, 200 repetitions."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import json
import numpy as np
import pandas as pd
from _common import analyze,seed_for
sys.path.insert(0,str(_REPO/'pipeline/5_domain_contributions'))
import importlib.util
spec=importlib.util.spec_from_file_location('contribution_inputs',_REPO/'pipeline/5_domain_contributions/_common.py');cc=importlib.util.module_from_spec(spec);spec.loader.exec_module(cc)
def length_matched_draw(P_obs, units, n, types, rng):
    """A형 행만 줄인 조성, 줄인 뒤 문장 수, 목표 길이를 돌려준다."""
    p_lengths = n[types == 'P']
    P = P_obs.copy()
    lengths = n.copy()
    targets = np.zeros(len(n), dtype=int)
    for i in np.flatnonzero(types == 'A'):
        t = int(p_lengths[rng.integers(0, len(p_lengths))])
        k = min(t, int(n[i]))
        sel = rng.choice(int(n[i]), size=k, replace=False)
        mass = units[i, sel].sum(0)
        if len(np.unique(sel)) != k or abs(mass.sum() - k) > 1e-9:
            raise ValueError(f'length-matched draw broken for report {i}')
        P[i] = mass / k
        lengths[i] = k
        targets[i] = t
    if not np.array_equal(P[types == 'P'], P_obs[types == 'P']):
        raise ValueError('P rows changed in test L')
    kept_ok = np.all(lengths[types == 'A'] == np.minimum(targets[types == 'A'], n[types == 'A']))
    if not kept_ok:
        raise ValueError('kept sentence count differs from min(t, n_i)')
    return P, lengths, targets


def main():
 m=cc.metadata();M,_=cc.label_tensor(m);keep=m['y']==1;mask=m['mask'][keep];units=M[keep];n=mask.sum(1);types=m['rtype'][keep]
 # Pack eligible sentences so random positions match the original analysis.
 packed=np.zeros_like(units)
 for i in range(len(units)):packed[i,:n[i]]=units[i,mask[i]]
 P=units.sum(1)/n[:,None];observed,_=analyze(P,types,n);observed.update(mean_A_length=float(n[types=='A'].mean()),n_A_shortened=0,A_to_P_tail_ratio=observed['tail_A_fraction']/observed['tail_P_fraction']);rows=[]
 for b in range(200):
  draw,lengths,_=length_matched_draw(P,packed,n,types,np.random.default_rng(seed_for(3,0,b)))
  metrics,_=analyze(draw,types,lengths);rows.append(dict(replicate=b+1,seed=seed_for(3,0,b),pseudocount=1e-6,mean_A_length=float(lengths[types=='A'].mean()),n_A_shortened=int((lengths[types=='A']<n[types=='A']).sum()),A_to_P_tail_ratio=metrics['tail_A_fraction']/metrics['tail_P_fraction'],status='ok',**metrics))
 df=pd.DataFrame(rows);out=paths.out(paths.LENGTH_RESULTS);df.to_csv(out/'length_matched_replicates.tsv',sep='\t',index=False)
 summary=[]
 for metric in df.select_dtypes(include='number').columns:
  if metric in ['replicate','seed','pseudocount']:continue
  lo,hi=np.quantile(df[metric],[.025,.975]);summary.append(dict(metric=metric,observed=observed.get(metric),length_matched_mean=float(df[metric].mean()),length_matched_median=float(df[metric].median()),length_matched95_low=float(lo),length_matched95_high=float(hi),n_requested=200,n_valid=200,n_failed=0))
 pd.DataFrame(summary).to_csv(out/'length_matched_summary.tsv',sep='\t',index=False)
if __name__=='__main__':main()
