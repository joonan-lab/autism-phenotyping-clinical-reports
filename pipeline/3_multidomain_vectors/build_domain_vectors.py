"""Build equal-mass domain proportions from every eligible labeled sentence."""

from pathlib import Path
import sys
_REPO = next(p for p in Path(__file__).resolve().parents if (p / "analysis_paths.py").is_file())
sys.path.insert(0, str(_REPO))
import analysis_paths as paths
import json
import numpy as np
import pandas as pd
import yaml

def main():
    spec = yaml.safe_load(paths.ONTOLOGY.read_text())
    domains = [d['id'] for c in spec['categories'] for s in c['subcategories'] for d in s['domains']]
    ids = np.load(paths.INTER / 'report_id_array.npy', allow_pickle=False).astype(str)
    labels = np.load(paths.INTER / 'labels_np.npy', allow_pickle=False)
    mask = np.load(paths.INTER / 'valid_sentence_mask_np.npy', allow_pickle=False).astype(bool)
    rows = {}; di = {d:j for j,d in enumerate(domains)}
    for line in paths.SILVER.open():
        r = json.loads(line); key = (str(r['report_id']), int(r['sentence_idx']))
        if key in rows: raise ValueError('Duplicate labeled sentence')
        ds = [x['domain_id'] for x in r['labels']]
        if not ds or len(ds) != len(set(ds)) or not set(ds) <= set(domains): raise ValueError('Invalid domain assignment')
        rows[key] = ds
    mass = np.zeros((len(ids),len(domains))); used=set()
    for i,rid in enumerate(ids):
        for j in np.flatnonzero(mask[i]):
            key=(rid,int(j)); ds=rows[key]; used.add(key)
            for d in ds: mass[i,di[d]] += 1 / len(ds)
    if used != set(rows): raise ValueError('Eligible and labeled sentence sets differ')
    P = mass / mask.sum(1,keepdims=True)
    if not np.allclose(P.sum(1),1): raise ValueError('Domain proportions do not sum to one')
    out=paths.out(paths.VECTORS)
    np.save(out/'domain_vectors_proportion.npy',P)
    df=pd.DataFrame(P,columns=domains);df.insert(0,'asd_label',labels);df.insert(0,'report_id',ids)
    df.to_csv(out/'domain_vectors_proportion_latest.tsv',sep='\t',index=False,float_format='%.17g')
    meta=dict(domain_columns=domains,n_reports=len(ids),n_domains=len(domains),n_sentences=int(mask.sum()),config=dict(sentence_selection='all_valid',label_allocation='equal',attention_weighting=False))
    (out/'domain_vectors_meta_latest.json').write_text(json.dumps(meta,indent=2))
    print(f'Wrote {len(ids)} report vectors from {mask.sum()} sentences to {out}')
if __name__=='__main__':main()
