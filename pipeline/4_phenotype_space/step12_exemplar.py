"""Recreate the author-selected RS-346 illustration using domain contributions E."""

from pathlib import Path
import os
from types import SimpleNamespace
_data=Path(os.environ.get('REPORT_LLM_DATA_ROOT','~/Dropbox/ReportLLM_analysis')).expanduser()
_output=Path(os.environ.get('REPORT_LLM_OUTPUT_ROOT',str(_data/'02_results'))).expanduser()
def _out(p):
    p.mkdir(parents=True,exist_ok=True)
    return p
paths=SimpleNamespace(TABLE_SOURCE=Path(os.environ.get('REPORT_LLM_TABLE_SOURCE_DIR',str(_data/'table_source'))).expanduser(),SPACE=_output/'phenotype_space',EVIDENCE=_output/'domain_contributions',SILVER_RESULTS=_output/'validation/silver_labels',out=_out)
import json
import numpy as np
import pandas as pd

SUBMISSION_ID='RS-346'

def main():
    table=paths.TABLE_SOURCE
    mapping=pd.read_csv(table/'internal/ESM_2_report_id_mapping.tsv',sep='\t')
    chosen=mapping.loc[mapping.submission_id==SUBMISSION_ID,'report_id']
    if len(chosen)!=1:raise ValueError('The selected submission ID must map to one report')
    rid=chosen.iloc[0]
    prof=pd.read_csv(paths.SPACE/'controlled_individual_profiles.tsv',sep='\t').set_index('report_id')
    fp=pd.read_csv(paths.EVIDENCE/'two_channel_fingerprints.tsv',sep='\t').set_index('report_id')
    a=np.load(paths.EVIDENCE/'channel2_core.npz',allow_pickle=False);ix=np.flatnonzero(a['report_id']==rid)
    if len(ix)!=1 or rid not in prof.index or rid not in fp.index:raise ValueError('Example report is missing from analysis inputs')
    i=int(ix[0]);fidelity=pd.read_csv(paths.SILVER_RESULTS/'per_domain_all_sentences.tsv',sep='\t').set_index('domain')
    sentence=json.loads((paths.SILVER_RESULTS/'sentence_level_validation_results.json').read_text());f1={r['domain']:r['f1'] for r in sentence['per_domain']}
    rows=[]
    for j,dom in enumerate(a['domain']):
        rho=float(fidelity.loc[dom,'spearman']);score=float(f1[dom])
        rows.append(dict(report_id=rid,submission_id=SUBMISSION_ID,fold=int(a['fold'][i]),report_type=a['report_type'][i],domain=dom,P=float(a['P'][i,j]),E=float(a['E'][i,j]),n_sentences=int(a['domain_sentence_count'][i,j]),layer2_z=float(prof.loc[rid,'z_'+dom]),gold_report_rho=rho,sentence_f1=score,reliable=bool(rho>=.3 and score>=.3)))
    detail=pd.DataFrame(rows);z=detail.layer2_z;top=detail.loc[z.idxmax()];largest=detail.loc[z.abs().idxmax()];distance=float(fp.loc[rid,'distance'])
    selection=dict(chosen_report=rid,submission_id=SUBMISSION_ID,selection_method='author-selected illustration',chosen=dict(report_id=rid,submission_id=SUBMISSION_ID,report_type=a['report_type'][i],fold=int(a['fold'][i]),n_valid_sentences=int(a['valid_mask'][i].sum()),distance=distance,distance_percentile=float((fp.distance<=distance).mean()*100),raw_logit=float(a['logit'][i]),model_probability=float(1/(1+np.exp(-a['logit'][i]))),top_emphasized_domain=top.domain,top_emphasized_z=float(top.layer2_z),largest_abs_domain=largest.domain,largest_abs_z=float(largest.layer2_z)))
    out=paths.out(paths.SPACE/'exemplar');detail.to_csv(out/'exemplar_domains.tsv',sep='\t',index=False);(out/'exemplar_selection.json').write_text(json.dumps(selection,indent=2))
if __name__=='__main__':main()
