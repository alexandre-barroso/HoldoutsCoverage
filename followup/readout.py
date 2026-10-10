"""Verification readout and explicitly post-result panel diagnostics.

No fitting; no new seed, budget, condition or model. Reads separately acquired
public input in memory. Never exports strings, labels, vocabulary or checkpoints.
"""
import argparse,json,sys
from pathlib import Path
from collections import Counter
from hashlib import sha256
import numpy as np
sys.dont_write_bytecode=True
if sys.flags.optimize:raise SystemExit('Assertions must remain enabled')
HERE=Path(__file__).resolve().parent;REPO=HERE.parent
sys.path.insert(0,str(REPO))


def compute(base,units,predictions,exact,summary,original):
    assess=[u for u in units if 2*base.rank('outer',0,u[0])>=base.SPACE]
    texts=[u[1] for u in assess];truth=np.array([u[2] for u in assess]);p=np.asarray(predictions,dtype=np.int64)
    difference=(p[:,0]!=truth).astype(np.int64)-(p[:,1]!=truth).astype(np.int64)
    disagreements=p[:,0]!=p[:,1]
    categories=[int((difference==k).sum()) for k in (-1,0,1)]
    fold_fixed=np.array([x==base.fold(x) for x in texts])
    fixed_no_disagreement=[];target_n=[];target_d=[]
    for seed in range(200):
        selected=np.array([2*base.rank('select',seed,x)<base.SPACE for x in texts])
        if not np.any(selected&fold_fixed&disagreements):fixed_no_disagreement.append(seed)
        target_n.append(int(selected.sum()));target_d.append(int((selected&disagreements).sum()))
    idx={(r['condition'],r['seed'],r['policy']):r for r in exact}
    seeds=[s for s in range(200) if idx['whitespace_half',s,'original_eligible']['exact'] is not None]
    common={}
    for policy in ('original_eligible','separate_eligible','uniform_target','disagreement_target'):
        records=[idx['whitespace_half',s,policy]['exact'] for s in seeds]
        common[policy]={'common_cases':len(seeds),'metrics':{name:float(np.mean([r['expectation'][name] for r in records])) for name in records[0]['expectation']},
                        'certificates':{name:float(np.mean([r['certificate_probability'][name] for r in records])) for name in records[0]['certificate_probability']}}
    cert_cases=[]
    strict=[];weak=[]
    for seed in range(200):
        r=idx['whitespace_half',seed,'disagreement_target'];probs=r['exact']['certificate_probability']
        q=probs['A_STRICTLY_BETTER']+probs['B_STRICTLY_BETTER'];w=probs['A_WEAKLY_BETTER']+probs['B_WEAKLY_BETTER']
        strict.append(q);weak.append(w)
        if q>0:cert_cases.append({'seed':seed,'target_n':r['target_n'],'disagreements':r['total_disagreements'],'strict_probability':q})
    result={'scope':'Original report readout plus diagnostics explicitly obtained after follow-up results; not new prospective outcomes or model fitting.',
        'units':'Canonical-group weighted finite targets; all-selection repeats V for every design seed.',
        'primary_common_case_ids':seeds,'common_primary_table':common,'capacity_profiles':summary['capacity_profiles'],
        'all_condition_and_paired_summaries':{'conditions':summary['conditions'],'paired_contrasts':summary['paired_contrasts']},
        'original_complete_configuration_summary':original['summary'],
        'post_result_diagnostics':{
          'assessment_n':len(assess),'assessment_loss_categories_minus_zero_plus':categories,'assessment_disagreements':int(disagreements.sum()),
          'minimum_difference_count_among_20_disagreements':20-2*min(20,categories[0]),
          'casefold_half_cases_with_zero_fixed_disagreements':len(fixed_no_disagreement),'casefold_half_zero_fixed_disagreement_seeds':fixed_no_disagreement,
          'primary_target_n':{'min':min(target_n),'max':max(target_n),'mean':float(np.mean(target_n))},
          'primary_target_disagreements':{'min':min(target_d),'max':max(target_d),'mean':float(np.mean(target_d))},
          'primary_disagreement_strict_certificate_mean_probability_all200':float(np.mean(strict)),
          'primary_disagreement_weak_only_certificate_mean_probability_all200':float(np.mean(weak)),
          'positive_strict_certificate_cases':cert_cases,
          'all_selection_max_observed_disagreements':20,'all_selection_missing_disagreements_after_20':int(disagreements.sum())-20,
          'interpretation':'The finite pair bound can stay uncertified despite good estimation; these probabilities are conditional sampling-law descriptions and not three-model robust-winner rates.'}}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--support-results',type=Path,required=True);ap.add_argument('--followup-results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    from reference import validate_sources,validate_result_directory,load,canonical
    from support_reference import validate_result_file
    validate_sources(REPO);validate_result_directory(a.followup_results)
    original,_=validate_result_file(a.support_results/'results.json')
    import support_experiment as base
    units,_=base.load_units(a.data)
    p=load(HERE/'prediction_cache.json')['predictions']
    exact=[json.loads(line) for line in (a.followup_results/'exact_expectations.jsonl').read_text().splitlines()]
    result=compute(base,units,p,exact,load(a.followup_results/'summary.json'),original)
    expected=load(HERE/'expected_readout.json')
    if canonical(result)!=canonical(expected):raise ValueError('Verification readout differs from frozen historical diagnostic/reference values')
    with a.output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'PASS_EXACT_READOUT','output_sha256':sha256(a.output.read_bytes()).hexdigest(),'scope':result['scope']}))
if __name__=='__main__':main()
