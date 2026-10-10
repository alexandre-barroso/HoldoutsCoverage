"""Original bounded post-result follow-up. Public input only; no protected reads.

Portable replay uses the bound historical prediction cache without fitting.
Design files are sealed before full-label outcome analysis in this replay; the
historical prospective chronology is documented separately in protocol.txt.
"""
import sys
sys.dont_write_bytecode = True
if not __debug__:
    raise SystemExit("Optimization mode is not supported")
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from analysis_math import METRICS, CERTIFICATES, exact_design, realized_metrics, composition_warning

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
POLICIES = ('original_eligible','separate_eligible','uniform_target','disagreement_target')
CONDITIONS = (
    ('whitespace_half','unicode_whitespace','cutoff_half'),
    ('casefold_half','nfc_casefold','cutoff_half'),
    ('whitespace_all','unicode_whitespace','all'),
)
BUDGET, REPETITIONS = 20, 32
PREDICTION_SHA = '1719751f8fc08749c21bdb31fecaf85cbbc5996e4e2c3dcd37c33c1fa09d7bc0'
COEFFICIENT_SHA = '876ea3650fff9d37720c05e34b5f1faec4a04214d2b16f98a115bdb1d1a0deda'
OLD_RESULT_SHA = 'a71d5bcabbf0c8e52dcdfdd4086de055c26581cc1e73d70766a13a34800dce9d'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')


def index_digest(indices):
    return sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest()


def permutation(n,selection,seed,repetition):
    payload=json.dumps(['en-support-followup-r1','acquisition',selection,seed,repetition],
                       ensure_ascii=False,separators=(',',':')).encode()
    random_seed=int.from_bytes(sha256(payload).digest()[:16],'little')
    return np.random.Generator(np.random.PCG64(random_seed)).permutation(n)


def describe(values):
    values=[v for v in values if v is not None]
    if not values: return {'defined':0}
    arr=np.array(values,dtype=float)
    return {'defined':len(values),'mean':float(arr.mean()),'median':float(np.median(arr)),
            'p05':float(np.quantile(arr,.05)),'p95':float(np.quantile(arr,.95)),
            'min':float(arr.min()),'max':float(arr.max())}


def sampling_plan(eligible,disagreement_policy=False):
    capacity=int(eligible.sum())
    if disagreement_policy:
        return min(BUDGET,capacity), ('SATURATED_ALL_DISAGREEMENTS' if capacity <= BUDGET else 'FEASIBLE')
    if capacity < BUDGET:
        return 0,'BUDGET_UNATTAINABLE'
    return BUDGET,'FEASIBLE'


def acquire(context,policy,repetition):
    eligible=context['eligible'][policy]
    k,status=sampling_plan(eligible,policy=='disagreement_target')
    if not k: return np.empty(0,dtype=np.int64),status
    order=permutation(len(eligible),context['selection'],context['seed'],repetition)
    selected=order[eligible[order]][:k]
    if len(selected)!=k or len(set(selected.tolist()))!=k:
        raise RuntimeError('Acquisition count or uniqueness failure')
    return selected,status


def verify_freeze(path):
    if Path(path).resolve(strict=True)!=(HERE/'portable_manifest.json').resolve(strict=True):
        raise ValueError('Only the included portable source manifest is accepted')
    from reference import validate_sources
    return validate_sources(ROOT)


def run(args):
    started=time.monotonic()
    frozen=verify_freeze(args.freeze)
    out=args.output.resolve()
    if out.exists() or out.is_relative_to(ROOT):
        raise RuntimeError('Use a new output directory outside the repository')
    out.mkdir(parents=True)
    write_json(out/'execution_start.json',{'started_utc':utc(),'freeze_sha256':digest(args.freeze),
               'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
               'scope':'Post-result fixed-panel follow-up; public data only; no protected reads'})
    # Import only the fully inspected original generated modules with frozen hashes.
    sys.path.insert(0,str(ROOT))
    import support_experiment as base
    from support_reference import validate_result_file
    old_path=args.support_results.resolve(strict=True)/'results.json'
    old,_=validate_result_file(old_path)
    old_records={(r['normalization'],r['selection'],r['seed'],r['arm']):r for r in old['records']}
    data=args.data.resolve(strict=True)
    units,input_info=base.load_units(data)
    train=[u for u in units if 2*base.rank('outer',0,u[0])<base.SPACE]
    assess=[u for u in units if 2*base.rank('outer',0,u[0])>=base.SPACE]
    if len(units)!=5159 or len(train)!=2538 or len(assess)!=2621:
        raise RuntimeError('Frozen pool dimensions changed')
    if set(u[0] for u in train)&set(u[0] for u in assess): raise RuntimeError('Pool overlap')
    texts=[u[1] for u in assess]
    truth=np.array([u[2] for u in assess],dtype=np.int64)
    print('Using the bound historical prediction matrix; no fitting.',flush=True)
    from cache_panel import cached_panel
    predictions,historical_fit=cached_panel([u[1] for u in train],np.array([u[2] for u in train]),texts)
    if historical_fit['prediction_sha256']!=PREDICTION_SHA: raise RuntimeError('Frozen prediction matrix changed')
    fit={'execution_mode':'CACHED_PREDICTIONS_NO_REFIT', 'historical_fit':historical_fit,
         'prediction_sha256':PREDICTION_SHA, 'cache_sha256':digest(HERE/'prediction_cache.json'),
         'scope':'Historical diagnostics are retained as provenance, not recomputed fitting or coefficient checks.'}
    write_json(out/'panel_replay.json',fit)
    write_json(out/'predictions_v.json',{'scope':'Derived fixed predictions only, no raw texts or labels',
               'ordered_matrix_encoding_for_digest':'little-endian signed int64, row-major',
               'prediction_sha256':PREDICTION_SHA,'predictions':predictions.tolist()})
    fixed={name:np.array([x==fn(x) for x in texts]) for name,fn in
           [('unicode_whitespace',base.whitespace),('nfc_casefold',base.fold)]}
    normalizers={'unicode_whitespace':base.whitespace,'nfc_casefold':base.fold}
    for name,fn in normalizers.items():
        tk=[fn(u[1]) for u in train]; vk=[fn(x) for x in texts]
        if len(set(tk))!=len(tk) or len(set(vk))!=len(vk) or set(tk)&set(vk):
            raise RuntimeError('Alias or boundary change')
    contexts=[]
    design_path=out/'acquisition_design.jsonl'
    capacity_rows=[]; integrity_rows=[]; design_slots=0
    with design_path.open('x') as stream:
        for seed in range(200):
            raw_ranks=[base.rank('select',seed,x) for x in texts]
            selected_half=np.array([2*h<base.SPACE for h in raw_ranks])
            held={}
            for norm_name,fn in normalizers.items():
                held[norm_name]={
                    'original_eligible':np.array([5*base.rank('select',seed,fn(x))>=4*base.SPACE for x in texts]),
                    'separate_eligible':np.array([5*base.rank('split',seed,fn(x))>=4*base.SPACE for x in texts])}
            for condition,norm_name,selection in CONDITIONS:
                available=np.flatnonzero(selected_half) if selection=='cutoff_half' else np.arange(len(texts))
                ids=np.array(sorted(available,key=lambda i:texts[i].encode('utf-8')),dtype=np.int64)
                if len(ids)<BUDGET: raise RuntimeError('Target smaller than frozen budget')
                disagreements=predictions[ids,0]!=predictions[ids,1]
                eligible={policy:held[norm_name][policy][ids] for policy in POLICIES[:2]}
                eligible['uniform_target']=np.ones(len(ids),dtype=bool)
                eligible['disagreement_target']=disagreements
                context={'condition':condition,'normalization':norm_name,'selection':selection,
                         'seed':seed,'ids':ids,'eligible':eligible,'disagreements':disagreements,
                         'fixed':fixed[norm_name][ids]}
                contexts.append(context)
                max_rank=max(raw_ranks[i] for i in ids)
                target_hash=index_digest(ids)
                target_fixed=int(context['fixed'].sum())
                warning=composition_warning(target_fixed,'select','select',
                           None if selection=='all' else max_rank,(4*base.SPACE+4)//5,target_n=len(ids))
                for policy,old_arm in [('original_eligible','reuse'),('separate_eligible','separate')]:
                    reference=old_records[norm_name,selection,seed,old_arm]
                    if (reference['audit']['target_n']!=len(ids)
                        or reference['audit']['observed_n']!=int(eligible[policy].sum())
                        or reference['support']['selected_fixed']!=target_fixed
                        or reference['support']['observed_fixed']!=int((eligible[policy]&context['fixed']).sum())):
                        raise RuntimeError('Original mask membership-count reconstruction failed')
                for policy in POLICIES:
                    k,status=sampling_plan(eligible[policy],policy=='disagreement_target')
                    capacity=int(eligible[policy].sum())
                    e_fixed=int((eligible[policy]&context['fixed']).sum())
                    capacity_rows.append({'condition':condition,'seed':seed,'policy':policy,
                         'target_n':len(ids),'capacity':capacity,'requested_budget':BUDGET,
                         'acquired_n':k,'status':status,'target_fixed':target_fixed,
                         'eligible_fixed':e_fixed,'target_disagreements':int(disagreements.sum()),
                         'eligible_disagreements':int((eligible[policy]&disagreements).sum()),
                         'target_index_sha256':target_hash,'eligible_index_sha256':index_digest(ids[eligible[policy]])})
                    integrity_rows.append({'condition':condition,'seed':seed,'policy':policy,
                         'duplicate_boundary':'PASS_DISJOINT_GROUPS',
                         'count_check':'BUDGET_UNATTAINABLE' if capacity<BUDGET and policy!='disagreement_target' else 'CAPACITY_SUFFICIENT_OR_INFORMATIVE_SET_SATURATED',
                         'stratum_coverage':'FIXED_STRATUM_ABSENT' if target_fixed and e_fixed==0 else 'NO_FIXED_STRATUM_ABSENCE',
                         'composition_witness':warning if policy=='original_eligible' else 'NO_FORCED_EXCLUSION_PROVED',
                         'scope':'Declared finite target and supplied roles; not arbitrary program verification'})
                for repetition in range(REPETITIONS):
                    for policy in POLICIES:
                        positions,status=acquire(context,policy,repetition)
                        observed_d=int(disagreements[positions].sum())
                        row={'condition':condition,'normalization':norm_name,'selection':selection,'seed':seed,
                             'repetition':repetition,'policy':policy,'status':status,'target_n':len(ids),
                             'eligible_n':int(eligible[policy].sum()),'acquired_n':len(positions),
                             'target_fixed':target_fixed,'acquired_fixed':int(context['fixed'][positions].sum()),
                             'total_disagreements':int(disagreements.sum()),'observed_disagreements':observed_d,
                             'missing_disagreements':int(disagreements.sum())-observed_d,
                             'target_index_sha256':target_hash,'selected_index_sha256':index_digest(ids[positions])}
                        stream.write(canonical(row)+'\n'); design_slots+=1
    if design_slots!=76800: raise RuntimeError('Incomplete acquisition design')
    profiles=[]
    for condition,_,_ in CONDITIONS:
        for policy in POLICIES:
            rows=[r for r in capacity_rows if r['condition']==condition and r['policy']==policy]
            profiles.append({'condition':condition,'policy':policy,'targets':len(rows),
                             'capacity':describe([r['capacity'] for r in rows]),
                             'capacity_histogram':dict(sorted(Counter(r['capacity'] for r in rows).items())),
                             'feasible_targets':sum(r['status']!='BUDGET_UNATTAINABLE' for r in rows),
                             'saturated_targets':sum(r['status']=='SATURATED_ALL_DISAGREEMENTS' for r in rows),
                             'actual_label_use':describe([r['acquired_n'] for r in rows if r['status']!='BUDGET_UNATTAINABLE'])})
    write_json(out/'eligibility_profile.json',{'budget':BUDGET,'choice':'Rounded prior mean availability 18.6; no deployment cost claim',
                    'profiles':profiles,'all_capacities':capacity_rows})
    write_json(out/'integrity_comparison.json',{'scope':'Deterministic report comparison; not measured deployment utility','records':integrity_rows})
    design_manifest={'status':'SEALED_BEFORE_NEW_FULL_LABEL_OUTCOMES','sealed_utc':utc(),
        'design_slots':design_slots,'target_policy_slots':len(capacity_rows),'contexts':len(contexts),
        'freeze_sha256':digest(args.freeze),'prediction_sha256':PREDICTION_SHA,
        'input_zip_sha256':base.ARCHIVE_SHA,'input_member_sha256':base.MEMBER_SHA,
        'files':{name:digest(out/name) for name in ['acquisition_design.jsonl','eligibility_profile.json','integrity_comparison.json','panel_replay.json','predictions_v.json']}}
    write_json(out/'design_manifest.json',design_manifest)
    sealed_sha=digest(out/'design_manifest.json')
    print('Sealed 76,800 label-free acquisition records; exact and realized outcomes begin now.',flush=True)

    # No new full-label category counts or performance computed above this line.
    exact_records=[]; seed_records=[]
    with (out/'acquisition_results.jsonl').open('x') as output, design_path.open() as design:
        for context in contexts:
            ids=context['ids']; n=len(ids)
            loss=(predictions[ids,0]!=truth[ids]).astype(np.int64)-(predictions[ids,1]!=truth[ids]).astype(np.int64)
            true_delta=int(loss.sum()); total_d=int(context['disagreements'].sum())
            old_ref=old_records[context['normalization'],context['selection'],context['seed'],'reuse']
            old_true=old_ref['oracle']['error_counts_S'][0]-old_ref['oracle']['error_counts_S'][1]
            if true_delta!=old_true: raise RuntimeError('Original target oracle reconstruction failed')
            local_metrics=defaultdict(list); local_certs=defaultdict(Counter)
            for policy in POLICIES:
                mask=context['eligible'][policy]
                k,status=sampling_plan(mask,policy=='disagreement_target')
                scale=Fraction(total_d,n) if policy=='disagreement_target' else Fraction(1)
                record={key:context[key] for key in ('condition','normalization','selection','seed')}
                record.update(policy=policy,status=status,target_n=n,eligible_n=int(mask.sum()),
                              acquired_n=k,true_difference_count=true_delta,total_disagreements=total_d,
                              estimator_scale=str(scale))
                if status!='BUDGET_UNATTAINABLE':
                    eligible_loss=loss[mask]
                    counts=tuple(int((eligible_loss==value).sum()) for value in (-1,0,1))
                    result=exact_design(counts,k,n,true_delta,total_d,scale)
                    record.update(category_counts_minus_zero_plus=counts,exact=result)
                    if policy in ('uniform_target','disagreement_target') and Fraction(result['expectation_rational']['signed_error'])!=0:
                        raise RuntimeError('Unbiased finite-target comparator has nonzero exact bias')
                else: record['exact']=None
                exact_records.append(record)
            for repetition in range(REPETITIONS):
                for policy in POLICIES:
                    row=json.loads(next(design)); positions,status=acquire(context,policy,repetition)
                    expected_key=(context['condition'],context['seed'],repetition,policy)
                    if (row['condition'],row['seed'],row['repetition'],row['policy'])!=expected_key:
                        raise RuntimeError('Design/outcome ordering mismatch')
                    if index_digest(ids[positions])!=row['selected_index_sha256'] or row['status']!=status:
                        raise RuntimeError('Sealed acquisition membership changed')
                    row['true_difference_count']=true_delta
                    if status=='BUDGET_UNATTAINABLE': row['outcome']=None
                    else:
                        sample_delta=int(loss[positions].sum())
                        scale=Fraction(total_d,n) if policy=='disagreement_target' else Fraction(1)
                        row['outcome']=realized_metrics(sample_delta,len(positions),n,true_delta,total_d,
                                           row['observed_disagreements'],scale)
                        local_metrics[policy].append(row['outcome'])
                        local_certs[policy][row['outcome']['certificate']]+=1
                    output.write(canonical(row)+'\n')
            for policy in POLICIES:
                items=local_metrics[policy]
                seed_records.append({'condition':context['condition'],'seed':context['seed'],'policy':policy,
                    'feasible':bool(items),'repetitions':len(items),
                    'mean':{metric:float(np.mean([row[metric] for row in items])) for metric in METRICS} if items else None,
                    'certificate_frequency':{key:local_certs[policy][key]/len(items) for key in CERTIFICATES} if items else None})
        if next(design,None) is not None: raise RuntimeError('Unprocessed design slots')
    with (out/'exact_expectations.jsonl').open('x') as stream:
        for row in exact_records: stream.write(canonical(row)+'\n')
    write_json(out/'seed_averages.json',seed_records)
    if len(exact_records)!=2400 or len(seed_records)!=2400: raise RuntimeError('Incomplete exact records')
    groups=[]
    for condition,_,_ in CONDITIONS:
        for policy in POLICIES:
            er=[r for r in exact_records if r['condition']==condition and r['policy']==policy and r['exact'] is not None]
            mr=[r for r in seed_records if r['condition']==condition and r['policy']==policy and r['feasible']]
            if len(er)!=len(mr): raise RuntimeError('Exact/realized feasibility mismatch')
            groups.append({'condition':condition,'policy':policy,'all_targets':200,'feasible_targets':len(er),
                'exact_design':{m:describe([r['exact']['expectation'][m] for r in er]) for m in METRICS},
                'realized_seed_averages':{m:describe([r['mean'][m] for r in mr]) for m in METRICS},
                'exact_certificate':{m:describe([r['exact']['certificate_probability'][m] for r in er]) for m in CERTIFICATES},
                'realized_certificate':{m:describe([r['certificate_frequency'][m] for r in mr]) for m in CERTIFICATES}})
    e_index={(r['condition'],r['seed'],r['policy']):r for r in exact_records}
    m_index={(r['condition'],r['seed'],r['policy']):r for r in seed_records}
    contrasts=[]
    pairs=[('original_eligible','separate_eligible'),('original_eligible','uniform_target'),
           ('uniform_target','separate_eligible'),('uniform_target','disagreement_target')]
    for condition,_,_ in CONDITIONS:
        for base_policy,other in pairs:
            seeds=[s for s in range(200) if e_index[condition,s,base_policy]['exact'] is not None and e_index[condition,s,other]['exact'] is not None]
            contrasts.append({'condition':condition,'contrast':other+' minus '+base_policy,
                'common_feasible_targets':len(seeds),'excluded_targets':200-len(seeds),'seeds':seeds,
                'exact_design':{m:describe([e_index[condition,s,other]['exact']['expectation'][m]-e_index[condition,s,base_policy]['exact']['expectation'][m] for s in seeds]) for m in METRICS},
                'realized_seed_averages':{m:describe([m_index[condition,s,other]['mean'][m]-m_index[condition,s,base_policy]['mean'][m] for s in seeds]) for m in METRICS}})
    # Sealed inputs must remain identical after all oracle access.
    if digest(out/'design_manifest.json')!=sealed_sha: raise RuntimeError('Manifest changed')
    for name,expected in design_manifest['files'].items():
        if digest(out/name)!=expected: raise RuntimeError('Sealed design file changed: '+name)
    summary={'status':'COMPUTED_PENDING_SEPARATE_SCIENTIFIC_ASSESSMENT','completed_utc':utc(),
             'scope':'Conditional finite acquisition design; no population inference or new fitted configuration',
             'budget':BUDGET,'repetitions':REPETITIONS,'design_slots':design_slots,'exact_slots':len(exact_records),
             'design_manifest_sha256':sealed_sha,'fit':fit,'capacity_profiles':profiles,
             'conditions':groups,'paired_contrasts':contrasts,'elapsed_seconds':time.monotonic()-started,
             'limitations':['Post-original-result follow-up; no external preregistration',
                'One fixed public target pool/prediction matrix; real calibration split not repaired',
                'Exact expectations conditional on simple random sampling design, not proof of PRNG randomness',
                'Common-feasible denominators and infeasible slots retained',
                'Familiar acquisition baselines and finite bounds; no novelty or utility certification']}
    write_json(out/'summary.json',summary)
    write_json(out/'manifest.json',{'files':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()},
                                    'freeze_sha256':digest(args.freeze),'new_fitted_configurations':0,'unchanged_fit_replays':0,'execution_mode':'CACHED_PREDICTIONS_NO_REFIT'})
    print(canonical({'status':summary['status'],'design_slots':design_slots,'exact_slots':len(exact_records),
                     'prediction_identity':True,'coefficient_identity_recomputed':False,
                     'elapsed_seconds':summary['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--freeze',type=Path,default=HERE/'portable_manifest.json')
    parser.add_argument('--support-results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
