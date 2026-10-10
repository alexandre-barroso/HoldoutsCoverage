"""Separate original reconstruction. Does not import fitting or analysis modules.

Parses the public input independently, checks all acquisition memberships/outcomes,
and checks the exact-law output through sequential SciPy hypergeometric factors.
"""
import sys
sys.dont_write_bytecode=True
if not __debug__: raise SystemExit('Optimization mode unsupported')
from collections import defaultdict,Counter
from datetime import datetime,timezone
from fractions import Fraction
from hashlib import sha256
from math import comb,fsum,floor
import json
import argparse
from pathlib import Path
from statistics import median
import unicodedata
import zipfile

import numpy as np
from scipy.stats import hypergeom

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
METRICS=('estimator','signed_error','absolute_error','squared_error','LR_order_probability',
         'NB_order_probability','tie_probability','wrong_strict_order_probability','best_regret','worst_regret')
CERTS=('A_STRICTLY_BETTER','B_STRICTLY_BETTER','EXACT_TIE','A_WEAKLY_BETTER','B_WEAKLY_BETTER','BOTH_STRICT_ORDERINGS_FEASIBLE')
CONDITIONS={'whitespace_half':('unicode_whitespace','cutoff_half'),
            'casefold_half':('nfc_casefold','cutoff_half'),'whitespace_all':('unicode_whitespace','all')}
POLICIES=('original_eligible','separate_eligible','uniform_target','disagreement_target')


def h(path): return sha256(Path(path).read_bytes()).hexdigest()
def ih(ids): return sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()
def rank(role,seed,text):
    b=json.dumps(['project8-support-v1',role,seed,text],ensure_ascii=False,separators=(',',':')).encode()
    return int.from_bytes(sha256(b).digest(),'big')
def fold(x): return unicodedata.normalize('NFC',unicodedata.normalize('NFC',x).casefold())
def norm(x,name): return ' '.join(x.split()) if name=='unicode_whitespace' else fold(x)
def cert(d,u):
    low,high=d-u,d+u
    return ('A_STRICTLY_BETTER' if high<0 else 'B_STRICTLY_BETTER' if low>0 else
            'EXACT_TIE' if low==high==0 else 'A_WEAKLY_BETTER' if high==0 else
            'B_WEAKLY_BETTER' if low==0 else 'BOTH_STRICT_ORDERINGS_FEASIBLE')


def close(a,b,tol=2e-12):
    if abs(a-b)>tol: raise AssertionError((a,b,abs(a-b)))


def describe(values):
    x=sorted(values);n=len(x)
    if not n:return {'defined':0}
    def q(p):
        pos=p*(n-1); j=floor(pos)
        return x[j]+(pos-j)*(x[min(j+1,n-1)]-x[j])
    return {'defined':n,'mean':fsum(x)/n,'median':median(x),'p05':q(.05),'p95':q(.95),'min':x[0],'max':x[-1]}


def check_describe(want,got):
    assert set(want)==set(got)
    for key,value in want.items(): close(value,got[key])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();RUN=args.results.resolve(strict=True)
    from reference import validate_sources, validate_result_directory
    validate_sources(ROOT);validate_result_directory(RUN)
    manifest=json.loads((RUN/'manifest.json').read_text())
    for name,entry in manifest['files'].items(): assert h(RUN/name)==entry['sha256']
    sealed=json.loads((RUN/'design_manifest.json').read_text())
    for name,digest in sealed['files'].items(): assert h(RUN/name)==digest
    if h(args.data)!='1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3':raise ValueError('Public ZIP identity mismatch')
    with zipfile.ZipFile(args.data) as z:
        raw=z.read('SMSSpamCollection')
    assert sha256(raw).hexdigest()=='7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d'
    grouped=defaultdict(list)
    for line in raw.decode().split('\n')[:-1]:
        label,text=line.split('\t',1); key=fold(' '.join(text.split()))
        grouped[key].append((text,{'ham':0,'spam':1}[label]))
    units=[]
    for key in sorted(grouped,key=lambda s:s.encode()):
        rows=grouped[key]; assert len({y for _,y in rows})==1
        if 2*rank('outer',0,key)>=1<<256:
            units.append((min((text for text,y in rows),key=lambda s:s.encode()),rows[0][1]))
    assert len(units)==2621
    pred_record=json.loads((RUN/'predictions_v.json').read_text());pred=pred_record['predictions']
    assert len(pred)==2621 and all(len(x)==3 and set(x)<={0,1} for x in pred)
    assert ih(np.asarray(pred))==pred_record['prediction_sha256']=='1719751f8fc08749c21bdb31fecaf85cbbc5996e4e2c3dcd37c33c1fa09d7bc0'
    losses=[int(p[0]!=y)-int(p[1]!=y) for p,(_,y) in zip(pred,units)]
    cache={}; permutations={}
    def context(condition,seed):
        key=(condition,seed)
        if key in cache:return cache[key]
        normalizer,selection=CONDITIONS[condition]
        ids=[i for i,(x,y) in enumerate(units) if selection=='all' or 2*rank('select',seed,x)<1<<256]
        ids.sort(key=lambda i:units[i][0].encode())
        e={
          'original_eligible':{j for j,i in enumerate(ids) if 5*rank('select',seed,norm(units[i][0],normalizer))>=4*(1<<256)},
          'separate_eligible':{j for j,i in enumerate(ids) if 5*rank('split',seed,norm(units[i][0],normalizer))>=4*(1<<256)},
          'uniform_target':set(range(len(ids))),
          'disagreement_target':{j for j,i in enumerate(ids) if pred[i][0]!=pred[i][1]},
        }
        fixed={j for j,i in enumerate(ids) if units[i][0]==norm(units[i][0],normalizer)}
        answer={'ids':ids,'eligible':e,'fixed':fixed,'D':len(e['disagreement_target']),
                'truth':sum(losses[i] for i in ids),'selection':selection}
        cache[key]=answer;return answer
    sums=defaultdict(lambda:defaultdict(list)); cert_counts=defaultdict(Counter)
    record_count=infeasible=0
    with (RUN/'acquisition_design.jsonl').open() as d, (RUN/'acquisition_results.jsonl').open() as r:
        for dline,rline in zip(d,r,strict=True):
            a=json.loads(dline);b=json.loads(rline)
            for key,value in a.items(): assert b[key]==value
            condition,seed,rep,policy=a['condition'],a['seed'],a['repetition'],a['policy']
            c=context(condition,seed); ids=c['ids'];eligible=c['eligible'][policy]
            n=len(ids);capacity=len(eligible)
            k=min(20,capacity) if policy=='disagreement_target' else (20 if capacity>=20 else 0)
            status=('SATURATED_ALL_DISAGREEMENTS' if policy=='disagreement_target' and capacity<=20 else
                    'BUDGET_UNATTAINABLE' if policy!='disagreement_target' and capacity<20 else 'FEASIBLE')
            pk=(c['selection'],seed,rep)
            if pk not in permutations:
                payload=json.dumps(['en-support-followup-r1','acquisition',c['selection'],seed,rep],ensure_ascii=False,separators=(',',':')).encode()
                s=int.from_bytes(sha256(payload).digest()[:16],'little')
                permutations[pk]=np.random.Generator(np.random.PCG64(s)).permutation(n).tolist()
            positions=[j for j in permutations[pk] if j in eligible][:k]
            assert a['target_n']==n and a['eligible_n']==capacity and a['acquired_n']==k and a['status']==status
            assert a['selected_index_sha256']==ih([ids[j] for j in positions]) and a['target_index_sha256']==ih(ids)
            assert a['target_fixed']==len(c['fixed']) and a['acquired_fixed']==sum(j in c['fixed'] for j in positions)
            obs_d=sum(j in c['eligible']['disagreement_target'] for j in positions)
            assert a['observed_disagreements']==obs_d and a['missing_disagreements']==c['D']-obs_d
            assert b['true_difference_count']==c['truth']
            record_count+=1
            if status=='BUDGET_UNATTAINABLE':
                assert b['outcome'] is None;infeasible+=1;continue
            delta=sum(losses[ids[j]] for j in positions)
            factor=Fraction(c['D'],n) if policy=='disagreement_target' else Fraction(1)
            estimate=factor*Fraction(delta,k) if k else Fraction(0)
            err=estimate-Fraction(c['truth'],n);wrong=estimate*c['truth']<0;tied=estimate==0
            vals=[estimate,err,abs(err),err*err,estimate<0,estimate>0,tied,wrong,
                  Fraction(abs(c['truth']),n) if wrong else 0,Fraction(abs(c['truth']),n) if wrong or tied else 0]
            metrics=dict(zip(METRICS,vals)); out=b['outcome']
            for key,value in metrics.items():
                close(float(value),out[key]);sums[condition,seed,policy][key].append(float(value))
            missing=c['D']-obs_d
            assert out['known_difference_count']==delta and out['missing_disagreements']==missing
            assert (out['lower_count'],out['upper_count'])==(delta-missing,delta+missing)
            assert out['certificate']==cert(delta,missing)
            cert_counts[condition,seed,policy][out['certificate']]+=1
    assert record_count==76800 and infeasible==3776
    exact=[]; pmf_max_error=metric_max_error=0.; checked_exact=0
    with (RUN/'exact_expectations.jsonl').open() as f:
        for line in f:
            r=json.loads(line);exact.append(r)
            c=context(r['condition'],r['seed']);e=c['eligible'][r['policy']]
            assert r['eligible_n']==len(e) and r['target_n']==len(c['ids'])
            if r['exact'] is None:
                assert r['status']=='BUDGET_UNATTAINABLE' and len(e)<20
                continue
            cat=Counter(losses[c['ids'][j]] for j in e)
            counts=(cat[-1],cat[0],cat[1]);assert tuple(r['category_counts_minus_zero_plus'])==counts
            m=sum(counts);k=r['acquired_n']; scale=float(Fraction(r['estimator_scale'])); n=r['target_n'];truth=c['truth']/n
            triples=[(a,k-a-z,z) for a in range(min(k,counts[0])+1) for z in range(min(k-a,counts[2])+1) if k-a-z<=counts[1]]
            if not m:
                probabilities=np.array([1.]);triples=[(0,0,0)]
            else:
                x=np.array(triples);p_a=hypergeom.pmf(x[:,0],m,counts[0],k)
                if m==counts[0]:p_c=np.ones(len(x))
                else:p_c=hypergeom.pmf(x[:,2],m-counts[0],counts[2],k-x[:,0])
                probabilities=p_a*p_c
            pmf_max_error=max(pmf_max_error,abs(float(probabilities.sum())-1))
            close(float(probabilities.sum()),1,5e-11)
            ref=r['exact'];assert ref['probability_mass_numerator']==ref['probability_mass_denominator']
            assert int(ref['probability_mass_denominator'])==(comb(m,k) if m else 1)
            expected=defaultdict(float);certp=defaultdict(float)
            for (a,b,z),prob in zip(triples,probabilities):
                est=scale*(z-a)/k if k else 0.;err=est-truth;wrong=est*truth<0;tie=est==0
                vals=[est,err,abs(err),err*err,est<0,est>0,tie,wrong,abs(truth) if wrong else 0.,abs(truth) if wrong or tie else 0.]
                for key,value in zip(METRICS,vals):expected[key]+=float(prob)*value
                certp[cert(z-a,c['D']-a-z)]+=float(prob)
            for key in METRICS:
                metric_max_error=max(metric_max_error,abs(expected[key]-ref['expectation'][key]))
                close(expected[key],ref['expectation'][key],5e-11)
                close(float(Fraction(ref['expectation_rational'][key])),ref['expectation'][key])
            for key in CERTS:close(certp[key],ref['certificate_probability'][key],5e-11)
            checked_exact+=1
    assert len(exact)==2400 and checked_exact==2282
    seed_saved=json.loads((RUN/'seed_averages.json').read_text())
    for r in seed_saved:
        key=(r['condition'],r['seed'],r['policy'])
        if not r['feasible']:assert key not in sums;continue
        for m in METRICS:close(fsum(sums[key][m])/32,r['mean'][m])
        for cert_name in CERTS:close(cert_counts[key][cert_name]/32,r['certificate_frequency'][cert_name])
    summary=json.loads((RUN/'summary.json').read_text());idx={(r['condition'],r['seed'],r['policy']):r for r in exact}
    midx={(r['condition'],r['seed'],r['policy']):r for r in seed_saved}
    for row in summary['conditions']:
        er=[r for r in exact if r['condition']==row['condition'] and r['policy']==row['policy'] and r['exact'] is not None]
        assert row['feasible_targets']==len(er) and row['all_targets']==200
        for metric in METRICS:
            check_describe(describe([r['exact']['expectation'][metric] for r in er]),row['exact_design'][metric])
            check_describe(describe([midx[r['condition'],r['seed'],r['policy']]['mean'][metric] for r in er]),row['realized_seed_averages'][metric])
        for name in CERTS:
            check_describe(describe([r['exact']['certificate_probability'][name] for r in er]),row['exact_certificate'][name])
            check_describe(describe([midx[r['condition'],r['seed'],r['policy']]['certificate_frequency'][name] for r in er]),row['realized_certificate'][name])
    for row in summary['paired_contrasts']:
        other,base=row['contrast'].split(' minus ');cond=row['condition']
        feasible=[s for s in range(200) if idx[cond,s,base]['exact'] is not None and idx[cond,s,other]['exact'] is not None]
        assert feasible==row['seeds'] and len(feasible)==row['common_feasible_targets']
        for metric in METRICS:
            check_describe(describe([idx[cond,s,other]['exact']['expectation'][metric]-idx[cond,s,base]['exact']['expectation'][metric] for s in feasible]),row['exact_design'][metric])
            check_describe(describe([midx[cond,s,other]['mean'][metric]-midx[cond,s,base]['mean'][metric] for s in feasible]),row['realized_seed_averages'][metric])
    profile=json.loads((RUN/'eligibility_profile.json').read_text())
    for row in profile['all_capacities']:
        c=context(row['condition'],row['seed']);assert row['capacity']==len(c['eligible'][row['policy']])
    for row in profile['profiles']:
        caps=[len(context(row['condition'],s)['eligible'][row['policy']]) for s in range(200)]
        assert {str(k):v for k,v in Counter(caps).items()}==row['capacity_histogram']
        check_describe(describe(caps),row['capacity'])
        feasible=[v for v in caps if row['policy']=='disagreement_target' or v>=20]
        assert row['feasible_targets']==len(feasible) and row['targets']==200
        assert row['saturated_targets']==sum(v<=20 for v in caps)*(row['policy']=='disagreement_target')
        check_describe(describe([min(v,20) if row['policy']=='disagreement_target' else 20 for v in feasible]),row['actual_label_use'])
    report={'status':'PASS_RECONSTRUCTION_NOT_NEW_FITS','completed_utc':datetime.now(timezone.utc).isoformat(),
            'acquisition_records':record_count,'infeasible_records':infeasible,'exact_records':len(exact),
            'feasible_exact_distributions':checked_exact,'conditional_hypergeometric_probability_max_error':pmf_max_error,
            'independent_float_expectation_max_error':metric_max_error,
            'all_12_condition_and_12_paired_summaries_checked':True,'all_capacity_histograms_checked':True,
            'model_fit_or_study_module_import':False,
            'limits':'Separate numerical reconstruction in same agent context, not independent human review or independent model fitting',
            'source_sha256':h(__file__),'results_manifest_sha256':h(RUN/'manifest.json')}
    out=args.output
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
