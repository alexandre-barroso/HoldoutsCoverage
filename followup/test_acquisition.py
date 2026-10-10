"""Original synthetic acquisition/estimator contract tests; no public data or fits."""
import sys
sys.dont_write_bytecode=True
if not __debug__: raise SystemExit('Optimization mode unsupported')
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
from run_followup import acquire, permutation, sampling_plan, POLICIES, index_digest
from analysis_math import realized_metrics


def main():
    n=57
    sets={'original_eligible':np.arange(n)%2==0,'separate_eligible':np.arange(n)%3!=0,
          'uniform_target':np.ones(n,dtype=bool),'disagreement_target':np.arange(n)<13}
    context={'selection':'cutoff_half','seed':17,'eligible':sets}
    checks=0
    for rep in range(32):
        order=permutation(n,'cutoff_half',17,rep)
        assert sorted(order.tolist())==list(range(n))
        for policy in POLICIES:
            selected,status=acquire(context,policy,rep)
            assert len(set(selected.tolist()))==len(selected)
            assert sets[policy][selected].all()
            k=13 if policy=='disagreement_target' else 20
            assert len(selected)==k
            assert selected.tolist()==order[sets[policy][order]][:k].tolist()
            altered=context|{'inaccessible_labels':[-1]*n}
            assert np.array_equal(selected,acquire(altered,policy,rep)[0])
            assert index_digest(selected)==sha256(np.asarray(selected,dtype='<i8').tobytes()).hexdigest()
            checks+=1
    below=np.array([True]*19+[False]*38)
    assert sampling_plan(below)==(0,'BUDGET_UNATTAINABLE')
    assert sampling_plan(below,True)==(19,'SATURATED_ALL_DISAGREEMENTS')
    context['eligible']['original_eligible']=below
    assert len(acquire(context,'original_eligible',0)[0])==0
    zero=np.zeros(n,dtype=bool)
    assert sampling_plan(zero,True)==(0,'SATURATED_ALL_DISAGREEMENTS')
    context['eligible']['disagreement_target']=zero
    assert len(acquire(context,'disagreement_target',0)[0])==0
    # Same half-selection target has the same coupled permutation in both normalizations.
    assert np.array_equal(permutation(n,'cutoff_half',17,3),permutation(n,'cutoff_half',17,3))
    assert not np.array_equal(permutation(n,'cutoff_half',17,3),permutation(n,'all',17,3))
    # Expansion belongs in the estimator; the finite logical interval uses raw d.
    x=realized_metrics(2,2,100,10,10,2,Fraction(1,10))
    assert x['estimator']==0.1 and x['absolute_error']==0
    assert x['known_difference_count']==2 and x['lower_count']==-6 and x['upper_count']==10
    assert x['certificate']=='BOTH_STRICT_ORDERINGS_FEASIBLE'
    here=Path(__file__).resolve().parent
    report={'status':'PASS_SYNTHETIC_NO_DATA_NO_FITS','date_utc':datetime.now(timezone.utc).isoformat(),
            'acquisition_membership_cases':checks,'infeasible_and_saturated_cases':True,
            'estimator_scaling_not_applied_to_logical_counts':True,
            'source_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),here/'run_followup.py',here/'analysis_math.py']}}
    out=Path(sys.argv[1]) if len(sys.argv)>1 else here/'acquisition_tests.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
