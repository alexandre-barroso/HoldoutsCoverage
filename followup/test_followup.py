"""Original synthetic tests; no public or protected data reads and no model fitting."""
import sys
sys.dont_write_bytecode = True
if not __debug__:
    raise SystemExit("Optimization mode is not supported")
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from datetime import datetime, timezone

from analysis_math import CERTIFICATES, METRICS, exact_design, realized_metrics, composition_warning


def direct_subsets(values, k, n, true_count, disagreements, scale):
    sums = {key:Fraction(0) for key in METRICS}
    certs = {key:Fraction(0) for key in CERTIFICATES}
    repetitions = 0
    for positions in combinations(range(len(values)),k):
        observed = [values[i] for i in positions]
        d = sum(observed)
        estimate = scale*Fraction(d,k)
        error = estimate-Fraction(true_count,n)
        wrong = estimate*true_count < 0
        tied = estimate == 0
        data = [estimate,error,abs(error),error*error,estimate < 0,estimate > 0,tied,wrong,
                Fraction(abs(true_count),n) if wrong else 0,
                Fraction(abs(true_count),n) if wrong or tied else 0]
        for key,value in zip(METRICS,data): sums[key] += value
        radius = disagreements-sum(abs(v) for v in observed)
        low, high = d-radius,d+radius
        name = ("A_STRICTLY_BETTER" if high < 0 else "B_STRICTLY_BETTER" if low > 0 else
                "EXACT_TIE" if low == high == 0 else "A_WEAKLY_BETTER" if high == 0 else
                "B_WEAKLY_BETTER" if low == 0 else "BOTH_STRICT_ORDERINGS_FEASIBLE")
        certs[name] += 1
        repetitions += 1
    return ({k:v/repetitions for k,v in sums.items()},
            {k:v/repetitions for k,v in certs.items()},repetitions)


def main():
    cases = subsets = 0
    for m in range(1,9):
        for minus in range(m+1):
            for plus in range(m-minus+1):
                zero = m-minus-plus
                values = [-1]*minus+[0]*zero+[1]*plus
                for k in range(1,m+1):
                    settings = [(m,plus-minus,minus+plus,Fraction(1)),
                                (m+3,plus-minus+1,minus+plus+1,Fraction(1))]
                    if zero == 0:
                        settings.append((m+3,plus-minus,minus+plus,Fraction(m,m+3)))
                    for n,truth,disagreements,scale in settings:
                        result = exact_design((minus,zero,plus),k,n,truth,disagreements,scale)
                        want,certs,count = direct_subsets(values,k,n,truth,disagreements,scale)
                        assert {key:Fraction(v) for key,v in result['expectation_rational'].items()} == want
                        assert {key:Fraction(v) for key,v in result['certificate_probability_rational'].items()} == certs
                        assert result['probability_mass_exact']
                        cases += 1; subsets += count
    empty = exact_design((0,0,0),0,7,0,0,Fraction(0))
    assert empty['expectation']['tie_probability'] == 1
    assert empty['certificate_probability']['EXACT_TIE'] == 1
    # Budget-20 populations exercise up to all 231 count triples.
    large = exact_design((100,100,100),20,300,0,200)
    assert large['nonzero_category_states'] == 231
    assert large['probability_mass_exact'] and large['scaled_population_mean_rational'] == '0'
    sat = exact_design((2,0,3),5,11,1,5,Fraction(5,11))
    assert sat['expectation_rational']['absolute_error'] == '0'
    assert sat['certificate_probability']['B_STRICTLY_BETTER'] == 1
    r = realized_metrics(1,5,11,1,5,5,Fraction(5,11))
    assert r['absolute_error'] == 0 and r['certificate'] == 'B_STRICTLY_BETTER'
    assert realized_metrics(0,0,7,0,0,0,Fraction(0))['certificate'] == 'EXACT_TIE'
    failures = 0
    for fn in [
        lambda:exact_design((1,0,0),2,3,-1,1),
        lambda:exact_design((1,0,0),0,3,-1,1),
        lambda:exact_design((1,0,0),1,0,-1,1),
        lambda:exact_design((1,0,0),1,3,1,1),
        lambda:realized_metrics(1,0,3,1,1,0),
        lambda:composition_warning(1,'a','a',0,1,target_n=0),
        lambda:composition_warning(1,'a','a',0,1,aliases=1),
    ]:
        try: fn()
        except ValueError: failures += 1
        else: raise AssertionError('Invalid fixture accepted')
    warnings = [
        composition_warning(5,'select','select',Fraction(1,2),Fraction(4,5),target_n=10),
        composition_warning(5,'select','select',None,Fraction(4,5),target_n=10),
        composition_warning(5,'select','split',Fraction(1,2),Fraction(4,5),target_n=10),
        composition_warning(5,'select','select',Fraction(9,10),Fraction(4,5),target_n=10),
        composition_warning(0,'select','select',Fraction(1,2),Fraction(4,5),target_n=10),
    ]
    assert warnings == ['FORCED_STRATUM_ABSENCE']+['NO_FORCED_EXCLUSION_PROVED']*4
    # Prediction-disagreement identity, empty visibility, and signed/weak boundaries.
    from analysis_math import certificate
    assert certificate(0,3) == 'BOTH_STRICT_ORDERINGS_FEASIBLE'
    assert certificate(4,3) == 'B_STRICTLY_BETTER'
    assert certificate(-4,3) == 'A_STRICTLY_BETTER'
    assert certificate(3,3) == 'B_WEAKLY_BETTER'
    assert certificate(-3,3) == 'A_WEAKLY_BETTER'
    assert certificate(0,0) == 'EXACT_TIE'
    here = Path(__file__).resolve().parent
    report = {'status':'PASS_SYNTHETIC_NOT_EMPIRICAL','date_utc':datetime.now(timezone.utc).isoformat(),
              'hypergeometric_cases':cases,'independently_enumerated_subsets':subsets,
              'invalid_fixtures_rejected':failures,'max_category_triples':231,
              'zero_and_saturated_disagreement_checked':True,'composition_fixtures':5,
              'signed_and_weak_certificates_checked':True,
              'source_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),here/'analysis_math.py']}}
    output = Path(sys.argv[1]) if len(sys.argv)>1 else here/'synthetic_tests.json'
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
