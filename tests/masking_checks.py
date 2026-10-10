"""Portable authored 584-case reenactment of the masked-information check.

Every binary two-model prediction matrix and visibility mask at n=1..3 is
covered. Visible labels alternate 0/1; all missing external labels are varied.
These are synthetic fixtures, not SMS observations or an unbounded proof.
"""
import sys
if sys.flags.optimize:
    raise SystemExit('Assertions must remain enabled for checks')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from itertools import product
import json
import numpy as np
from support_audit import audit, exhaustive_checks

checks=0
for n in range(1,4):
    for flat in product((0,1),repeat=2*n):
        pred=np.asarray(flat).reshape(n,2)
        for seen_tuple in product((False,True),repeat=n):
            seen=np.asarray(seen_tuple);baseline=np.arange(n)%2
            masked=np.where(seen,baseline,-1)
            original=audit(pred,masked)
            missing=np.flatnonzero(~seen)
            risks=[]
            for fill in product((0,1),repeat=len(missing)):
                truth=baseline.copy();truth[missing]=fill
                remasked=np.where(seen,truth,-1)
                assert audit(pred,remasked)==original
                risks.append((pred!=truth[:,None]).sum(axis=0).tolist())
            for strict in (False,True):
                compare=(lambda x,y:x<y) if strict else (lambda x,y:x<=y)
                possible=[a for a in range(2) if any(all(compare(r[a],r[b]) for b in range(2) if b!=a) for r in risks)]
                robust=[a for a in range(2) if all(all(compare(r[a],r[b]) for b in range(2) if b!=a) for r in risks)]
                suffix='strict' if strict else 'weak'
                assert original['possible_'+suffix+'_winners']==possible
                assert original['robust_'+suffix+'_winners']==robust
            if not any(seen):
                assert original['status']=='EMPTY_OBSERVED_SET'
                assert original['observed_winners']==[] and original['observed_risks']==[None,None]
            checks+=1
assert checks==584
for prediction,labels in [(np.empty((0,2),dtype=int),[]),([[0,-1]],[0])]:
    try:audit(prediction,labels)
    except ValueError:pass
    else:raise AssertionError('Invalid public audit target/class accepted')
print(json.dumps({'status':'PASS_SYNTHETIC_ONLY','masking_fixtures':checks,'scope':'Portable authored reenactment plus direct strict/weak API comparisons; no empirical fit.','public_exhaustive_checks':exhaustive_checks()}))
