"""Original data-free finite checks, ported from the reviewed protocol audit.
Synthetic fixtures only; no empirical data or unbounded proof.
"""
import sys
if sys.flags.optimize:
 raise SystemExit("Assertions must remain enabled for finite checks")
from itertools import product
import json
counts={'panels_masks_weightings':0,'label_completions':0,'ordered_pair_checks':0,'candidate_winner_checks':0}
for k,max_n in [(2,3),(3,2)]:
 for n in range(1,max_n+1):
  for flat in product(range(k),repeat=3*n):
   pred=[flat[a*n:(a+1)*n] for a in range(3)]
   for observed in product(range(-1,k),repeat=n):
    missing=[i for i,y in enumerate(observed) if y<0]
    for weights in [(1,)*n,tuple(range(1,n+1))]:
     records=[]
     for fill in product(range(k),repeat=len(missing)):
      y=list(observed)
      for i,label in zip(missing,fill): y[i]=label
      risks=[sum(w*(p!=t) for w,p,t in zip(weights,pa,y)) for pa in pred]
      records.append(risks)
     counts['panels_masks_weightings']+=1
     counts['label_completions']+=len(records)
     lower={}; upper={}
     for a in range(3):
      for b in range(3):
       if a==b: continue
       known=sum(weights[i]*((pred[a][i]!=observed[i])-(pred[b][i]!=observed[i])) for i in range(n) if observed[i]>=0)
       disagreement=sum(weights[i] for i in missing if pred[a][i]!=pred[b][i])
       values=[r[a]-r[b] for r in records]
       lower[a,b]=known-disagreement; upper[a,b]=known+disagreement
       assert min(values)==lower[a,b] and max(values)==upper[a,b]
       counts['ordered_pair_checks']+=1
     for a in range(3):
      witness=[pred[a][i] if observed[i]<0 else observed[i] for i in range(n)]
      witness_risk=[sum(w*(p!=y) for w,p,y in zip(weights,pa,witness)) for pa in pred]
      for strict in [False,True]:
       cmp=(lambda x,y:x<y) if strict else (lambda x,y:x<=y)
       possible=any(all(cmp(r[a],r[b]) for b in range(3) if a!=b) for r in records)
       lower_certificate=all(cmp(lower[a,b],0) for b in range(3) if a!=b)
       common_witness=all(cmp(witness_risk[a],witness_risk[b]) for b in range(3) if a!=b)
       robust=all(all(cmp(r[a],r[b]) for b in range(3) if a!=b) for r in records)
       upper_certificate=all(cmp(upper[a,b],0) for b in range(3) if a!=b)
       assert possible==lower_certificate==common_witness
       assert robust==upper_certificate
       counts['candidate_winner_checks']+=1
# Shared unknown label: the two classifiers each predict one 0 and one 1.
# Their total risk differences must be zero, despite itemwise disagreement two.
shared=[sum(p!=y for p in (0,1))-sum(p!=y for p in (1,0)) for y in (0,1)]
assert shared==[0,0]
# Three pairwise endpoints cannot be selected independently.
risks3=[tuple(int(p!=y) for p in range(3)) for y in range(3)]
assert not any(r[0]-r[1]==-1 and r[1]-r[2]==-1 for r in risks3)
# Test the exact integer inequalities with a tiny exhaustive hash space.
B=2**8
for h in range(B):
 heldout=5*h>=4*B
 assert not (2*h<B and heldout)
 assert not (5*h<B and heldout)
# A top-one selected fixed point can exceed the holdout threshold when all raw ranks do.
raw_ranks=[230,240]
assert 5*min(raw_ranks)>=4*B
print(json.dumps({'status':'PASS','counts':counts,'shared_label_itemwise_sharpness_counterexample':True,'nonjoint_pair_endpoints_counterexample':True,'integer_exclusion_checks':B*2,'top_k_unconditional_exclusion_counterexample':True,'empty_target_status':'UNDEFINED: normalized risk and bounds require positive total target weight'},indent=2))

