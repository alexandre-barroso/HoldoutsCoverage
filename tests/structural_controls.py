"""Original data-free finite checks, ported from the reviewed protocol audit.
Synthetic fixtures only; no empirical data or unbounded proof.
"""
import sys
if sys.flags.optimize:
 raise SystemExit("Assertions must remain enabled for finite checks")
import hashlib,json
B=1<<256
def h(role,seed,text):
 return int.from_bytes(hashlib.sha256(json.dumps(['project8-support-v1',role,seed,text],ensure_ascii=False,separators=(',',':')).encode('utf-8')).digest(),'big')
raw=[f'key{i}' for i in range(100)]
selected=[s for s in raw if 2*h('select',0,s)<B]
identity_visible=[s for s in selected if 5*h('select',0,s)>=4*B]
upper_visible=[s for s in selected if 5*h('select',0,s.upper())>=4*B]
all_visible=[s for s in raw if 5*h('select',0,s)>=4*B]
assert selected and not identity_visible and all(s.upper()!=s for s in raw)
assert len(set(s.upper() for s in raw))==len(raw)
assert upper_visible and all_visible
truth=(0,0,1,1); seen=(0,1)
def compare(a,b):
 delta_s=sum(x!=y for x,y in zip(a,truth))-sum(x!=y for x,y in zip(b,truth))
 delta_o=sum(a[i]!=truth[i] for i in seen)-sum(b[i]!=truth[i] for i in seen)
 return {'S_difference':delta_s/len(truth),'O_difference':delta_o/len(seen),'strict_reversal':delta_s*delta_o<0}
flip=compare((0,0,0,0),(1,0,1,1))
no_flip=compare((0,0,1,1),(1,0,0,1))
assert flip['strict_reversal'] and not no_flip['strict_reversal']
print(json.dumps({'status':'PASS_AUTHORED_SYNTHETIC_ONLY','raw_n':len(raw),'selected_n':len(selected),'identity_visible_n':len(identity_visible),'no_fixed_point_uppercase_visible_n':len(upper_visible),'no_selection_identity_visible_n':len(all_visible),'reversal_fixture':flip,'no_reversal_fixture':no_flip},indent=2))

