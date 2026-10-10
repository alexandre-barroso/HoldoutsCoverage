"""Data-free failures and metadata invariance of the reproduction gate."""
import sys
if sys.flags.optimize:
    raise SystemExit('Assertions must remain enabled for checks')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,subprocess,tempfile
from support_reference import (scientific_digest,load_reference,validate_sources,validate_inputs,
                               validate_manifest_schema,validate_companion_summary)
repo=Path(__file__).resolve().parents[1]
ref=load_reference();validate_sources(repo,repo/'research/support_protocol.txt')
base={'records':[{'count':3}],'fit':{'metric':.125}}
assert scientific_digest(base)==scientific_digest(base|{'date_utc':'changed','elapsed_seconds':999,'environment':{'platform':'different'},'protocol_filename':'renamed.txt'})
assert scientific_digest(base)!=scientific_digest(base|{'records':[{'count':4}]})
summary={k:v for k,v in base.items() if k!='records'}
validate_companion_summary(base,summary)
for changed in [summary|{'fit':{'metric':.25}},summary|{'date_utc':'changed'},
                summary|{'records':base['records']},summary|{'fit':{'metric':True}}]:
    try:validate_companion_summary(base,changed)
    except ValueError:pass
    else:raise AssertionError('Altered companion summary accepted')
manifest={'files':{name:{'sha256':'synthetic','bytes':0} for name in ['summary.json','results.json','masked_audits.json']},
          'source_files':{'support_experiment.py':'synthetic','support_audit.py':'synthetic'},'protocol_sha256':'synthetic'}
validate_manifest_schema(manifest)
for changed in [manifest|{'files':{}},manifest|{'source_files':{}},manifest|{'unexpected':0},
                manifest|{'files':manifest['files']|{'unexpected.json':{'sha256':'synthetic','bytes':0}}}]:
    try:validate_manifest_schema(changed)
    except ValueError:pass
    else:raise AssertionError('Incomplete or extended manifest accepted')
with tempfile.TemporaryDirectory(prefix='reference-checks-') as temporary:
    bad=Path(temporary)/'synthetic-invalid.bin';bad.write_bytes(b'authored invalid input fixture')
    for call in [lambda:validate_sources(repo,bad),lambda:validate_inputs(repo,repo/'research/support_protocol.txt',bad)]:
        try:call()
        except ValueError:pass
        else:raise AssertionError('Invalid protocol/data accepted')
for name in ('verify_support.py','support_readout.py'):
    p=subprocess.run([sys.executable,'-O','-B',str(repo/name),'--help'],capture_output=True,text=True)
    assert p.returncode!=0 and 'Optimized Python' in p.stdout+p.stderr
print(json.dumps({'status':'PASS_DATA_FREE_GATE_CHECKS','optimization_rejections':2,'invalid_bindings':2,'invalid_summaries':4,'invalid_manifests':4,'scope':'Scientific fixture mutation, companion-summary binding and metadata projection; actual 6,400-record positive/negative checks need an empirical run.'}))
