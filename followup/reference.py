"""Portable source and complete fixed-budget result contracts.

Exact record digests bind to independently reconstructed historical output.
Only explicitly named execution metadata differ; no numerical tolerance or
self-generated-reference update is a success route.
"""
from pathlib import Path
from hashlib import sha256
import json,math,re,sys
if sys.flags.optimize:raise SystemExit('Assertions must remain enabled')
HERE=Path(__file__).resolve().parent
REFERENCE_SHA='75299a320af5145da40be8a0293a98d1a9d9b94ee96f27d0156b1d37806dc3e5'
OUTPUTS={'acquisition_design.jsonl','acquisition_results.jsonl','design_manifest.json','eligibility_profile.json','exact_expectations.jsonl','execution_start.json','integrity_comparison.json','panel_replay.json','predictions_v.json','seed_averages.json','summary.json'}
DESIGN_FILES={'acquisition_design.jsonl','eligibility_profile.json','integrity_comparison.json','panel_replay.json','predictions_v.json'}
SUMMARY_METADATA={'completed_utc','elapsed_seconds','design_manifest_sha256','fit'}

def sha(path):return sha256(Path(path).read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def object_sha(x):return sha256(canonical(x).encode()).hexdigest()
def _pairs(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('Duplicate JSON key')
        d[k]=v
    return d

def load(path):return json.loads(Path(path).read_text(),object_pairs_hook=_pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def reference():
    path=HERE/'reference_results.json'
    if sha(path)!=REFERENCE_SHA:raise ValueError('Historical scientific reference changed')
    return load(path)

def validate_sources(repo):
    repo=Path(repo).resolve();ref=reference();p=HERE/'portable_manifest.json';m=load(p)
    if set(m)!={'schema','files','scope'} or m['schema']!='portable-followup-sources':raise ValueError('Source manifest schema')
    if set(m['files'])!=set(ref['required_source_files']):raise ValueError('Source manifest critical set mismatch')
    for name,digest in m['files'].items():
        f=repo/name
        if not re.fullmatch('[0-9a-f]{64}',digest) or f.is_symlink() or not f.resolve(strict=True).is_relative_to(repo):raise ValueError('Invalid source binding')
        if sha(f)!=digest:raise ValueError('Portable source changed: '+name)
    import numpy,scipy,unicodedata
    if numpy.__version__!='2.4.4' or scipy.__version__!='1.17.1' or sys.version_info[:3]!=(3,12,3) or unicodedata.unidata_version!='15.0.0':raise ValueError('Pinned Python/Unicode/NumPy/SciPy runtime required')
    cache=load(HERE/'prediction_cache.json')
    if cache['source_results_sha256']!=ref['original_support_results_sha256']:raise ValueError('Cache source binding')
    if cache['prediction_sha256']!=ref['prediction_sha256']:raise ValueError('Cache prediction binding')
    sys.path.insert(0,str(repo))
    from support_reference import validate_sources as original_sources
    original_sources(repo,repo/'research/support_protocol.txt')
    return m

def expected_panel():
    cache=load(HERE/'prediction_cache.json')
    return {'execution_mode':'CACHED_PREDICTIONS_NO_REFIT','historical_fit':cache['historical_fit'],
            'prediction_sha256':cache['prediction_sha256'],'cache_sha256':sha(HERE/'prediction_cache.json'),
            'scope':'Historical diagnostics are retained as provenance, not recomputed fitting or coefficient checks.'}

def validate_summary(summary):
    ref=reference()
    if set(summary)!=set(ref['summary_keys']):raise ValueError('Summary schema mismatch')
    if canonical(summary['fit'])!=canonical(expected_panel()):raise ValueError('Summary cache-mode/provenance mismatch')
    if type(summary['elapsed_seconds']) not in (float,int) or summary['elapsed_seconds']<0 or not math.isfinite(summary['elapsed_seconds']):raise ValueError('Invalid runtime duration')
    if not isinstance(summary['completed_utc'],str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T.+\+00:00',summary['completed_utc']):raise ValueError('Invalid runtime timestamp')
    if not re.fullmatch('[0-9a-f]{64}',summary['design_manifest_sha256']):raise ValueError('Invalid design binding')
    science={k:v for k,v in summary.items() if k not in SUMMARY_METADATA}
    if object_sha(science)!=ref['summary_scientific_sha256']:raise ValueError('Complete scientific summary differs from historical reference')

def validate_result_directory(directory):
    directory=Path(directory).resolve(strict=True);ref=reference();m=load(directory/'manifest.json')
    if set(m)!={'files','freeze_sha256','new_fitted_configurations','unchanged_fit_replays','execution_mode'}:raise ValueError('Output manifest schema')
    if set(m['files'])!=OUTPUTS:raise ValueError('Output manifest exact companion set')
    if m['execution_mode']!='CACHED_PREDICTIONS_NO_REFIT' or any(type(m[k])is not int or m[k]!=0 for k in ('new_fitted_configurations','unchanged_fit_replays')):raise ValueError('Execution-mode metadata mismatch')
    if m['freeze_sha256']!=sha(HERE/'portable_manifest.json'):raise ValueError('Recorded portable-source binding mismatch')
    for name,binding in m['files'].items():
        p=directory/name
        if p.is_symlink() or not p.resolve(strict=True).is_relative_to(directory):raise ValueError('Companion path escape')
        if set(binding)!={'sha256','bytes'} or type(binding['bytes'])is not int or binding['bytes']<0 or not re.fullmatch('[0-9a-f]{64}',binding['sha256']):raise ValueError('Invalid companion binding')
        if p.stat().st_size!=binding['bytes'] or sha(p)!=binding['sha256']:raise ValueError('Companion changed: '+name)
    for name,binding in ref['exact_record_files'].items():
        if sha(directory/name)!=binding['sha256'] or (directory/name).stat().st_size!=binding['bytes']:raise ValueError('Historical scientific records differ: '+name)
    panel=load(directory/'panel_replay.json')
    if canonical(panel)!=canonical(expected_panel()):raise ValueError('Cached panel provenance mismatch')
    start=load(directory/'execution_start.json')
    if set(start)!={'started_utc','freeze_sha256','environment','scope'} or start['freeze_sha256']!=m['freeze_sha256']:raise ValueError('Execution start schema/binding')
    if start['environment']!={'python':'3.12.3','numpy':'2.4.4','scipy':'1.17.1'}:raise ValueError('Recorded runtime mismatch')
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T.+\+00:00',start['started_utc']):raise ValueError('Invalid execution timestamp')
    if start['scope']!='Post-result fixed-panel follow-up; public data only; no protected reads':raise ValueError('Unexpected execution scope')
    design=load(directory/'design_manifest.json')
    if set(design)!={'status','sealed_utc','design_slots','target_policy_slots','contexts','freeze_sha256','prediction_sha256','input_zip_sha256','input_member_sha256','files'}:raise ValueError('Design schema')
    if set(design['files'])!=DESIGN_FILES:raise ValueError('Design critical set')
    for n,count in [('design_slots',76800),('target_policy_slots',2400),('contexts',600)]:
        if type(design[n])is not int or design[n]!=count:raise ValueError('Design count mismatch')
    for n,expected in [('status','SEALED_BEFORE_NEW_FULL_LABEL_OUTCOMES'),('freeze_sha256',m['freeze_sha256']),('prediction_sha256',ref['prediction_sha256']),('input_zip_sha256',ref['input_zip_sha256']),('input_member_sha256',ref['input_member_sha256'])]:
        if design[n]!=expected:raise ValueError('Design binding mismatch: '+n)
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T.+\+00:00',design['sealed_utc']):raise ValueError('Invalid sealing timestamp')
    for n,digest in design['files'].items():
        if digest!=sha(directory/n):raise ValueError('Sealed design companion changed')
    summary=load(directory/'summary.json');validate_summary(summary)
    if summary['design_manifest_sha256']!=sha(directory/'design_manifest.json'):raise ValueError('Summary design binding')
    return {'status':'EXACT_HISTORICAL_RECORDS_AND_SCIENTIFIC_SUMMARY','original_result_manifest_sha256':ref['historical_followup_manifest_sha256'],'actual_result_manifest_sha256':sha(directory/'manifest.json'),'source_manifest_sha256':m['freeze_sha256'],'fits_executed':0,'scope':'Record/reference and internal binding check; does not authenticate historical execution or recompute fitted coefficients.'}
