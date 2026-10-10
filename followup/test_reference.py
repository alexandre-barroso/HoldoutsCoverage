"""Authored corruption fixtures. No fitting, source mutation or protected input."""
import argparse,copy,json,shutil,sys
from pathlib import Path
from hashlib import sha256
if sys.flags.optimize:raise SystemExit('Assertions must remain enabled')
from reference import reference,validate_summary,expected_panel,validate_result_directory,load,sha,canonical

def reject(fn):
    try:fn()
    except (ValueError,KeyError,TypeError,FileNotFoundError):return
    raise AssertionError('Corrupted fixture was accepted')

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path);p.add_argument('--work-dir',type=Path);a=p.parse_args()
    ref=reference();fixture=copy.deepcopy(ref['summary_scientific'])
    fixture.update(completed_utc='2026-09-27T00:00:00+00:00',elapsed_seconds=1.0,design_manifest_sha256='0'*64,fit=expected_panel())
    validate_summary(fixture);tests=[]
    def summary_test(name,change):
        x=copy.deepcopy(fixture);change(x);reject(lambda:validate_summary(x));tests.append(name)
    summary_test('changed_primary_metric',lambda x:x['conditions'][0]['exact_design']['absolute_error'].__setitem__('mean',.123))
    summary_test('boolean_for_budget',lambda x:x.__setitem__('budget',True))
    summary_test('boolean_for_float',lambda x:x['conditions'][0]['exact_design']['absolute_error'].__setitem__('mean',False))
    summary_test('missing_condition',lambda x:x['conditions'].pop())
    summary_test('duplicated_condition',lambda x:x['conditions'].append(copy.deepcopy(x['conditions'][0])))
    summary_test('nonfinite_metric',lambda x:x['conditions'][0]['exact_design']['absolute_error'].__setitem__('mean',float('nan')))
    summary_test('false_fit_mode',lambda x:x['fit'].__setitem__('execution_mode','FRESH_FIT'))
    summary_test('changed_historical_diagnostic',lambda x:x['fit']['historical_fit']['convergence'].__setitem__('max_abs_gradient',0.0))
    summary_test('boolean_runtime_duration',lambda x:x.__setitem__('elapsed_seconds',True))
    summary_test('missing_summary_field',lambda x:x.pop('repetitions'))
    summary_test('unexpected_summary_field',lambda x:x.__setitem__('unbound_claim',42))
    if a.results:
        if not a.work_dir:raise ValueError('Explicit fixture directory required')
        a.work_dir.mkdir(parents=True,exist_ok=False);target=a.work_dir/'altered_run';shutil.copytree(a.results,target)
        validate_result_directory(target)
        originals={n:(target/n).read_bytes() for n in ('manifest.json','summary.json','design_manifest.json','execution_start.json','panel_replay.json')}
        def write(name,x):(target/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
        def refresh(name):
            m=load(target/'manifest.json');m['files'][name]={'sha256':sha(target/name),'bytes':(target/name).stat().st_size};write('manifest.json',m)
        def reset():
            for n,b in originals.items():(target/n).write_bytes(b)
        def companion_test(name,change):
            reset();change();reject(lambda:validate_result_directory(target));tests.append(name)
        def summary_corrupt():
            s=load(target/'summary.json');s['conditions'][0]['exact_design']['absolute_error']['mean']=.123;write('summary.json',s);refresh('summary.json')
        companion_test('self_consistent_wrong_summary',summary_corrupt)
        def dropped():
            m=load(target/'manifest.json');m['files'].pop('summary.json');write('manifest.json',m)
        companion_test('incomplete_companion_manifest',dropped)
        def manifest_bool():
            m=load(target/'manifest.json');m['unchanged_fit_replays']=False;write('manifest.json',m)
        companion_test('boolean_execution_count',manifest_bool)
        def design_missing():
            d=load(target/'design_manifest.json');d['files'].pop('eligibility_profile.json');write('design_manifest.json',d);refresh('design_manifest.json')
            s=load(target/'summary.json');s['design_manifest_sha256']=sha(target/'design_manifest.json');write('summary.json',s);refresh('summary.json')
        companion_test('self_consistent_incomplete_design',design_missing)
        def env_wrong():
            x=load(target/'execution_start.json');x['environment']['numpy']='0';write('execution_start.json',x);refresh('execution_start.json')
        companion_test('self_consistent_wrong_environment',env_wrong)
        def runtime_omitted():
            x=load(target/'execution_start.json');x.pop('scope');write('execution_start.json',x);refresh('execution_start.json')
        companion_test('self_consistent_missing_execution_metadata',runtime_omitted)
        def missing_record():
            name='acquisition_design.jsonl';p=target/name;original=p.read_bytes();first,rest=original.split(b'\n',1);p.write_bytes(rest);refresh(name)
            try:reject(lambda:validate_result_directory(target))
            finally:p.write_bytes(original)
        reset();missing_record();tests.append('missing_acquisition_record');reset()
        validate_result_directory(target)
    print(json.dumps({'status':'PASS_AUTHORED_CORRUPTION_CHECKS','cases':len(tests),'checks':tests,'scope':'Deliberate invalid fixtures, not model fitting or authentication of historical JSON execution'},indent=2))
if __name__=='__main__':main()
