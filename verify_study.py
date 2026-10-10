"""Combined original support and fixed-budget verification; no default model fit.

Outputs are private local verification records. Nothing is downloaded/uploaded.
"""
import argparse,json,os,re,shutil,subprocess,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
from hashlib import sha256
sys.dont_write_bytecode=True
if sys.flags.optimize:raise SystemExit('Use assertions enabled, without -O/PYTHONOPTIMIZE')
REPO=Path(__file__).resolve().parent
MANIFEST_SHA='16db95dae39d5fd34e4a771192cadcaadf1f2d02178edbb1e88351f4f7c17572'

def sha(p):return sha256(Path(p).read_bytes()).hexdigest()
def formal_preflight():
    elan=shutil.which('elan')
    if not elan:raise RuntimeError('Install Lean4.34.0 and compatible leanchecker separately; verifier never installs it')
    p=subprocess.run([elan,'toolchain','list'],capture_output=True,text=True,timeout=30)
    if p.returncode:raise RuntimeError('Cannot inspect installed Lean toolchains')
    entries=[re.sub(r'\s+\([^)]*\)\s*$','',line.strip()) for line in p.stdout.splitlines()]
    if 'leanprover/lean4:v4.34.0' not in entries:raise RuntimeError('Required Lean4.34.0 is not already installed; no install attempted')
    return {'installed_required_toolchain':True,'scope':'Read-only elan toolchain list preflight; subsequent unchanged support verifier builds and freshly replays the proof.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--existing-support-results',type=Path);p.add_argument('--existing-followup-results',type=Path)
    a=p.parse_args();work=a.work_dir.resolve()
    if work.is_relative_to(REPO):raise SystemExit('Choose a work directory outside this repository')
    work.mkdir(parents=True,exist_ok=True);fresh=Path(tempfile.mkdtemp(prefix='study-verify-',dir=work));commands=[]
    report={'status':'IN_PROGRESS','started_utc':datetime.now(timezone.utc).isoformat(),'fits_executed':0,'commands':commands,
        'mode':'FIXED_PREDICTION_RECONSTRUCTION_NO_REFIT','scope':'Full aggregate checks plus independent acquisition reconstruction and fresh finite/formal checks; not new model fitting, novelty or acceptance certification.'}
    def run(cmd,name):
        q=subprocess.run([str(x) for x in cmd],cwd=REPO,capture_output=True,text=True,timeout=600)
        (fresh/(name+'.log')).write_text(q.stdout+q.stderr);commands.append({'name':name,'exit_code':q.returncode,'log':name+'.log'})
        if q.returncode:raise RuntimeError(name+' failed; see '+name+'.log')
        return q.stdout
    try:
        manifest=REPO/'followup/portable_manifest.json'
        if sha(manifest)!=MANIFEST_SHA:raise ValueError('Portable source manifest changed')
        sys.path.insert(0,str(REPO/'followup'))
        from reference import validate_sources,validate_result_directory
        validate_sources(REPO);report['source_manifest_sha256']=sha(manifest)
        from support_reference import validate_inputs
        validate_inputs(REPO,REPO/'research/support_protocol.txt',a.data)
        report['formal_preflight']=formal_preflight()
        for name,out in [('test_followup.py','synthetic_tests.json'),('test_acquisition.py','acquisition_tests.json')]:
            run([sys.executable,'-B',REPO/'followup'/name,fresh/out],name.removesuffix('.py'))
        run([sys.executable,'-B',REPO/'followup/test_reference.py'], 'reference_negative_checks')
        support=a.existing_support_results.resolve(strict=True) if a.existing_support_results else fresh/'support'
        if not a.existing_support_results:
            run([sys.executable,'-B',REPO/'followup/run_support_cached.py','--data',a.data.resolve(strict=True),'--output',support,'--record',fresh/'support_cache_execution.json'],'support_from_cache')
        support_stdout=run([sys.executable,'-B',REPO/'verify_support.py','--data',a.data.resolve(strict=True),'--existing-results',support,'--work-dir',fresh/'support_verification'],'original_support_verification')
        support_record=json.loads(support_stdout.strip().splitlines()[-1]);vp=Path(support_record['verification']);formal=json.loads(vp.read_text())
        if formal['status']!='PASS':raise ValueError('Original support formal/evidence gate did not pass')
        report['support_verification']={'path':str(vp.relative_to(fresh)),'sha256':sha(vp),'scientific_reference_check':formal['scientific_reference_check'],'axioms':formal['axioms']}
        followup=a.existing_followup_results.resolve(strict=True) if a.existing_followup_results else fresh/'followup'
        if not a.existing_followup_results:
            run([sys.executable,'-B',REPO/'followup/run_followup.py','--data',a.data.resolve(strict=True),'--support-results',support,'--output',followup],'fixed_budget_from_cache')
        report['followup_reference_check']=validate_result_directory(followup)
        run([sys.executable,'-B',REPO/'followup/reconstruct.py','--data',a.data.resolve(strict=True),'--results',followup,'--output',fresh/'independent_reconstruction.json'],'independent_reconstruction')
        run([sys.executable,'-B',REPO/'followup/test_reference.py','--results',followup,'--work-dir',fresh/'corruption_fixtures'],'companion_corruption_checks')
        run([sys.executable,'-B',REPO/'followup/readout.py','--data',a.data.resolve(strict=True),'--support-results',support,'--followup-results',followup,'--output',fresh/'verification_readout.json'],'verification_readout')
        report['verification_readout_sha256']=sha(fresh/'verification_readout.json')
        report['independent_reconstruction']=json.loads((fresh/'independent_reconstruction.json').read_text())
        report['existing_support_results_supplied']=bool(a.existing_support_results);report['existing_followup_results_supplied']=bool(a.existing_followup_results)
        report['status']='PASS';report['finished_utc']=datetime.now(timezone.utc).isoformat()
    except Exception as e:report.update(status='FAIL',error=type(e).__name__+': '+str(e))
    path=fresh/'verification.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'verification':str(path)}));raise SystemExit(report['status']!='PASS')
if __name__=='__main__':main()
