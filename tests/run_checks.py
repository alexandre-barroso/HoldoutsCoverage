"""Run the data-free Python checks; no dataset or Lean installation required."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

if sys.flags.optimize:
    raise SystemExit('Assertions must remain enabled for checks')

repo = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env['PYTHONDONTWRITEBYTECODE'] = '1'

def run(script, *arguments):
    result = subprocess.run([sys.executable, '-B', str(repo/script), *map(str, arguments)],
                            cwd=repo, env=env, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)

with tempfile.TemporaryDirectory(prefix='verification-tests-') as temporary:
    work = Path(temporary)
    for name in ('finite_weighted.py', 'structural_controls.py', 'masking_checks.py',
                 'reference_checks.py', 'runtime_checks.py'):
        run('tests/'+name)
    run('followup/test_followup.py', work/'exact_sampling.json')
    run('followup/test_acquisition.py', work/'acquisition.json')
    run('followup/test_reference.py')
print('PASS: all data-free Python checks; no empirical replay or Lean build.')
