"""Verify original support-study evidence and replay its Lean kernel objects.

Default: recompute the prespecified public-data experiment, then build/replay.
--existing-results verifies bindings of an existing run instead of fitting again.
Neither mode certifies empirical truth or population generality.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from support_reference import (validate_sources, validate_inputs, validate_result_file,
                               validate_manifest_schema, validate_companion_summary)

if sys.flags.optimize:
    raise SystemExit('Verification requires assertions enabled; do not use -O or PYTHONOPTIMIZE')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--existing-results', type=Path)
    parser.add_argument('--synthetic-only', action='store_true', help='Data-free finite checks and formal build; no empirical reproduction claim')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    protocol = args.protocol or repo/'research/support_protocol.txt'
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    fresh = Path(tempfile.mkdtemp(prefix='support-verify-', dir=work))
    commands = []
    outcome = {'date_utc': datetime.now(timezone.utc).isoformat(), 'status': 'IN_PROGRESS',
               'commands': commands, 'scope': 'Execution, artifact bindings and fresh Lean kernel replay; separate scientific review required.'}
    def run(command, cwd, name, env=None):
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=240)
        log = fresh/(name+'.log')
        log.write_text(result.stdout+result.stderr)
        commands.append({'name': name, 'command': [str(x) for x in command],
                         'exit_code': result.returncode, 'log': log.name})
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {log.name}')
        return result.stdout
    try:
        validate_sources(repo, protocol)
        run([sys.executable, '-B', str(repo/'tests/run_checks.py')], repo, 'portable_finite_checks')
        if args.synthetic_only:
            if args.existing_results or args.data:
                raise ValueError('Synthetic-only mode does not accept data or existing results')
            outcome['experiment_mode'] = 'SYNTHETIC_AND_FORMAL_ONLY_NO_EMPIRICAL_REPRODUCTION'
        else:
            if args.data is None:
                raise ValueError('--data is required for empirical verification')
            validate_inputs(repo, protocol, args.data)
            verify_empirical(args, repo, protocol, fresh, outcome, run)
        verify_formal(repo, fresh, outcome, run)
    except Exception as exc:
        outcome.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
    path = fresh/'verification.json'
    path.write_text(json.dumps(outcome, indent=2)+'\n')
    print(json.dumps({'status': outcome['status'], 'verification': str(path)}))
    raise SystemExit(outcome['status'] != 'PASS')


def verify_empirical(args, repo, protocol, fresh, outcome, run):
        if args.existing_results:
            results = args.existing_results.resolve(strict=True)
            outcome['experiment_mode'] = 'EXISTING_RUN_BINDINGS_ONLY_NO_REFIT'
        else:
            results = fresh/'experiment'
            run([sys.executable, '-B', str(repo/'support_experiment.py'), '--data', str(args.data.resolve(strict=True)),
                 '--protocol', str(protocol.resolve(strict=True)), '--output', str(results)], repo, 'experiment')
            outcome['experiment_mode'] = 'FRESH_RECOMPUTATION'
        manifest = json.loads((results/'manifest.json').read_text())
        validate_manifest_schema(manifest)
        for name, binding in manifest['files'].items():
            assert Path(name).name == name
            assert sha(results/name) == binding['sha256']
            assert (results/name).stat().st_size == binding['bytes']
        for name, binding in manifest['source_files'].items():
            assert Path(name).name == name
            assert sha(repo/name) == binding, f'Original experiment code changed: {name}'
        assert sha(protocol) == manifest['protocol_sha256']
        evidence, reference_check = validate_result_file(results/'results.json')
        validate_companion_summary(evidence, json.loads((results/'summary.json').read_text()))
        outcome['scientific_reference_check'] = reference_check
        outcome['companion_summary_check'] = 'EXACT_RESULTS_PROJECTION_INCLUDING_RUNTIME_METADATA'
        outcome['manifest_file_set_check'] = 'EXACT_THREE_OUTPUTS_AND_TWO_SCIENTIFIC_SOURCES'
        assert sha(args.data) == evidence['input_zip_sha256']
        assert sha(results/'masked_audits.json') == evidence['masked_audits_sha256_before_oracle']
        assert len(evidence['records']) == 6400
        assert evidence['fit']['convergence']['success']
        assert evidence['fit']['convergence']['max_abs_gradient'] <= 1e-5
        assert all(r['oracle']['oracle_containment'] for r in evidence['records'])
        outcome['results_sha256'] = sha(results/'results.json')
        outcome['summary_sha256'] = sha(results/'summary.json')
        outcome['masked_audits_sha256'] = sha(results/'masked_audits.json')


def verify_formal(repo, fresh, outcome, run):
        lean_dir = repo/'lean'
        source = lean_dir/'SupportBounds.lean'
        # This scan supplements kernel/axiom checks; it is not a proof by itself.
        stripped = re.sub(r'/\-[\s\S]*?\-/', '', source.read_text())
        assert not re.search(r'\b(sorry|admit|axiom|native_decide|unsafe)\b', stripped)
        lean = shutil.which('lean')
        if not lean:
            raise RuntimeError('Lean is not installed')
        version = run([lean, '--version'], lean_dir, 'lean_version')
        assert 'version 4.34.0,' in version
        prefix = Path(run([lean, '--print-prefix'], lean_dir, 'lean_prefix').strip())
        build = run([lean, '-o', str(fresh/'SupportBounds.olean'), str(source)], lean_dir, 'lean_build')
        required = {'fixedpoint_exclusion', 'selected_max_exclusion', 'comparison_interval',
                    'lower_completion_attains', 'upper_completion_attains', 'common_lower_completion',
                    'possible_weak_winner_iff', 'robust_weak_winner_iff'}
        axioms = {}
        for line in build.splitlines():
            match = re.match(r"'SupportAudit\.(\w+)' (.*)", line)
            if not match:
                continue
            name, report = match.groups()
            if 'does not depend on any axioms' in report:
                dependencies = set()
            else:
                found = re.search(r'axioms: \[(.*)\]', report)
                assert found, 'Unknown axiom report format'
                dependencies = set(found.group(1).split(', '))
            assert dependencies <= {'propext', 'Classical.choice', 'Quot.sound'}, f'Unacceptable axioms: {name}'
            axioms[name] = sorted(dependencies)
        assert set(axioms) == required
        checker = prefix/'bin/leanchecker'
        assert checker.is_file(), 'Compatible fresh kernel replay tool unavailable'
        env = os.environ.copy(); env['LEAN_PATH'] = str(fresh)
        run([str(checker), '--fresh', '-v', 'SupportBounds'], lean_dir, 'lean_fresh_replay', env)
        outcome.update(status='PASS', proof_source_sha256=sha(source), axioms=axioms,
                       formal_scope='Integer equal-unit loss-difference counts on lists, endpoint completions, weak panel winners and deterministic rank exclusion. Repeated indices count repeatedly.',
                       outside_formal_scope=['Rational division by sample size', 'arbitrary real weights',
                                             'Unicode/key/data correctness', 'hash independence',
                                             'model fitting', 'empirical truth', 'novelty'],
                       environment_limit='Fresh module directory and kernel replay on the same installed host; no independent OS installation or external checker.')


if __name__ == '__main__':
    main()
