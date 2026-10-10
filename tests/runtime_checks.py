"""Data-free source, numerical, parser and predictor checks on authored inputs."""
from pathlib import Path
from hashlib import sha256
import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from unittest.mock import patch

if sys.flags.optimize:
    raise SystemExit('Assertions must remain enabled for checks')
sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/'followup'))

import numpy as np
from scipy import sparse
import support_experiment as experiment
import support_reference
import reference as followup_reference
import reconstruct


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def rejects(function):
    try:
        function()
    except (AssertionError, ValueError, UnicodeError, KeyError, TypeError,
            zipfile.BadZipFile):
        return
    raise AssertionError('An invalid fixture was accepted')


def main():
    ignored = {'.git', '.venv', 'venv', '__pycache__', '.test-work',
               '.pytest_cache', 'verification', 'plot-cache', 'node_modules'}
    files = []
    for directory, folders, names in os.walk(REPO):
        folders[:] = [name for name in folders if name not in ignored]
        files.extend(Path(directory)/name for name in names)
    scripts = sorted(p for p in files if p.suffix == '.py')
    for script in scripts:
        ast.parse(script.read_text(encoding='utf-8'), filename=str(script))
    json_files = sorted(p for p in files if p.suffix == '.json')
    for path in json_files:
        followup_reference.load(path)
    assert [p.relative_to(REPO).as_posix() for p in files if p.suffix.lower() == '.md'] == ['README.md']
    assert (REPO/'README.md').read_text().splitlines()[0] == '# Verification repository'

    spec = REPO/'research/support_protocol.txt'
    core = support_reference.validate_sources(REPO, spec)
    ref = followup_reference.reference()
    manifest = followup_reference.load(REPO/'followup/portable_manifest.json')
    assert manifest['schema'] == 'portable-followup-sources'
    assert set(manifest['files']) == set(ref['required_source_files'])
    for name, expected in manifest['files'].items():
        path = REPO/name
        assert not path.is_symlink() and path.resolve(strict=True).is_relative_to(REPO)
        assert digest(path) == expected, name
    assert followup_reference.object_sha(ref['summary_scientific']) == ref['summary_scientific_sha256']

    cache = followup_reference.load(REPO/'followup/prediction_cache.json')
    matrix = np.asarray(cache['predictions'], dtype=np.int64)
    assert matrix.shape == (2621, 3) and np.isin(matrix, (0, 1)).all()
    assert sha256(matrix.astype('<i8').tobytes()).hexdigest() == cache['prediction_sha256']
    assert cache['prediction_sha256'] == ref['prediction_sha256'] == core['prediction_sha256']
    assert cache['input_zip_sha256'] == core['data_zip_sha256']
    assert cache['input_member_sha256'] == core['data_member_sha256']
    assert cache['source_results_sha256'] == ref['original_support_results_sha256']

    # Verify the explicit specification-identifier mapping independently. No
    # numerical field or input binding is dropped by this operation.
    fixture = {'protocol_sha256': digest(spec), 'records': [{'metric': 0.125}]}
    projected = copy.deepcopy(fixture)
    projected['protocol_sha256'] = core['protocol_comparison_sha256']
    expected = sha256(json.dumps(projected, sort_keys=True, separators=(',', ':'),
                                ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    assert support_reference.scientific_digest(fixture) == expected
    assert support_reference.scientific_digest(fixture | {'environment': {'host': 'fixture'}}) == expected
    changed = fixture | {'records': [{'metric': 0.25}]}
    assert support_reference.scientific_digest(changed) != expected
    rejects(lambda: support_reference.scientific_digest(fixture | {'protocol_sha256': '0'*64}))

    numerical = experiment.numerical_checks()
    assert experiment.whitespace('  Hello\tworld\n') == 'Hello world'
    assert experiment.fold('Straße') == 'strasse'
    assert experiment.common_key('  HE\u0301LLO\tWORLD  ') == 'héllo world'
    for role in ('outer', 'select', 'split', 'budget'):
        for seed in (0, 1, 199):
            for text in ('', 'fixture', '  Café\tStraße  ', '\U0001f600'):
                assert experiment.rank(role, seed, text) == reconstruct.rank(role, seed, text)
    counts = sparse.csr_matrix([[0., 0.], [3., 4.]])
    normalized = experiment.tfidf(counts, np.ones(2)).toarray()
    assert np.array_equal(normalized[0], [0., 0.])
    assert np.allclose(normalized[1], [.6, .8], rtol=0, atol=1e-15)

    # Expected archive fingerprints are substituted only inside this synthetic
    # fixture scope; the distributed input contract is never changed.
    parser_cases = 0
    with tempfile.TemporaryDirectory(prefix='verification-parser-') as temporary:
        path = Path(temporary)/'authored.zip'
        def parse(content, extra_member=False):
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('SMSSpamCollection', content)
                archive.writestr('readme', 'Authored test fixture, not corpus data.')
                if extra_member:
                    archive.writestr('unexpected', '')
            with patch.object(experiment, 'ARCHIVE_SHA', digest(path)), \
                 patch.object(experiment, 'MEMBER_SHA', sha256(content).hexdigest()):
                return experiment.load_units(path)
        content = b'ham\thello world\nham\t HELLO  WORLD \nspam\tprize now\nspam\tPRIZE NOW\n'
        units, info = parse(content)
        assert info['rows'] == 4 and info['retained_common_groups'] == 2
        assert {u[0] for u in units} == {'hello world', 'prize now'}
        assert next(u[1] for u in units if u[0] == 'hello world') == ' HELLO  WORLD '
        parser_cases += 1
        for invalid in (b'ham\thello\nspam\tHELLO\n', b'ham\thello',
                        b'other\thello\n', b'ham\t\xff\n', b'ham\t\n'):
            rejects(lambda invalid=invalid: parse(invalid))
            parser_cases += 1
        rejects(lambda: parse(content, extra_member=True))
        parser_cases += 1
        rejects(lambda: experiment.load_units(path))
        parser_cases += 1

    # A real fit on small authored texts exercises the fitting implementation;
    # it is not a fit or reproduction of the external corpus.
    train = ([f'meeting notes home item{i}' for i in range(8)] +
             [f'offer prize claim item{i}' for i in range(8)])
    labels = np.array([0]*8 + [1]*8, dtype=np.int64)
    predictions, fit = experiment.fit_panel(train, labels,
                                            ['meeting notes home', 'offer prize claim', 'unseen_token'])
    assert predictions.shape == (3, 3) and np.isin(predictions, (0, 1)).all()
    assert fit['convergence']['success'] and fit['convergence']['max_abs_gradient'] <= 1e-5
    assert np.array_equal(predictions[:, 2], np.zeros(3, dtype=np.int64))
    assert predictions[0, 0] == predictions[0, 1] == 0
    assert predictions[1, 0] == predictions[1, 1] == 1

    entrypoints = ('verify_support.py', 'verify_study.py', 'support_experiment.py',
                  'support_readout.py', 'followup/run_followup.py',
                  'followup/run_support_cached.py', 'followup/reconstruct.py',
                  'followup/readout.py', 'followup/test_reference.py')
    for name in entrypoints:
        result = subprocess.run([sys.executable, '-B', str(REPO/name), '--help'],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (name, result.stderr)
        assert 'usage:' in result.stdout.lower(), name
    print(json.dumps({'status': 'PASS_DATA_FREE_RUNTIME_CHECKS',
                      'python_sources_parsed': len(scripts), 'json_files_parsed': len(json_files),
                      'source_bindings': len(manifest['files']), 'cached_prediction_rows': len(matrix),
                      'numerical_checks': numerical, 'parser_fixtures': parser_cases,
                      'independent_hash_cases': 48, 'cli_help_checks': len(entrypoints),
                      'authored_predictor_fit': 'PASS', 'protocol_identifier_mapping': 'PASS',
                      'scope': 'Authored inputs and static references only; no empirical replay or Lean build.'}, indent=2))


if __name__ == '__main__':
    main()
