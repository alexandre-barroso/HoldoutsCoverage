"""Reference integrity and exact scientific-content checks for the frozen study.

Runtime metadata may differ. Scientific results and the original experiment,
protocol, proof and data bindings must match the recorded reference.
This is a reproduction check, not independent reconstruction or scientific truth.
"""
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

if sys.flags.optimize:
    raise SystemExit('Optimized Python disables scientific assertions; use ordinary Python without -O/PYTHONOPTIMIZE')

REFERENCE_SHA256 = '86abbff5b171ba145ebfa408ec0342c366d2619dc807157be3f1ddcbc8de4a14'
EXCLUDED_METADATA = ('date_utc', 'elapsed_seconds', 'environment', 'protocol_filename')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference():
    path = Path(__file__).resolve().parent/'reference_expectations.json'
    if sha(path) != REFERENCE_SHA256:
        raise ValueError('Reference-expectation integrity check failed')
    reference = json.loads(path.read_text())
    if tuple(reference['excluded_metadata_fields']) != EXCLUDED_METADATA:
        raise ValueError('Unexpected metadata exclusion policy')
    return reference


def scientific_digest(data):
    payload = {k: v for k, v in data.items() if k not in EXCLUDED_METADATA}
    if 'protocol_sha256' in payload:
        reference = load_reference()
        if payload['protocol_sha256'] != reference['protocol_sha256']:
            raise ValueError('Scientific projection received an unbound protocol')
        payload['protocol_sha256'] = reference['protocol_comparison_sha256']
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'),
                           ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def validate_manifest_schema(manifest):
    if set(manifest) != {'files', 'source_files', 'protocol_sha256'}:
        raise ValueError('Unexpected run-manifest fields')
    if set(manifest['files']) != {'summary.json', 'results.json', 'masked_audits.json'}:
        raise ValueError('Missing or unexpected run-manifest file')
    if set(manifest['source_files']) != {'support_experiment.py', 'support_audit.py'}:
        raise ValueError('Missing or unexpected run-manifest source')
    for binding in manifest['files'].values():
        if set(binding) != {'sha256', 'bytes'}:
            raise ValueError('Unexpected run-manifest file binding')


def validate_companion_summary(evidence, summary):
    expected = {k: v for k, v in evidence.items() if k != 'records'}
    options = dict(sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    if json.dumps(summary, **options) != json.dumps(expected, **options):
        raise ValueError('Companion summary differs from the verified results projection')


def validate_sources(repo, protocol):
    ref = load_reference()
    for name, digest in ref['source_files'].items():
        if sha(Path(repo)/name) != digest:
            raise ValueError('Frozen source/proof mismatch: '+name)
    if sha(protocol) != ref['protocol_sha256']:
        raise ValueError('Frozen protocol mismatch')
    return ref


def validate_inputs(repo, protocol, data):
    ref = validate_sources(repo, protocol)
    if sha(data) != ref['data_zip_sha256']:
        raise ValueError('Public input ZIP digest mismatch')
    return ref


def validate_payload(data):
    ref = load_reference()
    records = data['records']
    dimensions = ('normalization', 'selection', 'arm', 'seed')
    actual = [tuple(r[k] for k in dimensions) for r in records]
    expected = set(product(*(ref['grid'][k] for k in dimensions)))
    if len(actual) != ref['record_count'] or len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Missing, duplicated or changed condition/seed identity')
    for key, refkey in [('input_zip_sha256', 'data_zip_sha256'),
                        ('input_member_sha256', 'data_member_sha256'),
                        ('protocol_sha256', 'protocol_sha256'),
                        ('masked_audits_sha256_before_oracle', 'masked_audits_sha256')]:
        if data[key] != ref[refkey]:
            raise ValueError('Result binding mismatch: '+key)
    for key in ('prediction_sha256', 'coefficient_sha256'):
        if data['fit'][key] != ref[key]:
            raise ValueError('Frozen fit mismatch: '+key)
    digest = scientific_digest(data)
    if digest != ref['scientific_projection_sha256']:
        raise ValueError('Scientific payload differs from the reference; exact reproduction not established. No tolerance or silent reference update is applied.')
    return {'status': 'EXACT_SCIENTIFIC_REFERENCE_MATCH', 'scientific_projection_sha256': digest,
            'excluded_metadata_fields': list(EXCLUDED_METADATA), 'records': len(records),
            'reference_expectations_sha256': REFERENCE_SHA256,
            'reference_results_sha256': ref['reference_results_sha256'],
            'actual_environment_record': data.get('environment'),
            'protocol_identifier_mapping': {'actual': ref['protocol_sha256'], 'comparison': ref['protocol_comparison_sha256']},
            'scope': 'All nonexcluded JSON fields match canonically after the declared protocol-identifier mapping. Runtime metadata and file byte digest are recorded separately; no empirical truth or independent reconstruction certified.'}


def validate_result_file(path, require_archived_bytes=False):
    data = json.loads(Path(path).read_text())
    result = validate_payload(data)
    actual = sha(path)
    if require_archived_bytes and actual != result['reference_results_sha256']:
        raise ValueError('Archived reference-byte check failed')
    result['actual_results_sha256'] = actual
    result['archived_reference_byte_match'] = actual == result['reference_results_sha256']
    return data, result
