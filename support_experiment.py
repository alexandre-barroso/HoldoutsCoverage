"""Controlled frozen-predictor SMS evaluation of composed hash sampling.

Reads only an explicitly provided public ZIP. Writes derived aggregates and
a source manifest. No raw texts,
vocabulary, model checkpoint, or full dataset is written. No pickle is loaded.
See research/support_protocol.txt for the prospective protocol and limits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import time
import unicodedata
import zipfile

import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit

from support_audit import audit, exhaustive_checks

ARCHIVE_SHA = '1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3'
MEMBER_SHA = '7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d'
SPACE = 1 << 256
MODEL_NAMES = ['logistic', 'naive_bayes', 'majority']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def whitespace(x):
    return ' '.join(x.split())


def fold(x):
    return unicodedata.normalize('NFC', unicodedata.normalize('NFC', x).casefold())


def common_key(x):
    return fold(whitespace(x))


def rank(role, seed, text):
    payload = json.dumps(['project8-support-v1', role, seed, text],
                         ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest(), 'big')


def load_units(path):
    assert digest(path) == ARCHIVE_SHA, 'Public dataset digest mismatch'
    with zipfile.ZipFile(path) as z:
        assert set(z.namelist()) == {'SMSSpamCollection', 'readme'}
        data = z.read('SMSSpamCollection')
    assert hashlib.sha256(data).hexdigest() == MEMBER_SHA
    lines = data.decode('utf-8').split('\n')
    assert lines[-1] == '' and all(lines[:-1])
    groups = defaultdict(list)
    raw_set = set()
    row_labels = Counter()
    for line in lines[:-1]:
        label, text = line.split('\t', 1)
        assert label in ('spam', 'ham') and text
        raw_set.add(text)
        row_labels[label] += 1
        groups[common_key(text)].append((text, int(label == 'spam')))
    units = []
    for key, values in groups.items():
        assert len({y for x, y in values}) == 1, 'Conflicting grouped labels'
        assert common_key(key) == key
        representative = min((x for x, y in values), key=lambda s: s.encode('utf-8'))
        units.append((key, representative, values[0][1]))
    units.sort(key=lambda row: row[0].encode('utf-8'))
    for normalizer in (whitespace, fold):
        keys = [normalizer(x) for key, x, y in units]
        assert len(keys) == len(set(keys)), 'Canonical alias survived grouping'
        assert all(normalizer(s) == s for s in keys)
    return units, {'rows': len(lines)-1, 'distinct_raw_texts': len(raw_set),
                   'retained_common_groups': len(units), 'row_label_counts': dict(row_labels),
                   'group_label_conflicts': 0,
                   'group_size_distribution': dict(Counter(len(v) for v in groups.values()))}


def features(text):
    tokens = re.findall(r'\w+', common_key(text), flags=re.UNICODE)
    counts = Counter(('u', word) for word in tokens)
    counts.update(('b', a, b) for a, b in zip(tokens, tokens[1:]))
    return counts


def count_matrix(documents, vocabulary):
    row, col, values = [], [], []
    for i, counts in enumerate(documents):
        for feature, count in counts.items():
            if feature in vocabulary:
                row.append(i); col.append(vocabulary[feature]); values.append(count)
    return sparse.csr_matrix((np.asarray(values, dtype=np.float64), (row, col)),
                             shape=(len(documents), len(vocabulary)))


def tfidf(counts, idf):
    x = counts.multiply(idf).tocsr()
    norms = np.sqrt(x.multiply(x).sum(axis=1)).A1
    scales = np.divide(1., norms, out=np.zeros_like(norms), where=norms > 0)
    return x.multiply(scales[:, None]).tocsr()


def logistic_objective(theta, x, y, c=1.):
    n = len(y)
    w, intercept = theta[:-1], theta[-1]
    logits = x @ w + intercept
    value = np.mean(np.logaddexp(0., logits)-y*logits) + (w @ w)/(2*c*n)
    residual = expit(logits)-y
    gradient = np.r_[np.asarray(x.T @ residual).ravel()/n + w/(c*n), residual.mean()]
    return float(value), gradient


def numerical_checks():
    x = sparse.csr_matrix([[0., 1., 2.], [1., 0., .5], [.3, .5, 0.], [2., 1., 0.]])
    y = np.array([0., 1., 0., 1.])
    t = np.array([-.2, .3, .1, -.4])
    analytical = logistic_objective(t, x, y)[1]
    epsilon = 1e-6
    numerical = []
    for j in range(len(t)):
        step = np.zeros_like(t); step[j] = epsilon
        numerical.append((logistic_objective(t+step, x, y)[0]-logistic_objective(t-step, x, y)[0])/(2*epsilon))
    error = float(np.max(np.abs(analytical-numerical)))
    assert error < 1e-8
    # Tiny invented count matrix; direct independent product-form NB score.
    counts = sparse.csr_matrix([[2., 0.], [1., 1.], [0., 3.], [0., 1.]])
    labels = np.array([0, 0, 1, 1])
    freq = np.vstack([np.asarray(counts[labels == k].sum(axis=0)).ravel()+1 for k in (0, 1)])
    log_p = np.log(freq/freq.sum(axis=1, keepdims=True))
    matrix_scores = np.asarray(counts @ log_p.T)+np.log(.5)
    direct = np.array([[np.log(.5)+sum(v*np.log(freq[k, j]/freq[k].sum()) for j, v in enumerate(row))
                        for k in (0, 1)] for row in counts.toarray()])
    assert np.allclose(matrix_scores, direct, atol=1e-14, rtol=0)
    # Exhaust finite integer boundaries, independent of SHA marginal behavior.
    boundaries = 0
    for space in (5, 10, 25):
        for h in range(space):
            for numerator, denominator in ((1, 5), (1, 2)):
                if denominator*h < numerator*space:
                    assert not 5*h >= 4*space
                boundaries += 1
    identity = [f'key{i}' for i in range(100)]
    selected = [s for s in identity if 2*rank('select', 0, s) < SPACE]
    assert selected and not [s for s in selected if 5*rank('select', 0, s) >= 4*SPACE]
    assert any(5*rank('select', 0, s) >= 4*SPACE for s in identity)
    assert all(s.upper() != s for s in identity)
    return {'status': 'PASS', 'logistic_gradient_max_error': error,
            'naive_bayes_direct_loop_checked': True, 'integer_boundary_cases': boundaries,
            'identity_empty_holdout_checked': True, 'no_selection_nonempty_checked': True}


def fit_panel(fit_texts, fit_labels, assess_texts):
    train_docs = [features(s) for s in fit_texts]
    assess_docs = [features(s) for s in assess_texts]
    df = Counter(f for d in train_docs for f in d)
    vocabulary = {feature: i for i, feature in enumerate(sorted(f for f, n in df.items() if n >= 2))}
    x_counts = count_matrix(train_docs, vocabulary)
    v_counts = count_matrix(assess_docs, vocabulary)
    n = len(fit_labels)
    idf = np.array([np.log((1+n)/(1+df[f]))+1 for f in vocabulary])
    x, v = tfidf(x_counts, idf), tfidf(v_counts, idf)
    result = minimize(logistic_objective, np.zeros(x.shape[1]+1), args=(x, fit_labels),
                      jac=True, method='L-BFGS-B',
                      options={'maxiter': 2000, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 50})
    value, gradient = logistic_objective(result.x, x, fit_labels)
    convergence = {'success': bool(result.success), 'message': str(result.message),
                   'iterations': int(result.nit), 'objective': value,
                   'max_abs_gradient': float(np.max(np.abs(gradient)))}
    if not result.success or convergence['max_abs_gradient'] > 1e-5:
        raise RuntimeError(json.dumps({'optimizer_failure': convergence}))
    lr = (v @ result.x[:-1]+result.x[-1] >= 0).astype(np.int64)
    class_counts = np.bincount(fit_labels, minlength=2)
    assert (class_counts > 0).all()
    frequencies = np.vstack([np.asarray(x_counts[fit_labels == k].sum(axis=0)).ravel()+1 for k in (0, 1)])
    log_probabilities = np.log(frequencies/frequencies.sum(axis=1, keepdims=True))
    nb_scores = np.asarray(v_counts @ log_probabilities.T) + np.log(class_counts/n)
    nb = (nb_scores[:, 1] > nb_scores[:, 0]).astype(np.int64)
    majority = np.full(len(assess_texts), int(class_counts[1] > class_counts[0]), dtype=np.int64)
    predictions = np.column_stack([lr, nb, majority])
    return predictions, {'fitting_n': n, 'assessment_n': len(assess_texts),
                         'fitting_classes': class_counts.tolist(), 'features': len(vocabulary),
                         'convergence': convergence,
                         'prediction_sha256': hashlib.sha256(predictions.astype('<i8').tobytes()).hexdigest(),
                         'coefficient_sha256': hashlib.sha256(result.x.astype('<f8').tobytes()).hexdigest()}


def describe(values):
    clean = [x for x in values if x is not None]
    if not clean:
        return {'defined': 0, 'undefined': len(values)}
    x = np.asarray(clean, dtype=float)
    return {'defined': len(clean), 'undefined': len(values)-len(clean),
            'mean': float(x.mean()), 'median': float(np.median(x)),
            'p05': float(np.quantile(x, .05)), 'p95': float(np.quantile(x, .95)),
            'min': float(x.min()), 'max': float(x.max())}


def oracle_fields(output, predictions, truth, full_risk):
    n = len(truth)
    count = (predictions != truth[:, None]).sum(axis=0)
    risk = count/n
    observed = output['observed_risks']
    s_error = [None if a is None else a-float(b) for a, b in zip(observed, risk)]
    v_error = [None if a is None else a-float(b) for a, b in zip(observed, full_risk)]
    winners = output['observed_winners']
    optimum = int(count.min())
    pair = output['pairwise'][0]
    true_difference = int(count[0]-count[1])
    for p in output['pairwise']:
        delta = int(count[p['a']]-count[p['b']])
        assert p['lower_count'] <= delta <= p['upper_count']
    return {'risk_S': risk.tolist(), 'error_counts_S': count.tolist(),
            'risk_estimation_error_S': s_error, 'risk_estimation_error_V': v_error,
            'primary_true_difference_S': true_difference/n,
            'primary_difference_estimation_error': (None if pair['observed_difference'] is None else
                                                   pair['observed_difference']-true_difference/n),
            'primary_strict_reversal': (None if not winners else pair['known_difference_count']*true_difference < 0),
            'observed_winners_contain_optimal': (None if not winners else any(count[w] == optimum for w in winners)),
            'regret_best': min((int(count[w])-optimum)/n for w in winners) if winners else None,
            'regret_worst': max((int(count[w])-optimum)/n for w in winners) if winners else None,
            'oracle_containment': True}


def summarize(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[(r['normalization'], r['selection'], r['arm'])].append(r)
    rows = []
    for (normalization, selection, arm), items in grouped.items():
        rows.append({'normalization': normalization, 'selection': selection, 'arm': arm,
                     'runs': len(items),
                     'observed_n': describe([r['audit']['observed_n'] for r in items]),
                     'fixed_visible_n': describe([r['support']['observed_fixed'] for r in items]),
                     'zero_fixed_visible_runs': sum(r['support']['observed_fixed'] == 0 for r in items),
                     'undefined_runs': sum(r['audit']['status'] != 'DEFINED' for r in items),
                     'primary_difference_error': describe([r['oracle']['primary_difference_estimation_error'] for r in items]),
                     'primary_difference_absolute_error': describe([None if r['oracle']['primary_difference_estimation_error'] is None
                                                                   else abs(r['oracle']['primary_difference_estimation_error']) for r in items]),
                     'strict_reversals': sum(r['oracle']['primary_strict_reversal'] is True for r in items),
                     'winner_set_excludes_optimal': sum(r['oracle']['observed_winners_contain_optimal'] is False for r in items),
                     'best_regret': describe([r['oracle']['regret_best'] for r in items]),
                     'worst_regret': describe([r['oracle']['regret_worst'] for r in items]),
                     'pair_interval_width': describe([r['audit']['pairwise'][0]['upper']-r['audit']['pairwise'][0]['lower'] for r in items]),
                     'pair_strict_certifications': sum(r['audit']['pairwise'][0]['certificate'] in ('A_STRICTLY_BETTER','B_STRICTLY_BETTER') for r in items),
                     'natural_separate_smaller_than_reuse': sum(r['support']['natural_separate_smaller'] for r in items)})
    paired = []
    indexed = {(r['normalization'], r['selection'], r['seed'], r['arm']): r for r in records}
    for normalization in ('unicode_whitespace', 'nfc_casefold'):
        for selection in ('cutoff_half', 'cutoff_fifth', 'all', 'top_fifth'):
            for base, other in [('reuse', 'separate'), ('reuse_budget', 'separate_budget')]:
                differences = []
                for seed in range(200):
                    a = indexed[normalization, selection, seed, base]['oracle']['primary_difference_estimation_error']
                    b = indexed[normalization, selection, seed, other]['oracle']['primary_difference_estimation_error']
                    differences.append(None if a is None or b is None else abs(b)-abs(a))
                paired.append({'normalization': normalization, 'selection': selection,
                               'contrast': other+' minus '+base+' absolute primary-difference error',
                               'summary': describe(differences)})
    return {'conditions': rows, 'paired_contrasts': paired}


def run(args):
    start = time.monotonic()
    out = args.output.resolve()
    assert not out.exists(), 'Preserve previous evidence; choose a new output directory'
    out.mkdir(parents=True)
    protocol = args.protocol.resolve(strict=True)
    checks = {'finite_algebra': exhaustive_checks(), 'numerical': numerical_checks()}
    units, input_info = load_units(args.data.resolve(strict=True))
    train = [u for u in units if 2*rank('outer', 0, u[0]) < SPACE]
    assess = [u for u in units if 2*rank('outer', 0, u[0]) >= SPACE]
    assert set(u[0] for u in train).isdisjoint(u[0] for u in assess)
    checks['group_boundaries'] = {}
    input_info['post_group_support'] = {}
    for name, normalizer in [('unicode_whitespace', whitespace), ('nfc_casefold', fold)]:
        fit_keys = [normalizer(u[1]) for u in train]
        assess_keys = [normalizer(u[1]) for u in assess]
        overlap = len(set(fit_keys) & set(assess_keys))
        assert overlap == 0 and len(set(fit_keys)) == len(fit_keys) and len(set(assess_keys)) == len(assess_keys)
        checks['group_boundaries'][name] = {'fitting_aliases': 0, 'assessment_aliases': 0, 'cross_pool_overlap': overlap}
        input_info['post_group_support'][name] = {
            pool: {'n': len(items), 'fixed': sum(u[1] == normalizer(u[1]) for u in items),
                   'changed': sum(u[1] != normalizer(u[1]) for u in items)}
            for pool, items in [('all', units), ('fitting', train), ('assessment', assess)]}
    texts = [u[1] for u in assess]
    truths = np.array([u[2] for u in assess], dtype=np.int64)
    pred, fit = fit_panel([u[1] for u in train], np.array([u[2] for u in train]), texts)
    fixed = {name: np.array([x == fn(x) for x in texts])
             for name, fn in [('unicode_whitespace', whitespace), ('nfc_casefold', fold)]}
    input_info['assessment_support'] = {name: {'fixed': int(v.sum()), 'changed': int((~v).sum())} for name, v in fixed.items()}
    records, selections = [], {}
    for seed in range(200):
        raw_ranks = [rank('select', seed, x) for x in texts]
        masks = {
            'cutoff_half': np.array([2*h < SPACE for h in raw_ranks]),
            'cutoff_fifth': np.array([5*h < SPACE for h in raw_ranks]),
            'all': np.ones(len(texts), dtype=bool),
            'top_fifth': np.zeros(len(texts), dtype=bool)}
        top_indices = sorted(range(len(texts)), key=lambda i: (raw_ranks[i], texts[i].encode('utf-8')))[:len(texts)//5]
        masks['top_fifth'][top_indices] = True
        budget_ranks = [rank('budget', seed, x) for x in texts]
        for norm_name, normalizer in [('unicode_whitespace', whitespace), ('nfc_casefold', fold)]:
            reuse_holdout = np.array([5*rank('select', seed, normalizer(x)) >= 4*SPACE for x in texts])
            separate_holdout = np.array([5*rank('split', seed, normalizer(x)) >= 4*SPACE for x in texts])
            for selection_name, selected in masks.items():
                ids = np.flatnonzero(selected)
                if not len(ids):
                    raise RuntimeError(f'EMPTY_TARGET: seed={seed}, selection={selection_name}; no metric defined')
                selections[seed, selection_name] = ids
                reused = selected & reuse_holdout
                separated = selected & separate_holdout
                matched = np.zeros(len(texts), dtype=bool)
                matched_reuse = np.zeros(len(texts), dtype=bool)
                to_keep = min(int(reused.sum()), int(separated.sum()))
                keep = sorted(np.flatnonzero(separated), key=lambda i: (budget_ranks[i], texts[i].encode('utf-8')))[:to_keep]
                matched[keep] = True
                keep_reuse = sorted(np.flatnonzero(reused), key=lambda i: (budget_ranks[i], texts[i].encode('utf-8')))[:to_keep]
                matched_reuse[keep_reuse] = True
                assert matched.sum() == matched_reuse.sum()
                maximum = max(raw_ranks[i] for i in ids)
                structural_antecedent = 5*maximum < 4*SPACE
                if structural_antecedent:
                    assert not np.any(reused & fixed[norm_name])
                for arm, visible in [('reuse', reused), ('separate', separated),
                                     ('reuse_budget', matched_reuse), ('separate_budget', matched)]:
                    local_labels = np.full(len(ids), -1, dtype=np.int64)
                    local_seen = visible[ids]
                    local_labels[local_seen] = truths[ids[local_seen]]
                    result = audit(pred[ids], local_labels)
                    records.append({'normalization': norm_name, 'selection': selection_name,
                                    'seed': seed, 'arm': arm,
                                    'selection_max_rank_hex': f'{maximum:064x}',
                                    'exclusion_antecedent': structural_antecedent,
                                    'support': {'selected_n': len(ids),
                                                'selected_fixed': int(fixed[norm_name][ids].sum()),
                                                'observed_fixed': int((visible & fixed[norm_name]).sum()),
                                                'observed_changed': int((visible & ~fixed[norm_name]).sum()),
                                                'natural_separate_smaller': bool(separated.sum() < reused.sum())},
                                    'audit': result})
    # Seal all masked-label audits before revealing full S or V risks.
    masked_path = out/'masked_audits.json'
    masked_path.write_text(json.dumps(records, sort_keys=True, allow_nan=False)+'\n')
    masked_sha = digest(masked_path)
    full_risk = (pred != truths[:, None]).mean(axis=0)
    for record in records:
        ids = selections[record['seed'], record['selection']]
        record['oracle'] = oracle_fields(record['audit'], pred[ids], truths[ids], full_risk)
    result = {'status': 'COMPUTED_PENDING_INDEPENDENT_REVIEW',
              'date_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': digest(protocol), 'protocol_filename': protocol.name,
              'input_zip_sha256': ARCHIVE_SHA, 'input_member_sha256': MEMBER_SHA,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                           'unicode': unicodedata.unidata_version, 'platform': platform.platform()},
              'models': MODEL_NAMES, 'checks': checks, 'input': input_info, 'fit': fit,
              'reference_V_risks': full_risk.tolist(),
              'masked_audits_sha256_before_oracle': masked_sha,
              'summary': summarize(records), 'records': records,
              'elapsed_seconds': time.monotonic()-start,
              'limitations': ['One finite, historically compiled public corpus; no current user-population risk claim.',
                              'Frozen fit; masks are paired design variation, not independent datasets.',
                              'Elementary finite bounds; no new-theory or confidence-interval claim.',
                              'Grouping representative and normalization chosen after input-only profiling.',
                              'No formal Lean gate, independent review, novel significance or submission readiness implied.']}
    (out/'results.json').write_text(json.dumps(result, sort_keys=True, allow_nan=False)+'\n')
    (out/'summary.json').write_text(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2, allow_nan=False)+'\n')
    files = [p for p in out.iterdir() if p.is_file()]
    source_files = [Path(__file__), Path(__file__).with_name('support_audit.py')]
    manifest = {'files': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in files},
                'source_files': {p.name: digest(p) for p in source_files},
                'protocol_sha256': digest(protocol)}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'conditions': len(records), 'input': input_info,
                      'fit': fit, 'reference_V_risks': full_risk.tolist(),
                      'elapsed_seconds': result['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args())
