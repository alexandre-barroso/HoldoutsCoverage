"""Finite zero-one comparison with partially observed labels.

Original implementation. These are identification envelopes, not confidence
intervals. Independent labels of retained units are unrestricted over classes.
No fitting, hash randomness or population transport is assumed by this module.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
import numpy as np


def audit(predictions, visible_labels, classes=(0, 1)):
    """Equal-weight finite target; -1 denotes every unavailable label."""
    pred = np.asarray(predictions, dtype=np.int64)
    labels = np.asarray(visible_labels, dtype=np.int64)
    if pred.ndim != 2 or labels.shape != (pred.shape[0],) or not len(labels):
        raise ValueError('Nonempty target and N by K predictions required')
    if not np.isin(pred, classes).all() or not np.isin(labels, (-1, *classes)).all():
        raise ValueError('Invalid class or missing-label encoding')
    n, k = pred.shape
    seen = labels != -1
    n_seen = int(seen.sum())
    errors = (pred[seen] != labels[seen, None]).sum(axis=0)
    pairwise = []
    lows = np.zeros((k, k), dtype=np.int64)
    highs = np.zeros((k, k), dtype=np.int64)
    for a in range(k):
        for b in range(k):
            known = int(errors[a] - errors[b])
            disagreement = int((pred[~seen, a] != pred[~seen, b]).sum())
            lo, hi = known - disagreement, known + disagreement
            lows[a, b], highs[a, b] = lo, hi
            if a < b:
                pairwise.append({
                    'a': a, 'b': b, 'known_difference_count': known,
                    'missing_disagreement_count': disagreement,
                    'lower_count': lo, 'upper_count': hi,
                    'lower': lo/n, 'upper': hi/n,
                    'mass_only_lower': (known-(n-n_seen))/n,
                    'mass_only_upper': (known+(n-n_seen))/n,
                    'observed_difference': known/n_seen if n_seen else None,
                    'certificate': ('A_STRICTLY_BETTER' if hi < 0 else
                                    'B_STRICTLY_BETTER' if lo > 0 else
                                    'EXACT_TIE' if lo == hi == 0 else
                                    'A_WEAKLY_BETTER' if hi == 0 else
                                    'B_WEAKLY_BETTER' if lo == 0 else
                                    'BOTH_STRICT_ORDERINGS_FEASIBLE')})
    return {
        'status': 'DEFINED' if n_seen else 'EMPTY_OBSERVED_SET',
        'target_n': n, 'observed_n': n_seen, 'missing_n': n-n_seen,
        'observed_error_counts': errors.tolist(),
        'observed_risks': (errors/n_seen).tolist() if n_seen else [None]*k,
        'observed_winners': np.flatnonzero(errors == errors.min()).tolist() if n_seen else [],
        'pairwise': pairwise,
        'robust_weak_winners': [a for a in range(k) if all(highs[a, b] <= 0 for b in range(k))],
        'robust_strict_winners': [a for a in range(k) if all(highs[a, b] < 0 for b in range(k) if a != b)],
        'possible_weak_winners': [a for a in range(k) if all(lows[a, b] <= 0 for b in range(k))],
        'possible_strict_winners': [a for a in range(k) if all(lows[a, b] < 0 for b in range(k) if a != b)]}


def exhaustive_checks():
    """Independent direct enumeration, including unequal integer weights."""
    cases = completions = 0
    for n in (1, 2):
        weights = tuple(range(1, n+1))
        vectors = list(product(range(3), repeat=n))
        for a, b, observed in product(vectors, repeat=3):
            for missing in product((False, True), repeat=n):
                known = sum(weights[i]*((a[i] != observed[i])-(b[i] != observed[i]))
                            for i in range(n) if not missing[i])
                radius = sum(weights[i] for i in range(n) if missing[i] and a[i] != b[i])
                ids = [i for i in range(n) if missing[i]]
                values = []
                for ys in product(range(3), repeat=len(ids)):
                    filled = list(observed)
                    for i, y in zip(ids, ys): filled[i] = y
                    values.append(sum(weights[i]*((a[i] != filled[i])-(b[i] != filled[i]))
                                      for i in range(n)))
                    completions += 1
                assert min(values) == known-radius and max(values) == known+radius
                cases += 1
    # Cross-check the public audit against all binary completions, three models.
    panel_cases = 0
    for n in (1, 2):
        vectors = list(product(range(2), repeat=n))
        for a, b, c in product(vectors, repeat=3):
            pred = np.array([a, b, c]).T
            for masked in product((-1, 0, 1), repeat=n):
                out = audit(pred, masked)
                missing = [i for i, v in enumerate(masked) if v == -1]
                all_errors = []
                for ys in product(range(2), repeat=len(missing)):
                    filled = list(masked)
                    for i, y in zip(missing, ys): filled[i] = y
                    all_errors.append((pred != np.array(filled)[:, None]).sum(axis=0))
                counts = np.array(all_errors)
                possible = [j for j in range(3) if np.any(np.all(counts[:, j, None] <= counts, axis=1))]
                robust = [j for j in range(3) if np.all(counts[:, j, None] <= counts)]
                assert out['possible_weak_winners'] == possible
                assert out['robust_weak_winners'] == robust
                for pair in out['pairwise']:
                    d = counts[:, pair['a']]-counts[:, pair['b']]
                    assert pair['lower_count'] == int(d.min())
                    assert pair['upper_count'] == int(d.max())
                panel_cases += 1
    # Itemwise bounds need not be sharp under a common alias label.
    alias_differences = [sum((a != y)-(b != y) for a, b in [(0, 1), (1, 0)]) for y in (0, 1)]
    assert alias_differences == [0, 0]
    p = Fraction(3, 5)
    assert p*Fraction(2, 5)+(1-p)*Fraction(1, 20) == Fraction(13, 50)
    assert p*Fraction(1, 10)+(1-p)*Fraction(1, 10) == Fraction(1, 10)
    return {'status': 'PASS_FINITE_TESTS_NOT_UNBOUNDED_PROOF',
            'weighted_pair_configurations': cases, 'label_completions': completions,
            'binary_panel_configurations': panel_cases,
            'shared_alias_counterexample': 'itemwise [-1,1], true shared-label difference 0',
            'authored_reversal': {'risk_A': 0.26, 'risk_B': 0.1, 'observed_A': 0.05, 'observed_B': 0.1}}
