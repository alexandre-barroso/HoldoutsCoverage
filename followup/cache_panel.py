"""Bound prediction-cache adapter. Historical fitting diagnostics are not rerun."""
from pathlib import Path
from hashlib import sha256
import copy,json,sys
import numpy as np
if sys.flags.optimize:raise SystemExit('Assertions must remain enabled')
HERE=Path(__file__).resolve().parent

def object_digest(value):
    return sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def cached_panel(train_texts,train_y,assessment_texts):
    from reference import validate_sources
    validate_sources(HERE.parent)
    cache=json.loads((HERE/'prediction_cache.json').read_text())
    if object_digest([[x,int(y)] for x,y in zip(train_texts,train_y,strict=True)])!=cache['ordered_fit_text_and_label_sha256']:
        raise ValueError('Cache training-order/input binding mismatch')
    if object_digest(assessment_texts)!=cache['ordered_assessment_text_sha256']:
        raise ValueError('Cache assessment-order binding mismatch')
    p=cache['predictions']
    if len(p)!=2621 or any(len(row)!=3 or any(type(x)is not int or x not in (0,1) for x in row) for row in p):
        raise ValueError('Invalid cached prediction matrix')
    matrix=np.asarray(p,dtype=np.int64)
    if sha256(matrix.astype('<i8').tobytes()).hexdigest()!=cache['prediction_sha256']:
        raise ValueError('Cache prediction matrix differs from recorded panel')
    return matrix,copy.deepcopy(cache['historical_fit'])
