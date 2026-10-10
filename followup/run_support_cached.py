"""Replay original masks/audits/oracles from bound predictions, without fitting.

The numerical calculation is retained. Its fit_panel callback is
replaced with a cache adapter that checks ordered input fingerprints. Historical
fit diagnostics remain inside its result schema, and are explicitly identified
as historical in the separate execution record. No coefficient check is rerun.
"""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
from hashlib import sha256
sys.dont_write_bytecode=True
if sys.flags.optimize:raise SystemExit('Assertions must remain enabled')
HERE=Path(__file__).resolve().parent;REPO=HERE.parent
sys.path.insert(0,str(REPO))
from reference import validate_sources
from cache_panel import cached_panel

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--record',type=Path,required=True);a=p.parse_args()
    validate_sources(REPO)
    if a.output.resolve().is_relative_to(REPO):raise ValueError('Outputs must be outside the repository')
    if a.record.exists():raise FileExistsError(a.record)
    import support_experiment as base
    original=base.fit_panel;base.fit_panel=cached_panel
    try:base.run(argparse.Namespace(data=a.data,output=a.output,protocol=REPO/'research/support_protocol.txt'))
    finally:base.fit_panel=original
    record={'mode':'FRESH_MASKS_AUDITS_AND_ORACLES_CACHED_PREDICTIONS_NO_REFIT','date_utc':datetime.now(timezone.utc).isoformat(),
        'new_fitted_configurations':0,'fits_executed':0,'historical_fit_diagnostics_recomputed':False,
        'original_source_sha256':sha256((REPO/'support_experiment.py').read_bytes()).hexdigest(),
        'cache_sha256':sha256((HERE/'prediction_cache.json').read_bytes()).hexdigest(),
        'scope':'Fresh numerical/finite checks and all6400 report records. Prediction identity is checked; historical convergence/coefficient diagnostics are provenance, not fresh fit evidence.'}
    a.record.write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':main()
