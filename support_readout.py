"""Original postrun tables and figures for the frozen support experiment.

No input records or fitted models are loaded. Quantiles describe the 200 design
seeds; they are not confidence intervals. Requires separately acquired results.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from support_reference import validate_result_file

if sys.flags.optimize:
    raise SystemExit('Readout requires assertions enabled; do not use optimized Python')

ARMS = ['reuse', 'separate', 'reuse_budget', 'separate_budget']
ARM_LABELS = ['Reuse', 'Separate', 'Reuse, matched', 'Separate, matched']
NORMALIZATIONS = ['unicode_whitespace', 'nfc_casefold']
SELECTIONS = ['cutoff_half', 'cutoff_fifth', 'top_fifth', 'all']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--author-metadata', default='', help='Optional author for a separate named figure build; anonymous by default')
    parser.add_argument('--require-archived-bytes', action='store_true', help='Require the original reference file bytes as well as scientific equality')
    args = parser.parse_args()
    data, reference_check = validate_result_file(args.results, args.require_archived_bytes)
    actual_results_sha = sha(args.results)
    rows = data['records']
    assert len(rows) == 6400
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for s in data['summary']['conditions']:
        subset = [r for r in rows if all(r[k] == s[k] for k in ('normalization','selection','arm'))]
        assert len(subset) == 200
        assert all(r['audit']['status'] == 'DEFINED' for r in subset)
        summaries.append({
            'normalization': s['normalization'], 'selection': s['selection'], 'arm': s['arm'],
            'runs': len(subset), 'mean_visible_n': s['observed_n']['mean'],
            'mean_fixed_visible_n': s['fixed_visible_n']['mean'],
            'mean_absolute_difference_error_pp': 100*s['primary_difference_absolute_error']['mean'],
            'mean_signed_difference_error_pp': 100*s['primary_difference_error']['mean'],
            'primary_observed_ties': sum(r['audit']['pairwise'][0]['known_difference_count'] == 0 for r in subset),
            'primary_reversals': s['strict_reversals'],
            'winner_excludes_optimum': s['winner_set_excludes_optimal'],
            'mean_worst_regret_pp': 100*s['worst_regret']['mean'],
            'mean_best_regret_pp': 100*s['best_regret']['mean'],
            'mean_disagreement_envelope_width_pp': 100*s['pair_interval_width']['mean'],
            'mean_missing_mass_envelope_width_pp': sum(200*r['audit']['missing_n']/r['audit']['target_n'] for r in subset)/200,
            'primary_certificate_counts': dict(Counter(r['audit']['pairwise'][0]['certificate'] for r in subset)),
            'observed_winner_sets': [{'members': list(k), 'count': v} for k,v in sorted(Counter(tuple(r['audit']['observed_winners']) for r in subset).items())]})
    result = {'status': 'POSTRUN_AGGREGATE_READOUT', 'source_sha256': actual_results_sha,
              'reference_check': reference_check,
              'scope': 'Predefined metrics, with ties/certificate readout added after outcomes. No refit or new selection.',
              'models': data['models'], 'conditions': summaries,
              'nonambiguous_primary_certificates': [
                  {k:r[k] for k in ('normalization','selection','arm','seed')} |
                  {'target_n':r['audit']['target_n'],'observed_n':r['audit']['observed_n'],
                   'pair':r['audit']['pairwise'][0]}
                  for r in rows if r['audit']['pairwise'][0]['certificate'] != 'BOTH_STRICT_ORDERINGS_FEASIBLE']}
    assert len(result['nonambiguous_primary_certificates']) == 2
    (output/'readout.json').write_text(json.dumps(result,indent=2)+'\n')
    artifacts = [output/'readout.json']
    if args.figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                             'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
        def save(fig, name):
            for suffix in ('pdf','png','jpg','svg'):
                target = output/(name+'.'+suffix)
                kw = {'metadata':{'Title':name.replace('_',' '),'Author':args.author_metadata}} if suffix=='pdf' else {}
                fig.savefig(target,dpi=300,facecolor='white',bbox_inches='tight',**kw)
                artifacts.append(target)
            plt.close(fig)
        colors=['#286A9B','#C56522','#286A9B','#C56522']
        fig, axes=plt.subplots(1,2,figsize=(7.0,3.8),sharex=False)
        for ax,norm,title in zip(axes,NORMALIZATIONS,['Whitespace normalization','NFC/casefold normalization']):
            values=[[100*abs(r['oracle']['primary_difference_estimation_error']) for r in rows
                     if r['normalization']==norm and r['selection']=='cutoff_half' and r['arm']==arm] for arm in ARMS]
            bp=ax.boxplot(values,vert=False,patch_artist=True,showfliers=True,widths=.55,
                          medianprops={'color':'black','linewidth':1.2},
                          flierprops={'marker':'.','markersize':2.5,'alpha':.45})
            for box,c in zip(bp['boxes'],colors): box.set_facecolor(c);box.set_alpha(.45)
            ax.set_yticks(range(1,5),ARM_LABELS);ax.invert_yaxis()
            ax.set_title(title,fontsize=10)
            ax.set_xlabel('Absolute error\n(percentage points)')
            ax.grid(axis='x',alpha=.2)
        fig.tight_layout(w_pad=2)
        save(fig,'primary_difference_error')
        fig,axes=plt.subplots(1,2,figsize=(7.0,3.8),sharex=True)
        select_labels=['Cutoff 1/2','Cutoff 1/5','Top fifth','All units']
        for ax,norm,title in zip(axes,NORMALIZATIONS,['Whitespace normalization','NFC/casefold normalization']):
            for j,selection in enumerate(SELECTIONS):
                matches=[s for s in data['summary']['paired_contrasts'] if s['normalization']==norm and s['selection']==selection]
                assert len(matches)==2
                for i,(s,c) in enumerate(zip(matches,['#286A9B','#C56522'])):
                    v=s['summary'];mean=100*v['mean'];lo=100*v['p05'];hi=100*v['p95'];y=j+(i-.5)*.22
                    ax.hlines(y,lo,hi,color=c,lw=1.5)
                    ax.plot(mean,y,'o',color=c,ms=4,label=('Natural masks' if i==0 else 'Matched counts') if j==0 else None)
            ax.axvline(0,color='0.5',lw=.8,ls='--');ax.set_yticks(range(4),select_labels);ax.invert_yaxis()
            ax.set_title(title,fontsize=10);ax.set_xlabel('Paired absolute-error change\nSeparate − reuse (percentage points)')
            ax.grid(axis='x',alpha=.2)
        handles, legend_labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,legend_labels,fontsize=8,loc='lower center',ncol=2,frameon=False)
        fig.tight_layout(w_pad=2,rect=(0,.09,1,1))
        save(fig,'paired_error_changes')
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'GENERATED_VISUAL_QA_PENDING' if args.figures else 'PASS',
              'source_results_sha256':actual_results_sha,'reference_check':reference_check,
              'anonymous_metadata':not bool(args.author_metadata),'generator_sha256':sha(Path(__file__)),
              'artifacts':[{'name':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in artifacts]}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'conditions':len(summaries),'artifacts':len(artifacts)}))


if __name__ == '__main__':
    main()
