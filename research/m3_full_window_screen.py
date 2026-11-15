"""One-shot full-window falsification screen for audit-controlled splitting.

Run with: bash scripts/run-macos.sh -m research.m3_full_window_screen
Results are write-once and are development evidence, not formal M4 admission.
"""
import itertools
import json
import platform
import time
from dataclasses import replace

import numpy as np

from metabolism.adaptive import AdaptiveConfig, AdaptiveSplit
from metabolism.continuation import ContinuationConfig, MacroSplit
from metabolism.m3_continuation import ROOT, TARGETS, digest, prepare, metrics
from metabolism.splitting import SplitFailure

OUT = ROOT / 'results/m3-full-window-screen'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reg_path = OUT / 'registration.json'
    if reg_path.exists():
        raise FileExistsError('Preserve existing evidence')
    source_files = ('research/m3_full_window_screen.py', 'metabolism/continuation.py',
        'metabolism/adaptive.py', 'metabolism/splitting.py', 'metabolism/m3_continuation.py')
    registration = dict(date='2026-11-15',
        hypothesis='The locally promising audit-only controller with local_fraction=1.0 may achieve globally acceptable error and useful actual split coverage on the frozen full windows.',
        scope='One candidate and one same-machine full-system control per model x target; original M1 initial states, full windows, public grids and scales. Development screen only; no warmup/seven-repeat formal claim.',
        candidate=dict(audit_control_only=True, local_fraction=1.0, initial_step=1e-5,
            max_step={'teusink': .1, 'kecoli': .01}, audit_rtol=1e-11,
            fallback_rtol=1e-11, max_attempts=512, max_rejections=16, timeout=180.),
        gates=dict(full_ode_audit_every_attempt=True, audit_all_public_and_quarter_points=True,
            original_global_accuracy_scored=True, pools_and_open_ledger=True,
            no_clipping=True, rejected_trial_rollback=True, real_split_coverage_recomputed=True,
            full_cost_includes_setup_classification_split_audit_fallback_output=True),
        source_sha256={p:digest(ROOT/p) for p in source_files},
        environment=dict(python=platform.python_version(), numpy=np.__version__))
    reg_path.write_text(json.dumps(registration, indent=2)+'\n')
    runs=[]
    (OUT/'runs.json').write_text(json.dumps(dict(runs=runs),indent=2)+'\n')
    for name in ('teusink','kecoli'):
        case=prepare(name)
        # Full-system controls use the fastest historically qualified same-machine configuration.
        baseline_data=json.loads((ROOT/'results/m3-continuation/baselines.json').read_text())
        historical=[r for r in baseline_data['runs'] if r['model']==name and r['repeat']>=0 and r['status']=='success']
        for target in TARGETS:
            qualified=[]
            for identity in sorted({r['config'] for r in historical}):
                subset=[r for r in historical if r['config']==identity and r['parameters']['target']==target]
                if len(subset)==7 and all(r['metrics']['passes'] for r in subset):
                    qualified.append(identity)
            if not qualified:
                raise RuntimeError(f'No historically qualified baseline for {name} target={target}')
            identity=min(qualified,key=lambda key: np.median([r['seconds'] for r in historical if r['config']==key and r['parameters']['target']==target]))
            base_row=next(r for r in historical if r['config']==identity and r['parameters']['target']==target)
            # Re-run the selected full-system solver on this machine under the same wrapper.
            baseline_cfg=AdaptiveConfig(**base_row['parameters'])
            for mode in ('baseline','candidate'):
                row=dict(model=name,target=target,mode=mode,repeat=0,baseline_config=identity)
                tick=time.perf_counter()
                try:
                    if mode=='baseline':
                        controller=AdaptiveSplit(case['system'],case['scale'],baseline_cfg,
                            fixed_groups=np.arange(len(case['system'].groups)),**case['options'])
                    else:
                        large=name=='kecoli'
                        cfg=ContinuationConfig(target=target,local_fraction=1.0,initial_step=1e-5,
                            max_step=.01 if large else .1,max_attempts=512,max_rejections=16,
                            timeout=180.,group_every=1,audit_method='CVODE-KLU' if large else 'Radau',
                            audit_rtol=1e-11,fallback_method='CVODE-KLU' if large else 'BDF',
                            fallback_rtol=1e-11,audit_control_only=True)
                        controller=MacroSplit(case['system'],case['scale'],cfg,**case['options'])
                    values,stats=controller.integrate(case['times'])
                    seconds=time.perf_counter()-tick
                    row.update(status='success',seconds=seconds,setup_seconds=case['setup_seconds'],
                               metrics=metrics(values,case,target,stats),stats=stats)
                    if mode=='candidate':
                        row['split_coverage_recomputed']=sum(a['h'] for a in stats['attempts'] if a['status']=='accepted')/(case['times'][-1]-case['times'][0])
                        row['every_attempt_audited']=stats['full_calls']==len(stats['attempts'])
                except (SplitFailure,RuntimeError,ValueError,FloatingPointError,TimeoutError) as exc:
                    seconds=time.perf_counter()-tick
                    row.update(status='failed',seconds=seconds,reason=str(exc),stats=getattr(exc,'stats',None))
                runs.append(row)
                (OUT/'runs.json').write_text(json.dumps(dict(runs=runs),indent=2)+'\n')
                print(name,target,mode,row['status'],round(row['seconds'],3),
                      (row.get('stats') or {}).get('split_coverage'),row.get('metrics'),flush=True)
    summary=[]
    for name,target in itertools.product(('teusink','kecoli'),TARGETS):
        items=[r for r in runs if r['model']==name and r['target']==target]
        cand=next((r for r in items if r['mode']=='candidate'),None)
        base=next((r for r in items if r['mode']=='baseline'),None)
        summary.append(dict(model=name,target=target,candidate_status=cand['status'] if cand else 'missing',
            baseline_status=base['status'] if base else 'missing',
            candidate_seconds=cand.get('seconds') if cand else None,
            baseline_seconds=base.get('seconds') if base else None,
            speedup=base['seconds']/cand['seconds'] if cand and base and cand['status']=='success' and base['status']=='success' else None,
            accuracy_pass=cand.get('metrics',{}).get('passes',False) if cand else False,
            split_coverage=cand.get('split_coverage_recomputed',0.) if cand else 0.,
            audit_all_attempts=cand.get('every_attempt_audited',False) if cand else False,
            split_coverage_pass=bool(cand and cand.get('split_coverage_recomputed',0.)>=.5)))
    result=dict(summary=summary,M4_admission=False,
        interpretation='One full-window development screen per configuration only. Formal admission still requires seven interleaved repeats, at least 50% real split coverage, all accuracy/structure gates and at least 1.5x median speedup in all four combinations.',
        source_sha256=registration['source_sha256'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'README.md').write_text('# Full-window candidate screen\n\nOne development candidate/control run per frozen model/target combination. Not formal admission. See registration.json, runs.json and summary.json.\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
