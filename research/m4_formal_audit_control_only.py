"""Formal M4 admission attempt for the frozen audit-control-only candidate.

Seven interleaved full-window repeats per model/target, plus one warmup per
method. The candidate is frozen to the preceding screening configuration.
"""
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np

from metabolism.adaptive import AdaptiveConfig, AdaptiveSplit
from metabolism.continuation import ContinuationConfig, MacroSplit
from metabolism.m3_continuation import ROOT, TARGETS, digest, prepare, metrics
from metabolism.splitting import SplitFailure

OUT = ROOT / 'results/m4-formal-audit-control-only'
REPEATS = 7
CANDIDATE = dict(audit_control_only=True, local_fraction=1.0, initial_step=1e-5,
    max_step={'teusink': .1, 'kecoli': .01}, max_attempts=512, max_rejections=16,
    timeout=180., group_every=1, audit_rtol=1e-11, fallback_rtol=1e-11)


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registration_path = OUT / 'registration.json'
    if registration_path.exists():
        raise FileExistsError('Preserve existing formal-attempt evidence')
    sources = ('research/m4_formal_audit_control_only.py', 'research/m3_full_window_screen.py',
        'metabolism/continuation.py', 'metabolism/adaptive.py', 'metabolism/splitting.py',
        'metabolism/m3_continuation.py', 'metabolism/m4_admission.py',
        'research/test_m4_admission.py', 'checks/test_continuation.py')
    registration = dict(date='2026-11-15', protocol='Frozen M4 protocol from docs/m4-admission-audit.md',
        purpose='Complete a formal admission attempt for the frozen audit-control-only/local_fraction=1 candidate despite its negative development screen; do not tune or relabel the candidate.',
        scope='Both original models x both original targets; original M1 initial conditions, full windows, public grids and scales; one warmup and seven interleaved candidate/baseline pairs per configuration.',
        candidate=CANDIDATE,
        control='Fastest historically qualified mature full-system configuration for the same model/target, re-run on the current machine with original fixed groups.',
        run_protocol=dict(formal_repeats=REPEATS, warmups_per_method=1, interleaved=True,
            preserve_every_failure=True, candidate_audit='Independent complete nonlinear ODE from each committed start at every candidate attempt, all public and quarter points',
            no_clipping=True, original_global_metrics=True, transactional_rollback=True,
            coverage_gate=.5, speedup_gate=1.5),
        source_sha256={p:digest(ROOT/p) for p in sources},
        environment=dict(python=platform.python_version(), numpy=np.__version__))
    save(registration_path, registration)
    runs_path = OUT / 'runs.json'
    save(runs_path, dict(runs=[]))
    runs=[]
    baseline_data=json.loads((ROOT/'results/m3-continuation/baselines.json').read_text())
    historical=baseline_data['runs']
    for name,target in itertools.product(('teusink','kecoli'),TARGETS):
        case=prepare(name)
        eligible=[]
        for identity in sorted({r['config'] for r in historical if r['model']==name and r['repeat']>=0}):
            subset=[r for r in historical if r['model']==name and r['config']==identity
                    and r['parameters']['target']==target and r['repeat']>=0]
            if len(subset)==7 and {r['repeat'] for r in subset}==set(range(7)) and all(r['metrics']['passes'] for r in subset):
                eligible.append(identity)
        if not eligible:
            raise RuntimeError(f'No qualified mature baseline for {name}/{target}')
        baseline_id=min(eligible,key=lambda key:np.median([r['seconds'] for r in historical
            if r['model']==name and r['config']==key and r['parameters']['target']==target and r['repeat']>=0]))
        baseline_row=next(r for r in historical if r['model']==name and r['config']==baseline_id
                          and r['parameters']['target']==target and r['repeat']==0)
        baseline_cfg=AdaptiveConfig(**baseline_row['parameters'])
        large=name=='kecoli'
        candidate_cfg=ContinuationConfig(target=target,local_fraction=1.0,initial_step=1e-5,
            max_step=.01 if large else .1,max_attempts=512,max_rejections=16,timeout=180.,
            group_every=1,audit_method='CVODE-KLU' if large else 'Radau',audit_rtol=1e-11,
            fallback_method='CVODE-KLU' if large else 'BDF',fallback_rtol=1e-11,
            audit_control_only=True)
        methods=('baseline','candidate')
        for method in methods:
            order=methods if method=='baseline' else methods[::-1]
            if method=='baseline':
                order=('baseline',)
            else:
                order=('candidate',)
            for which in order:
                row=dict(model=name,target=target,method=which,repeat=-1,
                    config=baseline_id if which=='baseline' else 'audit-control-only-local1-frozen',warmup=True)
                tick=time.perf_counter()
                try:
                    controller=(AdaptiveSplit(case['system'],case['scale'],baseline_cfg,
                        fixed_groups=np.arange(len(case['system'].groups)),**case['options']) if which=='baseline'
                        else MacroSplit(case['system'],case['scale'],candidate_cfg,**case['options']))
                    values,stats=controller.integrate(case['times'])
                    row.update(status='success',seconds=time.perf_counter()-tick,setup_seconds=case['setup_seconds'],
                               metrics=metrics(values,case,target,stats),stats=stats)
                    if which=='candidate':
                        row['split_coverage_recomputed']=sum(a['h'] for a in stats['attempts'] if a['status']=='accepted')/(case['times'][-1]-case['times'][0])
                        row['every_attempt_audited']=stats['full_calls']==len(stats['attempts'])
                except (SplitFailure,RuntimeError,ValueError,FloatingPointError,TimeoutError) as exc:
                    row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',None))
                runs.append(row); save(runs_path,dict(runs=runs))
                print(name,target,'warmup',which,row['status'],round(row['seconds'],3),flush=True)
        for repeat in range(REPEATS):
            order=('baseline','candidate') if repeat%2==0 else ('candidate','baseline')
            for which in order:
                row=dict(model=name,target=target,method=which,repeat=repeat,
                    config=baseline_id if which=='baseline' else 'audit-control-only-local1-frozen',warmup=False)
                tick=time.perf_counter()
                try:
                    controller=(AdaptiveSplit(case['system'],case['scale'],baseline_cfg,
                        fixed_groups=np.arange(len(case['system'].groups)),**case['options']) if which=='baseline'
                        else MacroSplit(case['system'],case['scale'],candidate_cfg,**case['options']))
                    values,stats=controller.integrate(case['times'])
                    row.update(status='success',seconds=time.perf_counter()-tick,setup_seconds=case['setup_seconds'],
                               metrics=metrics(values,case,target,stats),stats=stats)
                    if which=='candidate':
                        row['split_coverage_recomputed']=sum(a['h'] for a in stats['attempts'] if a['status']=='accepted')/(case['times'][-1]-case['times'][0])
                        row['every_attempt_audited']=stats['full_calls']==len(stats['attempts'])
                except (SplitFailure,RuntimeError,ValueError,FloatingPointError,TimeoutError) as exc:
                    row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',None))
                runs.append(row); save(runs_path,dict(runs=runs))
                print(name,target,'repeat',repeat,which,row['status'],round(row['seconds'],3),flush=True)
    save(OUT/'manifest.json',dict(registration_sha256=digest(registration_path),runs_sha256=digest(runs_path),source_sha256=registration['source_sha256']))
    print('formal repeat acquisition complete',flush=True)


if __name__=='__main__':
    main()
