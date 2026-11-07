"""Registered development ablation of stage Jacobian and exact LU sharing."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.history_polynomial import HistoryPolynomial
from research.mri_endpoint import MRIEndpoint
from research.shared_radau import SharedMRIEndpoint, shared_full_curve
from research.inner_dense import InnerCurve, physical_ledger, full_curve
from research.m3_ordered_cost import structural
from research.m3_priority import metrics
from research.m3_shadow_study import ROOT, digest

OUT = ROOT/'results/m3-shared-radau'
OLD = ROOT/'results/m3-inner-dense'
METHODS = ['endpoint', 'jacobian', 'shared', 'full', 'full_shared']


def run():
    OUT.mkdir(exist_ok=True)
    sources = ['research/m3_shared_cost.py', 'research/shared_radau.py', 'research/test_shared_radau.py',
               'research/mri_endpoint.py', 'research/ordered_radau.py', 'research/inner_dense.py',
               'research/history_polynomial.py', 'research/m3_ordered_cost.py', 'research/m3_priority.py']
    reg = dict(date='2026-11-08', paths=list(range(6)), step=4, degree=5,
               tolerances=[1e-8,1e-10], methods=METHODS, repeats=3, warmups=1,
               scope='Known diagnostic states, free actual history, cached setup; development only, not full-window admission.',
               hypothesis='Stage Jacobian seeds plus exact-matrix LU reuse reduce actual stage work without changing Radau convergence/error checks.',
               invalidation='Fresh factory every trial; fixed fast-variable ordering; borrowed Jacobian is stale and refreshable; LU requires exact dtype/shape/entries/CSC indices.',
               controls='Original endpoint, Jacobian-only, shared Jacobian+LU; original full Radau and full with identical LU cache. No step-size heuristic changes.',
               timing='Endpoint/internal guards + quintic history + continuous positivity/pool/physical ledger for candidates; matching guards and native dense output for full. Independent accuracy assessment outside both timers.',
               exclusions='Mandatory full audit, estimator, startup, grouping and cached setup excluded. No acceptance or reduced audit authorized.',
               stopping='Only consider whole-window engineering after qualified local speedup >=1.5 and a separate audit/estimator cost case. All failures retained.',
               source_sha256={p:digest(ROOT/p) for p in sources},
               input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in
                   [OLD/'history.json', ROOT/'results/m3-history-degree/manifest.json']+sorted(OLD.glob('history-[0-5].npz'))})
    registration = OUT/'registration.json'
    if registration.exists():
        assert json.loads(registration.read_text()) == reg
        if (OUT/'runs.json').exists():
            raise FileExistsError('Preserve completed/partial evidence; run in an isolated checkout')
    else:
        save(registration, reg)
    old = json.loads((OLD/'history.json').read_text())['runs']
    rows = []
    for index in reg['paths']:
        oldrow = next(r for r in old if r['shard']==f'history-{index}' and r['step']==4)
        saved = np.load(OLD/f'history-{index}.npz')
        case = prepare(oldrow['model'])
        y,h = saved['endpoints'][:,4],oldrow['h']
        query,ref = saved['4-query'],saved['4-reference']
        screen = GroupScreen(case['system'],case['scale'])
        original = MRIEndpoint(case,saved['groups'])
        shared = SharedMRIEndpoint(case,saved['groups'])
        arrays = {}
        times,previous = saved['endpoint_times'][:6],saved['endpoints'][:,:5]
        for repeat in (-1,0,1,2):
            offset = (repeat+1+index)%len(METHODS)
            order = METHODS[offset:]+METHODS[:offset]
            for method in order:
                tolerances = reg['tolerances'] if repeat%2==0 else reg['tolerances'][::-1]
                for tol in tolerances:
                    key = f'{method}-{tol}-{repeat}'
                    row = dict(key=key, source_shard=f'history-{index}', model=oldrow['model'], h=h,
                               threshold=oldrow['threshold'], method=method, tolerance=tol, repeat=repeat,
                               setup_seconds=shared.setup_seconds if method in ('jacobian','shared') else original.setup_seconds)
                    tick = time.perf_counter()
                    try:
                        if method not in ('full','full_shared'):
                            if method == 'endpoint':
                                trial = original.trial(y,h,tol,timeout=20.)
                            else:
                                trial = shared.trial(y,h,tol,timeout=20.,share_lu=method=='shared')
                            stats = trial['stats']
                            poly = HistoryPolynomial(times,np.column_stack([previous,trial['fine']]))
                            curve = InnerCurve([poly])
                            stats.update(physical_ledger(case,curve,y,trial['fine']))
                            stats['continuous_nonnegative'] = bool(np.all(poly.minima()>=-1e-12*case['scale']))
                        else:
                            fn = full_curve if method=='full' else shared_full_curve
                            trial = fn(case,y,h,tol,screen)
                            curve,stats = trial['pieces'][0],trial['stats']
                        values = curve(query)
                        guard = structural(case,values)
                        seconds = time.perf_counter()-tick
                        valid = stats['continuous_nonnegative'] and guard['nonnegative'] and max(stats['pool_error'],guard['pool'])<=1e-10
                        valid &= all(stats.get(k,0.)<=1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature'))
                        targets = {}
                        for target in (1e-4,1e-6):
                            actual = metrics(case,values,ref,target) if stats['continuous_nonnegative'] else None
                            targets[str(target)] = dict(actual=actual,passes=bool(valid and max(actual.values())<=1))
                        row.update(status='success',seconds=seconds,stats=stats,guard=guard,targets=targets)
                        arrays[key+'-values'] = values
                        arrays[key+'-endpoint'] = trial['fine']
                        if method not in ('full','full_shared'):
                            arrays[key+'-coefficients'] = poly.coefficients
                        else:
                            for k,v in curve.pack().items():
                                arrays[key+'-'+k] = v
                    except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                        row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
                    rows.append(row)
                    save(OUT/'runs.json',dict(runs=rows,source_sha256=reg['source_sha256']))
            print(index, repeat, 'done', flush=True)
        original.close(); shared.close()
        tmp = OUT/f'cost-{index}.tmp.npz'
        np.savez_compressed(tmp,**arrays)
        with np.load(tmp) as check:
            for key in check.files:
                check[key]
        tmp.replace(OUT/f'cost-{index}.npz')


if __name__ == '__main__': run()
