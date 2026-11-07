"""Pre-registered local screen: one fast flow with computed slow history."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.extrapolated_flow import ExtrapolatedFlow,pack
from research.inner_dense import full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_ordered_cost import structural
from research.m3_priority import metrics
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-extrapolated-flow';OLD=ROOT/'results/m3-inner-dense'


def run():
    OUT.mkdir(exist_ok=True)
    sources=['research/m3_extrapolated_study.py','research/extrapolated_flow.py','research/test_extrapolated_flow.py',
             'research/history_polynomial.py','research/shared_radau.py','research/fullstep_radau.py','research/inner_dense.py',
             'research/m3_priority.py','research/m3_ordered_cost.py']
    reg=dict(date='2026-11-08',paths=list(range(6)),step=4,orders=[2,3,4],tolerances=[1e-6,1e-8,1e-10],
        initial_steps=['automatic','whole'],repeats=1,scope='One-run correctness/development screen, not formal cost evidence.',
        history='Previous slow RHS from actual MRI3 endpoints; current slow RHS computed in timer. No reference-history injection. Method-generated history still requires separate trajectory validation.',
        formula='P(s) interpolates q slow RHS samples at prior normalized times; solve vdot=f_fast(v)+P(t/H) once over H, inactive coordinates use exact integrated polynomial.',
        timing='All output/internal/pool/modified and original physical ledger guards; independent error outside timer. Cached previous slow samples and grouping supplied, no startup, estimator or audit charged.',
        baseline='Full Radau automatic and whole-interval initial attempt; all three tolerances and same guards.',
        source_sha256={p:digest(ROOT/p) for p in sources},
        input_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in [OLD/'history.json',ROOT/'results/m3-shared-radau/manifest.json']+sorted(OLD.glob('history-[0-5].npz'))})
    if (OUT/'registration.json').exists():
        assert json.loads((OUT/'registration.json').read_text())==reg
        if (OUT/'runs.json').exists():raise FileExistsError('Preserve existing evidence')
    else:save(OUT/'registration.json',reg)
    old=json.loads((OLD/'history.json').read_text())['runs'];rows=[]
    for index in reg['paths']:
        oldrow=next(r for r in old if r['shard']==f'history-{index}' and r['step']==4)
        src=np.load(OLD/f'history-{index}.npz');case=prepare(oldrow['model']);y=src['endpoints'][:,4];h=oldrow['h']
        kernel=ExtrapolatedFlow(case,src['groups']);screen=GroupScreen(case['system'],case['scale'])
        slow=np.column_stack([kernel.slow(z) for z in src['endpoints'][:,:4].T])
        arrays={'past_slow':slow};query=src['4-query'];ref=src['4-reference']
        specs=[('extrapolated',q,t,whole) for q in reg['orders'] for t in reg['tolerances'] for whole in (False,True)]
        specs += [('full',0,t,whole) for t in reg['tolerances'] for whole in (False,True)]
        for method,order,tol,whole in specs:
            key=f'{method}-{order}-{tol}-{whole}';tick=time.perf_counter()
            row=dict(key=key,path=index,model=oldrow['model'],h=h,threshold=oldrow['threshold'],method=method,
                     order=order,tolerance=tol,whole_step=whole,setup_seconds=kernel.setup_seconds if method=='extrapolated' else None)
            try:
                if method=='extrapolated':
                    trial=kernel.trial(y,h,src['endpoint_times'][5-order:5],slow[:,5-order:4],tol,whole)
                else:trial=(fullstep_curve if whole else full_curve)(case,y,h,tol,screen)
                curve=trial['pieces'][0];values=curve(query);stats=trial['stats'];guard=structural(case,values)
                seconds=time.perf_counter()-tick
                valid=stats['continuous_nonnegative'] and guard['nonnegative'] and max(stats['pool_error'],guard['pool'])<=1e-10
                valid &= all(stats.get(k,0.)<=1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature'))
                targets={}
                for target in (1e-4,1e-6):
                    actual=metrics(case,values,ref,target) if stats['continuous_nonnegative'] else None
                    targets[str(target)]=dict(actual=actual,passes=bool(valid and max(actual.values())<=1))
                row.update(status='success',stats=stats,guard=guard,targets=targets,seconds=seconds)
                arrays[key+'-endpoint']=trial['fine'];arrays[key+'-values']=values
                packed=pack(curve) if method=='extrapolated' else curve.pack()
                for k,v in packed.items():arrays[key+'-'+k]=v
                if method=='extrapolated':arrays[key+'-forcing']=trial['forcing_coefficients']
            except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                row.update(status='failed',reason=str(exc),stats=getattr(exc,'stats',{}),seconds=time.perf_counter()-tick)
            rows.append(row);save(OUT/'runs.json',dict(runs=rows))
        kernel.close();np.savez_compressed(OUT/f'path-{index}.npz',**arrays)
        print(index,'done',flush=True)


if __name__=='__main__':run()
