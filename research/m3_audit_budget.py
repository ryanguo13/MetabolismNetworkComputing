"""Prospective same-accuracy full ODE audit cost challenge, no audit removal."""
import argparse
import itertools
import json
import platform
import sys
import time
import zipfile
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.ordered_radau import OrderedRadau
from research.inner_dense import InnerCurve, full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_priority import full, metrics
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_ledger_flow_report import registration, close, check_curve
from research.m3_collocation_report import timeless
from research.m3_shadow_study import ROOT, digest

OUT = ROOT/'results/m3-audit-budget'; SOURCE = ROOT/'results/m3-remainder-curve'


def read(p): return json.loads(p.read_text())


def audit_pair(case, screen, y, q, h, method):
    """Two uninterrupted original ODE solves; no candidate state/history used."""
    tick = time.perf_counter(); arrays = []; work = []
    for tol in (1e-12, 1e-11):
        start = time.perf_counter()
        if method == 'current':
            values = full(case, y, h*q, tol)
            stats = dict(backend='CVODE' if case['system'].flux_jacobian is not None else 'Radau')
        else:
            sys = case['system']
            r = solve_ivp(lambda t,z:sys.S@sys.flux(0.,z), (0., h), y,
                t_eval=h*q, rtol=tol, atol=.01*tol*case['scale'], method=OrderedRadau,
                jac=lambda t,z:sys.S@screen.gradient(z),
                first_step=h if method=='ordered-whole' else None)
            if not r.success: raise RuntimeError(r.message)
            values = r.y; stats = dict(nfev=r.nfev, njev=r.njev, nlu=r.nlu)
        work.append(dict(tolerance=tol, seconds=time.perf_counter()-start, **stats))
        arrays.append(values)
    return arrays, dict(seconds=time.perf_counter()-tick, solves=work)


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists(): raise FileExistsError('Preserve evidence')
    reg = dict(date='2026-11-08', reference_keys=['0-3-1','0-2-1','1-3-1','1-2-1'],
        h=.01, methods=['current','ordered-auto','ordered-whole'], repeats=[-1,0,1,2],
        audit_tolerances=[1e-12,1e-11], control_tolerances=[1e-6,1e-7,1e-8,1e-10],
        hypothesis='At the newly qualified H=.01 and two earlier states, challenge the audit cost with same-accuracy optimized full ODE solves before integrating rational/reaction corrections. Prior optimized-Radau audit work is acknowledged; this extends to earlier states, paired references and fresh matched baseline challenge.',
        scope='48 original-ODE audit pairs (12 warmups),128 mature controls (32 warmups), rotating method order. All fixed configurations retained. Three exploratory repeats, not seven formal repeats or full-window M4 evidence.',
        independent='Every audit starts from the same saved prefix initial state, solves original nonlinear ODE on the same65/public queries without candidate predictor/history, and retains1e-12/1e-11 tightening. Compare both solutions to independently saved full CVODE truth; each<=.05 development strict units. No clipping/reference injection and no runtime audit removal.',
        timing='Preparation and GroupScreen setup separate. Each audit includes allocation/library binding as applicable, current J/solve/output; error scoring measured separately and added back. Primary feasibility uses audit-pair-only cost as an optimistic lower budget. Mature solves include continuous positivity/pools/physical ledgers and sample guards. Same optimized solver offered to mature controls. Setup excluded from both local ratios; no formal end-to-end claim.',
        decision='For each state and target, select fastest median qualified complete-system control and fastest median qualified full-ODE audit pair. If audit alone exceeds baseline/1.5, do not attach a new kernel and claim feasibility. This does not prove every possible audit implementation or full trajectory impossible.',
        source_sha256={p:digest(ROOT/p) for p in ('research/m3_audit_budget.py','research/m3_priority.py','metabolism/cvode.py','research/ordered_radau.py','research/fullstep_radau.py','research/inner_dense.py','research/m3_extrapolated_paths.py','research/m3_ordered_cost.py')},
        input_sha256={p:digest(ROOT/p) for p in ('results/m3-remainder-curve/manifest.json','results/m3-remainder-curve/runs.json','results/m3-ordered-cost/audit-registration.json','results/m3-ordered-cost/optimized-audit.json')})
    save(OUT/'registration.json', reg)
    rows, controls, setups = [], [], []
    def persist(): save(OUT/'runs.json', dict(audits=rows, controls=controls, setups=setups))
    refs = {r['key']:r for r in read(SOURCE/'runs.json')['references']}
    for key in reg['reference_keys']:
        rr = refs[key]; case = prepare(rr['model']); tick = time.perf_counter()
        screen = GroupScreen(case['system'], case['scale'])
        setups.append(dict(reference_key=key, seconds=time.perf_counter()-tick))
        with np.load(SOURCE/f'reference-{key}.npz') as a:
            y, q, truth = a['initial'], a['query'], a['reference']
        methods = reg['methods']; configs = list(itertools.product(reg['control_tolerances'], (False,True)))
        for rep in reg['repeats']:
            shift = (rep+1)%len(methods)
            for method in methods[shift:]+methods[:shift]:
                ck=f'audit-{key}-{rep}-{method}'; row=dict(key=ck,reference_key=key,model=rr['model'],state_index=rr['state_index'],method=method,repeat=rep)
                tick=time.perf_counter()
                try:
                    values, stats=audit_pair(case,screen,y,q,rr['h'],method)
                    scoring=time.perf_counter(); budgets=[metrics(case,v,truth,1e-6) for v in values]
                    tightening=metrics(case,values[1],values[0],1e-6)
                    good=all(np.isfinite(v).all() and np.all(v>=-1e-12*case['scale'][:,None]) for v in values)
                    good &= max([*tightening.values(),*[x for b in budgets for x in b.values()]])<=.05
                    scoreseconds=time.perf_counter()-scoring
                    row.update(status='success',stats=stats,budgets=budgets,tightening=tightening,qualified=bool(good),scoring_seconds=scoreseconds,total_seconds=stats['seconds']+scoreseconds)
                    np.savez_compressed(OUT/f'{ck}.npz',tight=values[0],loose=values[1])
                except (ValueError,RuntimeError,FloatingPointError) as exc:
                    row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),qualified=False)
                rows.append(row);persist()
            shift=(rep+1)%len(configs)
            for tol,whole in configs[shift:]+configs[:shift]:
                ck=f'control-{key}-{rep}-{tol}-{whole}';row=dict(key=ck,reference_key=key,model=rr['model'],state_index=rr['state_index'],tolerance=tol,whole=whole,repeat=rep);tick=time.perf_counter()
                try:
                    r=(fullstep_curve if whole else full_curve)(case,y,rr['h'],tol,screen);c=r['pieces'][0];values=c(q);structural(case,values)
                    elapsed=time.perf_counter()-tick;guard,targets=qualified(case,c,values,truth,r['stats'])
                    row.update(status='success',seconds=elapsed,stats=r['stats'],guard=guard,targets=targets)
                    np.savez_compressed(OUT/f'{ck}.npz',endpoint=r['fine'],**c.pack())
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                    row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc))
                controls.append(row);persist()
            print('measured',key,rep,flush=True)


def verify():
    reg=registration(OUT/'registration.json');data=read(OUT/'runs.json')
    assert len(data['audits'])==48 and len(data['controls'])==128
    refs={r['key']:r for r in read(SOURCE/'runs.json')['references']};summary=[]
    for key in reg['reference_keys']:
        rr=refs[key];case=prepare(rr['model'])
        with np.load(SOURCE/f'reference-{key}.npz') as a:y,q,truth=a['initial'],a['query'],a['reference']
        audits=[r for r in data['audits'] if r['reference_key']==key]
        controls=[r for r in data['controls'] if r['reference_key']==key]
        assert {(r['method'],r['repeat']) for r in audits}==set(itertools.product(reg['methods'],reg['repeats']))
        assert {(r['tolerance'],r['whole'],r['repeat']) for r in controls}==set(itertools.product(reg['control_tolerances'],(False,True),reg['repeats']))
        for row in audits:
            assert row['status']=='success'
            with np.load(OUT/f"{row['key']}.npz") as a:values=[a['tight'],a['loose']]
            budgets=[metrics(case,v,truth,1e-6) for v in values];tightening=metrics(case,values[1],values[0],1e-6)
            close(budgets,row['budgets']);close(tightening,row['tightening'])
            good=all(np.isfinite(v).all() and np.all(v>=-1e-12*case['scale'][:,None]) for v in values)
            good &= max([*tightening.values(),*[x for b in budgets for x in b.values()]])<=.05
            assert bool(good)==row['qualified']
            # Numerical determinism across warmup and timing repeats, including
            # work counters; timing itself cannot be replayed bit for bit.
            first=next(r for r in audits if r['method']==row['method'] and r['repeat']==-1)
            close(timeless(row['stats']),timeless(first['stats']))
            with np.load(OUT/f"{first['key']}.npz") as a:
                np.testing.assert_array_equal(values[0],a['tight']);np.testing.assert_array_equal(values[1],a['loose'])
        for row in controls:
            if row['status']=='failed':
                assert row['reason']=='Invalid full RHS state'
                continue
            with np.load(OUT/f"{row['key']}.npz") as a:
                c=InnerCurve.unpack(a['lengths'],a['coefficients']);end=a['endpoint']
            check_curve(case,c,y,end,truth,q,row['stats'],row['guard'],row['targets'])
        for target in ('0.0001','1e-06'):
            baseline=[]
            for tol,whole in itertools.product(reg['control_tolerances'],(False,True)):
                rows=[r for r in controls if r['tolerance']==tol and r['whole']==whole and r['repeat']>=0]
                good=all(r['status']=='success' and r['targets'][target]['passes'] for r in rows)
                baseline.append(dict(tolerance=tol,whole=whole,qualified=good,median_seconds=float(np.median([r['seconds'] for r in rows]))))
            best=min((r for r in baseline if r['qualified']),key=lambda r:r['median_seconds'])
            costs=[]
            for method in reg['methods']:
                rows=[r for r in audits if r['method']==method and r['repeat']>=0]
                vals=[r['stats']['seconds'] for r in rows]
                costs.append(dict(method=method,qualified=all(r['qualified'] for r in rows),median_seconds=float(np.median(vals)),min_seconds=min(vals),max_seconds=max(vals),median_with_scoring_seconds=float(np.median([r['total_seconds'] for r in rows]))))
            fastest=min((r for r in costs if r['qualified']),key=lambda r:r['median_seconds'])
            summary.append(dict(reference_key=key,model=rr['model'],state_index=rr['state_index'],h=rr['h'],target=target,baselines=baseline,best_baseline=best,audits=costs,best_audit=fastest,residual_1p5_budget=best['median_seconds']/1.5-fastest['median_seconds'],audit_only_speedup_ceiling=best['median_seconds']/fastest['median_seconds']))
        print('verified audit budget',key,flush=True)
    crc=0
    for path in OUT.glob('*.npz'):
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        crc+=1
    return dict(audit_pairs=48,mature_controls=128,audits_qualified=sum(r['qualified'] for r in data['audits']),mature_failures=sum(r['status']=='failed' for r in data['controls']),npz_crc_checked=crc,comparisons=summary,M4_admission=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','verify','write']);args=p.parse_args()
    if args.action=='run':run();return
    result=verify()
    if args.action=='write':
        if (OUT/'manifest.json').exists():raise FileExistsError('Preserve manifest')
        save(OUT/'summary.json',result)
        save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
        files=[p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']+[Path(__file__),ROOT/'docs/m3-audit-budget.md']
        save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in files})
    elif (OUT/'summary.json').exists():
        close(result,read(OUT/'summary.json'))
        for path,sha in read(OUT/'manifest.json').items():assert digest(ROOT/path)==sha
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    from pathlib import Path
    main()
