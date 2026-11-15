"""Prospective tolerance calibration of independent nonlinear ODE audits.

The .05 external budget and original M4 gates are fixed. This is a local
cost screen; single runtime audit and offline tightening have distinct roles.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
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

OUT=ROOT/'results/m3-audit-calibration'
SOURCE=ROOT/'results/m3-remainder-curve'
OLD=ROOT/'results/m3-audit-budget'


def read(path):return json.loads(path.read_text())
def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def independent_solve(case,screen,y,q,h,tol,method):
    tick=time.perf_counter()
    if method=='current':
        values=full(case,y,h*q,tol)
        stats=dict(backend='CVODE' if case['system'].flux_jacobian is not None else 'Radau')
    else:
        system=case['system']
        r=solve_ivp(lambda t,z:system.S@system.flux(0.,z),(0.,h),y,
            t_eval=h*q,rtol=tol,atol=.01*tol*case['scale'],method=OrderedRadau,
            jac=lambda t,z:system.S@screen.gradient(z),
            first_step=h if method=='ordered-whole' else None)
        if not r.success:raise RuntimeError(r.message)
        values=r.y;stats=dict(nfev=r.nfev,njev=r.njev,nlu=r.nlu)
    stats['seconds']=time.perf_counter()-tick
    return values,stats


def certified(case,values,truth):
    budgets=[metrics(case,v,truth,1e-6) for v in values]
    tightening=metrics(case,values[0],values[1],1e-6)
    guards=[structural(case,v) for v in values]
    nonnegative=all(g['nonnegative'] and g['pool']<=1e-10 for g in guards)
    good=nonnegative and max([*tightening.values(),*[x for b in budgets for x in b.values()]])<=.05
    return dict(budgets=budgets,tightening=tightening,guards=guards,qualified=bool(good))


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    sources=['research/m3_audit_calibration.py','research/m3_audit_budget.py','research/m3_priority.py',
        'metabolism/cvode.py','research/ordered_radau.py','research/inner_dense.py',
        'research/fullstep_radau.py','research/m3_extrapolated_paths.py','research/m3_ordered_cost.py']
    inputs=[SOURCE/'manifest.json',SOURCE/'runs.json',OLD/'manifest.json',OLD/'summary.json']
    inputs += [SOURCE/f'reference-{k}.npz' for k in ['0-3-1','0-2-1','1-3-1','1-2-1']]
    reg=dict(date='2026-11-15',reference_keys=['0-3-1','0-2-1','1-3-1','1-2-1'],
        methods=['current','ordered-auto','ordered-whole'],loose_tolerances=[1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10],
        tightening_factor=.1,repeats=[-1,0,1,2],h=.01,
        control_tolerances=[1e-6,1e-7,1e-8,1e-10],external_reference_budget=.05,
        hypothesis='Existing 1e-12/1e-11 development reference pairs may over-solve; existing single runtime full ODE audit costs must be assessed separately. Calibrate a prespecified tolerance ladder against independently saved CVODE truth, without changing external concentration/flux/net-flux budgets or replacing original nonlinear ODE.',
        scope='336 paired audit attempts and128 matched complete controls, including84/32 warmups. Same known development states, not unseen validation/full windows/seven formal repeats. No PE or split coverage.',
        independent='Each pair starts at saved prefix state, with no candidate/history/reference injection; two original nonlinear full ODE solves, tenfold tightening. Both members and pair discrepancy must independently meet .05 strict development units. Sample finite/nonnegative and pool gates retained. Audit references are sampled ODE truth checks, not candidate dense-output certification.',
        timing='Single-thread BLAS/OpenMP, coordinated exclusive numerical work. Audit solve and external qualification scoring separate; pair scoring against each other separately timed and included in practical pair cost. Controls freshly timed with same optimized backend and unchanged continuous structural/physical-ledger checks. Preparation/setup and offline truth comparison excluded from optimistic solve-only budget. Repeated arrays stored once with exact hashes.',
        decision='Choose only configurations whose three measured repeats are certified. Positive baseline/1.5 minus paired solve cost is necessary, not sufficient, for reaction splitting. Also report paired solve+discrepancy scoring budget. Existing MacroSplit runtime uses one full ODE audit per candidate. Report single-member certified solve cost as its optimistic solve-only budget; candidate comparison and controller costs remain additional. Paired solve costs are offline tightening calibration costs, unless a future runtime strategy actually invokes both. Calibration at four known states does not certify general tolerance selection.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs})
    save(OUT/'registration.json',reg)
    rows=[];controls=[];setups=[]
    def persist():save(OUT/'runs.json',dict(audits=rows,controls=controls,setups=setups))
    refs={r['key']:r for r in read(SOURCE/'runs.json')['references']}
    for key in reg['reference_keys']:
        rr=refs[key];tick=time.perf_counter();case=prepare(rr['model']);screen=GroupScreen(case['system'],case['scale'])
        setups.append(dict(reference_key=key,seconds=time.perf_counter()-tick))
        with np.load(SOURCE/f'reference-{key}.npz') as a:y,q,truth=a['initial'],a['query'],a['reference']
        configs=list(itertools.product(reg['methods'],range(len(reg['loose_tolerances']))))
        baselineconfigs=list(itertools.product(reg['control_tolerances'],(False,True)))
        for rep in reg['repeats']:
            shift=(rep+1)%len(configs)
            for method,ti in configs[shift:]+configs[:shift]:
                tol=reg['loose_tolerances'][ti];ck=f'audit-{key}-{method}-{ti}'
                row=dict(key=ck,reference_key=key,model=rr['model'],state_index=rr['state_index'],method=method,tolerance=tol,repeat=rep)
                tick=time.perf_counter()
                try:
                    values=[];work=[]
                    for t in (tol,tol*reg['tightening_factor']):
                        v,stats=independent_solve(case,screen,y,q,rr['h'],t,method);values.append(v);work.append(stats)
                    # Runtime pair discrepancy is distinct from offline certification
                    # against immutable truth; retain its actual cost explicitly.
                    scoretick=time.perf_counter();discrepancy=metrics(case,values[0],values[1],1e-6)
                    discrepancy_seconds=time.perf_counter()-scoretick
                    scoretick=time.perf_counter();qualification=certified(case,values,truth);scoring_seconds=time.perf_counter()-scoretick
                    seconds=sum(w['seconds'] for w in work)
                    row.update(status='success',solves=work,seconds=seconds,discrepancy_seconds=discrepancy_seconds,
                        practical_seconds=seconds+discrepancy_seconds,qualification_scoring_seconds=scoring_seconds,
                        value_hashes=[array_hash(v) for v in values],**qualification)
                    close(discrepancy,row['tightening'])
                    if rep==-1:np.savez_compressed(OUT/f'{ck}.npz',loose=values[0],tight=values[1])
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                    row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),qualified=False)
                rows.append(row);persist()
            shift=(rep+1)%len(baselineconfigs)
            for tol,whole in baselineconfigs[shift:]+baselineconfigs[:shift]:
                ck=f'control-{key}-{tol}-{whole}';row=dict(key=ck,reference_key=key,model=rr['model'],state_index=rr['state_index'],tolerance=tol,whole=whole,repeat=rep);tick=time.perf_counter()
                try:
                    r=(fullstep_curve if whole else full_curve)(case,y,rr['h'],tol,screen);c=r['pieces'][0];values=c(q);structural(case,values)
                    seconds=time.perf_counter()-tick;guard,targets=qualified(case,c,values,truth,r['stats'])
                    row.update(status='success',seconds=seconds,stats=r['stats'],guard=guard,targets=targets,
                        curve_hash=array_hash(c.pack()['coefficients']))
                    if rep==-1:np.savez_compressed(OUT/f'{ck}.npz',endpoint=r['fine'],**c.pack())
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                    row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc))
                controls.append(row);persist()
            print('calibrated',key,rep,flush=True)


def verify():
    reg=registration(OUT/'registration.json');data=read(OUT/'runs.json')
    assert len(data['audits'])==336 and len(data['controls'])==128
    refs={r['key']:r for r in read(SOURCE/'runs.json')['references']};summary=[]
    for key in reg['reference_keys']:
        rr=refs[key];case=prepare(rr['model'])
        with np.load(SOURCE/f'reference-{key}.npz') as a:y,q,truth=a['initial'],a['query'],a['reference']
        audits=[r for r in data['audits'] if r['reference_key']==key];controls=[r for r in data['controls'] if r['reference_key']==key]
        assert {(r['method'],r['tolerance'],r['repeat']) for r in audits}==set(itertools.product(reg['methods'],reg['loose_tolerances'],reg['repeats']))
        assert {(r['tolerance'],r['whole'],r['repeat']) for r in controls}==set(itertools.product(reg['control_tolerances'],(False,True),reg['repeats']))
        for row in audits:
            first=next(r for r in audits if r['key']==row['key'] and r['repeat']==-1)
            assert row['status']==first['status']
            if row['status']=='failed':assert not row['qualified'] and row['reason']==first['reason'];continue
            with np.load(OUT/f"{row['key']}.npz") as a:values=[a['loose'],a['tight']]
            close(certified(case,values,truth),{k:row[k] for k in ('budgets','tightening','guards','qualified')})
            assert [array_hash(v) for v in values]==row['value_hashes']
            close(timeless(row['solves']),timeless(first['solves']))
            close(row['seconds'],sum(v['seconds'] for v in row['solves']))
            close(row['practical_seconds'],row['seconds']+row['discrepancy_seconds'])
        for row in controls:
            first=next(r for r in controls if r['key']==row['key'] and r['repeat']==-1)
            assert row['status']==first['status']
            if row['status']=='failed':assert row['reason']==first['reason'];continue
            with np.load(OUT/f"{row['key']}.npz") as a:c=InnerCurve.unpack(a['lengths'],a['coefficients']);end=a['endpoint']
            assert row['curve_hash']==array_hash(c.pack()['coefficients'])
            close(timeless(row['stats']),timeless(first['stats']))
            check_curve(case,c,y,end,truth,q,row['stats'],row['guard'],row['targets'])
        configs=[];singles=[]
        for method,tol in itertools.product(reg['methods'],reg['loose_tolerances']):
            rows=[r for r in audits if r['method']==method and r['tolerance']==tol and r['repeat']>=0]
            good=all(r['qualified'] for r in rows)
            configs.append(dict(method=method,tolerance=tol,qualified=good,median_seconds=float(np.median([r['seconds'] for r in rows])),
                median_practical_seconds=float(np.median([r.get('practical_seconds',r['seconds']) for r in rows])),
                min_seconds=min(r['seconds'] for r in rows),max_seconds=max(r['seconds'] for r in rows),
                max_budget=max((max([*r['tightening'].values(),*[v for b in r['budgets'] for v in b.values()]]) for r in rows if r['status']=='success'),default=None)))
            for member in (0,1):
                good=all(r['status']=='success' and max(r['budgets'][member].values())<=.05 and r['guards'][member]['nonnegative'] and r['guards'][member]['pool']<=1e-10 for r in rows)
                if good:singles.append(dict(method=method,tolerance=tol*(reg['tightening_factor'] if member else 1),median_seconds=float(np.median([r['solves'][member]['seconds'] for r in rows]))))
        qualified_pairs=[r for r in configs if r['qualified']]
        best_pair=min(qualified_pairs,key=lambda r:r['median_seconds']) if qualified_pairs else None
        best_practical=min(qualified_pairs,key=lambda r:r['median_practical_seconds']) if qualified_pairs else None
        best_single=min(singles,key=lambda r:r['median_seconds']) if singles else None
        for target in ('0.0001','1e-06'):
            baselines=[]
            for tol,whole in itertools.product(reg['control_tolerances'],(False,True)):
                rows=[r for r in controls if r['tolerance']==tol and r['whole']==whole and r['repeat']>=0]
                baselines.append(dict(tolerance=tol,whole=whole,qualified=all(r['status']=='success' and r['targets'][target]['passes'] for r in rows),median_seconds=float(np.median([r['seconds'] for r in rows]))))
            best=min((r for r in baselines if r['qualified']),key=lambda r:r['median_seconds'])
            summary.append(dict(reference_key=key,model=rr['model'],state_index=rr['state_index'],target=target,baselines=baselines,best_baseline=best,
                audits=configs,best_audit=best_pair,best_practical_audit=best_practical,best_single_diagnostic=best_single,
                residual_1p5_budget=None if best_pair is None else best['median_seconds']/1.5-best_pair['median_seconds'],
                practical_residual_1p5_budget=None if best_practical is None else best['median_seconds']/1.5-best_practical['median_practical_seconds'],
                single_only_residual_diagnostic=None if best_single is None else best['median_seconds']/1.5-best_single['median_seconds']))
        print('verified calibration',key,flush=True)
    crc=0
    for path in OUT.glob('*.npz'):
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        crc+=1
    return dict(date=reg['date'],audit_pairs=len(data['audits']),mature_controls=len(data['controls']),audit_pairs_qualified=sum(r['qualified'] for r in data['audits']),
        audit_failures=sum(r['status']=='failed' for r in data['audits']),mature_failures=sum(r['status']=='failed' for r in data['controls']),
        npz_crc_checked=crc,comparisons=summary,M4_admission=False,
        limitation='Known-state local calibration only; no general online reference-tolerance guarantee, no original full-window qualification, no reaction split coverage or formal seven-repeat speed evidence.')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','verify','write']);args=p.parse_args()
    if args.action=='run':run();return
    result=verify()
    if args.action=='write':
        if (OUT/'manifest.json').exists():raise FileExistsError('Preserve manifest')
        save(OUT/'summary.json',result)
        save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP; coordinated exclusive numerical workload'))
        files=[p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']+[Path(__file__)]
        save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in files})
    elif (OUT/'summary.json').exists():
        close(result,read(OUT/'summary.json'))
        for path,sha in read(OUT/'manifest.json').items():assert digest(ROOT/path)==sha,path
    print(json.dumps(dict(audit_pairs=result['audit_pairs'],audit_pairs_qualified=result['audit_pairs_qualified'],audit_failures=result['audit_failures'],mature_failures=result['mature_failures'],npz_crc_checked=result['npz_crc_checked'],comparisons=[{k:r[k] for k in ('reference_key','target','best_baseline','best_audit','best_practical_audit','best_single_diagnostic','residual_1p5_budget','practical_residual_1p5_budget','single_only_residual_diagnostic')} for r in result['comparisons']]),indent=2))


if __name__=='__main__':main()
