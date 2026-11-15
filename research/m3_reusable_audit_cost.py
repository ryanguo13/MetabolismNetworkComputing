"""Prospective full-audit allocation ablation with mandatory BDF resets."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import scipy

from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.reusable_audit import ReusableFullAudit
from research.m3_audit_output_cost import guarded_audit, runtime_score, finite_metrics
from research.m3_formal_audit_reanalysis import formal_metrics, rates, read
from research.m3_ledger_flow_report import registration, close
from research.inner_dense import InnerCurve, full_curve
from research.fullstep_radau import fullstep_curve
from research.extrapolated_flow import unpack
from research.remainder_storage import load_arrays
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT, digest

OUT=ROOT/'results/m3-reusable-audit-cost'
TRUTH=ROOT/'results/m3-remainder-curve'
PRIOR=ROOT/'results/m3-audit-output-cost'


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def register():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve prospective registration')
    previous=read(PRIOR/'registration.json')
    cases=[r for r in previous['cases'] if r['model']=='kecoli']
    sources=['research/reusable_audit.py','research/test_reusable_audit.py',
        'research/m3_reusable_audit_cost.py','research/mri_native.py','metabolism/cvode.py',
        'metabolism/__init__.py','metabolism/kecoli.py','metabolism/splitting.py',
        'metabolism/m3_continuation.py',
        'research/m3_audit_output_cost.py','research/m3_formal_audit_reanalysis.py',
        'research/inner_dense.py','research/fullstep_radau.py','research/ordered_radau.py']
    inputs=[PRIOR/'manifest.json',PRIOR/'registration.json',TRUTH/'manifest.json',TRUTH/'runs.json']
    inputs += [ROOT/'models/k-ecoli457.zip',ROOT/'results/m1-completion/kecoli-reference.npz']
    inputs += [TRUTH/f"reference-{r['reference_key']}.npz" for r in cases]
    save(OUT/'registration.json',dict(date='2026-11-15',cases=cases,repeats=[-1,0,1,2],
        audit_tolerances=[1e-6,1e-7,1e-8,1e-9,1e-10,1e-11],
        tightening_pairs=[[1e-6,1e-7],[1e-8,1e-9],[1e-10,1e-11]],
        native_methods=['original-cold','adapter-cold','adapter-warm'],
        control_tolerances=[1e-4,1e-6,1e-8,1e-10],control_radau_methods=['ordered-auto','ordered-whole'],
        formal_targets=[1e-3,1e-5],formal_net_target=1e-3,local_fraction=.1,
        reference_budget_in_formal_units=.005,
        hypothesis='Repeated native library binding and context/vector/matrix/solver allocation costs may be reduced without retaining inconsistent BDF history. Extend existing NativeSession to mandatory same-state ReInit and complete required sampled queries. Compare identical adapter cold/warm, plus original cold binding.',
        scope='Two previously used k-ecoli states, H=.01, required public output plus quarters, one warmup and three exploratory repeats. No original full-window, held-out, split coverage or seven formal repeats. SBML without analytic Jacobian explicitly unsupported.',
        independence='Every solve begins at the same explicitly supplied true candidate initial state. Every adapter call resets BDF history via ReInit, even after failure/rejection or changed initial/tolerance. Only native bindings/allocations/fixed sparsity survive; no numeric LU/KLU-history reuse claim. Original cold binding has no explicit final stop boundary; same-adapter cold/warm both set it, isolating allocation effects fairly.',
        truth='Immutable independent original nonlinear ODE truth only offline. Each standalone audit and its tenfold tightening must each meet .005 formal units, i.e. .05 of original .1-formal local acceptance unit; pair discrepancy must meet same. All metrics explicitly finite/nonnegative. This is finite-state calibration, not an error bound or general no-false-acceptance guarantee.',
        runtime='Audit solve includes actual allocation/reinit and required sampled nonnegative/pool guards. Fixed saved MERB candidate comparison is separately timed at each formal target, never uses truth, and is added to runtime budget. Candidate generation remains additional work; no candidate accepted/split coverage claim.',
        fairness='Same native original/cold/warm resources available to mature full ODE controls at four tolerances. Also mature optimized Radau auto/whole with continuous nonnegative/pool checks. Native accepted internal states unexposed; original internal RHS domain and sampled output guards retained, no ledger applicable to admitted large case. Setup and repeated model preparation separate, runtime cold allocation included; warm setup is reported and must be amortized in any future full-window claim.',
        timing='Single-thread BLAS/OpenMP, parent-coordinated exclusive numerical window. Rotate config order. Save each warmup array/curve and every repeat exact output hash, counters, metrics and cost. Recompute metrics and hashes for all repeated records. No controller/formal speed claim.',
        source_sha256={p:digest(ROOT/p) for p in sources},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs}))


def native(case,screen,y,h,q,tol,method,warm):
    if method=='original-cold':
        values,stats,guard=guarded_audit(case,screen,y,h,q,tol,'current')
        return values,stats,guard
    tick=time.perf_counter()
    audit=warm if method=='adapter-warm' else ReusableFullAudit(case)
    try:values,stats=audit.solve(y,h,q,tol)
    finally:
        if method=='adapter-cold':audit.close()
    stats['runtime_seconds']=time.perf_counter()-tick
    return values,stats,structural(case,values)


def spec_list(reg):
    specs=[('audit',m,t) for m,t in itertools.product(reg['native_methods'],reg['audit_tolerances'])]
    specs += [('control',m,t) for m,t in itertools.product(reg['native_methods']+reg['control_radau_methods'],reg['control_tolerances'])]
    return specs


def run():
    reg=registration(OUT/'registration.json')
    if (OUT/'runs.json').exists():raise FileExistsError('Preserve observations')
    rows=[];setups=[]
    for cfg in reg['cases']:
        key=cfg['reference_key'];case=prepare('kecoli');screen=GroupScreen(case['system'],case['scale'])
        q=np.array(cfg['required_queries']);h=cfg['h']
        with np.load(TRUTH/f'reference-{key}.npz') as a:
            y=a['initial'];truth=a['reference'][:,cfg['required_truth_indices']]
        with load_arrays(TRUTH/f"{cfg['scoring_candidate']}.npz") as a:
            candidate=unpack(a['lengths'],a['coefficients'])
        warm=ReusableFullAudit(case);setups.append(dict(reference_key=key,seconds=warm.setup_seconds))
        configs=spec_list(reg)
        try:
            for rep in reg['repeats']:
                shift=(rep+1)%len(configs)
                for kind,method,tol in configs[shift:]+configs[:shift]:
                    name=f'{key}-{kind}-{method}-{tol}'
                    row=dict(key=name,reference_key=key,kind=kind,method=method,tolerance=tol,repeat=rep)
                    tick=time.perf_counter()
                    try:
                        if method in reg['native_methods']:
                            values,stats,guard=native(case,screen,y,h,q,tol,method,warm);curve=None
                            valid=guard['nonnegative'] and math.isfinite(guard['pool']) and guard['pool']<=1e-10
                        else:
                            r=(fullstep_curve if method=='ordered-whole' else full_curve)(case,y,h,tol,screen)
                            curve=r['pieces'][0];values=curve(q);stats=r['stats'];guard=structural(case,values)
                            valid=bool(stats['continuous_nonnegative'] and guard['nonnegative'] and max(stats['pool_error'],guard['pool'])<=1e-10)
                        seconds=time.perf_counter()-tick
                        targets={}
                        for target in reg['formal_targets']:
                            actual=finite_metrics(formal_metrics(case,values,truth,target))
                            info=dict(actual=actual,qualified=bool(valid and max(actual.values())<=(reg['reference_budget_in_formal_units'] if kind=='audit' else 1.)))
                            if kind=='audit':
                                score,elapsed=runtime_score(case,candidate,q,values,target)
                                info.update(candidate_error=score,candidate_pass=bool(max(score.values())<=1),scoring_seconds=elapsed,total_runtime_seconds=seconds+elapsed)
                            targets[str(target)]=info
                        row.update(status='success',seconds=seconds,stats=stats,guard=guard,targets=targets,value_hash=array_hash(values))
                        if rep==-1:np.savez_compressed(OUT/f'{name}.npz',values=values,**(curve.pack() if curve else {}))
                    except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                        row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc))
                    rows.append(row);save(OUT/'runs.json',dict(runs=rows,setups=setups))
                print('measured reusable audit',key,rep,flush=True)
        finally:warm.close()


def verify():
    reg=registration(OUT/'registration.json');data=read(OUT/'runs.json');rows=data['runs'];specs=spec_list(reg)
    expected={(cfg['reference_key'],kind,method,tol,rep) for cfg,rep,spec in itertools.product(reg['cases'],reg['repeats'],specs) for kind,method,tol in [spec]}
    actual=[(r['reference_key'],r['kind'],r['method'],r['tolerance'],r['repeat']) for r in rows]
    assert len(actual)==len(set(actual))==len(expected) and set(actual)==expected
    comparisons=[]
    for cfg in reg['cases']:
        key=cfg['reference_key'];case=prepare('kecoli');q=np.array(cfg['required_queries'])
        with np.load(TRUTH/f'reference-{key}.npz') as a:truth=a['reference'][:,cfg['required_truth_indices']]
        with load_arrays(TRUTH/f"{cfg['scoring_candidate']}.npz") as a:candidate=unpack(a['lengths'],a['coefficients'])
        local=[r for r in rows if r['reference_key']==key]
        arrays={}
        for row in local:
            assert math.isfinite(row['seconds']) and row['seconds']>0
            first=next(r for r in local if r['key']==row['key'] and r['repeat']==-1)
            assert row['status']==first['status']
            if row['status']=='failed':assert row['reason']==first['reason'];continue
            with np.load(OUT/f"{row['key']}.npz") as a:
                values=a['values'];curve=InnerCurve.unpack(a['lengths'],a['coefficients']) if 'lengths' in a else None
            arrays[row['method'],row['tolerance']]=values
            assert np.isfinite(values).all() and array_hash(values)==row['value_hash']
            guard=structural(case,values);close(guard,row['guard'])
            valid=guard['nonnegative'] and guard['pool']<=1e-10
            if curve is not None:
                np.testing.assert_array_equal(curve(q),values)
                positive=bool(np.all(curve.minima()>=-1e-12*case['scale']))
                assert positive==row['stats']['continuous_nonnegative']
                valid &= positive and row['stats']['pool_error']<=1e-10
            for target in reg['formal_targets']:
                info=row['targets'][str(target)]
                actual_error=finite_metrics(formal_metrics(case,values,truth,target));close(actual_error,info['actual'])
                limit=reg['reference_budget_in_formal_units'] if row['kind']=='audit' else 1.
                assert info['qualified']==bool(valid and max(actual_error.values())<=limit)
                if row['kind']=='audit':
                    score,_=runtime_score(case,candidate,q,values,target);close(score,info['candidate_error'])
                    assert info['candidate_pass']==bool(max(score.values())<=1)
                    assert math.isfinite(info['scoring_seconds']) and info['scoring_seconds']>=0
                    close(info['total_runtime_seconds'],row['seconds']+info['scoring_seconds'])
            if row['method'] in ('adapter-cold','adapter-warm'):
                assert row['stats']['history_reset'] is True
                assert math.isfinite(row['stats']['reinit_seconds']) and row['stats']['reinit_seconds']>=0
        for target in reg['formal_targets']:
            costs=[];audits=[]
            for kind,method,tol in specs:
                samples=[r for r in local if (r['kind'],r['method'],r['tolerance'])==(kind,method,tol) and r['repeat']>=0]
                good=all(r['status']=='success' and r['targets'][str(target)]['qualified'] for r in samples)
                if kind=='control':
                    costs.append(dict(method=method,tolerance=tol,qualified=good,median_seconds=float(np.median([r['seconds'] for r in samples]))))
            for method,(loose,tight) in itertools.product(reg['native_methods'],reg['tightening_pairs']):
                ss=[[r for r in local if r['kind']=='audit' and r['method']==method and r['tolerance']==t and r['repeat']>=0] for t in (loose,tight)]
                good=all(r['status']=='success' and r['targets'][str(target)]['qualified'] for s in ss for r in s)
                discrepancy=None
                if good:
                    discrepancy=finite_metrics(formal_metrics(case,arrays[method,loose],arrays[method,tight],target))
                    good &= max(discrepancy.values())<=reg['reference_budget_in_formal_units']
                audits.append(dict(method=method,tolerance=loose,tightening_tolerance=tight,qualified=bool(good),tightening=discrepancy,
                    median_seconds=float(np.median([r['seconds'] for r in ss[0]])),
                    median_with_scoring_seconds=float(np.median([r.get('targets',{}).get(str(target),{}).get('total_runtime_seconds',r['seconds']) for r in ss[0]]))))
            best=min((c for c in costs if c['qualified']),key=lambda r:r['median_seconds'],default=None)
            per_method={m:min((a for a in audits if a['method']==m and a['qualified']),key=lambda r:r['median_with_scoring_seconds'],default=None) for m in reg['native_methods']}
            warm=per_method['adapter-warm']
            comparisons.append(dict(reference_key=key,target=target,query_count=len(q)-1,controls=costs,audits=audits,best_baseline=best,best_audits=per_method,
                residual_1p5_budget=None if best is None or warm is None else best['median_seconds']/1.5-warm['median_with_scoring_seconds']))
        print('verified reusable audit',key,flush=True)
    crc=0
    for path in OUT.glob('*.npz'):
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        crc+=1
    return dict(records=len(rows),failures=sum(r['status']=='failed' for r in rows),npz_crc_checked=crc,comparisons=comparisons,M4_admission=False,
        scope='Local known-state allocation ablation with mandatory history resets. Candidate generation/startup/partition/control costs remain unpaid; no original full-window or reaction-split coverage qualification.')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','write','verify']);args=p.parse_args()
    if args.action=='register':register();return
    if args.action=='run':run();return
    result=verify()
    if args.action=='write':
        if (OUT/'manifest.json').exists():raise FileExistsError('Preserve evidence')
        save(OUT/'summary.json',result)
        save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP, coordinated exclusive numerical workload'))
        save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
