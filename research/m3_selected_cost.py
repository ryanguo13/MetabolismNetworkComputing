"""Prospective fixed-column kinetics ablation; no audit/control removal."""
import argparse
from dataclasses import replace
import itertools
import json
import platform
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import scipy
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.selected_reactions import SelectedReactions
from research.selected_extrapolated import SelectedExtrapolatedFlow
from research.extrapolated_flow import ExtrapolatedFlow,SlowPolynomial,pack,unpack
from research.inner_dense import InnerCurve,full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_priority import full,metrics
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_ledger_flow_report import close,check_curve,modified,registration
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-selected-cost';OLD=ROOT/'results/m3-inner-dense'


def read(path):return json.loads(path.read_text())


def selected_full(case):
    rates=SelectedReactions(case['model'],np.arange(case['system'].S.shape[1]),1e-12*case['scale'])
    out=case.copy()
    out['system']=replace(case['system'],flux=rates.flux,flux_jacobian=rates.gradient)
    return out


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    sources=['research/m3_selected_cost.py','research/selected_reactions.py','research/selected_extrapolated.py',
        'research/test_selected_reactions.py','research/extrapolated_flow.py','research/mri_kernel.py',
        'research/inner_dense.py','research/fullstep_radau.py','research/shared_radau.py','research/m3_extrapolated_paths.py',
        'metabolism/sbml.py','metabolism/kecoli.py','metabolism/adaptive.py']
    reg=dict(date='2026-11-15',paths=[2,4,5],step=4,order=3,repeats=[-1,0,1,2],
        candidate_tolerances=[1e-6,1e-8],control_tolerances=[1e-6,1e-8,1e-10],control_whole=[False,True],
        hypothesis='Fast RHS/J evaluate only fixed selected reaction columns, preserving full-state kinetic dependencies. Slow forcing uses the exact complementary columns. Test if removed complete-rate evaluation changes cost without changing physical qualification.',
        independence='Same previously generated real MRI3 history, never reference-driven. Fresh independent complete original ODE pair1e-12/1e-11; compare pair and archived independent truth within .05 development strict units. Original local targets and protections unchanged. Selected curves get original full-rate independent scoring and ledger reconstruction.',
        timing='One warmup plus3 exploratory repeats in rotating order. Candidate includes current slow RHS/fast integration/output and all original pool/physical+modified ledger/sample guards. Setup/history preparation separate. Independent reference pair and error scoring separate; add audit cost for optimistic feasibility. No startup/controller/estimator overhead removed from a formal claim; this is local ablation only.',
        fairness='Complete mature Radau gets all-column selected kinetics/gradient as well as original kinetics; automatic/whole first-step and3 tolerances. Choose fastest qualified median among all. Each candidate gets fixed real reaction groups and dependencies saved. Small-model constant slow groups excluded by path choice2.',
        stopping='No formal repetitions or new controller unless both models have positive baseline/1.5 minus candidate and required audit budget. Failures and all unsuccessful qualifications remain in runs. These are local supplied-history paths, not M4 coverage.',
        source_sha256={p:digest(ROOT/p) for p in sources},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in [OLD/'history.json',ROOT/'results/m3-extrapolated-flow/manifest.json']+[OLD/f'history-{i}.npz' for i in (2,4,5)]})
    save(OUT/'registration.json',reg)
    runs=[];references=[];setups=[]
    old=read(OLD/'history.json')['runs']
    for index in reg['paths']:
        source=next(r for r in old if r['shard']==f'history-{index}' and r['step']==4)
        case=prepare(source['model']);chosen=selected_full(case)
        with np.load(OLD/f'history-{index}.npz') as a:
            y=a['endpoints'][:,4];groups=a['groups'];q=a['4-query'];archived=a['4-reference']
            times=a['endpoint_times'][2:5];states=a['endpoints'][:,2:4]
        h=source['h'];kernels=[ExtrapolatedFlow(case,groups),SelectedExtrapolatedFlow(case,groups)]
        tick=time.perf_counter();histories=[np.column_stack([k.slow(z) for z in states.T]) for k in kernels]
        history_seconds=time.perf_counter()-tick
        tick=time.perf_counter();ref=full(case,y,h*q,1e-12);loose=full(case,y,h*q,1e-11)
        audit_seconds=time.perf_counter()-tick
        budgets=dict(tightening=metrics(case,loose,ref,1e-6),archived=metrics(case,ref,archived,1e-6))
        assert max(v for b in budgets.values() for v in b.values())<=.05
        history_difference=float(np.max(abs(histories[0]-histories[1])/case['scale'][:,None]))
        np.savez_compressed(OUT/f'reference-{index}.npz',initial=y,query=q,reference=ref,loose=loose,archived=archived,
            groups=groups,times=times,original_history=histories[0],selected_history=histories[1],history_states=states,
            selected_columns=kernels[1].cols,complement_columns=kernels[1].slow_columns,
            selected_dependencies=kernels[1].fast_rates.dependencies.toarray())
        references.append(dict(path=index,model=source['model'],h=h,budgets=budgets,audit_seconds=audit_seconds,
            slow_history_difference_scaled=history_difference,fast_reactions=len(kernels[1].cols),slow_reactions=len(kernels[1].slow_columns),
            active=len(kernels[1].active),total_reactions=case['system'].S.shape[1]))
        setups.append(dict(path=index,original_seconds=kernels[0].setup_seconds,selected_seconds=kernels[1].setup_seconds,
            history_seconds=history_seconds))
        screens=[GroupScreen(c['system'],c['scale']) for c in (case,chosen)]
        specs=[('candidate',v,tol,True) for v,tol in itertools.product(range(2),reg['candidate_tolerances'])]
        specs += [('full',v,tol,whole) for v,tol,whole in itertools.product(range(2),reg['control_tolerances'],reg['control_whole'])]
        for repeat in reg['repeats']:
            shift=(repeat+1)%len(specs)
            for kind,version,tol,whole in specs[shift:]+specs[:shift]:
                key=f'{index}-{repeat}-{kind}-{version}-{tol}-{whole}'
                row=dict(key=key,path=index,repeat=repeat,kind=kind,version=version,tolerance=tol,whole=whole)
                tick=time.perf_counter()
                try:
                    if kind=='candidate':trial=kernels[version].trial(y,h,times,histories[version],tol,whole)
                    else:trial=(fullstep_curve if whole else full_curve)((case,chosen)[version],y,h,tol,screens[version])
                    curve=trial['pieces'][0];values=curve(q);structural(case,values)
                    seconds=time.perf_counter()-tick
                    scoring=time.perf_counter();guard,targets=qualified(case,curve,values,ref,trial['stats'])
                    row.update(status='success',seconds=seconds,scoring_seconds=time.perf_counter()-scoring,
                        stats=trial['stats'],guard=guard,targets=targets)
                    arrays=(pack(curve) if kind=='candidate' else curve.pack())
                    if kind=='candidate':arrays['forcing']=trial['forcing_coefficients']
                    np.savez_compressed(OUT/f'{key}.npz',endpoint=trial['fine'],**arrays)
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                    row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
                runs.append(row);save(OUT/'runs.json',dict(runs=runs,references=references,setups=setups))
            print('measured selected kinetics',index,repeat,flush=True)
        for kernel in kernels:kernel.close()


def verify():
    reg=registration(OUT/'registration.json');data=read(OUT/'runs.json');summaries=[]
    assert len(data['runs'])==192
    for rr in data['references']:
        index=rr['path'];case=prepare(rr['model'])
        with np.load(OUT/f'reference-{index}.npz') as a:
            y=a['initial'];q=a['query'];ref=a['reference'];groups=a['groups'];times=a['times']
            histories=[a['original_history'],a['selected_history']];states=a['history_states']
            close(metrics(case,a['loose'],ref,1e-6),rr['budgets']['tightening'])
            close(metrics(case,ref,a['archived'],1e-6),rr['budgets']['archived'])
            kernels=[ExtrapolatedFlow(case,groups),SelectedExtrapolatedFlow(case,groups)]
            np.testing.assert_array_equal(a['selected_columns'],kernels[1].cols)
            np.testing.assert_array_equal(a['complement_columns'],kernels[1].slow_columns)
            np.testing.assert_array_equal(a['selected_dependencies'],kernels[1].fast_rates.dependencies.toarray())
        for i,kernel in enumerate(kernels):
            close(histories[i],np.column_stack([kernel.slow(z) for z in states.T]))
        rows=[r for r in data['runs'] if r['path']==index]
        assert len(rows)==64
        for row in rows:
            if row['status']=='failed':continue
            with np.load(OUT/f"{row['key']}.npz") as a:
                curve=(unpack if row['kind']=='candidate' else InnerCurve.unpack)(a['lengths'],a['coefficients'])
                end=a['endpoint']
                check_curve(case,curve,y,end,ref,q,row['stats'],row['guard'],row['targets'])
                if row['kind']=='candidate':
                    forcing=SlowPolynomial(times,np.column_stack([histories[row['version']],kernels[row['version']].slow(y)]),rr['h'])
                    close(forcing.coefficients,a['forcing'])
                    modified(case,kernels[row['version']],curve,y,end,forcing,row['stats'])
        for target in ('0.0001','1e-06'):
            costs=[]
            for kind,version,tol,whole in dict.fromkeys((r['kind'],r['version'],r['tolerance'],r['whole']) for r in rows):
                samples=[r for r in rows if r['repeat']>=0 and (r['kind'],r['version'],r['tolerance'],r['whole'])==(kind,version,tol,whole)]
                good=len(samples)==3 and all(r['status']=='success' and r['targets'][target]['passes'] for r in samples)
                costs.append(dict(kind=kind,version=version,tolerance=tol,whole=whole,qualified=good,
                    median_seconds=float(np.median([r['seconds'] for r in samples]))))
            baselines=[r for r in costs if r['kind']=='full' and r['qualified']]
            best=min(baselines,key=lambda r:r['median_seconds']) if baselines else None
            choices={str(v):min((r for r in costs if r['kind']=='candidate' and r['version']==v and r['qualified']),key=lambda r:r['median_seconds'],default=None) for v in range(2)}
            candidate=choices['1'];positive=best is not None and candidate is not None
            residual=(best['median_seconds']/1.5-candidate['median_seconds']-rr['audit_seconds']) if positive else None
            summaries.append(dict(path=index,model=rr['model'],target=target,costs=costs,best_baseline=best,
                best_candidate=choices,residual_with_audit_budget=residual,
                positive_budget=bool(residual is not None and residual>0)))
        for kernel in kernels:kernel.close()
    crc=0
    for path in OUT.glob('*.npz'):
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        crc+=1
    return dict(records=len(data['runs']),references=data['references'],failures=sum(r['status']=='failed' for r in data['runs']),
        npz_crc_checked=crc,comparisons=summaries,M4_admission=False,
        scope='Local supplied own-history ablation; startup/control/estimator/full-window coverage/formal repeats absent. All original guards and independent ODE references retained.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['run','verify','write']);args=parser.parse_args()
    if args.action=='run':run();return
    result=verify()
    if args.action=='write':
        if (OUT/'manifest.json').exists():raise FileExistsError('Preserve evidence')
        save(OUT/'summary.json',result)
        save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
        files=[p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']
        files += [ROOT/p for p in read(OUT/'registration.json')['source_sha256']]
        save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in files})
    elif (OUT/'summary.json').exists():
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
