"""Prospective slow-tangent/fast-nonlinear coupling mechanism screen.

Local development only: original formal targets and stronger local fractions
are scored separately. No implicit full-vector work is counted as M4 coverage.
"""
import argparse
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
from metabolism.m2_experiments import error, save
from metabolism.m3_continuation import prepare
from research.slow_tangent import SlowTangent
from research.inner_dense import InnerCurve, physical_ledger, full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_priority import full, metrics
from research.m3_formal_audit_reanalysis import formal_metrics, rates
from research.m3_ordered_cost import structural
from research.m3_ledger_flow_report import close, registration
from research.m3_shadow_study import ROOT, digest

OUT=ROOT/'results/m3-slow-tangent'
OLD=ROOT/'results/m3-remainder-curve'


def read(path): return json.loads(path.read_text())


def finite_errors(value):
    if not value or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<0 for v in value.values()):
        raise ValueError('Nonfinite or negative error indicator')
    return value


def assess(case, curve, end, initial, q, truth, stats, reference_ok):
    values=curve(q); guard=structural(case,values)
    valid=(stats['continuous_nonnegative'] is True and guard['nonnegative'] is True
        and math.isfinite(guard['pool']) and math.isfinite(stats['pool_error'])
        and max(guard['pool'],stats['pool_error'])<=1e-10
        and math.isfinite(curve.join_error(case['scale'])) and curve.join_error(case['scale'])<=1e-10)
    if 'ledger_weights' in case['options']:
        required=['physical_ledger_residual','physical_ledger_quadrature']
        if 'modified_ledger_residual' in stats:
            required += ['modified_ledger_residual','modified_ledger_quadrature']
        valid &= all(k in stats and math.isfinite(stats[k]) and 0<=stats[k]<=1e-10 for k in required)
    vf,vr=rates(case,values),rates(case,truth)
    targets={}
    for target in (1e-3,1e-5):
        actual=finite_errors(formal_metrics(case,values,truth,target,vf,vr))
        local=dict(concentration=error(values,truth,case['scale'],.1*target),
            flux=error(vf,vr,case['options']['flux_scale'],.1*target))
        if 'net_map' in case['options']:
            local['net_flux']=error(case['options']['net_map']@vf,case['options']['net_map']@vr,case['options']['net_scale'],1e-4)
        finite_errors(local)
        targets[str(target)]=dict(actual=actual,local_actual=local,
            formal_pass=bool(valid and reference_ok and max(actual.values())<=1),
            local_pass=bool(valid and reference_ok and max(local.values())<=1))
    return guard,targets


def register():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists(): raise FileExistsError('Preserve prospective evidence')
    sources=['research/m3_slow_tangent_study.py','research/slow_tangent.py','research/test_slow_tangent.py',
        'research/extrapolated_flow.py','research/inner_dense.py','research/fullstep_radau.py',
        'research/ordered_radau.py','research/shared_radau.py','research/m3_priority.py',
        'research/mri_kernel.py','research/remainder_curve.py',
        'research/m3_formal_audit_reanalysis.py','metabolism/adaptive.py',
        'metabolism/splitting.py','metabolism/sbml.py','metabolism/kecoli.py','metabolism/cvode.py']
    refs=[r for r in read(OLD/'runs.json')['references'] if r['h'] in (.01,.03)]
    inputs=[ROOT/'results/m3-slow-tangent-initial/registration.json',ROOT/'results/m3-slow-tangent-initial/initial-source.py',ROOT/'results/m3-slow-tangent-initial/failure.json',OLD/'manifest.json',OLD/'runs.json']+[OLD/f"reference-{r['key']}.npz" for r in refs]
    save(OUT/'registration.json',dict(date='2026-11-15',prospective=True,initial_attempt='results/m3-slow-tangent-initial; serialization implementation failure preserved and excluded',reference_keys=[r['key'] for r in refs],
        group_thresholds=dict(teusink=[1.,3.],kecoli=[.1,1.]),classification_step=.001,
        sweeps=[0,1,2],tolerances=[1e-6,1e-8],control_tolerances=[1e-5,1e-6,1e-8],control_whole=[False,True],
        hypothesis='Retain exact nonlinear fast reactions and frozen current slow reaction tangent in one modified IVP. Zero-sweep prediction; one/two cubic slow-remainder corrections at self-generated H/2,H. No full-system nonlinear predictor before fast correction. Test whether stiff linear slow coupling is resolved with fewer slow nonlinear evaluations.',
        formula='zprime=fF(z)+fS(y)+Js*(z-y)+q^2*(A+q*B), Js=Ss*Dvs(y), A=8*Nhalf-Nend,B=2*Nend-8*Nhalf. Reactionwise N=vs(z)-vs(y)-Dvs(y)*(z-y). Current Jacobian fFprime(z)+Js; full implicit vector dimension explicitly retained.',
        scope='96 tangent candidates and48 mature controls across8 known-state intervals. Single exploratory timings, not formal repeats. Preparation/classification separate, all candidate solves/guards/ledgers included. This method is an additive modified-equation diagnostic; reaction_split_coverage=0 pending independently qualified controller and nondegenerate work reduction.',
        qualification='Original formal concentration/one-way1e-3/1e-5 and net1e-3 plus separate .1 local fractions; pools/physical and modified ledgers1e-10, no clipping, all curves/internal states checked. Independent fresh original ODE1e-12/1e-11 pair plus old archived independent truth, both <=.05 development strict units;65/public queries unchanged.',
        fairness='Mature full Radau gets same complete gradient information, ordered factorization and auto/whole first-step. Single timings screen only; missing BDF/CVODE mature options prevent positive results from granting admission. Any baseline faster than a candidate already vetoes cost prospect. Previous complete audits remain required at runtime, fresh reference solve cost shown separately as a nonminimal conservative diagnostic.',
        stopping='Only if both models have nondegenerate local-pass candidates with baseline/candidate>1.5 before audit, then prospective repeated cost and audited own-state paths may be considered. Otherwise retain all failures and do not build a production controller or formal repetitions. No PE.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs}))


def run():
    reg=registration(OUT/'registration.json')
    if (OUT/'runs.json').exists(): raise FileExistsError('Preserve runs')
    oldrefs={r['key']:r for r in read(OLD/'runs.json')['references']}
    candidates=[];controls=[];references=[];setups=[]
    def persist(): save(OUT/'runs.json',dict(candidates=candidates,controls=controls,references=references,setups=setups))
    for key in reg['reference_keys']:
        rr=oldrefs[key];case=prepare(rr['model']);screen=GroupScreen(case['system'],case['scale'])
        with np.load(OLD/f'reference-{key}.npz') as a:
            y,q,archived=a['initial'],a['query'],a['reference']
        h=rr['h'];stamp=time.perf_counter()
        truth=full(case,y,h*q,1e-12);loose=full(case,y,h*q,1e-11)
        seconds=time.perf_counter()-stamp
        budgets=dict(tightening=finite_errors(metrics(case,loose,truth,1e-6)),archived=finite_errors(metrics(case,truth,archived,1e-6)))
        refok=all(v<=.05 for b in budgets.values() for v in b.values())
        references.append(dict(key=key,model=rr['model'],h=h,start=rr['start'],budgets=budgets,reference_ok=refok,seconds=seconds))
        stamp=time.perf_counter();kappa,_=screen.evaluate(y);groups={}
        for threshold in reg['group_thresholds'][rr['model']]:
            groups[threshold]=np.flatnonzero(reg['classification_step']*kappa>=threshold)
        setups.append(dict(reference_key=key,classification_seconds=time.perf_counter()-stamp,model_setup_seconds=case['setup_seconds']))
        np.savez_compressed(OUT/f'reference-{key}.npz',initial=y,query=q,reference=truth,loose=loose,archived=archived,kappa=kappa)
        for threshold,sweeps,tol in itertools.product(reg['group_thresholds'][rr['model']],reg['sweeps'],reg['tolerances']):
            ck=f'candidate-{key}-{threshold}-{sweeps}-{tol}'
            row=dict(key=ck,reference_key=key,model=rr['model'],threshold=threshold,sweeps=sweeps,tolerance=tol)
            setup=time.perf_counter();kernel=SlowTangent(case,groups[threshold]);row['setup_seconds']=time.perf_counter()-setup
            row.update(group_count=len(groups[threshold]),total_groups=len(case['system'].groups),implicit_dimension=len(y))
            stamp=time.perf_counter()
            try:
                r=kernel.trial(y,h,tol=tol,sweeps=sweeps,whole_step=True);curve=r['pieces'][0];values=curve(q);structural(case,values)
                elapsed=time.perf_counter()-stamp
                guard,targets=assess(case,curve,r['fine'],y,q,truth,r['stats'],refok)
                row.update(status='success',seconds=elapsed,stats=r['stats'],guard=guard,targets=targets)
                arrays=dict(initial=y,groups=groups[threshold],endpoint=r['fine'],**curve.pack())
                for i,c in enumerate(r['curves']): arrays.update({f'curve{i}_'+k:v for k,v in c.pack().items()})
                for name in ('stages','rates','remainders','forcings','initial_rates'):
                    arrays[name]=r[name]
                np.savez_compressed(OUT/f'{ck}.npz',**arrays)
            except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                row.update(status='failed',seconds=time.perf_counter()-stamp,reason=str(exc),stats=getattr(exc,'stats',{}))
            candidates.append(row);persist()
        for tol,whole in itertools.product(reg['control_tolerances'],reg['control_whole']):
            ck=f'control-{key}-{tol}-{whole}';row=dict(key=ck,reference_key=key,model=rr['model'],tolerance=tol,whole=whole)
            stamp=time.perf_counter()
            try:
                r=(fullstep_curve if whole else full_curve)(case,y,h,tol,screen);curve=r['pieces'][0];structural(case,curve(q));elapsed=time.perf_counter()-stamp
                guard,targets=assess(case,curve,r['fine'],y,q,truth,r['stats'],refok)
                row.update(status='success',seconds=elapsed,stats=r['stats'],guard=guard,targets=targets)
                np.savez_compressed(OUT/f'{ck}.npz',endpoint=r['fine'],**curve.pack())
            except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                row.update(status='failed',seconds=time.perf_counter()-stamp,reason=str(exc))
            controls.append(row);persist()
        print('slow tangent measured',key,flush=True)


def verify():
    reg=registration(OUT/'registration.json');data=read(OUT/'runs.json');comparisons=[]
    expected_c=set();expected_b=set()
    for rr in data['references']:
        key=rr['key'];case=prepare(rr['model']);screen=GroupScreen(case['system'],case['scale'])
        with np.load(OUT/f'reference-{key}.npz') as a:
            y,q,truth,kappa=a['initial'],a['query'],a['reference'],a['kappa']
            budgets=dict(tightening=finite_errors(metrics(case,a['loose'],truth,1e-6)),archived=finite_errors(metrics(case,truth,a['archived'],1e-6)))
        close(budgets,rr['budgets']);assert rr['reference_ok']==all(v<=.05 for b in budgets.values() for v in b.values())
        np.testing.assert_allclose(screen.evaluate(y)[0],kappa,rtol=1e-13,atol=1e-13)
        for threshold,sweeps,tol in itertools.product(reg['group_thresholds'][rr['model']],reg['sweeps'],reg['tolerances']):
            expected_c.add(f'candidate-{key}-{threshold}-{sweeps}-{tol}')
        for tol,whole in itertools.product(reg['control_tolerances'],reg['control_whole']):expected_b.add(f'control-{key}-{tol}-{whole}')
        for kind in ('candidates','controls'):
            for row in (r for r in data[kind] if r['reference_key']==key):
                assert math.isfinite(row['seconds']) and row['seconds']>0
                if row['status']=='failed': assert row['reason'];continue
                assert row['status']=='success'
                with np.load(OUT/f"{row['key']}.npz") as a:
                    curve=InnerCurve.unpack(a['lengths'],a['coefficients']);end=a['endpoint']
                    np.testing.assert_allclose(curve(np.array([0.,1.])),np.column_stack([y,end]),rtol=2e-12,atol=1e-12)
                    assert bool(np.all(curve.minima()>=-1e-12*case['scale']))==row['stats']['continuous_nonnegative']
                    if 'ledger_weights' in case['options']:
                        for k,v in physical_ledger(case,curve,y,end).items():close(v,row['stats'][k])
                    guard,targets=assess(case,curve,end,y,q,truth,row['stats'],rr['reference_ok']);close(guard,row['guard']);close(targets,row['targets'])
                    if kind=='candidates':
                        groups=np.flatnonzero(reg['classification_step']*kappa>=row['threshold']);np.testing.assert_array_equal(groups,a['groups'])
                        cols=np.concatenate([case['system'].groups[g] for g in groups]) if len(groups) else np.array([],int)
                        slow=np.setdiff1d(np.arange(case['system'].S.shape[1]),cols);sm=case['system'].S[:,slow];gradient=screen.gradient(y)[slow];v0=case['system'].flux(0.,y)[slow]
                        np.testing.assert_allclose(a['initial_rates'][slow],v0,rtol=2e-13,atol=1e-13)
                        for i in range(row['sweeps']+1):
                            prev=InnerCurve.unpack(a[f'curve{i}_lengths'],a[f'curve{i}_coefficients'])
                            assert np.all(prev.minima()>=-1e-12*case['scale']) and prev.join_error(case['scale'])<=1e-10
                            if i==row['sweeps']:continue
                            stages=prev(np.array([.5,1.]));rv=np.column_stack([case['system'].flux(0.,z)[slow] for z in stages.T]);n=rv-v0[:,None]-gradient@(stages-y[:,None])
                            forcing=sm@np.column_stack([8*n[:,0]-n[:,1],2*n[:,1]-8*n[:,0]])
                            for name,value in [('stages',stages),('remainders',n),('forcings',forcing)]:close(value,a[name][i])
                            close(rv,a['rates'][i][slow])
                        if 'ledger_weights' in case['options']:
                            from research.mri_kernel import GAUSS
                            force=a['forcings'][-1] if row['sweeps'] else np.zeros((len(y),2))
                            jf=case['system'].S[:,cols];js=sm@gradient;s0=sm@v0;totals=[0.,0.];qd=0.;left=0.
                            for piece in curve.pieces:
                                ints=[]
                                for degree in (8,16):
                                    nodes,w=GAUSS[degree];frac=(nodes+1)/2;tau=(left+piece.h*frac)/rr['h'];zs=piece(frac)
                                    velocity=jf@np.column_stack([case['system'].flux(0.,z)[cols] for z in zs.T])+s0[:,None]+js@(zs-y[:,None])+force[:,:1]*tau**2+force[:,1:]*tau**3
                                    ints.append(piece.h/2*float(w@(case['options']['ledger_weights']@velocity)))
                                for j in (0,1):totals[j]+=ints[j]
                                qd+=abs(ints[1]-ints[0]);left+=piece.h
                            den=max(1.,abs(float(case['options']['ledger_weights']@y)))
                            close(abs(float(case['options']['ledger_weights']@(end-y))-totals[1])/den,row['stats']['modified_ledger_residual'])
                            close(qd/den,row['stats']['modified_ledger_quadrature'])
        for target in ('0.001','1e-05'):
            cc=[r for r in data['candidates'] if r['reference_key']==key and r['status']=='success' and r['targets'][target]['local_pass'] and 0<r['group_count']<r['total_groups']]
            bb=[r for r in data['controls'] if r['reference_key']==key and r['status']=='success' and r['targets'][target]['local_pass']]
            best=min(bb,key=lambda r:r['seconds'],default=None);candidate=min(cc,key=lambda r:r['seconds'],default=None)
            brief=lambda r:None if r is None else {k:r[k] for k in ('key','seconds')}
            comparisons.append(dict(reference_key=key,model=rr['model'],target=target,qualified_candidates=len(cc),best_baseline=brief(best),best_candidate=brief(candidate),
                optimistic_speedup=None if not best or not candidate else best['seconds']/candidate['seconds'],
                candidate_plus_current_reference_speedup=None if not best or not candidate else best['seconds']/(candidate['seconds']+rr['seconds'])))
    for kind,expected in [('candidates',expected_c),('controls',expected_b)]:
        keys=[r['key'] for r in data[kind]];assert len(keys)==len(set(keys)) and set(keys)==expected
    assert {r['key'] for r in data['references']}==set(reg['reference_keys']) and len(data['references'])==len(reg['reference_keys'])
    crc=0
    for p in OUT.glob('*.npz'):
        with zipfile.ZipFile(p) as z:assert z.testzip() is None
        crc+=1
    return dict(date=reg['date'],candidate_records=len(data['candidates']),control_records=len(data['controls']),candidate_failures=sum(r['status']=='failed' for r in data['candidates']),control_failures=sum(r['status']=='failed' for r in data['controls']),comparisons=comparisons,npz_crc_checked=crc,reaction_split_coverage=0.,M4_admission=False,
        scope='Known-state local mechanism screen with original and .1-local formal targets; full implicit vector; no runtime controller, no formal repeat or full-window qualification.')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','write','verify']);args=p.parse_args()
    if args.action=='register':register();return
    if args.action=='run':run();return
    result=verify()
    if args.action=='write':
        if (OUT/'manifest.json').exists():raise FileExistsError('Preserve manifest')
        save(OUT/'summary.json',result);save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),execution='scripts/run-macos.sh single-thread BLAS/OpenMP; exclusive numerical workload'))
        save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
