"""Re-solve all nonlinear trials and verify saved implicit-history evidence."""
import argparse
import collections
import json
import platform
import subprocess
import sys
import numpy as np
import scipy
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.implicit_flow import EndpointImplicitFlow
from research.implicit_history import EndpointPolynomial
from research.extrapolated_flow import unpack
from research.inner_dense import InnerCurve,physical_ledger
from research.m3_growth_study import audit
from research.m3_implicit_stability import evaluate
from research.m3_implicit_local_corrected import history_values
from research.m3_ledger_flow_report import close,registration,check_curve
from research.mri_kernel import GAUSS
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-implicit-history';LOCAL=ROOT/'results/m3-implicit-local-corrected'
def read(p):return json.loads(p.read_text())


def verify_stability():
    registration(OUT/'stability-registration.json');rows,arrays=evaluate()
    close(rows,read(OUT/'stability.json')['runs']);assert len(rows)==300
    with np.load(OUT/'stability.npz') as a:
        assert set(a.files)==set(arrays)
        for k,v in arrays.items():np.testing.assert_allclose(v,a[k],rtol=1e-12,atol=1e-13)
    bounds=[]
    for state in ('early','late'):
        for h in (1e-4,3e-4,1e-3,3e-3,1e-2):
            for q in (2,3,4):
                valid=[r for r in rows if r['state']==state and r['h']==h and r['order']==q and r['passes_frozen_linear_screen']]
                bounds.append(dict(state=state,h=h,order=q,minimum_fast_active=min((r['fast_active'] for r in valid),default=None)))
    return dict(configurations=300,minimum_dimension=bounds,scope='Exact fast flow, frozen constant-step linear analysis only.')


def modified_audit(kernel,curve,y,end,forcing,stats):
    totals=np.zeros(2);absolute=0.;left=0.
    for piece in curve.pieces:
        integrals=[]
        for n in (4,8):
            q=(GAUSS[n][0]+1)/2
            vals=[kernel.weights@(kernel.sf@kernel.system.flux(0.,z)[kernel.cols]+forcing.value(left+piece.h*t)) for t,z in zip(q,piece(q).T)]
            integrals.append(piece.h/2*float(GAUSS[n][1]@vals))
        totals+=integrals;absolute+=abs(integrals[1]-integrals[0]);left+=piece.h
    den=max(1.,abs(float(kernel.weights@y)))
    close(abs(float(kernel.weights@(end-y))-totals[1])/den,stats['modified_ledger_residual'])
    close(abs(totals[1]-totals[0])/den,stats['modified_ledger_quadrature'])
    # Also enforce the historical stricter sum of absolute piecewise differences.
    assert absolute/den<=1e-10
    return absolute/den


def verify_local():
    registration(OUT/'local-registration.json');original=read(OUT/'local.json')
    assert len(original['runs'])==108
    invalid=[r for r in original['runs'] if r['order']>2]
    assert len(invalid)==72 and all(r['status']=='failed' and r['reason']=='Invalid implicit controls/history' for r in invalid)
    reg=registration(LOCAL/'local-registration.json');data=read(LOCAL/'local.json')
    assert len(data['runs'])==108 and len(data['baselines'])==18
    assert {(r['h'],r['threshold'],r['order'],r['iteration'],r['tolerance']) for r in data['runs']}=={
        (h,g,q,m,t) for h in reg['steps'] for g in reg['thresholds'] for q in reg['orders'] for m in reg['iterations'] for t in reg['tolerances']}
    case=prepare('teusink');screen=GroupScreen(case['system'],case['scale'])
    origin=np.load(ROOT/'results/m3-priority/p0.npz')['teusink-prefix'][:,3]
    max_quad=0.;decisions=[]
    for h in reg['steps']:
        with np.load(LOCAL/f'reference-{h}.npz') as base:
            np.testing.assert_array_equal(origin,base['origin']);y=base['initial'];query=base['query']
            start=1+2*h;observed=(case['times'][(case['times']>start)&(case['times']<start+h)]-start)/h
            np.testing.assert_array_equal(query,np.unique(np.r_[np.linspace(0,1,65),observed]))
            ref,budget,_=audit(case,y,query,h);assert max(budget.values())<=.05
            np.testing.assert_allclose(ref,base['reference'],rtol=2e-12,atol=1e-13)
            hist=next(r for r in data['histories'] if r['h']==h);close(budget,hist['budget'])
            boot=InnerCurve.unpack(base['bootstrap-lengths'],base['bootstrap-coefficients'])
            states=boot(np.array([0.,.5,1.]));states[:,-1]=y
            np.testing.assert_array_equal(states,base['history'])
            assert np.all(boot.minima()>=-1e-12*case['scale']) and boot.join_error(case['scale'])<=1e-10
            assert hist['stats']['pool_error']<=1e-10
            for k,v in physical_ledger(case,boot,origin,y).items():close(v,hist['stats'][k]);assert v<=1e-10
            for row in [r for r in data['baselines'] if r['h']==h]:
                assert row['status']=='success';k=row['key'];curve=InnerCurve.unpack(base[k+'-lengths'],base[k+'-coefficients'])
                check_curve(case,curve,y,base[k+'-endpoint'],ref,query,row['stats'],row['guard'],row['targets'])
            kappa,_=screen.evaluate(y)
            for row in [r for r in data['runs'] if r['h']==h]:
                groups=np.flatnonzero(1e-3*kappa>=row['threshold']);np.testing.assert_array_equal(groups,row['groups'])
                kernel=EndpointImplicitFlow(case,groups);q=row['order'];times=np.arange(-(q-2),1)*h
                past=history_values(np.column_stack([kernel.slow(z) for z in states[:,:-1].T]),q)
                with np.load(LOCAL/f"{row['name']}.npz") as a:
                    np.testing.assert_array_equal(y,a['initial']);np.testing.assert_array_equal(times,a['history_times'])
                    np.testing.assert_array_equal(past,a['past_slow']);np.testing.assert_array_equal(groups,a['groups'])
                    try:
                        again=kernel.trial(y,h,times,past,row['tolerance'],row['iteration'],residual_tol=reg['residual_tolerance'],maxiter=reg['maxiter'])
                    except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:
                        assert row['status']=='failed' and str(exc)==row['reason']
                        close(exc.stats['iteration_residuals'],row['stats']['iteration_residuals'])
                    else:
                        assert row['status']=='success'
                        np.testing.assert_allclose(again['fine'],a['endpoint'],rtol=1e-12,atol=1e-13)
                        for k,v in again['stats'].items():
                            if k!='seconds':close(v,row['stats'][k])
                        curve=unpack(a['lengths'],a['coefficients']);guess=a['implicit_guess'];end=a['endpoint']
                        forcing=EndpointPolynomial(np.r_[times,h],np.column_stack([past,kernel.slow(y),kernel.slow(guess)]),0.,h)
                        np.testing.assert_array_equal(forcing.coefficients,a['forcing'])
                        assert np.max(abs(end-guess)/case['scale'])<=reg['residual_tolerance']
                        check_curve(case,curve,y,end,ref,query,row['stats'],row['guard'],row['targets'])
                        max_quad=max(max_quad,modified_audit(kernel,curve,y,end,forcing,row['stats']))
                kernel.close()
        for target in ('0.0001','1e-06'):
            candidates=[r for r in data['runs'] if r['h']==h and r['status']=='success' and r['targets'][target]['passes']]
            baselines=[r for r in data['baselines'] if r['h']==h and r['targets'][target]['passes']]
            c=min(candidates,key=lambda r:r['seconds']) if candidates else None;b=min(baselines,key=lambda r:r['seconds'])
            decisions.append(dict(h=h,target=target,qualified=len(candidates),candidate=None if c is None else dict(name=c['name'],seconds=c['seconds'],active=c['active'],
                work={k:c['stats'][k] for k in ('subflows','nfev','njev','nlu','endpoint_njev','endpoint_nlu','endpoint_solves')}),baseline=dict(key=b['key'],seconds=b['seconds']),
                optimistic_single_run_ratio=b['seconds']/c['seconds'] if c else None))
        print('verified local h',h,flush=True)
    return dict(candidates=108,baselines=18,converged=sum(r['status']=='success' for r in data['runs']),
        qualified={t:sum(r['status']=='success' and r['targets'][t]['passes'] for r in data['runs']) for t in ('0.0001','1e-06')},
        failures=dict(collections.Counter(r['reason'] for r in data['runs'] if r['status']=='failed')),decisions=decisions,
        max_absolute_piecewise_modified_quadrature=max_quad,excluded_original_records=108,input_shape_failures_in_original=72)


def verify():
    result=dict(date='2026-11-08',stability=verify_stability(),local=verify_local(),M4_admission=False)
    for p,sha in read(ROOT/'results/m3-growth-history/manifest.json').items():assert digest(ROOT/p)==sha,p
    result.update(previous_manifest_unchanged=True,scope='Small-model local development, single timings excluding history/grouping/independent audit; no full-window evidence.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args();result=verify()
    if args.write:
        save(OUT/'summary.json',result)
        if not (OUT/'environment.json').exists():save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
        files=[p for d in (OUT,LOCAL) for p in d.iterdir() if p.is_file() and p.name!='manifest.json']
        files += [ROOT/'research'/p for p in ('implicit_history.py','test_implicit_history.py','implicit_flow.py','test_implicit_flow.py',
            'm3_implicit_stability.py','m3_implicit_local.py','m3_implicit_local_corrected.py','test_implicit_local_inputs.py','m3_implicit_report.py')]
        files += [ROOT/'docs/m3-implicit-history.md']
        save(OUT/'manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
