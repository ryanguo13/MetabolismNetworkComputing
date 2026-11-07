"""Verify SPC stages, forcing, raw/final curves, audits and fair comparisons."""
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
from research.spc_mri import SPC,AffineForcing,tableau
from research.extrapolated_flow import unpack
from research.inner_dense import InnerCurve
from research.mri_kernel import Hermite
from research.m3_growth_study import audit
from research.m3_ledger_flow_report import close,registration,check_curve,modified
from research.m3_spc_order import evaluate as order_check
from research.spc_residual_diagnostic import trace
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-spc';EXTRA=ROOT/'results/m3-spc-solver-tolerance'
def read(p):return json.loads(p.read_text())
def sources(record):
    for p,sha in record['source_sha256'].items():assert digest(ROOT/p)==sha,p


def verify_study(directory,cases,predictions):
    reg=registration(directory/'registration.json');data=read(directory/'runs.json');rows=data['runs'];extra=directory==EXTRA
    assert len(rows)==(174 if extra else 192)
    assert collections.Counter(r['method'] for r in rows)==({'spc':144,'sdirk':12,'radau':18} if extra else {'spc':144,'sdirk':12,'radau':36})
    assert len({r['key'] for r in rows})==len(rows)
    if extra:
        expected={(h,g,q,o,t,p) for h in reg['steps'] for g in reg['thresholds'] for q in reg['orders'] for o in reg['outputs'] for t in reg['inner_tolerances'] for p in reg['predictor_residuals']}
        assert {(r['h'],r['threshold'],r['order'],r['output'],r['tolerance'],r['predictor_residual']) for r in rows if r['method']=='spc'}==expected
    else:
        expected={(m,h,g,q,o,t) for m,v in reg['models'].items() for h in reg['steps'] for g in v['thresholds'] for q in reg['orders'] for o in reg['outputs'] for t in reg['tolerances']}
        assert {(r['model'],r['h'],r['threshold'],r['order'],r['output'],r['tolerance']) for r in rows if r['method']=='spc'}==expected
    decisions=[];prefix=np.load(ROOT/'results/m3-priority/p0.npz')
    for model in ('kecoli',) if extra else ('teusink','kecoli'):
        case=cases[model];screen=GroupScreen(case['system'],case['scale']);origin=prefix[model+'-prefix'][:,3];kappa,_=screen.evaluate(origin)
        for h in reg['steps']:
            filename=f'reference-{h}.npz' if extra else f'reference-{model}-{h}.npz'
            with np.load(directory/filename) as base:
                y=base['initial'];np.testing.assert_array_equal(y,origin);query=base['query']
                start=.1 if model=='kecoli' else 1.
                np.testing.assert_array_equal(query,np.unique(np.r_[np.linspace(0,1,65),(case['times'][(case['times']>start)&(case['times']<start+h)]-start)/h]))
                ref,budget,_=audit(case,y,query,h);np.testing.assert_allclose(ref,base['reference'],rtol=2e-12,atol=1e-13)
                rr=next(r for r in data['references'] if r['h']==h and (extra or r['model']==model));close(budget,rr['budget'])
                assert rr['reference_ok']==(max(budget.values())<=.05)
                # Every registered reference in these completed batches certified.
                assert rr['reference_ok']
                for row in [r for r in rows if r['model']==model and r['h']==h]:
                    kernel=None;pred=None
                    if row['method']!='radau':
                        groups=np.flatnonzero(1e-3*kappa>=row['threshold']) if row['method']=='spc' else np.array([],dtype=int)
                        np.testing.assert_array_equal(groups,row['groups']);kernel=SPC(case,groups);assert len(kernel.active)==row['active']
                        tolerance=row['predictor_residual'] if extra else reg['predictor_residual'];cachekey=(model,h,row['order'],tolerance)
                        if cachekey not in predictions:
                            try:predictions[cachekey]=(True,kernel.predictor(y,h,row['order'],residual_tol=tolerance))
                            except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:predictions[cachekey]=(False,str(exc))
                        success,pred=predictions[cachekey]
                        if not success:
                            assert row['status']=='failed' and pred==row['reason'];kernel.close();continue
                    assert row['status']=='success'
                    with np.load(directory/f"{row['key']}.npz") as a:
                        curve=(InnerCurve.unpack if row['method']=='radau' else unpack)(a['lengths'],a['coefficients']);end=a['endpoint']
                        check_curve(case,curve,y,end,ref,query,row['stats'],row['guard'],row['targets'])
                        if pred is not None:
                            for key in ('stages','full','rates'):np.testing.assert_array_equal(pred[key],a['predictor_'+key])
                            np.testing.assert_array_equal(pred['initial_rhs'],a['initial_rhs'])
                            for key in ('predictor_nfev','predictor_njev','predictor_nlu','predictor_solves','predictor_residuals','predictor_iterations'):close(pred['stats'][key],row['stats'][key])
                            aa,gamma=tableau(row['order']);stages=a['predictor_stages'];fulls=a['predictor_full'];rates=a['predictor_rates']
                            for i,z in enumerate(stages.T):
                                np.testing.assert_array_equal(case['system'].flux(0.,z),rates[:,i])
                                np.testing.assert_array_equal(case['system'].S@rates[:,i],fulls[:,i])
                                base=y+h*sum((aa[i,j]*fulls[:,j] for j in range(i)),start=np.zeros_like(y))
                                residual=float(np.max(abs(base+h*aa[i,i]*fulls[:,i]-z)/case['scale']))
                                close(residual,row['stats']['predictor_residuals'][i]);assert residual<=tolerance
                                assert np.all(z>=-1e-12*case['scale'])
                            if row['method']=='sdirk':np.testing.assert_array_equal(stages[:,-1],end);assert row['stats']['subflows']==0
                            else:
                                assert row['stats']['subflows']==1
                                coeff=(fulls-kernel.sf@rates[kernel.cols])@gamma.T
                                np.testing.assert_array_equal(coeff,a['forcing']);forcing=AffineForcing(coeff,h)
                                raw=unpack(a['raw-lengths'],a['raw-coefficients'])
                                np.testing.assert_allclose(raw(np.array([0.,1.])),np.column_stack([y,end]),rtol=2e-12,atol=1e-12)
                                assert np.all(raw.minima()>=-1e-12*case['scale']) and raw.join_error(case['scale'])<=1e-10
                                modified(case,kernel,raw,y,end,forcing,row['stats'])
                                if row['output']=='inner':np.testing.assert_array_equal(curve(query),raw(query))
                            if row['output']=='hermite':
                                hermite=Hermite(y,end,case['system'].S@case['system'].flux(0.,y),case['system'].S@case['system'].flux(0.,end),h)
                                np.testing.assert_allclose(curve(query),hermite(query),rtol=2e-12,atol=1e-12)
                    if kernel is not None:kernel.close()
            for target in ('0.0001','1e-06'):
                eligible=[r for r in rows if r['model']==model and r['h']==h and r['status']=='success' and r['targets'][target]['passes']]
                cand=[r for r in eligible if r['method']=='spc'];radau=[r for r in eligible if r['method']=='radau'];controls=[r for r in eligible if r['method']!='spc']
                c=min(cand,key=lambda r:r['seconds']) if cand else None;b=min(controls,key=lambda r:r['seconds']) if controls else None;m=min(radau,key=lambda r:r['seconds']) if radau else None
                decisions.append(dict(model=model,h=h,target=target,qualified_candidates=len(cand),
                    candidate=None if c is None else dict(key=c['key'],order=c['order'],output=c['output'],active=c['active'],seconds=c['seconds'],
                        work={k:c['stats'][k] for k in ('predictor_nfev','predictor_njev','predictor_nlu','predictor_seconds','nfev','njev','nlu','subflows','accepted_steps')}),
                    fastest_control=None if b is None else dict(key=b['key'],method=b['method'],seconds=b['seconds']),
                    fastest_mature=None if m is None else dict(key=m['key'],method=m['method'],seconds=m['seconds']),
                    optimistic_control_ratio=b['seconds']/c['seconds'] if b and c else None,optimistic_mature_ratio=m['seconds']/c['seconds'] if m and c else None,
                    fixed_work_zero_corrector_ratio=m['seconds']/c['stats']['predictor_seconds'] if m and c else None))
            print('verified',directory.name,model,h,flush=True)
    return dict(records=len(rows),candidate_successes=sum(r['method']=='spc' and r['status']=='success' for r in rows),
        qualified_candidates={t:sum(r['method']=='spc' and r['status']=='success' and r['targets'][t]['passes'] for r in rows) for t in ('0.0001','1e-06')},
        failures=dict(collections.Counter(r['reason'] for r in rows if r['status']=='failed')),decisions=decisions)


def verify():
    order=read(OUT/'order.json');sources(order);close(order_check(),order['results'])
    traced=read(OUT/'predictor-trace.json');sources(traced);close(trace(),traced['runs'])
    cases={name:prepare(name) for name in ('teusink','kecoli')};predictions={}
    result=dict(date='2026-11-08',order=order['results'],initial=verify_study(OUT,cases,predictions),solver_tolerance=verify_study(EXTRA,cases,predictions),M4_admission=False)
    for p,sha in read(ROOT/'results/m3-implicit-history/manifest.json').items():assert digest(ROOT/p)==sha,p
    result.update(previous_manifest_unchanged=True,scope='Two-model local development with single optimistic timings; no full-window, coverage or formal repeat qualification.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args();result=verify()
    if args.write:
        save(OUT/'summary.json',result)
        if not (OUT/'environment.json').exists():save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
        files=[p for d in (OUT,EXTRA) for p in d.iterdir() if p.is_file() and p.name!='manifest.json']
        files += [ROOT/'research'/p for p in ('spc_mri.py','test_spc_mri.py','m3_spc_study.py','spc_residual_diagnostic.py','m3_spc_solver_tolerance.py','m3_spc_order.py','m3_spc_report.py')]
        files += [ROOT/'docs/m3-spc.md']
        save(OUT/'manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
