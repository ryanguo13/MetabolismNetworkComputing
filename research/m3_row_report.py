"""Reconstruct registered ROW predictors, distinct ledgers and physical outputs."""
import argparse
import collections
import json
import platform
import subprocess
import sys
import zipfile
import numpy as np
import scipy
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.row_spc import ROWSPC,coefficients
from research.row_continuous import ROWContinuous,coupling,PolynomialForcing
from research.spc_mri import AffineForcing
from research.extrapolated_flow import unpack
from research.inner_dense import InnerCurve
from research.mri_kernel import Hermite
from research.m3_growth_study import audit
from research.m3_ledger_flow_report import close,check_curve,modified,registration
from research.m3_row_spc_diagnostic import evaluate
from research.test_row_continuous import evaluate_order
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-row-spc'
DIRECTORIES=[OUT,ROOT/'results/m3-row-continuous',ROOT/'results/m3-row-continuous-growth']
def read(p):return json.loads(p.read_text())


def initial_registration():
    reg=read(OUT/'registration.json');rev=read(OUT/'test-revision.json')
    assert rev['original_path']=='research/test_row_spc.py'
    assert reg['source_sha256'][rev['original_path']]==rev['original_sha256']==digest(ROOT/rev['archived_source'])
    assert digest(ROOT/rev['original_path'])==rev['current_sha256']
    for p,sha in (reg['source_sha256']|reg['input_sha256']).items():
        if p!=rev['original_path']:assert digest(ROOT/p)==sha,p
    # Original failed test is preserved, not silently reclassified as passing.
    old=read(OUT/'initial-order.json');assert old['source_sha256']==rev['original_sha256']
    q=next(r for r in old['results']['nonlinear'] if not r['base_only'] and r['theta']==.5 and r['steps']==64)
    assert q['observed_order']<2.8
    return reg


def verify_study(directory,cases):
    cubic=directory!=OUT;reg=registration(directory/'registration.json') if cubic else initial_registration()
    data=read(directory/'runs.json');rows=data['runs'];expected=78 if directory.name=='m3-row-continuous' else 52 if cubic else 150
    assert len(rows)==expected and len({r['key'] for r in rows})==len(rows)
    expected_configs={(m,h,t,g) for m,v in reg['models'].items() for h in reg['steps'] for t in reg['tolerances'] for g in v['thresholds']}
    if not cubic:expected_configs={(*x,theta) for x in expected_configs for theta in reg['theta']}
    actual={(r['model'],r['h'],r['tolerance'],r['threshold'])+(() if cubic else (r['theta'],)) for r in rows if r['method']=='row-spc'}
    assert actual==expected_configs
    assert sum(r['method']=='radau' for r in rows)==12*len(reg['steps'])
    assert sum(r['method']=='row' for r in rows)==2*len(reg['steps'])
    decisions=[];prefix=np.load(ROOT/'results/m3-priority/p0.npz')
    for model,spec in reg['models'].items():
        case=cases[model];screen=GroupScreen(case['system'],case['scale']);origin=prefix[model+'-prefix'][:,3];kappa,_=screen.evaluate(origin)
        for h in reg['steps']:
            with np.load(directory/f'reference-{model}-{h}.npz') as refdata:
                y=refdata['initial'];np.testing.assert_array_equal(y,origin);query=refdata['query']
                start=spec['start'];np.testing.assert_array_equal(query,np.unique(np.r_[np.linspace(0,1,65),(case['times'][(case['times']>start)&(case['times']<start+h)]-start)/h]))
                ref,budget,_=audit(case,y,query,h);np.testing.assert_allclose(ref,refdata['reference'],rtol=2e-12,atol=2e-12)
            record=next(r for r in data['references'] if r['model']==model and r['h']==h)
            close(budget,record['budget']);assert record['reference_ok'] and max(budget.values())<=.05
            gradient=screen.gradient(y);jac=(case['system'].S@gradient).tocsc()
            for row in [r for r in rows if r['model']==model and r['h']==h]:
                kernel=None
                if row['method']!='radau':
                    groups=np.flatnonzero(1e-3*kappa>=row['threshold']) if row['method']=='row-spc' else np.array([],dtype=int)
                    np.testing.assert_array_equal(groups,row['groups']);kernel=(ROWContinuous if cubic else ROWSPC)(case,groups);assert len(kernel.active)==row['active']
                if row['status']=='failed':
                    assert kernel is not None
                    try:
                        if cubic:kernel.trial(y,h,row['tolerance'],base_only=row['method']=='row')
                        else:kernel.trial(y,h,row['tolerance'],row['theta'],base_only=row['method']=='row')
                    except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:assert str(exc)==row['reason']
                    else:raise AssertionError('Recorded integration failure did not reproduce')
                    finally:kernel.close()
                    continue
                with np.load(directory/f"{row['key']}.npz") as datafile:
                    z=datafile;end=z['endpoint'];curve=(InnerCurve.unpack if row['method']=='radau' else unpack)(z['lengths'],z['coefficients'])
                    check_curve(case,curve,y,end,ref,query,row['stats'],row['guard'],row['targets'])
                    if kernel is None:continue
                    a,g,b,mu=coupling() if cubic else coefficients(row['theta']);ks=z['slow_increments'];ki=z['increments'];stages=z['stages'];rates=z['rates'];full=z['full']
                    slowj=jac-kernel.sf@gradient[kernel.cols];residuals=[]
                    for i in range(4):
                        state=y+sum((a[i,k]*ki[:,k] for k in range(i)),start=np.zeros_like(y))
                        np.testing.assert_array_equal(state,stages[:,i]);assert np.all(state>=-1e-12*case['scale'])
                        np.testing.assert_array_equal(case['system'].flux(0.,state),rates[:,i]);np.testing.assert_array_equal(case['system'].S@rates[:,i],full[:,i])
                        previous=sum((g[i,k]*ki[:,k] for k in range(i)),start=np.zeros_like(y))
                        rhs=h*full[:,i]+h*(jac@previous)
                        from scipy.sparse import eye
                        residuals.append(float(np.max(abs((eye(len(y),format='csc')-h*g[i,i]*jac)@ki[:,i]-rhs)/case['scale'])))
                        expected_ks=h*(full[:,i]-kernel.sf@rates[kernel.cols,i])+h*(slowj@(previous+g[i,i]*ki[:,i]))
                        np.testing.assert_allclose(expected_ks,ks[:,i],rtol=2e-12,atol=2e-14)
                    close(residuals,row['stats']['linear_residuals'])
                    for key,value in dict(predictor_nfev=4,predictor_njev=1,predictor_nlu=1,predictor_solves=4).items():assert row['stats'][key]==value
                    if row['method']=='row':
                        np.testing.assert_allclose(end,y+ki@b,rtol=2e-12,atol=2e-14);assert row['stats']['subflows']==0
                    else:
                        assert row['stats']['subflows']==1
                        forcing=PolynomialForcing((ks@mu)*np.arange(1,4)/h,h) if cubic else AffineForcing(np.column_stack([ks@mu/h,np.zeros(len(y))]),h)
                        np.testing.assert_array_equal(forcing.coefficients,z['forcing']);raw=unpack(z['raw-lengths'],z['raw-coefficients']);rawend=z['raw_endpoint']
                        np.testing.assert_allclose(raw(np.array([0.,1.])),np.column_stack([y,rawend]),rtol=2e-12,atol=2e-12)
                        assert np.all(raw.minima()>=-1e-12*case['scale']) and raw.join_error(case['scale'])<=1e-10
                        modified(case,kernel,raw,y,rawend,forcing,row['stats'])
                        if cubic:
                            np.testing.assert_array_equal(rawend,end);np.testing.assert_array_equal(raw(query),curve(query))
                        else:np.testing.assert_allclose(end,rawend+ks@(b-mu),rtol=2e-12,atol=2e-14)
                    if not cubic:
                        hermite=Hermite(y,end,full[:,0],kernel.full(end),h)
                        np.testing.assert_allclose(curve(query),hermite(query),rtol=2e-12,atol=2e-12)
                    elif row['method']=='row':np.testing.assert_allclose(z['coefficients'][0],np.column_stack([y,ki@mu]),rtol=2e-12,atol=2e-14)
                if kernel is not None:kernel.close()
            for target in ('0.0001','1e-06'):
                eligible=[r for r in rows if r['model']==model and r['h']==h and r['status']=='success' and r['targets'][target]['passes']]
                candidates=[r for r in eligible if r['method']=='row-spc'];mature=[r for r in eligible if r['method']=='radau'];controls=[r for r in eligible if r['method']!='row-spc']
                c=min(candidates,key=lambda r:r['seconds']) if candidates else None;m=min(mature,key=lambda r:r['seconds']) if mature else None;b=min(controls,key=lambda r:r['seconds']) if controls else None
                decisions.append(dict(model=model,h=h,target=target,qualified_candidates=len(candidates),candidate=None if c is None else dict(key=c['key'],active=c['active'],seconds=c['seconds'],
                    predictor_seconds=c['stats']['predictor_seconds'],work={k:c['stats'][k] for k in ('predictor_nfev','predictor_njev','predictor_nlu','predictor_solves','nfev','njev','nlu','accepted_steps')}),
                    fastest_mature=None if m is None else dict(key=m['key'],seconds=m['seconds']),fastest_control=None if b is None else dict(key=b['key'],method=b['method'],seconds=b['seconds']),
                    optimistic_mature_ratio=m['seconds']/c['seconds'] if m and c else None,optimistic_control_ratio=b['seconds']/c['seconds'] if b and c else None))
            print('verified',directory.name,model,h,flush=True)
    return dict(records=len(rows),statuses=dict(collections.Counter(r['status'] for r in rows)),failures=dict(collections.Counter(r['reason'] for r in rows if r['status']=='failed')),
        candidate_successes=sum(r['method']=='row-spc' and r['status']=='success' for r in rows),
        qualified_candidates={t:sum(r['method']=='row-spc' and r['status']=='success' and r['targets'][t]['passes'] for r in rows) for t in ('0.0001','1e-06')},decisions=decisions)


def verify():
    reg=registration(OUT/'diagnostic-registration.json');diag=read(OUT/'diagnostic.json');close(evaluate(reg),diag)
    order=read(DIRECTORIES[1]/'order.json');close(evaluate_order(),order['results'])
    for p,sha in order['source_sha256'].items():assert digest(ROOT/p)==sha,p
    preg=registration(DIRECTORIES[1]/'profile-registration.json');profiles=read(DIRECTORIES[1]/'profile.json')['results']
    assert len(profiles)==6
    for spec,profile in zip(preg['configs'],profiles):
        for key,value in spec.items():assert profile[key]==value
        assert profile['endpoint_scaled_difference']<1e-12 and profile['instrumented_seconds']>0
    cases={n:prepare(n) for n in ('teusink','kecoli')};results={p.name:verify_study(p,cases) for p in DIRECTORIES}
    for p,sha in read(ROOT/'results/m3-spc/manifest.json').items():assert digest(ROOT/p)==sha,p
    for p,sha in read(ROOT/'results/m3-implicit-history/manifest.json').items():assert digest(ROOT/p)==sha,p
    count=0
    for folder in DIRECTORIES:
        for p in folder.glob('*.npz'):
            with zipfile.ZipFile(p) as z:assert z.testzip() is None,p
            count+=1
    return dict(date='2026-11-08',studies=results,nonlinear_order=diag['order'],continuous_order=order['results'],
        frozen_linear=dict(records=len(diag['stability']),passes=sum(r['passes_frozen_linear_screen'] for r in diag['stability'])),
        profiling_configurations=len(profiles),npz_crc_checked=count,historical_manifests_unchanged=True,M4_admission=False,
        scope='Single-run optimistic local timings only. Full registered windows, original formal targets, split coverage,7 repeats and1.5x fastest qualified mature baseline remain unfulfilled.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args();result=verify()
    if args.write:
        if (OUT/'manifest.json').exists():raise FileExistsError('Preserve completed evidence')
        save(OUT/'summary.json',result)
        save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
        files=[p for folder in DIRECTORIES for p in folder.iterdir() if p.is_file() and p.name!='manifest.json']
        files += [ROOT/'research'/p for p in ('row_spc.py','test_row_spc.py','m3_row_spc_study.py','m3_row_spc_diagnostic.py','row_continuous.py','test_row_continuous.py','m3_row_continuous_study.py','m3_row_continuous_growth.py','m3_row_profile.py','m3_row_report.py')]
        files += [ROOT/'docs/m3-row-spc.md']
        save(OUT/'manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
