"""Replay correction equations, all local trials and the baseline challenge."""
import argparse
import json
import platform
import subprocess
import sys
import zipfile
from collections import Counter
import numpy as np
import scipy
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.collocation_correction import CorrectedROW,tableau
from research.extrapolated_flow import DensePower,pack,unpack
from research.inner_dense import InnerCurve,full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_growth_study import audit
from research.m3_ledger_flow_report import registration,close,check_curve
from research.m3_shadow_study import ROOT,digest
from research.mri_kernel import GAUSS

OUT=ROOT/'results/m3-collocation-correction'
BROKEN=ROOT/'results/m3-collocation-tolerance'
FIXED=ROOT/'results/m3-collocation-tolerance-fixed'
FAIR=ROOT/'results/m3-collocation-fairness'
def read(p):return json.loads(p.read_text())

def timeless(value):
    if isinstance(value,dict):return {k:timeless(v) for k,v in value.items() if not k.endswith('seconds')}
    if isinstance(value,list):return [timeless(v) for v in value]
    return value


def correction_check(case,r,initial,h,stages):
    c,a,l,integ,*_=tableau(stages);curve=r['pieces'][0];rates=r['corrected_rates']
    np.testing.assert_allclose(curve(c),r['stage_values'],rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(curve(c),initial[:,None]+h*rates@a.T,rtol=2e-12,atol=2e-12)
    q=np.linspace(0,1,33);np.testing.assert_allclose(curve.derivative(q),rates@l.T@np.vander(q,stages,increasing=True).T,rtol=2e-11,atol=2e-11)
    weights=case['options'].get('ledger_weights')
    if weights is not None:
        ints=[]
        for n in (4,8):
            q=(GAUSS[n][0]+1)/2;rhs=rates@l.T@np.vander(q,stages,increasing=True).T;ints.append(h/2*float(GAUSS[n][1]@(weights@rhs)))
        den=max(1.,abs(float(weights@initial)))
        close(abs(float(weights@(r['fine']-initial))-ints[1])/den,r['stats']['modified_ledger_residual'])
        close(abs(ints[1]-ints[0])/den,r['stats']['modified_ledger_quadrature'])


def verify_local(cases):
    reg=registration(OUT/'registration.json');data=read(OUT/'runs.json');assert len(data['references'])==6 and len(data['candidates'])==48 and len(data['controls'])==36
    comparisons=[]
    for mi,spec in enumerate(reg['models']):
        case=cases[spec['name']];y=np.load(ROOT/'results/m3-priority/p0.npz')[spec['name']+'-prefix'][:,3].copy();screen=GroupScreen(case['system'],case['scale']);kappa,_=screen.evaluate(y);groups=np.flatnonzero(1e-3*kappa>=spec['threshold']);kernel=CorrectedROW(case,groups)
        for hi,h in enumerate(reg['steps']):
            key=f'{mi}-{hi}';public=case['times'][(case['times']>spec['start'])&(case['times']<spec['start']+h)]-spec['start'];q=np.unique(np.r_[np.linspace(0,1,65),public/h]);ref,budget,_=audit(case,y,q,h);assert max(budget.values())<=.05
            rr=next(x for x in data['references'] if x['key']==key);close(budget,rr['budget']);assert rr['active']==len(kernel.active) and rr['groups']==len(groups)
            with np.load(OUT/f'reference-{key}.npz') as a:
                np.testing.assert_array_equal(a['initial'],y);np.testing.assert_array_equal(a['groups'],groups);np.testing.assert_array_equal(a['query'],q);np.testing.assert_allclose(a['reference'],ref,rtol=2e-12,atol=2e-12)
            candidates=[x for x in data['candidates'] if x['reference_key']==key];controls=[x for x in data['controls'] if x['reference_key']==key]
            assert {(r['predictor'],r['stages'],r['sweeps']) for r in candidates}=={(p,s,k) for p in reg['predictors'] for s in reg['stages'] for k in reg['sweeps']}
            assert {(r['tolerance'],r['whole']) for r in controls}=={(t,w) for t in (1e-6,1e-8,1e-10) for w in (False,True)}
            for row in candidates:
                assert row['status']=='success'
                r=kernel.trial(y,h,reg['inner_tolerance'],row['stages'],row['sweeps'],row['predictor']=='row');correction_check(case,r,y,h,row['stages']);close(timeless(r['stats']),timeless(row['stats']))
                with np.load(OUT/f"{row['key']}.npz") as a:
                    c=unpack(a['lengths'],a['coefficients']);np.testing.assert_array_equal(a['endpoint'],r['fine']);np.testing.assert_array_equal(a['stage_values'],r['stage_values']);np.testing.assert_array_equal(a['corrected_rates'],r['corrected_rates']);np.testing.assert_array_equal(c(q),r['pieces'][0](q))
                    pc=unpack(a['predictor_lengths'],a['predictor_coefficients']);np.testing.assert_array_equal(a['predictor_endpoint'],r['predictor']['fine']);np.testing.assert_array_equal(pc(q),r['predictor']['pieces'][0](q))
                    check_curve(case,pc,y,a['predictor_endpoint'],ref,q,row['stats']['predictor_stats'],row['predictor_guard'],row['predictor_targets'])
                    check_curve(case,c,y,a['endpoint'],ref,q,row['stats'],row['guard'],row['targets'])
            for row in controls:
                assert row['status']=='success';r=(fullstep_curve if row['whole'] else full_curve)(case,y,h,row['tolerance'],screen);close(timeless(r['stats']),timeless(row['stats']))
                with np.load(OUT/f"{row['key']}.npz") as a:
                    c=InnerCurve.unpack(a['lengths'],a['coefficients']);np.testing.assert_array_equal(a['endpoint'],r['fine']);np.testing.assert_array_equal(c(q),r['pieces'][0](q));check_curve(case,c,y,a['endpoint'],ref,q,row['stats'],row['guard'],row['targets'])
            for target in ('0.0001','1e-06'):
                mature=min([r for r in controls if r['targets'][target]['passes']],key=lambda r:r['seconds'])
                for seed in ('mri','row'):
                    good=[r for r in candidates if r['predictor']==seed and r['targets'][target]['passes']];best=min(good,key=lambda r:r['seconds']) if good else None
                    comparisons.append(dict(model=spec['name'],h=h,target=target,predictor=seed,qualified=len(good),best_key=None if best is None else best['key'],best_seconds=None if best is None else best['seconds'],mature_key=mature['key'],mature_seconds=mature['seconds'],local_ratio=None if best is None else mature['seconds']/best['seconds']))
        kernel.close();print('verified local',spec['name'],flush=True)
    return dict(candidates=48,controls=36,passes={t:sum(r['targets'][t]['passes'] for r in data['candidates']) for t in ('0.0001','1e-06')},comparisons=comparisons)


def verify_followup(directory,cases):
    reg=registration(directory/'registration.json');data=read(directory/'runs.json');expected=len(reg['models'])*len(reg['methods'])*len(reg['repeats']);assert len(data['runs'])==expected;broken=directory==BROKEN;summaries=[];failures=0
    for mi,spec in enumerate(reg['models']):
        case=cases[spec['name']];h=reg['h'];screen=GroupScreen(case['system'],case['scale'])
        with np.load(OUT/f"reference-{spec['reference_key']}.npz") as a:y=a['initial'];groups=a['groups'];q=a['query']
        ref,budget,_=audit(case,y,q,h);assert max(budget.values())<=.05;close(budget,data['references'][mi]['budget']);kernel=CorrectedROW(case,groups)
        with np.load(directory/f'reference-{mi}.npz') as a:
            np.testing.assert_array_equal(y,a['initial']);np.testing.assert_array_equal(groups,a['groups']);np.testing.assert_array_equal(q,a['query']);np.testing.assert_allclose(ref,a['reference'],rtol=2e-12,atol=2e-12)
        for rep in reg['repeats']:
            rows=[r for r in data['runs'] if r['index']==mi and r['repeat']==rep];shift=rep+1;methods=reg['methods'][shift:]+reg['methods'][:shift]
            assert [r['method'] for r in rows]==methods and [r['position'] for r in rows]==list(range(len(methods)))
            for row in rows:
                method=row['method']
                if broken and method.startswith('radau'):
                    try:_,tol,initial=method.split('-')
                    except ValueError as exc:assert row['status']=='failed' and row['reason']==str(exc)
                    else:raise AssertionError('Expected recorded dispatch failure')
                    assert not (directory/f"{row['key']}.npz").exists();failures+=1;continue
                assert row['status']=='success'
                if method.startswith('radau'):
                    tol,initial=method.removeprefix('radau-').rsplit('-',1);r=(fullstep_curve if initial=='whole' else full_curve)(case,y,h,float(tol),screen)
                else:
                    tol=float(method.removeprefix('mri-')) if method.startswith('mri-') else 1e-8;r=kernel.trial(y,h,tol,spec['stages'],reg['sweeps'],method=='row');correction_check(case,r,y,h,spec['stages'])
                close(timeless(r['stats']),timeless(row['stats']))
                with np.load(directory/f"{row['key']}.npz") as a:
                    c=unpack(a['lengths'],a['coefficients']);np.testing.assert_array_equal(a['endpoint'],r['fine']);np.testing.assert_allclose(c(q),r['pieces'][0](q),rtol=2e-13,atol=2e-13);check_curve(case,c,y,a['endpoint'],ref,q,row['stats'],row['guard'],row['targets'])
        for target in ('0.0001','1e-06'):
            timings=[]
            for method in reg['methods']:
                rs=[r for r in data['runs'] if r['index']==mi and r['repeat']>=0 and r['method']==method];assert len(rs)==3
                valid=all(r['status']=='success' and r['targets'][target]['passes'] for r in rs)
                timings.append(dict(method=method,qualified=valid,median_seconds=float(np.median([r['seconds'] for r in rs])),range_seconds=[min(r['seconds'] for r in rs),max(r['seconds'] for r in rs)]))
            good=[t for t in timings if t['method'].startswith('radau') and t['qualified']]
            best=min(good,key=lambda x:x['median_seconds']) if good else None
            for t in timings:t['ratio_to_best_mature']=None if broken or best is None or not t['qualified'] else best['median_seconds']/t['median_seconds']
            summaries.append(dict(model=spec['name'],target=target,timings=timings,best_mature=None if best is None else best['method'],excluded_from_cost_comparison=broken))
        kernel.close();print('verified',directory.name,spec['name'],flush=True)
    assert failures==(32 if broken else 0)
    return dict(records=expected,dispatch_failures=failures,summaries=summaries)


def verify():
    cases={n:prepare(n) for n in ('teusink','kecoli')};result=dict(local=verify_local(cases),first_tolerance=verify_followup(BROKEN,cases),corrected_tolerance=verify_followup(FIXED,cases),fairness=verify_followup(FAIR,cases))
    for p,sha in read(ROOT/'results/m3-checkpoint-controller/manifest.json').items():assert digest(ROOT/p)==sha,p
    crc=0
    for folder in (OUT,BROKEN,FIXED,FAIR):
        for p in folder.glob('*.npz'):
            with zipfile.ZipFile(p) as z:assert z.testzip() is None,p
            crc+=1
    result.update(npz_crc_checked=crc,historical_manifest_unchanged=True,M4_admission=False,scope='Local continuous-correction/cost experiments only. Original full windows, target accuracy, split coverage,7 formal repeats and1.5 speedup gates unfulfilled.')
    return result


def write(result):
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'summary.json',result);save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
    files=[p for d in (OUT,BROKEN,FIXED,FAIR) for p in d.iterdir() if p.is_file() and p.name!='manifest.json']
    files += [ROOT/'research'/p for p in ('collocation_correction.py','test_collocation_correction.py','test_collocation_guards.py','m3_collocation_study.py','m3_collocation_tolerance.py','m3_collocation_tolerance_fixed.py','m3_collocation_fairness.py','m3_collocation_report.py')]
    files += [ROOT/'docs/m3-collocation-correction.md']
    save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args();result=verify()
    if args.write:write(result)
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
