"""Reproduce algebra, frozen-linear maps, all three local screens and guards."""
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
from research.rodas_continuous import RodasContinuous
from research.rodas_increment import IncrementRodas
from research.rodas_centered import CenteredRodas
from research.rodas_coupling import RodasForcing,coupling
from research.m3_rodas_study import algebra_diagnostic,linear_diagnostic
from research.test_rodas_continuous import order_diagnostic
from research.extrapolated_flow import unpack
from research.inner_dense import InnerCurve,full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_growth_study import audit
from research.m3_ledger_flow_report import registration,close,check_curve,modified
from research.m3_collocation_report import timeless
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-rodas-continuous';RAW=ROOT/'results/m3-rodas-increment';CENTER=ROOT/'results/m3-rodas-centered'
def read(p):return json.loads(p.read_text())


def verify_folder(folder,cls,cases):
    reg=registration(folder/'registration.json');data=read(folder/'runs.json');assert len(data['candidates'])==48 and len(data['controls'])==64 and len(data['references'])==8
    records=[];counts=Counter();failure_reasons=Counter();paired=[]
    for mi,spec in enumerate(reg['models']):
        case=cases[spec['name']];y=np.load(ROOT/'results/m3-priority/p0.npz')[spec['name']+'-prefix'][:,3].copy();screen=GroupScreen(case['system'],case['scale']);kappa,_=screen.evaluate(y);groups=np.flatnonzero(1e-3*kappa>=spec['threshold']);kernel=cls(case,groups)
        for hi,h in enumerate(reg['steps']):
            key=f'{mi}-{hi}';public=case['times'][(case['times']>spec['start'])&(case['times']<spec['start']+h)]-spec['start'];q=np.unique(np.r_[np.linspace(0,1,65),public/h]);ref,budget,_=audit(case,y,q,h);assert max(budget.values())<=.05
            rr=next(r for r in data['references'] if r['key']==key);close(budget,rr['budget']);assert rr['groups']==len(groups) and rr['active']==len(kernel.active)
            with np.load(folder/f'reference-{key}.npz') as a:
                np.testing.assert_array_equal(y,a['initial']);np.testing.assert_array_equal(groups,a['groups']);np.testing.assert_array_equal(q,a['query']);np.testing.assert_allclose(ref,a['reference'],rtol=2e-12,atol=2e-12)
            controls=[r for r in data['controls'] if r['reference_key']==key];candidates=[r for r in data['candidates'] if r['reference_key']==key]
            assert {(r['whole'],r['tolerance']) for r in controls}=={(w,t) for w in (False,True) for t in reg['controls']['tolerances']}
            assert {(r['augmented'],r['whole'],r['tolerance']) for r in candidates}=={(c['augmented'],c['whole'],c['tolerance']) for c in reg['configurations']}
            for row in controls:
                assert row['status']=='success';r=(fullstep_curve if row['whole'] else full_curve)(case,y,h,row['tolerance'],screen);close(timeless(r['stats']),timeless(row['stats']))
                with np.load(folder/f"{row['key']}.npz") as a:
                    c=InnerCurve.unpack(a['lengths'],a['coefficients']);np.testing.assert_array_equal(a['endpoint'],r['fine']);np.testing.assert_array_equal(c(q),r['pieces'][0](q));check_curve(case,c,y,a['endpoint'],ref,q,row['stats'],row['guard'],row['targets'])
            for row in candidates:
                args=dict(base_only=row['whole'],augmented=row['augmented'])
                try:r=kernel.trial(y,h,row['tolerance'],**args)
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                    assert row['status']=='failed' and str(exc)==row['reason'];close(timeless(exc.stats),timeless(row['stats']));assert not (folder/f"{row['key']}.npz").exists();counts['failed']+=1;failure_reasons[str(exc)]+=1;continue
                assert row['status']=='success';counts['success']+=1;close(timeless(r['stats']),timeless(row['stats']))
                stages=9 if row['augmented'] else 6
                assert r['stats']['predictor_nlu']==r['stats']['predictor_njev']==1
                assert r['stats']['predictor_nfev']==r['stats']['predictor_solves']==stages
                ctx=r['rodas_linearization'];np.testing.assert_allclose(ctx['gamma'],.25,rtol=0.,atol=2e-15)
                np.testing.assert_allclose(ctx['matrix']@ctx['solve'](case['scale']),case['scale'],rtol=2e-10,atol=1e-11)
                with np.load(folder/f"{row['key']}.npz") as a:
                    c=unpack(a['lengths'],a['coefficients'])
                    np.testing.assert_array_equal(c(q),r['pieces'][0](q));np.testing.assert_array_equal(a['endpoint'],r['fine'])
                    for k in ('increments','slow_increments','stages','rates','full'):np.testing.assert_array_equal(a[k],r[k])
                    check_curve(case,c,y,a['endpoint'],ref,q,row['stats'],row['guard'],row['targets'])
                    if not row['whole']:
                        np.testing.assert_array_equal(a['forcing'],r['forcing_coefficients']);modified(case,kernel,c,y,r['fine'],RodasForcing(a['forcing'],h),r['stats'])
                if folder!=OUT:
                    with np.load(OUT/f"{row['key']}.npz") as a:
                        np.testing.assert_array_equal(a['increments'],r['increments']);np.testing.assert_array_equal(a['slow_increments'],r['slow_increments'])
                        if not row['whole']:np.testing.assert_array_equal(a['forcing'],r['forcing_coefficients'])
                for target in ('0.0001','1e-06'):counts['passes_'+target]+=row['targets'][target]['passes']
                if not row['whole'] and row['augmented']:
                    pool=case['options']['pools'];den=max(1e-12,float(np.max(abs(pool@y))));coefficient_pool=max(float(np.max(abs(pool@p.coefficients[:,1:])))/den for p in c.pieces)
                    paired.append(dict(model=spec['name'],h=h,tolerance=row['tolerance'],internal_pool=row['stats']['pool_error'],query_pool=row['guard']['pool'],coefficient_pool=coefficient_pool,actual=row['targets']['1e-06']['actual'],passes=row['targets']['1e-06']['passes'],physical_ledger=row['stats'].get('physical_ledger_residual')))
            for target in ('0.0001','1e-06'):
                mature=min((r for r in controls if r['targets'][target]['passes']),key=lambda r:r['seconds'])
                for augmented,whole in ((False,False),(True,False),(False,True),(True,True)):
                    good=[r for r in candidates if r['augmented']==augmented and r['whole']==whole and r['status']=='success' and r['targets'][target]['passes']]
                    best=min(good,key=lambda r:r['seconds']) if good else None
                    records.append(dict(model=spec['name'],h=h,target=target,augmented=augmented,whole=whole,qualified=len(good),candidate=None if best is None else best['key'],candidate_seconds=None if best is None else best['seconds'],mature=mature['key'],mature_seconds=mature['seconds'],ratio=None if best is None else mature['seconds']/best['seconds']))
        kernel.close();print('verified',folder.name,spec['name'],flush=True)
    return dict(counts=dict(counts),failure_reasons=dict(failure_reasons),comparisons=records,pool_diagnostics=paired)


def verify():
    algebra=algebra_diagnostic();close(algebra,read(OUT/'algebra.json'));order=order_diagnostic();close(order,read(OUT/'order.json'));linear=linear_diagnostic();close(linear,read(OUT/'linear.json'))
    cases={n:prepare(n) for n in ('teusink','kecoli')};result=dict(direct=verify_folder(OUT,RodasContinuous,cases),raw_increment=verify_folder(RAW,IncrementRodas,cases),centered=verify_folder(CENTER,CenteredRodas,cases))
    for p,sha in read(ROOT/'results/m3-collocation-correction/manifest.json').items():assert digest(ROOT/p)==sha,p
    crc=0
    for folder in (OUT,RAW,CENTER):
        for p in folder.glob('*.npz'):
            with zipfile.ZipFile(p) as z:assert z.testzip() is None,p
            crc+=1
    result.update(algebra=dict(base_rank=algebra['base_rank'],base_target_rank=algebra['base_target_rank'],two_stage_valid=sum(r['valid'] for r in algebra['two_stage_candidates']),three_stage_valid=sum(r['valid'] for r in algebra['three_stage_candidates']),selected=algebra['selected']),order_cases=len(order),linear_cases=len(linear),linear_unstable=sum(r['spectral_radius']>1+1e-8 for r in linear),linear_negative_samples=sum(r['minimum_sampled'] < -1e-12 for r in linear),npz_crc_checked=crc,historical_manifest_unchanged=True,M4_admission=False,scope='Local continuous extension and equivalent-variable screens only; original full-window/formal-target/coverage/7-repeat/1.5-speedup requirements remain unfulfilled.')
    return result


def write(result):
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'summary.json',result);save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
    files=[p for d in (OUT,RAW,CENTER) for p in d.iterdir() if p.is_file() and p.name!='manifest.json']
    files += [ROOT/'research'/p for p in ('rodas_coupling.py','rodas_continuous.py','rodas_increment.py','rodas_centered.py','test_rodas_continuous.py','test_rodas_increment.py','test_rodas_centered.py','m3_rodas_study.py','m3_rodas_increment.py','m3_rodas_centered.py','m3_rodas_report.py','licenses/OrdinaryDiffEq-Rodas.md')]
    files += [ROOT/'docs/m3-rodas-continuous.md']
    save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args();result=verify()
    if args.write:write(result)
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
