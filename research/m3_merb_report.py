"""Replay MERB predictor screen and one-shift affine action diagnostics."""
import argparse
import json
import platform
import subprocess
import sys
import zipfile
import numpy as np
import scipy
from scipy.linalg import expm
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.merb_predictor import MERBPredictor
from research.test_merb_predictor import order_diagnostic
from research.rational_affine import action
from research.affine_partition import freeze
from research.extrapolated_flow import unpack
from research.inner_dense import InnerCurve,full_curve,physical_ledger
from research.fullstep_radau import fullstep_curve
from research.mri_kernel import GAUSS
from research.m3_growth_study import audit
from research.m3_priority import metrics
from research.m3_ordered_cost import structural
from research.m3_ledger_flow_report import registration,close,check_curve
from research.m3_collocation_report import timeless
from research.m3_merb_study import ARRAYS
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-merb-predictor';RATIONAL=ROOT/'results/m3-rational-affine'
def read(p):return json.loads(p.read_text())


def ledger(case,curve,y,end,j,f0,forcing):
    weights=case['options'].get('ledger_weights')
    if weights is None:return {}
    totals=[]
    for n in (8,16):
        q=(GAUSS[n][0]+1)/2;total=0.;left=0.
        for piece in curve.pieces:
            z=piece(q);t=left+piece.h*q
            values=weights@(j@(z-y[:,None])+f0[:,None]+forcing[:,None]*(t/curve.h)**2)
            total+=piece.h/2*float(GAUSS[n][1]@values);left+=piece.h
        totals.append(total)
    den=max(1.,abs(float(weights@y)))
    return dict(modified_ledger_residual=abs(float(weights@(end-y))-totals[1])/den,modified_ledger_quadrature=abs(totals[1]-totals[0])/den)


def tight(a,b):
    for k,v in a.items():np.testing.assert_allclose(v,b[k],rtol=2e-8,atol=2e-17)


def verify():
    # Validate reference provenance before using it for any comparison.
    for name in ('results/m3-affine-attribution/manifest.json','results/m3-subcycled-predictor/manifest.json'):
        for p,sha in read(ROOT/name).items():assert digest(ROOT/p)==sha,p
    reg=registration(OUT/'registration.json');data=read(OUT/'runs.json');close(order_diagnostic(),read(OUT/'order.json'))
    assert len(data['candidates'])==18 and len(data['controls'])==48 and len(data['references'])==6
    ratios=[]
    for mi,name in enumerate(reg['models']):
        case=prepare(name);kernel=MERBPredictor(case);screen=GroupScreen(case['system'],case['scale'])
        for hi,h in enumerate(reg['steps']):
            key=f'{mi}-{hi}';rr=next(r for r in data['references'] if r['key']==key)
            with np.load(OUT/f'reference-{key}.npz') as a:y=a['initial'];q=a['query'];savedref=a['reference']
            ref,budget,_=audit(case,y,q,h);close(budget,rr['budget']);assert max(budget.values())<=.05 and rr['reference_ok'];np.testing.assert_array_equal(ref,savedref)
            gradient=screen.gradient(y);j=(case['system'].S@gradient).tocsc();f0=case['system'].S@case['system'].flux(0,y)
            subset=[r for r in data['candidates'] if r['reference_key']==key];assert {r['tolerance'] for r in subset}==set(reg['tolerances'])
            for row in subset:
                assert row['status']=='success';r=kernel.trial(y,h,row['tolerance']);close(timeless(r['stats']),timeless(row['stats']))
                assert r['stats']['nonlinear_nfev']==2 and r['stats']['nonlinear_njev']==1 and r['stats']['subflows']==2
                assert r['stats']['lu_cache_misses']==r['stats']['nlu']
                with np.load(OUT/f"{row['key']}.npz") as a:
                    c=unpack(a['lengths'],a['coefficients']);stagecurve=unpack(a['stage_lengths'],a['stage_coefficients']);np.testing.assert_array_equal(c(q),r['pieces'][0](q));np.testing.assert_array_equal(a['endpoint'],r['fine'])
                    for k in ARRAYS:np.testing.assert_array_equal(a[k],r[k])
                    np.testing.assert_array_equal(stagecurve(q),r['stage_curve'](q))
                plain={k:{a:b for a,b in v.items() if a!='endpoint'} for k,v in row['targets'].items()}
                check_curve(case,c,y,r['fine'],ref,q,row['stats'],row['guard'],plain)
                for t,v in row['targets'].items():close(metrics(case,r['fine'][:,None],ref[:,-1:],float(t)),v['endpoint'])
                np.testing.assert_allclose(stagecurve(np.array([0.,1.])),np.column_stack([y,r['stage']]),rtol=2e-12,atol=1e-12)
                assert np.all(stagecurve.minima()>=-1e-12*case['scale']) and stagecurve.join_error(case['scale'])<=1e-10
                np.testing.assert_array_equal(r['remainder'],case['system'].flux(0,r['stage'])-r['initial_rates']-gradient@(r['stage']-y))
                np.testing.assert_allclose(r['forcing'],(16/9)*(case['system'].S@r['remainder']),rtol=2e-12,atol=1e-14)
                tight(physical_ledger(case,c,y,r['fine']),row['stats']);tight(ledger(case,c,y,r['fine'],j,f0,r['forcing']),row['stats'])
            controls=[r for r in data['controls'] if r['reference_key']==key]
            assert {(r['tolerance'],r['whole']) for r in controls}=={(t,w) for t in reg['controls']['tolerances'] for w in (False,True)}
            for row in controls:
                assert row['status']=='success';r=(fullstep_curve if row['whole'] else full_curve)(case,y,h,row['tolerance'],screen);close(timeless(r['stats']),timeless(row['stats']))
                with np.load(OUT/f"{row['key']}.npz") as a:
                    c=InnerCurve.unpack(a['lengths'],a['coefficients']);np.testing.assert_array_equal(c(q),r['pieces'][0](q));np.testing.assert_array_equal(a['endpoint'],r['fine'])
                check_curve(case,c,y,r['fine'],ref,q,row['stats'],row['guard'],row['targets']);tight(physical_ledger(case,c,y,r['fine']),row['stats'])
            for target in ('0.0001','1e-06'):
                best=min([r for r in controls if r['targets'][target]['passes']],key=lambda r:r['seconds'])
                for row in subset:ratios.append(dict(model=name,h=h,tolerance=row['tolerance'],target=target,passes=row['targets'][target]['passes'],actual=row['targets'][target]['actual'],endpoint=row['targets'][target]['endpoint'],physical_ledger=row['stats'].get('physical_ledger_residual'),candidate_seconds=row['seconds'],predictor_seconds=row['stats']['predictor_seconds'],baseline_seconds=best['seconds'],baseline_key=best['key'],ratio=best['seconds']/row['seconds'],predictor_ratio=best['seconds']/row['stats']['predictor_seconds']))
            print('verified MERB',name,h,flush=True)
    rreg=registration(RATIONAL/'registration.json');rows=read(RATIONAL/'runs.json')['runs'];assert len(rows)==60;rational=[]
    for mi,name in enumerate(rreg['models']):
        case=prepare(name);screen=GroupScreen(case['system'],case['scale'])
        for hi,h in enumerate(rreg['steps']):
            key=f'{mi}-{hi+1}';refrow=next(r for r in read(ROOT/'results/m3-affine-attribution/runs.json')['references'] if r['key']==key);assert refrow['reference_ok']
            with np.load(ROOT/f'results/m3-affine-attribution/reference-{key}.npz') as a:y=a['initial'];q=a['query'];reference=a['affine_reference']
            frozen,ctx=freeze(case,y);subset=[r for r in rows if r['model']==name and r['h']==h]
            assert {(r['dimension'],r['shift']) for r in subset}=={(m,s) for m in rreg['dimensions'] for s in rreg['shifts']}
            for row in subset:
                assert row['status']=='success';gradient=screen.gradient(y);j=(case['system'].S@gradient).tocsc();rhs=case['system'].S@case['system'].flux(0,y)
                r=action(j,rhs,y,case['scale'],h,q,row['dimension'],row['shift']);close(timeless(r['stats']),timeless(row['stats']))
                with np.load(RATIONAL/f"{row['key']}.npz") as a:
                    for k in ('values','derivatives','basis','projection','drive'):np.testing.assert_array_equal(r[k],a[k])
                actual=metrics(frozen,r['values'],reference,1e-6);close(actual,row['actual']);close(structural(frozen,r['values']),row['guard'])
                assert row['sampled_accurate']==(max(actual.values())<=1) and not row['M4_eligible']
                assert r['stats']['nlu']==1 and r['stats']['solves']<=min(len(y),row['dimension'])-1
                # Reconstruct projected J and sampled residual independently.
                v=r['basis'];scaledj=(j.multiply(case['scale'][None,:])).multiply(1/case['scale'][:,None])
                np.testing.assert_allclose(v.T@(scaledj@v),r['projection'],rtol=2e-12,atol=2e-8)
                # Independent direct expm at each query; no semigroup/cache
                # reuse. Avoid subtracting the large saved concentration
                # offset: the stiff J amplifies its float64 rounding error.
                m=v.shape[1];block=np.zeros((m+1,m+1));block[:m,:m]=h*r['projection'];block[:m,-1]=h*r['drive']
                coefficients=np.column_stack([(expm(t*block)@np.r_[np.zeros(m),1.])[:m] for t in q])
                independent_values=y[:,None]+case['scale'][:,None]*(v@coefficients)
                np.testing.assert_allclose(independent_values,r['values'],rtol=2e-10,atol=2e-12)
                residual=(v@r['projection']-scaledj@v)@coefficients+(v@r['drive']-rhs/case['scale'])[:,None]
                measured=float(np.max(abs(h*residual)))
                np.testing.assert_allclose(measured,row['stats']['residual_scaled'],rtol=1e-6,atol=3e-10)
                spectral=float(max(np.linalg.eigvals(r['projection']).real))
                rational.append(dict(key=row['key'],model=name,h=h,dimension=row['dimension'],actual_dimension=r['stats']['actual_dimension'],shift=row['shift'],actual=actual,sampled_accurate=row['sampled_accurate'],sampled_nonnegative=row['guard']['nonnegative'],sampled_pool=row['guard']['pool'],seconds=row['seconds'],residual_strict=r['stats']['residual_scaled']/1e-6,projected_spectral_abscissa=spectral))
            print('verified rational',name,h,flush=True)
    crc=0
    for directory in (OUT,RATIONAL):
        for p in directory.glob('*.npz'):
            with zipfile.ZipFile(p) as z:assert z.testzip() is None,p
            crc+=1
    return dict(merb_candidates=18,mature_controls=48,merb_passes={t:sum(r['targets'][t]['passes'] for r in data['candidates']) for t in ('0.0001','1e-06')},merb_comparisons=ratios,rational_records=rational,rational_samples_accurate=sum(r['sampled_accurate'] for r in rational),rational_sampled_negative=sum(not r['sampled_nonnegative'] for r in rational),npz_crc_checked=crc,historical_manifests_unchanged=True,M4_admission=False,
        conclusion='MERB4 nonlinear-call reduction alone fails larger-step accuracy/ledger and linear cost. One-shift rational affine actions offer potential cost headroom but nonnormal projection has nonmonotonic errors and spurious unstable modes; all outcomes retained. Need certified action control and nonlinear/continuous reconstruction before reaction-partition coupling. No reaction coverage or M4 admission claimed.')


def write(result):
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'summary.json',result);save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
    files=[p for d in (OUT,RATIONAL) for p in d.iterdir() if p.is_file() and p.name!='manifest.json']
    files += [ROOT/'research'/p for p in ('merb_predictor.py','test_merb_predictor.py','m3_merb_study.py','rational_affine.py','test_rational_affine.py','m3_rational_affine_study.py','m3_merb_report.py')]
    files += [ROOT/'docs/m3-merb-predictor.md']
    save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args();result=verify()
    if args.write:write(result)
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
