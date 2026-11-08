"""Replay real-partition affine attribution without altering acceptance gates."""
import argparse
import json
import platform
import subprocess
import sys
import zipfile
import numpy as np
import scipy
from metabolism.m2_experiments import save,error
from metabolism.m3_continuation import prepare
from research.affine_partition import freeze,displacement_reference,affine_predictor,decomposition
from research.rodas_centered import CenteredRodas
from research.rodas_coupling import RodasForcing
from research.extrapolated_flow import unpack
from research.m3_growth_study import audit
from research.m3_priority import metrics
from research.m3_ledger_flow_report import registration,close,check_curve,modified
from research.m3_collocation_report import timeless
from research.m3_affine_ledger import signed_ledger
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-affine-attribution';LEDGER=ROOT/'results/m3-affine-ledger';OLD=ROOT/'results/m3-rodas-centered'
def read(p):return json.loads(p.read_text())


def verify():
    reg=registration(OUT/'registration.json');data=read(OUT/'runs.json');oldreg=registration(OLD/'registration.json');old=read(OLD/'runs.json')['candidates'];lreg=registration(LEDGER/'registration.json');ledgerrows=read(LEDGER/'runs.json')['runs']
    assert len(data['candidates'])==32 and len(data['references'])==8 and len(ledgerrows)==16
    assert reg['models']==oldreg['models'] and reg['steps']==oldreg['steps']
    summaries=[];ledger_summary=[];budgets=[];largest_inner=0.;counts=0
    for mi,spec in enumerate(reg['models']):
        case=prepare(spec['name'])
        for hi,h in enumerate(reg['steps']):
            key=f'{mi}-{hi}';rr=next(r for r in data['references'] if r['key']==key)
            with np.load(OLD/f'reference-{key}.npz') as a:y=a['initial'];groups=a['groups'];q=a['query']
            frozen,ctx=freeze(case,y);kernel=CenteredRodas(frozen,groups);nonlinear_kernel=CenteredRodas(case,groups)
            nr,nb,_=audit(case,y,q,h);lr,lw=displacement_reference(ctx['jacobian'],lambda t:ctx['rhs'],y,h,q,case['scale'],1e-12);lo,lw2=displacement_reference(ctx['jacobian'],lambda t:ctx['rhs'],y,h,q,case['scale'],1e-11);ir,ib,_=audit(frozen,y,q,h)
            lb=metrics(frozen,lo,lr,1e-6);cross=metrics(frozen,ir,lr,1e-6);refok=max([*nb.values(),*lb.values(),*ib.values(),*cross.values()])<=.05
            assert refok==rr['reference_ok'] and refok
            for k,v in [('nonlinear_budget',nb),('affine_budget',lb),('independent_budget',ib),('cross_budget',cross)]:close(v,rr[k])
            close(timeless(lw),timeless(rr['affine_work']));close(timeless(lw2),timeless(rr['affine_loose_work']))
            den=1e-6*(.01*case['scale'][:,None]+np.maximum(abs(nr),case['scale'][:,None]));residual=np.column_stack([case['system'].S@(case['system'].flux(0,z)-frozen['system'].flux(0,z)) for z in nr.T])
            close(float(np.max(abs(h*residual/den))),rr['scaled_rhs_remainder']);close(error(lr,nr,case['scale'],1e-6),rr['truth_gap']);close(float(lr.min()),rr['affine_minimum'])
            assert rr['affine_negative_rate_samples']==sum(np.count_nonzero(frozen['system'].flux(0,z)<0) for z in lr.T)
            assert rr['groups']==len(groups) and rr['active']==len(kernel.active) and rr['jacobian_nnz']==ctx['jacobian'].nnz
            with np.load(OUT/f'reference-{key}.npz') as a:
                np.testing.assert_array_equal(y,a['initial']);np.testing.assert_array_equal(groups,a['groups']);np.testing.assert_array_equal(q,a['query']);np.testing.assert_array_equal(ctx['rates'],a['rates'])
                for k in ('data','indices','indptr'):np.testing.assert_array_equal(getattr(ctx['gradient'],k),a['gradient_'+k])
                np.testing.assert_array_equal(ctx['gradient'].shape,a['gradient_shape'])
                for k,v in [('nonlinear_reference',nr),('affine_reference',lr),('affine_loose',lo),('independent_reference',ir)]:np.testing.assert_allclose(a[k],v,rtol=2e-12,atol=2e-12)
            rows=[r for r in data['candidates'] if r['reference_key']==key];assert {(r['augmented'],r['whole']) for r in rows}=={(a,w) for a in (False,True) for w in (False,True)}
            for augmented in reg['augmented']:
                pred=affine_predictor(ctx,kernel.sf,kernel.cols,h,augmented);forcing=RodasForcing(pred['forcing'],h)
                oracle,ow=displacement_reference(pred['fastj'],lambda t:pred['fast']+forcing.value(t),y,h,q,case['scale'],1e-12)
                oracle_loose,ow2=displacement_reference(pred['fastj'],lambda t:pred['fast']+forcing.value(t),y,h,q,case['scale'],1e-11)
                ob=metrics(frozen,oracle_loose,oracle,1e-6);assert max(ob.values())<=.05
                for whole in reg['whole']:
                    row=next(r for r in rows if r['augmented']==augmented and r['whole']==whole);original=next(r for r in old if r['key']==row['nonlinear_key']);assert original['reference_key']==key and original['augmented']==augmented and original['whole']==whole and original['tolerance']==(1e-6 if whole else reg['inner_tolerance'])
                    nonlinear=nonlinear_kernel.trial(y,h,original['tolerance'],base_only=whole,augmented=augmented)
                    with np.load(OLD/f"{row['nonlinear_key']}.npz") as a:
                        nc=unpack(a['lengths'],a['coefficients']);nv=nc(q);ne=a['endpoint'];np.testing.assert_array_equal(ne,nonlinear['fine']);np.testing.assert_array_equal(nv,nonlinear['pieces'][0](q));check_curve(case,nc,y,ne,nr,q,original['stats'],original['guard'],original['targets'])
                    close(metrics(case,nv,nr,1e-6),row['nonlinear_actual']);close(original['targets'],row['nonlinear_targets']);close(ob,row['oracle_budget']);assert row['reference_ok']==refok
                    close(timeless(ow),timeless(row['oracle_work']));close(timeless(ow2),timeless(row['oracle_loose_work']))
                    assert row['status']=='success';r=kernel.trial(y,h,reg['inner_tolerance'],base_only=whole,augmented=augmented);curve=r['pieces'][0];values=curve(q);close(timeless(r['stats']),timeless(row['stats']))
                    close(float(np.max(abs(r['increments']-pred['increments'])/case['scale'][:,None])),row['stage_roundoff'])
                    # This is an independently reordered stage calculation,
                    # not the ledger residual. Bound its difference by .001
                    # of the strict concentration scale; keep measured values.
                    assert row['stage_roundoff']/1e-6<.001
                    with np.load(OUT/f"{row['key']}.npz") as a:
                        c=unpack(a['lengths'],a['coefficients']);np.testing.assert_array_equal(c(q),values);np.testing.assert_array_equal(a['endpoint'],r['fine'])
                        for k in ('increments','slow_increments'):np.testing.assert_array_equal(a[k],r[k])
                        if not whole:
                            np.testing.assert_array_equal(a['forcing'],pred['forcing']);np.testing.assert_allclose(a['oracle'],oracle,rtol=2e-12,atol=2e-12);np.testing.assert_allclose(a['oracle_loose'],oracle_loose,rtol=2e-12,atol=2e-12)
                    check_curve(frozen,curve,y,r['fine'],lr,q,row['stats'],row['guard'],row['targets'])
                    if not whole:modified(frozen,kernel,curve,y,r['fine'],RodasForcing(r['forcing_coefficients'],h),r['stats'])
                    dec=decomposition(nv,nr,values,lr,case['scale'],active=kernel.active);close(dec,row['decomposition']);close(metrics(frozen,values,lr,1e-6),row['affine_actual'])
                    if not whole:
                        od=decomposition(nv,nr,oracle,lr,case['scale'],active=kernel.active);close(od,row['oracle_decomposition']);inner=metrics(frozen,values,oracle,1e-6);close(inner,row['inner_error']);largest_inner=max(largest_inner,inner['concentration'])
                    if spec['name']=='teusink':
                        saved=next(x for x in ledgerrows if x['key']==row['key']);nn=signed_ledger(case,nc,y,ne);ll=signed_ledger(frozen,curve,y,r['fine']);nl=signed_ledger(case,curve,y,r['fine']);ln=signed_ledger(frozen,nc,y,ne)
                        for k,v in [('nonlinear_on_nonlinear',nn),('affine_on_affine',ll),('nonlinear_on_affine',nl),('affine_on_nonlinear',ln)]:
                            # Tight absolute residual check: default report
                            # tolerance1e-12 would be too loose near ledger1e-10.
                            for field in v:np.testing.assert_allclose(v[field],saved[k][field],rtol=2e-8,atol=2e-17)
                        vector=nn['signed_residual']-ln['signed_residual'];change=ln['signed_residual']-ll['signed_residual'];difference=nn['signed_residual']-ll['signed_residual']
                        np.testing.assert_allclose([vector,change,difference],[saved['vector_field_effect'],saved['curve_effect'],saved['difference']],rtol=2e-8,atol=2e-17);assert abs(vector+change-difference)<1e-20
                        if h==.03:ledger_summary.append(saved)
                    summaries.append(dict(model=spec['name'],h=h,augmented=augmented,whole=whole,nonlinear_pass=original['targets']['1e-06']['passes'],affine_pass=row['targets']['1e-06']['passes'],decomposition=dec,inner_error=None if whole else row['inner_error'],oracle_decomposition=None if whole else row['oracle_decomposition']))
                    counts+=1
            kernel.close();nonlinear_kernel.close();budgets.append(dict(key=key,nonlinear=nb,affine=lb,independent=ib,cross=cross));print('verified',spec['name'],h,flush=True)
    for p,sha in read(ROOT/'results/m3-rodas-continuous/manifest.json').items():assert digest(ROOT/p)==sha,p
    crc=0
    for d in (OUT,LEDGER):
        for p in d.glob('*.npz'):
            with zipfile.ZipFile(p) as z:assert z.testzip() is None,p
            crc+=1
    return dict(pairs=counts,references=budgets,oracle_concentration_error_max=largest_inner,comparisons=summaries,ledger_h03=ledger_summary,npz_crc_checked=crc,historical_manifest_unchanged=True,M4_admission=False,scope='Mechanism attribution on frozen local affine models; nonlinear reference never replaced. Original formal windows,targets,coverage,7 repeats and1.5 speedup remain unfulfilled.')


def write(result):
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'summary.json',result);save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
    files=[p for d in (OUT,LEDGER) for p in d.iterdir() if p.is_file() and p.name!='manifest.json']
    files += [ROOT/'research'/p for p in ('affine_partition.py','test_affine_partition.py','m3_affine_attribution.py','m3_affine_ledger.py','m3_affine_report.py')]
    files += [ROOT/'docs/m3-affine-attribution.md']
    save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args();result=verify()
    if args.write:write(result)
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
