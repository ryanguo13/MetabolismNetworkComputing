"""Replay subcycled predictors, all guards, reference budgets and cost decisions."""
import argparse
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
from research.affine_partition import freeze,displacement_reference
from research.rodas_subcycled import SubcycledRodas
from research.rodas_subcycled_refreshed import RefreshedSubcycledRodas
from research.rodas_coupling import RodasForcing
from research.test_rodas_subcycled import order_diagnostic
from research.test_rodas_subcycled_refreshed import order_diagnostic as refreshed_order
from research.extrapolated_flow import unpack
from research.inner_dense import InnerCurve,full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_growth_study import audit
from research.m3_priority import metrics
from research.m3_ledger_flow_report import registration,close,check_curve,modified
from research.m3_collocation_report import timeless
from research.m3_subcycled_study import ARRAYS
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-subcycled-predictor';REFRESH=ROOT/'results/m3-subcycled-refreshed'
def read(p):return json.loads(p.read_text())


class PiecewiseForcing:
    def __init__(self,coefficients,dt):
        self.pieces=[RodasForcing(c,dt) for c in coefficients];self.dt=dt
    def value(self,t):
        i=min(int(t/self.dt),len(self.pieces)-1)
        return self.pieces[i].value(t-i*self.dt)


def tight_ledger_stats(a,b):
    for k in a:
        if 'ledger' in k:np.testing.assert_allclose(a[k],b[k],rtol=2e-8,atol=2e-17)


def verify():
    reports=[]
    for directory,cls,order in ((OUT,SubcycledRodas,order_diagnostic),(REFRESH,RefreshedSubcycledRodas,refreshed_order)):
        reg=registration(directory/'registration.json');data=read(directory/'runs.json')
        close(order(),read(directory/'order.json'))
        assert len(data['candidates'])==len(reg['affine'])*32 and len(data['controls'])==32 and len(data['references'])==4
        counts=dict(candidates=0,controls=0);decisions=[]
        for mi,name in enumerate(reg['models']):
            original=prepare(name)
            for hi,h in enumerate(reg['steps']):
                key=f'{mi}-{hi}';rr=next(r for r in data['references'] if r['key']==key)
                with np.load(directory/f'reference-{key}.npz') as src:
                    y=src['initial'];groups=src['groups'];q=src['query'];frozen,ctx=freeze(original,y)
                    nr,nb,_=audit(original,y,q,h)
                    lr,lw=displacement_reference(ctx['jacobian'],lambda t:ctx['rhs'],y,h,q,original['scale'],1e-12)
                    ll,lw2=displacement_reference(ctx['jacobian'],lambda t:ctx['rhs'],y,h,q,original['scale'],1e-11)
                    for k,v in [('nonlinear_reference',nr),('affine_reference',lr),('affine_loose',ll)]:np.testing.assert_array_equal(src[k],v)
                with np.load(ROOT/f"results/m3-affine-attribution/reference-{rr['oldkey']}.npz") as old:
                    for k,v in [('initial',y),('groups',groups),('query',q),('nonlinear_reference',nr),('affine_reference',lr),('affine_loose',ll)]:np.testing.assert_array_equal(old[k],v)
                lb=metrics(frozen,ll,lr,1e-6);assert max([*nb.values(),*lb.values()])<=.05 and rr['reference_ok']
                close(nb,rr['nonlinear_budget']);close(lb,rr['affine_budget']);close(timeless(lw),timeless(rr['affine_work']));close(timeless(lw2),timeless(rr['affine_loose_work']))
                local=[r for r in data['candidates'] if r['reference_key']==key]
                assert {(r['affine'],r['substeps'],r['whole']) for r in local}=={(a,m,w) for a in reg['affine'] for m in reg['substeps'] for w in reg['whole']}
                for affine in reg['affine']:
                    case=frozen if affine else original;reference=lr if affine else nr;kernel=cls(case,groups)
                    for row in [r for r in local if r['affine']==affine]:
                        assert row['status']=='success'
                        m=row['substeps'];whole=row['whole'];r=kernel.trial(y,h,reg['inner_tolerance'],substeps=m,base_only=whole)
                        close(timeless(r['stats']),timeless(row['stats']));tight_ledger_stats(r['stats'],row['stats'])
                        assert r['stats']['predictor_nlu']==(1 if directory==OUT else m)
                        assert r['stats']['predictor_njev']==r['stats']['predictor_nlu'] and r['stats']['predictor_solves']==9*m
                        assert r['stats']['subflows']==(0 if whole else m)
                        with np.load(directory/f"{row['key']}.npz") as a:
                            for k in ARRAYS:np.testing.assert_array_equal(a[k],r[k])
                            c=unpack(a['lengths'],a['coefficients']);np.testing.assert_array_equal(a['endpoint'],r['fine']);np.testing.assert_array_equal(c(q),r['pieces'][0](q))
                        check_curve(case,c,y,r['fine'],reference,q,row['stats'],row['guard'],row['targets'])
                        if not whole:
                            modified(case,kernel,c,y,r['fine'],PiecewiseForcing(r['forcing_coefficients'],h/m),r['stats'])
                            np.testing.assert_allclose(c(np.linspace(0,1,m+1)),r['correction_endpoints'],rtol=2e-12,atol=1e-12)
                            assert int(r['segment_counts'].sum())==len(c.pieces)
                        else:np.testing.assert_allclose(c(np.linspace(0,1,m+1)),r['predictor_endpoints'],rtol=2e-12,atol=1e-12)
                        counts['candidates']+=1
                    kernel.close()
                controls=[r for r in data['controls'] if r['reference_key']==key];screen=GroupScreen(original['system'],original['scale'])
                assert {(r['tolerance'],r['whole']) for r in controls}=={(t,w) for t in reg['controls']['tolerances'] for w in (False,True)}
                for row in controls:
                    assert row['status']=='success'
                    r=(fullstep_curve if row['whole'] else full_curve)(original,y,h,row['tolerance'],screen)
                    close(timeless(r['stats']),timeless(row['stats']));tight_ledger_stats(r['stats'],row['stats'])
                    with np.load(directory/f"{row['key']}.npz") as a:
                        c=InnerCurve.unpack(a['lengths'],a['coefficients']);np.testing.assert_array_equal(c(q),r['pieces'][0](q));np.testing.assert_array_equal(a['endpoint'],r['fine'])
                    check_curve(original,c,y,r['fine'],nr,q,row['stats'],row['guard'],row['targets']);counts['controls']+=1
                for target in ('0.0001','1e-06'):
                    qualified=[r for r in controls if r['targets'][target]['passes']];assert qualified
                    best=min(qualified,key=lambda r:r['seconds'])
                    for row in [r for r in local if not r['affine'] and not r['whole']]:
                        standalone=next(r for r in local if not r['affine'] and r['whole'] and r['substeps']==row['substeps'])
                        decisions.append(dict(model=name,h=h,substeps=row['substeps'],target=target,passes=row['targets'][target]['passes'],actual=row['targets'][target]['actual'],physical_ledger=row['stats'].get('physical_ledger_residual'),candidate_seconds=row['seconds'],predictor_seconds=row['stats']['predictor_seconds'],baseline_key=best['key'],baseline_seconds=best['seconds'],speed_ratio=best['seconds']/row['seconds'],predictor_only_ratio=best['seconds']/row['stats']['predictor_seconds'],standalone_passes=standalone['targets'][target]['passes'],standalone_ratio=standalone['seconds']/row['seconds'] if standalone['targets'][target]['passes'] else None))
                print('verified',directory.name,name,h,flush=True)
        success={t:sum(r['targets'][t]['passes'] for r in data['candidates']) for t in ('0.0001','1e-06')}
        reports.append(dict(directory=directory.name,counts=counts,passes=success,decisions=decisions))
    crc=0
    for directory in (OUT,REFRESH):
        for p in directory.glob('*.npz'):
            with zipfile.ZipFile(p) as z:assert z.testzip() is None,p
            crc+=1
    for manifest in ('results/m3-affine-attribution/manifest.json','results/m3-rodas-continuous/manifest.json'):
        for p,sha in read(ROOT/manifest).items():assert digest(ROOT/p)==sha,p
    return dict(batches=reports,npz_crc_checked=crc,historical_manifests_unchanged=True,M4_admission=False,
        conclusion='Frozen-LU subcycling lowers stiff linear error but loses nonlinear accuracy. Refreshed Jacobians restore local accuracy at extra cost. Qualified H=.03,m>=4 refreshed predictors alone already exceed the local1.5 speedup budget, before fast correction/controller/audits. Single-run screening is not formal performance evidence; original M4 gates remain unmet.')


def write(result):
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'summary.json',result);save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
    files=[p for d in (OUT,REFRESH) for p in d.iterdir() if p.is_file() and p.name!='manifest.json']
    files += [ROOT/'research'/p for p in ('rodas_subcycled.py','rodas_subcycled_refreshed.py','test_rodas_subcycled.py','test_rodas_subcycled_refreshed.py','m3_subcycled_study.py','m3_subcycled_refreshed.py','m3_subcycled_report.py')]
    files += [ROOT/'docs/m3-subcycled-predictor.md']
    save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args();result=verify()
    if args.write:write(result)
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
