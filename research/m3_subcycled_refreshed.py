"""Prospective local test of LU-sharing predictor subcycling."""
import itertools
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.affine_partition import freeze,displacement_reference
from research.rodas_subcycled_refreshed import RefreshedSubcycledRodas as SubcycledRodas
from research.test_rodas_subcycled_refreshed import order_diagnostic
from research.extrapolated_flow import pack
from research.inner_dense import full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_growth_study import audit
from research.m3_priority import metrics
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-subcycled-refreshed';OLD=ROOT/'results/m3-rodas-centered'
ARRAYS=('increments','slow_increments','forcing_coefficients','predictor_endpoints','correction_endpoints','segment_counts')


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve existing evidence')
    reg=dict(date='2026-11-08',models=['teusink','kecoli'],steps=[.01,.03],substeps=[1,2,4,8],whole=[False,True],affine=[False],inner_tolerance=1e-10,
        construction='Nine-stage predictor, equal substeps, current substep-origin Jacobian and actual LU of I-.25*(H/m)*J. Carry predictor endpoints through full prediction; then carry corrected state through m fast solves with predetermined piecewise slow forcing. No reference input. Fast Radau restarts at each forcing boundary; all m subflows counted.',
        scope='32 nonlinear local candidates,32 nonlinear uninterrupted mature Radau controls. Frozen systems are diagnostics only. Same optimized standalone predictor. Single exploratory timings exclude references and controller, no M4 claim.',
        order='Refresh current Jacobian each substep; check measured local convergence. Record24 weakly stiff nonlinear continuous/endpoint convergence values for m=1,2,4.',
        controls=dict(tolerances=[1e-6,1e-7,1e-8,1e-10],initial_steps=['auto','whole']),
        references='Independent full nonlinear audit pair at1e-12/1e-11. Affine displacement Radau pair1e-12/1e-11, checked against previously frozen cross-method audit evidence. Original65-point/public query grids. Budget<=.05 strict.',
        priority='First require accuracy, physical/modified ledgers, positivity and pools. If qualified local multi is slower than matched complete or mature control, do not escalate to formal runs. All original M4 gates unchanged.',
        source_sha256={p:digest(ROOT/p) for p in ('research/rodas_subcycled.py','research/test_rodas_subcycled.py','research/rodas_subcycled_refreshed.py','research/test_rodas_subcycled_refreshed.py','research/m3_subcycled_refreshed.py','research/affine_partition.py','research/rodas_coupling.py','research/rodas_centered.py','research/shared_radau.py','research/m3_extrapolated_paths.py','research/m3_growth_study.py','research/m3_priority.py','research/inner_dense.py','research/fullstep_radau.py')},
        input_sha256={p:digest(ROOT/p) for p in ('results/m3-subcycled-predictor/registration.json','results/m3-subcycled-predictor/runs.json','results/m3-affine-attribution/manifest.json','results/m3-rodas-continuous/manifest.json','results/m3-rodas-centered/registration.json','results/m3-rodas-centered/runs.json')})
    save(OUT/'registration.json',reg);save(OUT/'order.json',order_diagnostic());rows=[];controls=[];references=[]
    def persist():save(OUT/'runs.json',dict(candidates=rows,controls=controls,references=references))
    for mi,name in enumerate(reg['models']):
        original=prepare(name)
        for hi,h in enumerate(reg['steps']):
            key=f'{mi}-{hi}';oldkey=f'{mi}-{hi+2}'
            with np.load(OLD/f'reference-{oldkey}.npz') as src:y=src['initial'];groups=src['groups'];q=src['query']
            frozen,ctx=freeze(original,y);nr,nb,ns=audit(original,y,q,h)
            lr,lw=displacement_reference(ctx['jacobian'],lambda t:ctx['rhs'],y,h,q,original['scale'],1e-12)
            ll,lw2=displacement_reference(ctx['jacobian'],lambda t:ctx['rhs'],y,h,q,original['scale'],1e-11)
            lb=metrics(frozen,ll,lr,1e-6);refok=max([*nb.values(),*lb.values()])<=.05
            references.append(dict(key=key,model=name,h=h,oldkey=oldkey,nonlinear_budget=nb,nonlinear_seconds=ns,affine_budget=lb,affine_work=lw,affine_loose_work=lw2,reference_ok=refok))
            np.savez_compressed(OUT/f'reference-{key}.npz',initial=y,groups=groups,query=q,nonlinear_reference=nr,affine_reference=lr,affine_loose=ll)
            for affine in reg['affine']:
                case=frozen if affine else original;reference=lr if affine else nr;kernel=SubcycledRodas(case,groups)
                for count,whole in itertools.product(reg['substeps'],reg['whole']):
                    ck=f'candidate-{key}-{affine}-{count}-{whole}';row=dict(key=ck,reference_key=key,model=name,h=h,affine=affine,substeps=count,whole=whole);tick=time.perf_counter()
                    try:
                        r=kernel.trial(y,h,reg['inner_tolerance'],substeps=count,base_only=whole);c=r['pieces'][0];values=c(q);structural(case,values);seconds=time.perf_counter()-tick
                        guard,targets=qualified(case,c,values,reference,r['stats'])
                        for t in targets.values():t['passes'] &= refok
                        row.update(status='success',seconds=seconds,stats=r['stats'],guard=guard,targets=targets)
                        np.savez_compressed(OUT/f'{ck}.npz',endpoint=r['fine'],**pack(c),**{k:r[k] for k in ARRAYS})
                    except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
                    rows.append(row);persist()
                kernel.close()
            screen=GroupScreen(original['system'],original['scale'])
            for tol,whole in itertools.product(reg['controls']['tolerances'],(False,True)):
                ck=f'control-{key}-{tol}-{whole}';row=dict(key=ck,reference_key=key,model=name,h=h,tolerance=tol,whole=whole);tick=time.perf_counter()
                try:
                    r=(fullstep_curve if whole else full_curve)(original,y,h,tol,screen);c=r['pieces'][0];values=c(q);structural(original,values);seconds=time.perf_counter()-tick
                    guard,targets=qualified(original,c,values,nr,r['stats'])
                    for t in targets.values():t['passes'] &= refok
                    row.update(status='success',seconds=seconds,stats=r['stats'],guard=guard,targets=targets);np.savez_compressed(OUT/f'{ck}.npz',endpoint=r['fine'],**c.pack())
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
                controls.append(row);persist()
            print(name,h,[(r['affine'],r['substeps'],r['whole'],r['status'],r.get('targets',{}).get('1e-06',{}).get('passes'),round(r['seconds'],5)) for r in rows[-16:]],flush=True)


if __name__=='__main__':run()
