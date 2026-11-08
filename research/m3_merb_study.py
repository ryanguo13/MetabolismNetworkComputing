"""Bounded predictor-only screen before attempting reaction-partition coupling."""
import itertools
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.merb_predictor import MERBPredictor
from research.test_merb_predictor import order_diagnostic
from research.extrapolated_flow import pack
from research.inner_dense import full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_growth_study import audit
from research.m3_priority import metrics
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-merb-predictor';OLD=ROOT/'results/m3-rodas-centered'
ARRAYS=('stage','initial_rates','stage_rates','remainder','forcing')


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    reg=dict(date='2026-11-08',models=['teusink','kecoli'],steps=[.003,.01,.03],tolerances=[1e-6,1e-8,1e-10],
        method='Autonomous MERB4 eq2.63 arXiv2106.05385v2: full J at macro origin; linear stage to.75H; reactionwise D=v(stage)-v0-Dv0*(stage-y); final linear forcing16/9*(t/H)^2*S*D. Two nonlinear rate evaluations plus guards. Full linear IVPs solved by adaptive ordered Radau with exact-matrix LU cache.',
        scope='18 complete predictors plus48 fresh uninterrupted mature controls. NOT reaction split coverage or production solver. Published endpoint order does not imply uniform dense order; audit full output, physical and modified ledgers independently. Preserve all M4 gates.',
        sources='Literature identity verified using arXiv Atom and Crossref. Read v2 sec2.2,2.3.4,2.4.3,2.5, not full paper review. Reference provenance in sources.json.',
        controls=dict(tolerances=[1e-6,1e-7,1e-8,1e-10],initial_steps=['auto','whole']),
        reference='Independent complete ODE audit pair1e-12/1e-11, budget<=.05 strict; original65-point/public grids. Recheck endpoint separately from continuous output.',
        timing='One exploratory timing including all output guards, excluding setup/reference/controller. Predictor time separately measured before guards; if it cannot meet cost budget, do not add reaction-fast correction.',
        source_sha256={p:digest(ROOT/p) for p in ('research/merb_predictor.py','research/test_merb_predictor.py','research/m3_merb_study.py','research/ordered_radau.py','research/shared_radau.py','research/inner_dense.py','research/fullstep_radau.py','research/m3_extrapolated_paths.py','research/m3_growth_study.py','research/m3_priority.py')},
        input_sha256={p:digest(ROOT/p) for p in ('results/m3-subcycled-predictor/manifest.json','results/m3-affine-attribution/manifest.json','results/m3-rodas-centered/registration.json','results/m3-rodas-centered/runs.json')})
    save(OUT/'registration.json',reg);save(OUT/'order.json',order_diagnostic());rows=[];controls=[];references=[]
    def persist():save(OUT/'runs.json',dict(candidates=rows,controls=controls,references=references))
    for mi,name in enumerate(reg['models']):
        case=prepare(name);kernel=MERBPredictor(case);screen=GroupScreen(case['system'],case['scale'])
        for hi,h in enumerate(reg['steps']):
            key=f'{mi}-{hi}';oldkey=f'{mi}-{hi+1}'
            with np.load(OLD/f'reference-{oldkey}.npz') as src:y=src['initial'];q=src['query']
            ref,budget,seconds=audit(case,y,q,h);refok=max(budget.values())<=.05
            references.append(dict(key=key,model=name,h=h,oldkey=oldkey,budget=budget,seconds=seconds,reference_ok=refok));np.savez_compressed(OUT/f'reference-{key}.npz',initial=y,query=q,reference=ref)
            for tol in reg['tolerances']:
                ck=f'candidate-{key}-{tol}';row=dict(key=ck,reference_key=key,model=name,h=h,tolerance=tol);tick=time.perf_counter()
                try:
                    r=kernel.trial(y,h,tol);c=r['pieces'][0];values=c(q);structural(case,values);elapsed=time.perf_counter()-tick;guard,targets=qualified(case,c,values,ref,r['stats'])
                    for t,v in targets.items():v['passes'] &= refok;v['endpoint']=metrics(case,r['fine'][:,None],ref[:,-1:],float(t))
                    row.update(status='success',seconds=elapsed,stats=r['stats'],guard=guard,targets=targets)
                    arrays=dict(endpoint=r['fine'],**pack(c),**{k:r[k] for k in ARRAYS});arrays.update({'stage_'+k:v for k,v in pack(r['stage_curve']).items()});np.savez_compressed(OUT/f'{ck}.npz',**arrays)
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
                rows.append(row);persist()
            for tol,whole in itertools.product(reg['controls']['tolerances'],(False,True)):
                ck=f'control-{key}-{tol}-{whole}';row=dict(key=ck,reference_key=key,model=name,h=h,tolerance=tol,whole=whole);tick=time.perf_counter()
                try:
                    r=(fullstep_curve if whole else full_curve)(case,y,h,tol,screen);c=r['pieces'][0];values=c(q);structural(case,values);elapsed=time.perf_counter()-tick;guard,targets=qualified(case,c,values,ref,r['stats'])
                    for v in targets.values():v['passes'] &= refok
                    row.update(status='success',seconds=elapsed,stats=r['stats'],guard=guard,targets=targets);np.savez_compressed(OUT/f'{ck}.npz',endpoint=r['fine'],**c.pack())
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
                controls.append(row);persist()
            print(name,h,[(r['tolerance'],r['status'],r.get('targets',{}).get('1e-06',{}).get('passes'),round(r['seconds'],5)) for r in rows[-3:]],flush=True)


if __name__=='__main__':run()
