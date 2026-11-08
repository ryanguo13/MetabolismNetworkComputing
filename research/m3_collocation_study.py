"""Registered local cost challenge for full-vector collocation correction."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.collocation_correction import CorrectedROW
from research.extrapolated_flow import pack
from research.inner_dense import full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_growth_study import audit
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-collocation-correction'


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    sources=['research/'+p for p in ('m3_collocation_study.py','collocation_correction.py','test_collocation_correction.py','linear_context_row.py','row_continuous.py','inner_dense.py','m3_extrapolated_paths.py','m3_growth_study.py','m3_priority.py')]
    reg=dict(date='2026-11-08',models=[dict(name='teusink',start=1.,threshold=3.),dict(name='kecoli',start=.1,threshold=1.)],steps=[.003,.01,.03],predictors=['mri','row'],stages=[3,5],sweeps=[1,2],inner_tolerance=1e-8,queries=65,
        initial='Unperturbed registered prefix column3; no reference injection. Same state and H for all candidates/controls.',
        method='One or two frozen full-J simplified Newton iterations of right-Radau collocation residual, seeded by MRI continuous output or standalone ROW cubic output. Actual polynomial integrates final corrected stage rates. Independent full-vector correction; not a scalar ledger fix and not a claimed published SDC tableau.',
        controls='Uninterrupted full Radau over each H; tolerances1e-6/1e-8/1e-10 and automatic/whole initial steps. Full ROW seed receives identical correction opportunity.',
        gate='All original development concentration/flux/positivity/pool/physical/modified ledger checks; 65 uniform queries plus public observations; independent complete ODE1e-12/1e-11 reference budget<=.05. No global/controller/formal M4 qualification from local results.',
        next_gate='Only consider longer paths if locally qualified MRI-corrected costs have credible advantage over both corrected full-ROW and uninterrupted mature controls. Always retain failures. Original M4 gates unchanged.',
        timing='Full trial incl predictor/guards/correction, plus query guard; setup separately recorded. Shared external independent audit pair recorded separately. Single exploratory timing, no statistical speedup claim.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={p:digest(ROOT/p) for p in ('results/m3-priority/p0.npz','results/m3-checkpoint-controller/summary.json','results/m3-checkpoint-controller/manifest.json')})
    save(OUT/'registration.json',reg);rows=[];controls=[];references=[]
    def persist():save(OUT/'runs.json',dict(candidates=rows,controls=controls,references=references))
    for mi,spec in enumerate(reg['models']):
        name=spec['name'];case=prepare(name);y=np.load(ROOT/'results/m3-priority/p0.npz')[name+'-prefix'][:,3].copy();tick=time.perf_counter();screen=GroupScreen(case['system'],case['scale']);kappa,_=screen.evaluate(y);groups=np.flatnonzero(1e-3*kappa>=spec['threshold']);kernel=CorrectedROW(case,groups);setup=time.perf_counter()-tick
        for hi,h in enumerate(reg['steps']):
            key=f'{mi}-{hi}';public=case['times'][(case['times']>spec['start'])&(case['times']<spec['start']+h)]-spec['start'];q=np.unique(np.r_[np.linspace(0,1,reg['queries']),public/h]);ref,budget,secs=audit(case,y,q,h)
            references.append(dict(key=key,model=name,h=h,budget=budget,seconds=secs,setup_seconds=setup,groups=len(groups),active=len(kernel.active)))
            np.savez_compressed(OUT/f'reference-{key}.npz',initial=y,groups=groups,query=q,reference=ref)
            for tol in (1e-6,1e-8,1e-10):
                for whole in (False,True):
                    ck=f'control-{key}-{tol}-{whole}';row=dict(key=ck,reference_key=key,tolerance=tol,whole=whole);tick=time.perf_counter()
                    try:
                        r=(fullstep_curve if whole else full_curve)(case,y,h,tol,screen);c=r['pieces'][0];values=c(q);structural(case,values);seconds=time.perf_counter()-tick;guard,targets=qualified(case,c,values,ref,r['stats'])
                        for t in targets.values():t['passes'] &= max(budget.values())<=.05
                        row.update(status='success',seconds=seconds,stats=r['stats'],guard=guard,targets=targets);np.savez_compressed(OUT/f'{ck}.npz',endpoint=r['fine'],**c.pack())
                    except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
                    controls.append(row);persist()
            for seed in reg['predictors']:
                for stages in reg['stages']:
                    for sweeps in reg['sweeps']:
                        ck=f'candidate-{key}-{seed}-{stages}-{sweeps}';row=dict(key=ck,reference_key=key,model=name,h=h,predictor=seed,stages=stages,sweeps=sweeps);tick=time.perf_counter()
                        try:
                            r=kernel.trial(y,h,reg['inner_tolerance'],stages,sweeps,seed=='row');c=r['pieces'][0];values=c(q);structural(case,values);seconds=time.perf_counter()-tick;guard,targets=qualified(case,c,values,ref,r['stats'])
                            for t in targets.values():t['passes'] &= max(budget.values())<=.05
                            pred=r['predictor'];pc=pred['pieces'][0];pg,pt=qualified(case,pc,pc(q),ref,pred['stats'])
                            for t in pt.values():t['passes'] &= max(budget.values())<=.05
                            row.update(status='success',seconds=seconds,stats=r['stats'],guard=guard,targets=targets,predictor_guard=pg,predictor_targets=pt)
                            arrays=dict(endpoint=r['fine'],stage_values=r['stage_values'],corrected_rates=r['corrected_rates'],predictor_endpoint=pred['fine'],**pack(c))
                            arrays.update({'predictor_'+k:v for k,v in pack(pc).items()});np.savez_compressed(OUT/f'{ck}.npz',**arrays)
                        except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
                        rows.append(row);persist()
            print(name,h,[(r['predictor'],r['stages'],r['sweeps'],r['status'],r.get('targets',{}).get('1e-06',{}).get('passes'),round(r['seconds'],4)) for r in rows[-8:]],flush=True)
        kernel.close()


if __name__=='__main__':run()
