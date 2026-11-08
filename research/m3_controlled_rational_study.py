"""Prospective enrichment/residual proposals with mandatory independent audit."""
import json
import time
import numpy as np
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.affine_partition import freeze,displacement_reference
from research.controlled_rational import RationalBasis,proposal
from research.m3_priority import STATES,metrics
from research.m3_growth_study import audit
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-controlled-rational'


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    for p,sha in json.loads((ROOT/'results/m3-merb-predictor/manifest.json').read_text()).items():assert digest(ROOT/p)==sha,p
    reg=dict(date='2026-11-08',models=['teusink','kecoli'],state_indices=[3,2],steps=[.01,.03],dimensions=[4,8,12,20,32,48,64],shifts=[.25,1.],projections=['direct','inverse'],budget=.05,target=1e-6,
        states='Saved own original prefix indices3 and2, latter is an earlier state not used for selecting preceding rational configurations. Not globally unseen development data. No reference resets inside actions.',
        projection='Same growing inverse-Arnoldi basis and actual LU for both diagnostics. Direct T=V^T A V; inverse T=(I-(V^T K V)^-1)/(sigma H). m shifted solves needed for m images, including final vector. Preserve all dimensions even after first proposal for mechanism audit.',
        proposal='Require H*sampled full-space residual and change from previous dimension <=.05 strict fixed-initial scale; sample positivity/pool checks. Indicators are heuristic, not certified. Earliest proposal selected before reading reference scores. Mandatory independent full-space affine reference can veto; record hypothetical false accepts, no silent acceptance.',
        reference='Each frozen reaction model: displacement Radau pair1e-12/1e-11 and independent original-state CVODE pair, all budget/cross differences<=.05 strict. Every candidate scored on65 uniform/public queries. Not nonlinear candidate or reaction coverage.',
        timing='Shared basis cost cumulative, per-projection evaluation/indicator/sampled-guard cost cumulative. Current gradient/J formation and all independent audits reported separately. No formal speed or M4 claim. Continuing after first proposal is diagnostic overhead separately identifiable.',
        storage='Save every reduced projection/drive/coefficients and final shared basis/images per sequence; replay all values and residuals without duplicating full lifted arrays.',
        source_sha256={p:digest(ROOT/p) for p in ('research/controlled_rational.py','research/test_controlled_rational.py','research/m3_controlled_rational_study.py','research/affine_partition.py','research/m3_priority.py','research/m3_growth_study.py','research/m3_ordered_cost.py')},
        input_sha256={p:digest(ROOT/p) for p in ('results/m3-merb-predictor/manifest.json','results/m3-priority/p0.npz','results/m3-priority/p0.json')})
    save(OUT/'registration.json',reg);rows=[];references=[];decisions=[]
    def persist():save(OUT/'runs.json',dict(runs=rows,references=references,decisions=decisions))
    for mi,name in enumerate(reg['models']):
        case=prepare(name)
        for si in reg['state_indices']:
            y=np.load(ROOT/'results/m3-priority/p0.npz')[name+'-prefix'][:,si];start=STATES[name][si]
            for hi,h in enumerate(reg['steps']):
                key=f'{mi}-{si}-{hi}';public=(case['times'][(case['times']>start)&(case['times']<start+h)]-start)/h;q=np.unique(np.r_[np.linspace(0,1,65),public])
                tick=time.perf_counter();frozen,ctx=freeze(case,y);formation=time.perf_counter()-tick
                ref,work=displacement_reference(ctx['jacobian'],lambda t:ctx['rhs'],y,h,q,case['scale'],1e-12)
                loose,worklo=displacement_reference(ctx['jacobian'],lambda t:ctx['rhs'],y,h,q,case['scale'],1e-11)
                independent,ib,seconds=audit(frozen,y,q,h);budget=metrics(frozen,loose,ref,1e-6);cross=metrics(frozen,independent,ref,1e-6)
                refok=max([*budget.values(),*ib.values(),*cross.values()])<=.05
                references.append(dict(key=key,model=name,state_index=si,h=h,start=start,budget=budget,independent_budget=ib,cross_budget=cross,reference_ok=refok,formation_seconds=formation,affine_work=work,affine_loose_work=worklo,independent_seconds=seconds))
                np.savez_compressed(OUT/f'reference-{key}.npz',initial=y,query=q,reference=ref,loose=loose,independent=independent)
                # Logarithmic infinity norm is a valid semigroup bound but may
                # be too pessimistic. Report H*mu, never exponentiate overflow.
                a=ctx['jacobian'].multiply(case['scale'][None,:]).multiply(1/case['scale'][:,None]).tocsr()
                mu=float(np.max(a.diagonal()+np.asarray(abs(a).sum(axis=1)).ravel()-abs(a.diagonal())))
                for shift in reg['shifts']:
                    seq=f'{key}-{shift}';basis=RationalBasis(ctx['jacobian'],ctx['rhs'],y,case['scale'],h,shift);previous={p:None for p in reg['projections']};elapsed={p:0. for p in reg['projections']};chosen={p:None for p in reg['projections']}
                    dimensions=sorted(set(min(len(y),d) for d in reg['dimensions']))
                    for dim in dimensions:
                        basis.extend(dim)
                        for method in reg['projections']:
                            ck=f'{seq}-{dim}-{method}';row=dict(key=ck,sequence=seq,reference_key=key,model=name,state_index=si,h=h,shift=shift,dimension_requested=dim,method=method,h_log_norm=h*mu);tick=time.perf_counter()
                            try:
                                result=basis.evaluate(q,method);indicator=proposal(result,previous[method],h,case['scale'],y,budget=reg['budget']);guard=structural(frozen,result['values']);elapsed[method]+=time.perf_counter()-tick
                                offered=bool(indicator['proposed'] and guard['nonnegative'] and guard['pool']<=1e-10)
                                if offered and chosen[method] is None:chosen[method]=ck
                                # Reference only sees a finished proposal, and
                                # cannot change its dimension or coefficients.
                                actual=metrics(frozen,result['values'],ref,1e-6);auditok=bool(refok and max(actual.values())<=1 and guard['nonnegative'] and guard['pool']<=1e-10)
                                row.update(status='success',stats=result['stats'],indicator=indicator,guard=guard,offered=offered,audit_ok=auditok,actual=actual,unsafe_proposal=bool(offered and not auditok),evaluation_cumulative_seconds=elapsed[method],optimistic_seconds=formation+result['stats']['basis_seconds']+elapsed[method])
                                np.savez_compressed(OUT/f'{ck}.npz',**{k:result[k] for k in ('coefficients','projection','drive')})
                                previous[method]=result
                            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                                elapsed[method]+=time.perf_counter()-tick;previous[method]=None
                                row.update(status='failed',reason=str(exc),evaluation_cumulative_seconds=elapsed[method])
                            rows.append(row);persist()
                    np.savez_compressed(OUT/f'basis-{seq}.npz',vectors=np.column_stack(basis.vectors),images=np.column_stack(basis.images),av=np.column_stack(basis.av))
                    for method in reg['projections']:
                        selected=next((r for r in rows if r['key']==chosen[method]),None)
                        decisions.append(dict(sequence=seq,method=method,selected_key=chosen[method],outcome='fallback_required' if selected is None else ('audited_linear_samples_only' if selected['audit_ok'] else 'audit_veto'),M4_eligible=False))
                    persist()
                print(name,si,h,'reference',refok,decisions[-4:],flush=True)


if __name__=='__main__':run()
