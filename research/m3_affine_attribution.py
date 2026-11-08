"""Prospective frozen-affine attribution on actual14/3003-state partitions."""
import json
import time
import numpy as np
from scipy.integrate import solve_ivp
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save,error
from metabolism.m3_continuation import prepare
from research.affine_partition import freeze,displacement_reference,affine_predictor,decomposition
from research.rodas_centered import CenteredRodas
from research.rodas_coupling import RodasForcing
from research.extrapolated_flow import pack,unpack
from research.m3_extrapolated_paths import qualified
from research.m3_growth_study import audit
from research.m3_priority import metrics
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-affine-attribution';OLD=ROOT/'results/m3-rodas-centered'


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    old=json.loads((OLD/'registration.json').read_text());oldrows=json.loads((OLD/'runs.json').read_text())['candidates']
    reg=dict(date='2026-11-08',models=old['models'],steps=old['steps'],augmented=[False,True],whole=[False,True],inner_tolerance=1e-10,
        freeze='Reactionwise v_L(z)=v(y0)+Dv(y0)(z-y0), same stoichiometry and reaction groups. No trajectory resets or reference injection; frozen model used only as diagnostic.',
        paired='Reuse frozen saved nonlinear trials (inner1e-10; whole1e-6). Reintegrate nonlinear truth independently. Recompute affine candidate with same kernel and independent affine stage algebra.',
        references='Affine displacement full Radau at1e-12/1e-11 plus independent original-state CVODE pair via existing full audit. All compared reference differences<=.05 strict concentration/flux/net budget. Fast-forcing oracle separately integrated in displacement form at1e-12/1e-11. Failed reference or candidate retained.',
        decomposition='Signed concentration errors E_N and E_L use same nonlinear-reference denominator. Report E_N-E_L, worst species/query, endpoint, active/inactive and cosine; never infer additive causal percentages from sup norms.',
        scope='32 local frozen candidates paired with32 historical nonlinear candidates plus16 accurately integrated frozen fast oracles. Not a candidate replacement or formal performance test; all original M4 gates remain.',
        source_sha256={p:digest(ROOT/p) for p in ('research/m3_affine_attribution.py','research/affine_partition.py','research/test_affine_partition.py','research/rodas_centered.py','research/rodas_coupling.py','research/m3_growth_study.py','research/m3_priority.py','research/m3_extrapolated_paths.py')},input_sha256={str(p.relative_to(ROOT)):digest(p) for p in [OLD/'registration.json',OLD/'runs.json',ROOT/'results/m3-rodas-continuous/manifest.json']})
    save(OUT/'registration.json',reg);rows=[];references=[]
    def persist():save(OUT/'runs.json',dict(candidates=rows,references=references))
    for mi,spec in enumerate(reg['models']):
        case=prepare(spec['name'])
        for hi,h in enumerate(reg['steps']):
            key=f'{mi}-{hi}'
            with np.load(OLD/f'reference-{key}.npz') as a:y=a['initial'];groups=a['groups'];q=a['query']
            frozen,ctx=freeze(case,y);kernel=CenteredRodas(frozen,groups)
            reference,nbudget,nseconds=audit(case,y,q,h)
            linear,lwork=displacement_reference(ctx['jacobian'],lambda t:ctx['rhs'],y,h,q,case['scale'],1e-12)
            loose,lwork2=displacement_reference(ctx['jacobian'],lambda t:ctx['rhs'],y,h,q,case['scale'],1e-11)
            independent,ibudget,iseconds=audit(frozen,y,q,h)
            lbudget=metrics(frozen,loose,linear,1e-6);cross=metrics(frozen,independent,linear,1e-6)
            reference_ok=max([*nbudget.values(),*lbudget.values(),*ibudget.values(),*cross.values()])<=.05
            den=1e-6*(.01*case['scale'][:,None]+np.maximum(abs(reference),case['scale'][:,None]))
            # Nonlinear remainder evaluated on independent nonlinear truth, with
            # H*R scaled as a diagnostic, not an integrated error estimate.
            residual=np.column_stack([case['system'].S@(case['system'].flux(0,z)-frozen['system'].flux(0,z)) for z in reference.T])
            refrow=dict(key=key,model=spec['name'],h=h,reference_ok=reference_ok,nonlinear_budget=nbudget,affine_budget=lbudget,independent_budget=ibudget,cross_budget=cross,nonlinear_seconds=nseconds,affine_work=lwork,affine_loose_work=lwork2,independent_seconds=iseconds,truth_gap=error(linear,reference,case['scale'],1e-6),scaled_rhs_remainder=float(np.max(abs(h*residual/den))),affine_minimum=float(linear.min()),affine_negative_rate_samples=int(sum(np.count_nonzero(frozen['system'].flux(0,z)<0) for z in linear.T)),groups=len(groups),active=len(kernel.active),jacobian_nnz=ctx['jacobian'].nnz)
            references.append(refrow);np.savez_compressed(OUT/f'reference-{key}.npz',initial=y,groups=groups,query=q,nonlinear_reference=reference,affine_reference=linear,affine_loose=loose,independent_reference=independent,rates=ctx['rates'],gradient_data=ctx['gradient'].data,gradient_indices=ctx['gradient'].indices,gradient_indptr=ctx['gradient'].indptr,gradient_shape=ctx['gradient'].shape)
            for augmented in reg['augmented']:
                predictor=affine_predictor(ctx,kernel.sf,kernel.cols,h,augmented);forcing=RodasForcing(predictor['forcing'],h)
                exact_fast,work=displacement_reference(predictor['fastj'],lambda t:predictor['fast']+forcing.value(t),y,h,q,case['scale'],1e-12)
                loose_fast,work2=displacement_reference(predictor['fastj'],lambda t:predictor['fast']+forcing.value(t),y,h,q,case['scale'],1e-11)
                oracle_budget=metrics(frozen,loose_fast,exact_fast,1e-6)
                for whole in reg['whole']:
                    original=next(r for r in oldrows if r['reference_key']==key and r['augmented']==augmented and r['whole']==whole and r['tolerance']==(1e-6 if whole else reg['inner_tolerance']))
                    with np.load(OLD/f"{original['key']}.npz") as a:nonlinear_curve=unpack(a['lengths'],a['coefficients']);nonlinear_values=nonlinear_curve(q)
                    ck=f'{key}-{augmented}-{whole}';row=dict(key=ck,reference_key=key,nonlinear_key=original['key'],model=spec['name'],h=h,augmented=augmented,whole=whole,reference_ok=reference_ok,nonlinear_actual=metrics(case,nonlinear_values,reference,1e-6),nonlinear_targets=original['targets'],oracle_budget=oracle_budget,oracle_work=work,oracle_loose_work=work2)
                    tick=time.perf_counter()
                    try:
                        r=kernel.trial(y,h,reg['inner_tolerance'],base_only=whole,augmented=augmented);curve=r['pieces'][0];values=curve(q);guard,targets=qualified(frozen,curve,values,linear,r['stats']);row.update(status='success',seconds=time.perf_counter()-tick,stats=r['stats'],guard=guard,targets=targets,
                            decomposition=decomposition(nonlinear_values,reference,values,linear,case['scale'],active=kernel.active),affine_actual=metrics(frozen,values,linear,1e-6),stage_roundoff=float(np.max(abs(r['increments']-predictor['increments'])/case['scale'][:,None])))
                        arrays=dict(endpoint=r['fine'],increments=r['increments'],slow_increments=r['slow_increments'],**pack(curve))
                        if not whole:
                            row.update(oracle_decomposition=decomposition(nonlinear_values,reference,exact_fast,linear,case['scale'],active=kernel.active),inner_error=metrics(frozen,values,exact_fast,1e-6))
                            arrays.update(oracle=exact_fast,oracle_loose=loose_fast,forcing=predictor['forcing'])
                        for t in targets.values():t['passes'] &= reference_ok
                        np.savez_compressed(OUT/f'{ck}.npz',**arrays)
                    except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
                    rows.append(row);persist()
            kernel.close();print(spec['name'],h,'reference',reference_ok,'gap',refrow['truth_gap'],[(r['augmented'],r['whole'],r['status'],r.get('decomposition',{}).get('nonlinear'),r.get('decomposition',{}).get('frozen'),r.get('decomposition',{}).get('difference')) for r in rows[-4:]],flush=True)


if __name__=='__main__':run()
