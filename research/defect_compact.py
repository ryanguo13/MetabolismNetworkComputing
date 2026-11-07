"""Frozen-linearized defect estimate with explicit research grid choices.

Same equation, score and guards as defect_fast; grid reduction is experimental.
Never authorizes a production integration step.
"""
import time
import numpy as np
from metabolism.m2_experiments import error
from research.defect_fast import BatchRates,propagate_fast,interpolate_fast


def estimate_compact(case,pieces,values,query,grids=(4,8)):
    if len(grids)!=2 or not (0<grids[0]<grids[1]):
        raise ValueError('Two increasing positive grids required')
    tick=time.perf_counter()
    scale=case['scale']
    if np.any(np.minimum.reduce([p.minima() for p in pieces]) < -1e-12*scale):
        return dict(status='guard_reject',reason='negative_candidate_curve',seconds=time.perf_counter()-tick),{}
    arrays={}
    try:
        batch=BatchRates(case)
        jacobians=[batch.jacobian(p(np.array([.5]))[:,0]) for p in pieces]
        histories,counts=[],[]
        for steps in grids:
            nodes,errors,stats=propagate_fast(pieces,batch,jacobians,steps)
            arrays[f'nodes-{steps}']=nodes
            arrays[f'error-{steps}']=errors
            histories.append(interpolate_fast(nodes,errors,query))
            counts.append(stats)
        low,high=histories
        corrected,corrected_low=values+high,values+low
        if any(not np.isfinite(v).all() or np.any(v < -1e-12*scale[:,None]) for v in (corrected,corrected_low)):
            return dict(status='guard_reject',reason='negative_or_nonfinite_linearized_correction',
                        seconds=time.perf_counter()-tick,counts=counts),arrays
        # Share rate evaluations across both target levels; no change to scales.
        fv,fc,fl=[batch.flux(v) for v in (values,corrected,corrected_low)]
        def observe(a,b,fa,fb,target):
            result=dict(concentration=error(a,b,scale,target),
                        flux=error(fa,fb,case['options']['flux_scale'],target))
            if 'net_map' in case['options']:
                result['net_flux']=error(case['options']['net_map']@fa,case['options']['net_map']@fb,
                                         case['options']['net_scale'],1e-4)
            return result
        result=dict(status='success',targets={},counts=counts)
        for target in (1e-4,1e-6):
            central=observe(values,corrected,fv,fc,target)
            refinement=observe(corrected_low,corrected,fl,fc,target)
            score={k:4*(central[k]+refinement[k]) for k in central}
            result['targets'][str(target)]=dict(central=central,refinement=refinement,score=score,
                                               would_accept=max(score.values())<=1)
        result['seconds']=time.perf_counter()-tick
        return result,arrays
    except (ValueError,RuntimeError,FloatingPointError) as exc:
        return dict(status='failed',reason=str(exc),seconds=time.perf_counter()-tick),arrays
