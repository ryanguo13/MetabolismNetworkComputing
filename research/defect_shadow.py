"""Offline shadow estimator; NEVER authorizes an integration step.

For a candidate P, e = y-P satisfies e' = F(P+e)-F(P)-r,
r = P'-F(P). We approximate this by e'=J_*e-r with the full
Jacobian frozen at each Hermite piece midpoint, using L-stable SDIRK2.
This is an experimental approximation, not a certified error bound.
"""
import time

import numpy as np
from scipy.sparse import eye,csc_matrix
from scipy.sparse.linalg import splu

from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import error

GAMMA=1-1/np.sqrt(2)


def propagate(pieces,full,jacobian,substeps):
    if substeps<1:
        raise ValueError('Positive substep count required')
    n=len(pieces[0].d)
    e=np.zeros(n)
    errors=[e.copy()]
    fractions=[0.]
    total=sum(p.h for p in pieces)
    elapsed=0.
    stats=dict(factorizations=0,linear_solves=0,defect_rhs=0,jacobians=0)
    for piece in pieces:
        j=csc_matrix(jacobian(piece(np.array([.5]))[:,0]))
        stats['jacobians']+=1
        dt=piece.h/substeps
        lu=splu(eye(n,format='csc')-GAMMA*dt*j)
        stats['factorizations']+=1
        def residual(s):
            value=piece(np.array([s]))[:,0]
            stats['defect_rhs']+=1
            return piece.derivative(np.array([s]))[:,0]-full(value)
        for i in range(substeps):
            k1=lu.solve(j@e-residual((i+GAMMA)/substeps))
            k2=lu.solve(j@(e+(1-GAMMA)*dt*k1)-residual((i+1)/substeps))
            e=e+dt*((1-GAMMA)*k1+GAMMA*k2)
            errors.append(e.copy())
            fractions.append((elapsed+(i+1)*dt)/total)
            stats['linear_solves']+=2
        elapsed+=piece.h
    return np.asarray(fractions),np.column_stack(errors),stats


def interpolate(fractions,errors,query):
    return np.vstack([np.interp(query,fractions,row) for row in errors])


def observables(case,a,b,target):
    vf=np.column_stack([case['system'].flux(0.,z) for z in a.T])
    vr=np.column_stack([case['system'].flux(0.,z) for z in b.T])
    result=dict(concentration=error(a,b,case['scale'],target),
                flux=error(vf,vr,case['options']['flux_scale'],target))
    if 'net_map' in case['options']:
        result['net_flux']=error(case['options']['net_map']@vf,case['options']['net_map']@vr,
                                 case['options']['net_scale'],1e-4)
    return result


def estimate(case,pieces,values,query,screen=None):
    tick=time.perf_counter()
    scale=case['scale']
    minima=np.minimum.reduce([p.minima() for p in pieces])
    if np.any(minima < -1e-12*scale):
        return dict(status='guard_reject',reason='negative_candidate_curve',seconds=time.perf_counter()-tick),{}
    screen=screen or GroupScreen(case['system'],scale)
    full=lambda y:case['system'].S@case['system'].flux(0.,y)
    jac=lambda y:case['system'].S@screen.gradient(y)
    arrays={}
    try:
        histories=[]
        counts=[]
        for substeps in (8,16):
            nodes,err,stats=propagate(pieces,full,jac,substeps)
            arrays[f'error-{substeps}']=err
            arrays[f'nodes-{substeps}']=nodes
            histories.append(interpolate(nodes,err,query))
            counts.append(stats)
        low,high=histories
        corrected=values+high
        corrected_low=values+low
        if any(not np.isfinite(v).all() or np.any(v < -1e-12*scale[:,None])
               for v in (corrected,corrected_low)):
            return dict(status='guard_reject',reason='negative_or_nonfinite_linearized_correction',
                        seconds=time.perf_counter()-tick,counts=counts),arrays
        result=dict(status='success',targets={},counts=counts)
        for target in (1e-4,1e-6):
            central=observables(case,values,corrected,target)
            refinement=observables(case,corrected_low,corrected,target)
            # Registered before development: no fitting to observed reference errors.
            score={k:4*(central[k]+refinement[k]) for k in central}
            result['targets'][str(target)]=dict(central=central,refinement=refinement,
                                               score=score,would_accept=max(score.values())<=1)
        result['seconds']=time.perf_counter()-tick
        return result,arrays
    except (RuntimeError,ValueError,FloatingPointError) as exc:
        return dict(status='failed',reason=str(exc),seconds=time.perf_counter()-tick),arrays
