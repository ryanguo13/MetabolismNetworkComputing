"""Same shadow equations, batched rates and alternative exact LU ordering.

No new acceptance rule: compare against defect_shadow before timing or use.
"""
import time
from types import FunctionType

import numpy as np
from scipy.linalg import lu_factor,lu_solve
from scipy.sparse import csc_matrix,eye
from scipy.sparse.linalg import splu

from metabolism.m2_experiments import error
from research.defect_shadow import GAMMA


class BatchRates:
    def __init__(self,case):
        self.case=case
        self.model=case['model']
        self.system=case['system']
        if self.system.flux_jacobian is None:
            original=self.system.flux
            env=dict(original.__globals__)
            env['array']=lambda rows:np.stack(np.broadcast_arrays(*rows))
            # Reuse the already validated SBML expression code unchanged.
            self.expression=FunctionType(original.__code__,env)

    def flux(self,states):
        if self.system.flux_jacobian is None:
            result=self.expression(0.,states)
            if result.ndim==1:
                result=np.broadcast_to(result[:,None],(len(result),states.shape[1]))
            return result
        m=self.model
        with np.errstate(invalid='raise',over='raise',divide='raise'):
            return (m.k[:,None]*np.power(states[m.indices[0]],m.orders[0,:,None])
                    *np.power(states[m.indices[1]],m.orders[1,:,None]))

    def jacobian(self,y):
        if self.system.flux_jacobian is not None:
            return csc_matrix(self.system.S@self.system.flux_jacobian(0.,y))
        probes=y[:,None].astype(complex)+1j*1e-24*np.eye(len(y))
        return csc_matrix(self.system.S@(self.flux(probes).imag/1e-24))


def propagate_fast(pieces,batch,jacobians,substeps):
    e=np.zeros(len(pieces[0].d))
    errors=[e.copy()]
    nodes=[0.]
    elapsed=0.
    total=sum(p.h for p in pieces)
    stats=dict(factorizations=0,linear_solves=0,defect_rhs_batches=0)
    for piece,j in zip(pieces,jacobians):
        dt=piece.h/substeps
        grid=np.column_stack([(np.arange(substeps)+GAMMA)/substeps,
                              (np.arange(substeps)+1)/substeps]).ravel()
        residual=piece.derivative(grid)-batch.system.S@batch.flux(piece(grid))
        stats['defect_rhs_batches']+=1
        if len(e)<=32:
            matrix=j.toarray()
            factors=lu_factor(np.eye(len(e))-GAMMA*dt*matrix,check_finite=False)
            solve=lambda v:lu_solve(factors,v,check_finite=False)
            multiply=lambda v:matrix@v
        else:
            factors=splu(eye(len(e),format='csc')-GAMMA*dt*j,permc_spec='MMD_AT_PLUS_A')
            solve=factors.solve
            multiply=lambda v:j@v
        stats['factorizations']+=1
        for i in range(substeps):
            k1=solve(multiply(e)-residual[:,2*i])
            k2=solve(multiply(e+(1-GAMMA)*dt*k1)-residual[:,2*i+1])
            e=e+dt*((1-GAMMA)*k1+GAMMA*k2)
            nodes.append((elapsed+(i+1)*dt)/total)
            errors.append(e.copy())
            stats['linear_solves']+=2
        elapsed+=piece.h
    return np.asarray(nodes),np.column_stack(errors),stats


def interpolate_fast(nodes,values,query):
    right=np.clip(np.searchsorted(nodes,query,side='right'),1,len(nodes)-1)
    left=right-1
    weight=(query-nodes[left])/(nodes[right]-nodes[left])
    return values[:,left]+weight*(values[:,right]-values[:,left])


def estimate_fast(case,pieces,values,query,screen=None):
    tick=time.perf_counter()
    scale=case['scale']
    if np.any(np.minimum.reduce([p.minima() for p in pieces]) < -1e-12*scale):
        return dict(status='guard_reject',reason='negative_candidate_curve',seconds=time.perf_counter()-tick),{}
    arrays={}
    try:
        batch=BatchRates(case)
        jacobians=[batch.jacobian(p(np.array([.5]))[:,0]) for p in pieces]
        histories,counts=[],[]
        for steps in (8,16):
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
