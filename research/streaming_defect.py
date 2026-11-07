"""Transactional streaming frozen-J defect estimate; never replaces full audits.

Error equation e'=J_n e + f(P)-P', independently advanced on two grids. A
three-stage third-order SDIRK has the SAME diagonal gamma as ROS34PW2, so a
whole-native-piece step can reuse the actual predictor LU when lengths match.
Internal polynomial boundaries are respected. Error Hermite output is checked
against grid refinement; this is an experimental estimator, not an error bound.
"""
import time
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from metabolism.m2_experiments import error
from research.row_spc import coefficients
from research.mri_kernel import Hermite
from research.inner_dense import InnerCurve
from research.extrapolated_flow import DensePower
from research.shared_ledger import BatchedRates
from research.defect_fast import BatchRates


def tableau():
    d=coefficients()[1][0,0]
    a=np.array([[d,0.,0.],[(1-d)/2,d,0.],[(-6*d*d+16*d-1)/4,(6*d*d-20*d+5)/4,d]])
    return a,a[-1].copy(),a.sum(axis=1)


def advance(curve,jacobian,initial,batch,count,context=None,scale=None):
    """Integrate linear nonautonomous error ODE with actual stage forcing."""
    a,b,c=tableau();d=a[0,0];e=initial.copy();pieces=[];factors={};stats=dict(factorizations=0,predictor_factor_hits=0,linear_solves=0,defect_rate_states=0,max_linear_residual=0.)
    if context is not None:
        if context['jacobian'] is not jacobian:raise ValueError('Factor context must own the current Jacobian')
        factors[float(context['h'])]=context['solve']
    for p in curve.pieces:
        dt=p.h/count
        if dt in factors:
            solve=factors[dt]
            if context is not None and dt==context['h']:stats['predictor_factor_hits']+=1
        else:
            matrix=eye(len(e),format='csc')-d*dt*jacobian
            if len(e)<=32:
                factor=lu_factor(matrix.toarray());solve=lambda r,factor=factor:lu_solve(factor,r)
            else:solve=splu(matrix,permc_spec='MMD_AT_PLUS_A').solve
            factors[dt]=solve;stats['factorizations']+=1
        matrix=eye(len(e),format='csc')-d*dt*jacobian
        den=None if scale is None else np.maximum(scale,np.asarray(abs(matrix)@scale).ravel())
        nodes=np.unique(np.r_[np.arange(count+1)/count,*[(i+c)/count for i in range(count)]])
        forcing=batch.system.S@batch.flux(p(nodes))-p.derivative(nodes);stats['defect_rate_states']+=len(nodes)
        def drive(s):
            index=int(np.argmin(abs(nodes-s)))
            if abs(nodes[index]-s)>1e-14:raise ValueError('Missing forcing sample')
            return forcing[:,index]
        for step in range(count):
            start=e.copy();ks=[]
            for i in range(3):
                stage=start+dt*sum((a[i,k]*ks[k] for k in range(i)),start=np.zeros_like(e))
                rhs=jacobian@stage+drive((step+c[i])/count);ki=solve(rhs);ks.append(ki);stats['linear_solves']+=1
                if scale is not None:
                    residual=ki-d*dt*(jacobian@ki)-rhs
                    stats['max_linear_residual']=max(stats['max_linear_residual'],float(np.max(abs(residual)/den)))
            e=start+dt*sum((b[k]*ks[k] for k in range(3)),start=np.zeros_like(e))
            left=jacobian@start+drive(step/count);right=jacobian@e+drive((step+1)/count)
            hermite=Hermite(start,e,left,right,dt);pieces.append(DensePower(np.column_stack([hermite.d,hermite.c,hermite.b,hermite.a]),dt))
    if not np.isfinite(e).all():raise FloatingPointError('Nonfinite cumulative error state')
    return e,InnerCurve(pieces),stats


class Rates:
    def __init__(self,case):
        self.system=case['system'];self.batch=BatchRates(case) if 'model' in case else None;self.generic=BatchedRates(self.system)
    def flux(self,values):return self.batch.flux(values) if self.batch is not None else self.generic(values)


class StreamingDefect:
    def __init__(self,case,initial,grids=(1,2)):
        if len(grids)!=2 or not 1<=grids[0]<grids[1]:raise ValueError('Two increasing positive grids required')
        self.case=case;self.batch=Rates(case);self.grids=tuple(grids);self.errors=[np.zeros_like(initial),np.zeros_like(initial)];self.expected=initial.copy();self.generation=0
        self.scores={str(t):{} for t in (1e-4,1e-6)}

    def trial(self,trial,query):
        tick=time.perf_counter();curve=trial['pieces'][0];ctx=trial['linearization'];scale=self.case['scale']
        np.testing.assert_array_equal(ctx['initial'],self.expected)
        if not trial['stats']['continuous_nonnegative']:raise ValueError('Negative candidate curve')
        if abs(curve.h-ctx['h'])>1e-13*ctx['h']:raise ValueError('Inconsistent candidate length')
        endpoints=[];curves=[];stats=[]
        for initial,count in zip(self.errors,self.grids):
            end,output,work=advance(curve,ctx['jacobian'],initial,self.batch,count,ctx,scale)
            endpoints.append(end);curves.append(output);stats.append(work)
        values=curve(query);low=values+curves[0](query);high=values+curves[1](query)
        physical_positive=all(np.isfinite(v).all() and np.all(v>=-1e-12*scale[:,None]) for v in (low,high))
        target_results={};next_scores={k:v.copy() for k,v in self.scores.items()}
        if physical_positive:
            fv,fl,fh=[self.batch.flux(v) for v in (values,low,high)]
            def observe(a,b,fa,fb,target):
                result=dict(concentration=error(a,b,scale,target),flux=error(fa,fb,self.case['options']['flux_scale'],target))
                if 'net_map' in self.case['options']:
                    result['net_flux']=error(self.case['options']['net_map']@fa,self.case['options']['net_map']@fb,self.case['options']['net_scale'],1e-4)
                return result
            for target in (1e-4,1e-6):
                key=str(target);central=observe(values,high,fv,fh,target);refinement=observe(low,high,fl,fh,target)
                score={k:4*(central[k]+refinement[k]) for k in central}
                running={k:max(score[k],self.scores[key].get(k,0.)) for k in score};next_scores[key]=running
                target_results[key]=dict(central=central,refinement=refinement,score=score,running_score=running,would_accept=max(running.values())<=1)
        result=dict(status='success' if physical_positive else 'guard_reject',targets=target_results,stats=stats,seconds=time.perf_counter()-tick,
            errors=endpoints,error_curves=curves,_owner=self,_generation=self.generation,_discarded=False,_next=trial['fine'].copy(),_scores=next_scores)
        return result

    def commit(self,result):
        if result['_owner'] is not self or result['_generation']!=self.generation or result['_discarded']:raise ValueError('Stale or foreign error transaction')
        if result['status']!='success':raise ValueError('Cannot commit failed error guard')
        # In forced diagnostics, commit propagates even a score>1, with explicit
        # reporting. Production acceptance must additionally require its target.
        self.errors=[e.copy() for e in result['errors']];self.expected=result['_next'].copy();self.scores=result['_scores'];self.generation+=1

    def discard(self,result):
        if result['_owner'] is not self:raise ValueError('Foreign error transaction')
        result['_discarded']=True
