"""Frozen-J simplified Newton correction of MRI/ROW stages toward Radau IIA.

This is not a published MRI tableau or SDC sweep. Solve the FULL vector stage
residual, then construct the polynomial by integrating the corrected stage
rates. All state/physical-ledger guards remain; no scalar ledger projection.
The predictor LU cannot be reused: the Radau eigenvalue shifts differ.
"""
from functools import lru_cache
import time
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from scipy.special import roots_jacobi
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from research.linear_context_row import LinearContextROW
from research.extrapolated_flow import DensePower
from research.inner_dense import InnerCurve,physical_ledger
from research.mri_kernel import GAUSS


@lru_cache(None)
def tableau(stages):
    if stages not in (3,5):raise ValueError('Three or five right-Radau nodes required')
    nodes=np.r_[(roots_jacobi(stages-1,1,0)[0]+1)/2,1.]
    lagrange=np.linalg.inv(np.vander(nodes,stages,increasing=True))
    integrated=lagrange/np.arange(1,stages+1)[:,None]
    a=np.vander(nodes,stages+1,increasing=True)[:,1:]@integrated
    values,vectors=np.linalg.eig(a);inverse=np.linalg.inv(vectors)
    return nodes,a,lagrange,integrated,values,vectors,inverse


class StageSolve:
    def __init__(self,jacobian,h,stages):
        self.j=jacobian;self.h=h;self.n=jacobian.shape[0]
        _,self.a,_,_,self.values,self.vectors,self.inverse=tableau(stages)
        self.solvers={};self.partners={};self.solves=0;self.max_residual=0.
        for k,value in enumerate(self.values):
            if value.imag < -1e-12:
                partner=int(np.argmin(abs(self.values-value.conjugate())))
                if abs(self.values[partner]-value.conjugate())>1e-12:raise ValueError('Missing conjugate eigenvalue')
                self.partners[k]=partner;continue
            real=abs(value.imag)<1e-12;value=value.real if real else value
            matrix=eye(self.n,format='csc')-h*value*jacobian
            if self.n<=32:
                factor=lu_factor(matrix.toarray());solve=lambda r,factor=factor:lu_solve(factor,r)
            else:solve=splu(matrix,permc_spec='MMD_AT_PLUS_A').solve
            self.solvers[k]=(solve,real)

    def solve(self,residual,scale):
        transformed=residual@self.inverse.T;z=np.zeros_like(transformed,dtype=complex)
        for k,(solve,real) in self.solvers.items():
            z[:,k]=solve(transformed[:,k].real if real else transformed[:,k]);self.solves+=1
        for k,partner in self.partners.items():z[:,k]=z[:,partner].conjugate()
        delta=z@self.vectors.T
        if np.max(abs(delta.imag)/scale[:,None])>1e-10:raise FloatingPointError('Nonreal correction')
        delta=delta.real
        remainder=delta-self.h*(self.j@delta)@self.a.T-residual
        denom=scale[:,None]+self.h*(abs(self.j)@scale)[:,None]*np.sum(abs(self.a),axis=1)[None,:]
        defect=float(np.max(abs(remainder)/denom));self.max_residual=max(self.max_residual,defect)
        if defect>1e-10:raise FloatingPointError('Block linear solve residual failed')
        return delta


def correct(case,initial,predictor,jacobian,h,stages,sweeps):
    if sweeps not in (1,2):raise ValueError('One or two registered sweeps required')
    tick=time.perf_counter();nodes,a,lagrange,integrated,*_=tableau(stages);scale=case['scale'];system=case['system'];pool=case['options']['pools'];pool0=pool@initial;poolden=max(1e-12,float(np.max(abs(pool0))))
    stage=predictor(nodes);solver=StageSolve(jacobian,h,stages);records=[];pool_error=0.;min_internal=float(initial.min());curve=None
    def check(values):
        nonlocal pool_error,min_internal
        if not np.isfinite(values).all() or np.any(values < -1e-12*scale[:,None]):raise ValueError('Invalid collocation stage; no clipping')
        pool_error=max(pool_error,float(np.max(abs(pool@values-pool0[:,None])))/poolden);min_internal=min(min_internal,float(values.min()))
    for sweep in range(sweeps):
        check(stage);rates=np.column_stack([system.S@system.flux(0.,z) for z in stage.T])
        residual=initial[:,None]+h*rates@a.T-stage;delta=solver.solve(residual,scale)
        corrected_rates=rates+jacobian@delta
        coefficients=np.column_stack([initial,h*corrected_rates@integrated.T])
        curve=InnerCurve([DensePower(coefficients,h)]);new_stage=curve(nodes);check(new_stage)
        consistency=float(np.max(abs(new_stage-stage-delta)/scale[:,None]))
        if consistency>1e-9:raise FloatingPointError('Polynomial-stage consistency failed')
        records.append(dict(residual=float(np.max(abs(residual)/scale[:,None])),correction=float(np.max(abs(delta)/scale[:,None])),consistency=consistency))
        stage=new_stage
    end=curve(np.array([1.]))[:,0];check(end[:,None]);positive=bool(np.all(curve.minima()>=-1e-12*scale))
    stats=dict(correction_stages=stages,correction_sweeps=sweeps,correction_nfev=stages*sweeps,correction_nlu=len(solver.solvers),correction_complex_nlu=sum(not r for _,r in solver.solvers.values()),correction_solves=solver.solves,correction_linear_residual=solver.max_residual,correction_iterations=records,pool_error=pool_error,min_internal=min_internal,continuous_nonnegative=positive,dense_join_error=0.)
    if not positive:raise ValueError('Negative corrected continuous output')
    stats.update(physical_ledger(case,curve,initial,end))
    weights=case['options'].get('ledger_weights')
    if weights is not None:
        # Audit the actual polynomial forcing of the final linearized equation,
        # independently of its endpoint. This does not replace the physical f.
        sums=[]
        for n in (4,8):
            q=(GAUSS[n][0]+1)/2
            rhs=corrected_rates@lagrange.T@np.vander(q,stages,increasing=True).T
            sums.append(h/2*float(GAUSS[n][1]@(weights@rhs)))
        den=max(1.,abs(float(weights@initial)))
        stats.update(modified_ledger_residual=abs(float(weights@(end-initial))-sums[1])/den,modified_ledger_quadrature=abs(sums[1]-sums[0])/den)
    stats['correction_seconds']=time.perf_counter()-tick
    return dict(fine=end,pieces=[curve],stats=stats,stage_values=stage,corrected_rates=corrected_rates)


class CorrectedROW(LinearContextROW):
    def trial(self,y,h,tol=1e-8,stages=3,sweeps=1,base_only=False,timeout=20.):
        tick=time.perf_counter();pred=super().trial(y,h,tol,base_only,timeout)
        try:
            if not pred['stats']['continuous_nonnegative']:raise ValueError('Negative predictor continuous output')
            result=correct(self.case,y,pred['pieces'][0],pred['linearization']['jacobian'],h,stages,sweeps)
        except Exception as exc:
            exc.stats=dict(predictor_stats=pred['stats'],seconds=time.perf_counter()-tick,correction_work='Not completed; elapsed cost retained, partial operation counts unavailable')
            raise
        result['predictor']=pred;result['stats']['predictor_stats']=pred['stats'];result['stats']['seconds']=time.perf_counter()-tick
        result['stats']['pool_error']=max(result['stats']['pool_error'],pred['stats']['pool_error'])
        result['stats']['min_internal']=min(result['stats']['min_internal'],pred['stats']['min_internal'])
        return result
