"""Fresh-Jacobian cubic ROW-SPC with transactional preconditioner reuse.

Every predictor uses CURRENT J and every fast solve retains Radau refresh/error
control. Old LU factors solve only as bounded-refinement preconditioners for
current matrices. Trial caches are published explicitly after caller accuracy
acceptance. Both ledgers and exact output guards are retained.
"""
import time
import numpy as np
from scipy.sparse import eye
from research.mri_kernel import MRI
from research.extrapolated_flow import DensePower
from research.inner_dense import InnerCurve
from research.row_continuous import coupling,PolynomialForcing
from research.shared_radau import SharedRadau
from research.shared_ledger import optimized_case,ledgers
from research.refined_lu import RefinedLU


class RefinedRadau(SharedRadau):
    def __init__(self,*args,refined_cache,linear_scale,**kwargs):
        super().__init__(*args,**kwargs)
        def lu(matrix):
            self.nlu+=1
            # Current-system weighted backward residual. Correction convergence
            # is relative to the solved increment with a1e-14*scale floor.
            backward=np.maximum(linear_scale,np.asarray(abs(matrix)@linear_scale).ravel())
            return refined_cache.operator(matrix,linear_scale,backward)
        self.lu=lu;self.solve_lu=lambda operator,rhs:operator.solve(rhs)


class CachedROW(MRI):
    def __init__(self,case,groups,reuse=True):
        super().__init__(optimized_case(case),groups)
        self.predictor_cache=RefinedLU(reuse);self.fast_cache=RefinedLU(reuse);self.generation=0

    def cache_state(self):
        return dict(generation=self.generation,predictor=self.predictor_cache.signatures(),fast=self.fast_cache.signatures())

    def commit(self,trial):
        if trial['_owner'] is not self or trial['_generation']!=self.generation or trial['_discarded']:
            raise ValueError('Stale or foreign cache transaction')
        stats=trial['stats']
        if not stats['continuous_nonnegative'] or stats['pool_error']>1e-10 or any(stats.get(k,0.)>1e-10 for k in ('physical_ledger_residual','physical_ledger_quadrature','modified_ledger_residual','modified_ledger_quadrature')):
            raise ValueError('Cannot commit failed structural guard')
        self.predictor_cache,self.fast_cache=trial['_pending'];self.generation+=1

    def discard(self,trial):
        if trial['_owner'] is not self:raise ValueError('Foreign cache transaction')
        trial['_discarded']=True

    def trial(self,y,h,tol=1e-8,base_only=False,timeout=20.):
        tick=time.perf_counter();a,g,b,mu=coupling();y=np.asarray(y,dtype=float).copy()
        if not np.isfinite(h) or h<=0 or not np.isfinite(tol) or tol<=0:raise ValueError('Positive finite step and tolerance required')
        stats=dict(predictor_nfev=0,predictor_njev=0,predictor_nlu=0,predictor_solves=0,nfev=0,njev=0,nlu=0,
            subflows=0,accepted_steps=0,output_nfev=0,pool_error=0.,min_internal=float(y.min()),linear_residuals=[])
        predictor_cache=self.predictor_cache.fork();fast_cache=self.fast_cache.fork()
        stats['cache_before']=self.cache_state()
        pool0=self.pools@y;poolden=max(1e-12,float(np.max(abs(pool0))))
        def check(z):
            if time.perf_counter()-tick>timeout:raise TimeoutError('Continuous ROW SPC budget')
            if not np.isfinite(z).all() or np.any(z < -1e-12*self.scale):raise ValueError('Invalid ROW state; no clipping')
        def accepted(z):
            check(z);stats['min_internal']=min(stats['min_internal'],float(z.min()))
            stats['pool_error']=max(stats['pool_error'],float(np.max(abs(self.pools@z-pool0)))/poolden)
        try:
            check(y);gradient=self.screen.gradient(y);j=(self.system.S@gradient).tocsc();fastj=(self.sf@gradient[self.cols]).tocsc();slowj=j-fastj
            matrix=eye(len(y),format='csc')-h*g[0,0]*j
            op=predictor_cache.operator(matrix.toarray() if len(y)<=32 else matrix,self.scale);solve=op.solve
            stats.update(predictor_njev=1)
            increments=[];ks=[];stages=[];rates=[];full=[]
            for i in range(4):
                state=y+sum((a[i,k]*increments[k] for k in range(i)),start=np.zeros_like(y));accepted(state)
                rate=self.system.flux(0.,state);f=self.system.S@rate;stats['predictor_nfev']+=1
                prev=sum((g[i,k]*increments[k] for k in range(i)),start=np.zeros_like(y))
                rhs=h*f+h*(j@prev);ki=solve(rhs);stats['predictor_solves']+=1
                stats['linear_residuals'].append(float(np.max(abs(matrix@ki-rhs)/self.scale)))
                increments.append(ki);ks.append(h*(f-self.sf@rate[self.cols])+h*(slowj@(prev+g[i,i]*ki)))
                stages.append(state);rates.append(rate);full.append(f)
            increments=np.column_stack(increments);ks=np.column_stack(ks)
            stats['predictor_nlu']=predictor_cache.stats['factorizations']
            stats['predictor_seconds']=time.perf_counter()-tick
            raw=None;forcing=None
            if base_only:end=y+increments@b;accepted(end)
            else:
                forcing=PolynomialForcing((ks@mu)*np.arange(1,4)/h,h);stats['subflows']=1
                def expand(t,u):
                    z=y+forcing.integral(t);z[self.active]=u;return z
                def rhs(t,u):
                    z=expand(t,u);check(z);return self.sfa@self.system.flux(0.,z)[self.cols]+forcing.value(t)[self.active]
                def jac(t,u):
                    z=expand(t,u);check(z);return (self.sfa@self.screen.gradient(z)[self.cols][:,self.active]).tocsc()
                pieces=[]
                if len(self.active):
                    solver=RefinedRadau(rhs,0.,y[self.active],h,refined_cache=fast_cache,linear_scale=self.scale[self.active],rtol=tol,atol=.01*tol*self.scale[self.active],jac=jac,seed=fastj[self.active][:,self.active].tocsc(),first_step=h)
                    while solver.status=='running':
                        left=solver.t;message=solver.step()
                        if solver.status=='failed':raise RuntimeError(message)
                        accepted(expand(solver.t,solver.y));pieces.append(forcing.dense(y,left,solver.t-left,self.active,solver.dense_output()))
                    rawend=expand(h,solver.y)
                    for k in ('nfev','njev','nlu'):stats[k]=getattr(solver,k)
                else:
                    rawend=y+forcing.integral(h);accepted(rawend);pieces=[forcing.dense(y,0.,h,self.active)]
                raw=InnerCurve(pieces);stats['accepted_steps']=len(pieces)
                stats['raw_nonnegative']=bool(np.all(raw.minima()>=-1e-12*self.scale));stats['raw_join_error']=raw.join_error(self.scale)
                if not stats['raw_nonnegative'] or stats['raw_join_error']>1e-10:raise ValueError('Invalid ROW raw curve')
                end=rawend;accepted(end)
                stats['endpoint_correction_scaled']=0.
            # Cubic stage extension for standalone ROW; modified fast curve
            # for SPC. This is an independently audited candidate physical output.
            curve=(InnerCurve([DensePower(np.column_stack([y,increments@mu]),h)]) if base_only else raw)
            stats.update(continuous_nonnegative=bool(np.all(curve.minima()>=-1e-12*self.scale)),dense_join_error=curve.join_error(self.scale))
            if stats['continuous_nonnegative']:stats.update(ledgers(self.case,curve,y,end,self,forcing))
            stats.update(predictor_linear=predictor_cache.stats.copy(),fast_linear=fast_cache.stats.copy())
            stats['cache_pending']=dict(predictor=predictor_cache.signatures(),fast=fast_cache.signatures())
            stats['seconds']=time.perf_counter()-tick
            return dict(fine=end,pieces=[curve],raw=raw,raw_endpoint=None if base_only else rawend,forcing_coefficients=None if base_only else forcing.coefficients,
                _owner=self,_generation=self.generation,_discarded=False,_pending=(predictor_cache,fast_cache),increments=increments,slow_increments=ks,stages=np.column_stack(stages),rates=np.column_stack(rates),full=np.column_stack(full),stats=stats)
        except Exception as exc:
            stats.update(predictor_linear=predictor_cache.stats.copy(),fast_linear=fast_cache.stats.copy())
            stats['seconds']=time.perf_counter()-tick;exc.stats=stats;raise
