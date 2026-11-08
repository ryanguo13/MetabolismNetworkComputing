"""Autonomous MERB4 predictor, Luan et al. arXiv:2106.05385v2 eq2.63.

Two nonlinear rate evaluations; linear IVPs use the full analytic Jacobian.
This is a complete-system predictor screen, NOT reaction-partition coverage.
The final linear IVP is audited as physical output without claiming uniform
fourth-order dense accuracy. Source endpoint order is not a dense-order proof.
"""
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from research.ordered_radau import OrderedRadau
from research.shared_radau import ExactLUCache
from research.extrapolated_flow import DensePower
from research.inner_dense import InnerCurve,physical_ledger
from research.mri_kernel import GAUSS


class MERBPredictor:
    def __init__(self,case):
        self.case=case;self.system=case['system'];self.scale=case['scale'];self.screen=GroupScreen(self.system,self.scale)
        if self.system.time_dependent:raise ValueError('Autonomous systems only')

    def trial(self,y,h,tol=1e-8,timeout=20.):
        tick=time.perf_counter();y=np.asarray(y,dtype=float).copy()
        if not np.isfinite(h) or h<=0 or not np.isfinite(tol) or tol<=0:raise ValueError('Positive finite step/tolerance required')
        stats=dict(nonlinear_nfev=0,nonlinear_njev=0,nfev=0,njev=0,nlu=0,subflows=0,accepted_steps=0,pool_error=0.,min_internal=float(y.min()))
        pools=self.case['options']['pools'];p0=pools@y;pd=max(1e-12,float(np.max(abs(p0))));cache=ExactLUCache()
        def check(z):
            if time.perf_counter()-tick>timeout:raise TimeoutError('MERB predictor budget')
            if not np.isfinite(z).all() or np.any(z < -1e-12*self.scale):raise ValueError('Invalid MERB state; no clipping')
        def accepted(z):
            check(z);stats['min_internal']=min(stats['min_internal'],float(z.min()));stats['pool_error']=max(stats['pool_error'],float(np.max(abs(pools@z-p0)))/pd)
        try:
            check(y);v0=self.system.flux(0.,y);stats['nonlinear_nfev']+=1
            gradient=self.screen.gradient(y);j=(self.system.S@gradient).tocsc();f0=self.system.S@v0;stats['nonlinear_njev']+=1
            def linear(duration,forcing):
                stats['subflows']+=1
                def rhs(t,z):
                    check(z);return j@(z-y)+f0+(t/h)**2*forcing
                solver=OrderedRadau(rhs,0.,y,duration,rtol=tol,atol=.01*tol*self.scale,jac=j,first_step=duration)
                factorize=solver.lu;solver.lu=lambda a:cache.factor(a,factorize);pieces=[]
                while solver.status=='running':
                    left=solver.t;message=solver.step()
                    if solver.status=='failed':raise RuntimeError(message)
                    accepted(solver.y);dense=solver.dense_output();pieces.append(DensePower(np.column_stack([dense.y_old,dense.Q]),solver.t-left))
                curve=InnerCurve(pieces)
                if np.any(curve.minima() < -1e-12*self.scale):raise ValueError('Negative MERB linear curve')
                if curve.join_error(self.scale)>1e-10:raise ValueError('Discontinuous MERB linear curve')
                for k in ('nfev','njev','nlu'):stats[k]+=getattr(solver,k)
                stats['accepted_steps']+=len(pieces)
                return solver.y.copy(),curve
            stage,stagecurve=linear(.75*h,np.zeros_like(y));accepted(stage)
            rates=self.system.flux(0.,stage);stats['nonlinear_nfev']+=1
            # Subtract reactionwise before applying S, avoiding affine offset
            # f(z)-J*z. Cancellation is still measured in independent tests.
            remainder=rates-v0-gradient@(stage-y);forcing=(16/9)*(self.system.S@remainder)
            end,curve=linear(h,forcing)
            stats['predictor_seconds']=time.perf_counter()-tick
            stats.update(continuous_nonnegative=True,dense_join_error=curve.join_error(self.scale),lu_cache_hits=cache.hits,lu_cache_misses=cache.misses)
            stats.update(physical_ledger(self.case,curve,y,end))
            weights=self.case['options'].get('ledger_weights')
            if weights is not None:
                totals=[]
                for n in (8,16):
                    nodes,w=GAUSS[n];q=(nodes+1)/2;total=0.;left=0.
                    for piece in curve.pieces:
                        values=[weights@(j@(z-y)+f0+((left+piece.h*t)/h)**2*forcing) for t,z in zip(q,piece(q).T)]
                        total+=piece.h/2*float(w@values);left+=piece.h
                    totals.append(total)
                den=max(1.,abs(float(weights@y)))
                stats.update(modified_ledger_residual=abs(float(weights@(end-y))-totals[1])/den,modified_ledger_quadrature=abs(totals[1]-totals[0])/den)
            stats['seconds']=time.perf_counter()-tick
            return dict(fine=end,pieces=[curve],stage_curve=stagecurve,stage=stage,initial_rates=v0,stage_rates=rates,remainder=remainder,forcing=forcing,stats=stats)
        except Exception as exc:
            stats.update(seconds=time.perf_counter()-tick,lu_cache_hits=cache.hits,lu_cache_misses=cache.misses);exc.stats=stats;raise
