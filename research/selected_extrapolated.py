"""Exact selected-reaction version of frozen ExtrapolatedFlow.

Original forcing, Radau, output, nonnegative/pool and physical/modified ledger
guards are retained. Only reaction-rate/derivative evaluation and the equivalent
slow complement sum change. No runtime audits are removed or coverage claimed.
"""
import time
import numpy as np
from research.mri_kernel import MRI,GAUSS
from research.extrapolated_flow import SlowPolynomial
from research.inner_dense import InnerCurve,physical_ledger
from research.shared_radau import SharedFactory
from research.fullstep_radau import FullStepFactory
from research.selected_reactions import SelectedReactions


class SelectedExtrapolatedFlow(MRI):
    def __init__(self, case, groups, **kwargs):
        tick=time.perf_counter()
        super().__init__(case,groups,**kwargs)
        model=case['model']
        if self.system.flux is not model.flux and self.system.flux != model.flux:
            raise ValueError('Selected kinetics require the admitted original model flux')
        self.fast_rates=SelectedReactions(model,self.cols,1e-12*self.scale)
        self.slow_columns=np.setdiff1d(np.arange(self.system.S.shape[1]),self.cols)
        self.slow_rates=SelectedReactions(model,self.slow_columns,1e-12*self.scale)
        self.slow_matrix=self.system.S[:,self.slow_columns]
        self.setup_seconds=time.perf_counter()-tick

    def slow(self,y):
        return self.slow_matrix@self.slow_rates.flux(0.,y)

    def trial(self,y,h,history_times,past_slow,tol=1e-10,whole_step=False,timeout=20.):
        tick=time.perf_counter();y=np.asarray(y,dtype=float).copy()
        if self.backend!='radau' or not np.isfinite(tol) or tol<=0:
            raise ValueError('Positive tolerance and Radau required')
        stats=dict(subflows=1,nfev=0,njev=0,nlu=0,accepted_steps=0,rhs_seconds=0.,jacobian_seconds=0.,
                   min_rhs=float(y.min()),min_internal=float(y.min()),pool_error=0.,ledger_seconds=0.)
        pool0=self.pools@y;poolden=max(1e-12,float(np.max(abs(pool0))))
        def check(z):
            if time.perf_counter()-tick>timeout:raise TimeoutError('Extrapolated-flow budget')
            if not np.isfinite(z).all() or np.any(z < -1e-12*self.scale):raise ValueError('Invalid extrapolated-flow state; no clipping')
        def accepted(z):
            check(z);stats['min_internal']=min(stats['min_internal'],float(z.min()))
            stats['pool_error']=max(stats['pool_error'],float(np.max(abs(self.pools@z-pool0)))/poolden)
        check(y)
        times=np.asarray(history_times);past=np.asarray(past_slow)
        if times.ndim!=1 or past.shape!=(len(y),len(times)-1):raise ValueError('History shape mismatch')
        forcing=SlowPolynomial(times,np.column_stack([past,self.slow(y)]),h)
        def expand(t,u):
            z=y+forcing.integral(t);z[self.active]=u;return z
        def rhs(t,u):
            before=time.perf_counter();z=expand(t,u);check(z)
            stats['min_rhs']=min(stats['min_rhs'],float(z.min()))
            value=self.sfa@self.fast_rates.flux(0.,z)+forcing.value(t)[self.active]
            stats['rhs_seconds']+=time.perf_counter()-before;return value
        def jac(t,u):
            before=time.perf_counter();z=expand(t,u);check(z)
            value=(self.sfa@self.fast_rates.gradient(0.,z)[:,self.active]).tocsc()
            stats['jacobian_seconds']+=time.perf_counter()-before;return value
        try:
            pieces=[];ledger=0.;quadrature=0.
            if len(self.active):
                factory=FullStepFactory() if whole_step else SharedFactory()
                solver=factory(rhs,0.,y[self.active],h,rtol=tol,atol=.01*tol*self.scale[self.active],jac=jac)
                while solver.status=='running':
                    left=solver.t;message=solver.step()
                    if solver.status=='failed':raise RuntimeError(message)
                    accepted(expand(solver.t,solver.y))
                    pieces.append(forcing.dense(y,left,solver.t-left,self.active,solver.dense_output()))
                end=expand(h,solver.y)
                for key in ('nfev','njev','nlu'):stats[key]=getattr(solver,key)
                stats.update(factory.stats())
            else:
                end=y+forcing.integral(h);accepted(end)
                pieces=[forcing.dense(y,0.,h,self.active)]
            stats['accepted_steps']=len(pieces)
            curve=InnerCurve(pieces);left=0.
            if self.weights is not None:
                before=time.perf_counter()
                for piece in pieces:
                    integrals=[]
                    for n in (4,8):
                        nodes,w=GAUSS[n];q=(nodes+1)/2;zs=piece(q);ts=left+piece.h*q;vals=[]
                        for t,z in zip(ts,zs.T):
                            check(z);vals.append(self.weights@(self.sf@self.fast_rates.flux(0.,z)+forcing.value(t)))
                        integrals.append(float(piece.h/2*(w@vals)))
                    ledger+=integrals[1];quadrature+=abs(integrals[1]-integrals[0]);left+=piece.h
                den=max(1.,abs(float(self.weights@y)))
                stats.update(modified_ledger_residual=abs(float(self.weights@(end-y))-ledger)/den,
                             modified_ledger_quadrature=quadrature/den)
                stats['ledger_seconds']=time.perf_counter()-before
            stats['continuous_nonnegative']=bool(np.all(curve.minima()>=-1e-12*self.scale))
            stats['dense_join_error']=curve.join_error(self.scale)
            if stats['dense_join_error']>1e-10:raise ValueError('Discontinuous extrapolated-flow output')
            if stats['continuous_nonnegative']:stats.update(physical_ledger(self.case,curve,y,end))
            stats['seconds']=time.perf_counter()-tick
            return dict(fine=end,pieces=[curve],stats=stats,forcing_coefficients=forcing.coefficients)
        except Exception as exc:
            stats['seconds']=time.perf_counter()-tick;exc.stats=stats;raise
