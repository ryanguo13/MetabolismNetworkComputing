"""One fast IVP driven by past slow RHS values (Adams limit when fast is zero).

Research prototype: polynomial extrapolation of additive forcing, not a claimed
implementation of a particular published MR-LMM tableau. Caller retains audits.
"""
import math
import time
import numpy as np
from research.mri_kernel import MRI,GAUSS
from research.history_polynomial import HistoryPolynomial
from research.inner_dense import InnerCurve,physical_ledger
from research.shared_radau import SharedFactory
from research.fullstep_radau import FullStepFactory


class DensePower(HistoryPolynomial):
    def __init__(self,coefficients,h):
        self.coefficients=np.asarray(coefficients).copy()
        self.h=float(h);self.degree=self.coefficients.shape[1]-1;self.d=self.coefficients[:,0]


class SlowPolynomial:
    def __init__(self,times,values,h):
        times=np.asarray(times,dtype=float);values=np.asarray(values,dtype=float)
        if len(times) not in (2,3,4) or values.ndim!=2 or values.shape[1]!=len(times):
            raise ValueError('Two to four slow RHS samples required')
        if not np.isfinite(h) or h<=0 or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or not np.isfinite(values).all():
            raise ValueError('Finite ordered history and positive step required')
        nodes=(times-times[-1])/h
        self.coefficients=np.linalg.solve(np.vander(nodes,len(times),increasing=True),values.T).T
        self.h=h;self.degree=len(times)-1
        self.integrated=np.column_stack([np.zeros(values.shape[0]),h*self.coefficients/np.arange(1,len(times)+1)])

    def value(self,t):return np.polynomial.polynomial.polyval(t/self.h,self.coefficients.T)
    def integral(self,t):return np.polynomial.polynomial.polyval(t/self.h,self.integrated.T)

    def dense(self,initial,left,dt,active,dense=None):
        degree=max(3,self.degree+1);c=np.zeros((len(initial),degree+1));c[:,0]=initial
        for j in range(1,self.degree+2):
            for k in range(j+1):
                c[:,k]+=self.integrated[:,j]*math.comb(j,k)*(left/self.h)**(j-k)*(dt/self.h)**k
        if dense is not None:
            c[active]=0.;c[active,0]=dense.y_old;c[active,1:4]=dense.Q
        return DensePower(c,dt)


class ExtrapolatedFlow(MRI):
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
            value=self.sfa@self.system.flux(0.,z)[self.cols]+forcing.value(t)[self.active]
            stats['rhs_seconds']+=time.perf_counter()-before;return value
        def jac(t,u):
            before=time.perf_counter();z=expand(t,u);check(z)
            value=(self.sfa@self.screen.gradient(z)[self.cols][:,self.active]).tocsc()
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
                            check(z);vals.append(self.weights@(self.sf@self.system.flux(0.,z)[self.cols]+forcing.value(t)))
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


def pack(curve):return dict(lengths=np.array([p.h for p in curve.pieces]),coefficients=np.stack([p.coefficients for p in curve.pieces]))
def unpack(lengths,coefficients):return InnerCurve([DensePower(c,h) for h,c in zip(lengths,coefficients)])
