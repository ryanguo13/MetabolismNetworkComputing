"""Nonlinear endpoint slow interpolation with auditable repeated fast solves.

The frozen slow-only Jacobian is an iteration preconditioner, not a replacement
for the nonlinear equation. Failure to reach the registered residual is failure.
"""
import time
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from research.mri_kernel import MRI,GAUSS
from research.implicit_history import EndpointPolynomial
from research.fullstep_radau import FullStepFactory
from research.inner_dense import InnerCurve,physical_ledger


class EndpointImplicitFlow(MRI):
    def trial(self,y,h,history_times,past_slow,tol=1e-8,iteration='newton',residual_tol=1e-11,maxiter=12,timeout=20.):
        tick=time.perf_counter();y=np.asarray(y,dtype=float).copy();times=np.asarray(history_times,dtype=float)
        past=np.asarray(past_slow,dtype=float)
        if (iteration not in ('newton','fixed') or times.ndim!=1 or len(times) not in (1,2,3)
                or past.shape!=(len(y),len(times)-1) or not np.isfinite(tol) or tol<=0
                or not np.isfinite(residual_tol) or residual_tol<=0 or maxiter<1):raise ValueError('Invalid implicit controls/history')
        stats=dict(subflows=0,nfev=0,njev=0,nlu=0,endpoint_njev=0,endpoint_nlu=0,endpoint_solves=0,
                   accepted_steps=0,pool_error=0.,min_rhs=float(y.min()),min_internal=float(y.min()),iteration_residuals=[])
        pool0=self.pools@y;poolden=max(1e-12,float(np.max(abs(pool0))))
        def check(z):
            if time.perf_counter()-tick>timeout:raise TimeoutError('Implicit history budget')
            if not np.isfinite(z).all() or np.any(z < -1e-12*self.scale):raise ValueError('Invalid implicit state; no clipping')
        def accepted(z):
            check(z);stats['min_internal']=min(stats['min_internal'],float(z.min()))
            stats['pool_error']=max(stats['pool_error'],float(np.max(abs(self.pools@z-pool0)))/poolden)
        def flow(forcing):
            stats['subflows']+=1
            def expand(t,u):
                z=y+forcing.integral(t);z[self.active]=u;return z
            def rhs(t,u):
                z=expand(t,u);check(z);stats['min_rhs']=min(stats['min_rhs'],float(z.min()))
                return self.sfa@self.system.flux(0.,z)[self.cols]+forcing.value(t)[self.active]
            def jac(t,u):
                z=expand(t,u);check(z)
                return (self.sfa@self.screen.gradient(z)[self.cols][:,self.active]).tocsc()
            pieces=[]
            if len(self.active):
                solver=FullStepFactory()(rhs,0.,y[self.active],h,rtol=tol,atol=.01*tol*self.scale[self.active],jac=jac)
                while solver.status=='running':
                    left=solver.t;message=solver.step()
                    if solver.status=='failed':raise RuntimeError(message)
                    accepted(expand(solver.t,solver.y))
                    pieces.append(forcing.dense(y,left,solver.t-left,self.active,solver.dense_output()))
                end=expand(h,solver.y)
                for key in ('nfev','njev','nlu'):stats[key]+=getattr(solver,key)
            else:
                end=y+forcing.integral(h);accepted(end);pieces=[forcing.dense(y,0.,h,self.active)]
            stats['accepted_steps']+=len(pieces)
            curve=InnerCurve(pieces)
            if curve.join_error(self.scale)>1e-10:raise ValueError('Discontinuous implicit flow')
            if np.any(curve.minima() < -1e-12*self.scale):raise ValueError('Negative implicit flow output')
            return end,curve
        try:
            check(y);current=self.slow(y);values=np.column_stack([past,current]);nodes=np.r_[times,times[-1]+h]
            # Validate history before constructing the nonlinear system.
            basis=EndpointPolynomial(nodes,np.eye(len(nodes))[-1:],times[-1],h)
            future_weight=float(basis.integral(h)[0])
            factor=None
            if iteration=='newton':
                gradient=self.screen.gradient(y);slowjac=(self.system.S@gradient-self.sf@gradient[self.cols]).tocsc()
                factor=splu(eye(len(y),format='csc')-future_weight*slowjac)
                stats['endpoint_njev']=1;stats['endpoint_nlu']=1
            guess=y+h*self.full(y)
            if not np.isfinite(guess).all() or np.any(guess < -1e-12*self.scale):guess=y.copy()
            for count in range(maxiter):
                check(guess);forcing=EndpointPolynomial(nodes,np.column_stack([values,self.slow(guess)]),times[-1],h)
                end,curve=flow(forcing);residual=end-guess
                norm=float(np.max(abs(residual)/self.scale));stats['iteration_residuals'].append(norm)
                if norm<=residual_tol:
                    stats.update(endpoint_residual=norm,continuous_nonnegative=True,dense_join_error=curve.join_error(self.scale))
                    stats.update(physical_ledger(self.case,curve,y,end))
                    if self.weights is not None:
                        totals=[]
                        for n in (4,8):
                            points,w=GAUSS[n];left=0.;total=0.
                            for piece in curve.pieces:
                                q=(points+1)/2;zs=piece(q)
                                vals=[self.weights@(self.sf@self.system.flux(0.,z)[self.cols]+forcing.value(left+piece.h*s)) for s,z in zip(q,zs.T)]
                                total+=piece.h/2*float(w@vals);left+=piece.h
                            totals.append(total)
                        den=max(1.,abs(float(self.weights@y)))
                        stats.update(modified_ledger_residual=abs(float(self.weights@(end-y))-totals[1])/den,
                                     modified_ledger_quadrature=abs(totals[1]-totals[0])/den)
                    stats['seconds']=time.perf_counter()-tick
                    return dict(fine=end,pieces=[curve],stats=stats,forcing_coefficients=forcing.coefficients,implicit_guess=guess.copy())
                delta=factor.solve(residual) if factor is not None else residual
                stats['endpoint_solves']+=int(factor is not None)
                alpha=1.
                while np.any(guess+alpha*delta < -1e-12*self.scale) and alpha>2**-12:alpha*=.5
                guess=guess+alpha*delta
            raise RuntimeError('Implicit endpoint iteration did not converge')
        except Exception as exc:
            stats['seconds']=time.perf_counter()-tick;exc.stats=stats;raise
