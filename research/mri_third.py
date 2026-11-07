"""MRI-GARK-ERK33a (Sandu, arXiv:1808.02759v3, section 6.2).

Physical-time subintervals have length H/3; final forcing is affine.
Inactive coordinates follow its exact integral, not a frozen value.
Full independent audits are still required; no online acceptance here.
"""
import time
from types import SimpleNamespace
import numpy as np
from research.mri_kernel import Hermite,GAUSS
from research.mri_single import SingleMRI
from research.mri_native import NativeSession
from research.ordered_radau import OrderedRadau as Radau


class MRI3(SingleMRI):
    def trial(self,y,h,tol=1e-10,timeout=20.,atol=None):
        if not np.isfinite(h) or h <= 0 or not np.isfinite(tol) or tol <= 0:
            raise ValueError('Invalid step/tolerance')
        tick = time.perf_counter()
        y = np.asarray(y,dtype=float).copy()
        stats = dict(subflows=0,nfev=0,njev=0,nlu=0,linear_solver_setups=0,
                     accepted_steps=0,rhs_seconds=0.,jacobian_seconds=0.,
                     native_allocation_seconds=0.,reinit_seconds=0.,ledger_seconds=0.,
                     min_rhs=float(y.min()),min_internal=float(y.min()),pool_error=0.)
        initial_pool = np.asarray(self.pools@y).ravel()
        pool_scale = max(1e-12,float(np.max(abs(initial_pool))))
        def check(z,where):
            if time.perf_counter()-tick > timeout:
                raise TimeoutError('MRI local attempt budget')
            if not np.isfinite(z).all() or np.any(z < -1e-12*self.scale):
                raise ValueError(f'Nonfinite/significant negative {where}; no clipping')
        def accepted(t,z):
            check(z,'accepted state')
            stats['min_internal'] = min(stats['min_internal'],float(z.min()))
            stats['pool_error'] = max(stats['pool_error'],float(np.max(abs(self.pools@z-initial_pool)))/pool_scale)
        check(y,'initial state')
        absolute = .01*tol*self.scale if atol is None else np.asarray(atol)

        def flow(z,duration,q,slope=None):
            slope=np.zeros_like(q) if slope is None else slope
            def forcing(t):return q+(t/duration)*slope
            stats['subflows'] += 1
            def expand(t,u):
                full = z+t*q+(.5*t*t/duration)*slope
                full[self.active] = u
                return full
            def rhs(t,u):
                before = time.perf_counter()
                full = expand(t,u)
                stats['min_rhs'] = min(stats['min_rhs'],float(full.min()))
                check(full,'RHS argument')
                answer = self.sfa@self.system.flux(0.,full)[self.cols]+forcing(t)[self.active]
                stats['rhs_seconds'] += time.perf_counter()-before
                return answer
            def jac(t,u):
                before = time.perf_counter()
                full = expand(t,u)
                check(full,'Jacobian argument')
                answer = (self.sfa@self.screen.gradient(full)[self.cols][:,self.active]).tocsc()
                stats['jacobian_seconds'] += time.perf_counter()-before
                return answer
            def record(t,u):
                accepted(t,expand(t,u))
            if not len(self.active):
                answer = z+duration*(q+.5*slope)
                accepted(duration,answer)
                return answer,(float(self.weights@(q+.5*slope))*duration if self.weights is not None else 0.),0.
            if self.backend == 'native':
                session = self.native if self.reuse else NativeSession(self.pattern)
                if not self.reuse:
                    stats['native_allocation_seconds'] += session.setup_seconds
                try:
                    model = SimpleNamespace(y0=z[self.active],rhs=rhs,jacobian=jac,accepted=record)
                    value,counts = session.solve(model,duration,tol,absolute[self.active])
                    for key in ('nfev','njev','linear_solver_setups','accepted_steps','reinit_seconds'):
                        stats[key] += counts[key]
                    return expand(duration,value),0.,0.
                finally:
                    if not self.reuse:
                        session.close()
            solver = Radau(rhs,0.,z[self.active],duration,rtol=tol,atol=absolute[self.active],jac=jac)
            ledger,quad = 0.,0.
            while solver.status == 'running':
                left = solver.t
                message = solver.step()
                if solver.status == 'failed':
                    raise RuntimeError(message)
                record(solver.t,solver.y)
                stats['accepted_steps'] += 1
                if self.weights is not None:
                    before = time.perf_counter()
                    dense = solver.dense_output()
                    integral = []
                    for n in (4,8):
                        nodes,w = GAUSS[n]
                        ts = left+(solver.t-left)*(nodes+1)/2
                        zs = dense(ts)
                        vals = []
                        for t,u in zip(ts,zs.T):
                            full = expand(t,u)
                            check(full,'ledger quadrature state')
                            vals.append(self.weights@(self.sf@self.system.flux(0.,full)[self.cols]+forcing(t)))
                        integral.append(float((solver.t-left)/2*(w@vals)))
                    ledger += integral[1]
                    quad += abs(integral[1]-integral[0])
                    stats['ledger_seconds'] += time.perf_counter()-before
            for key in ('nfev','njev','nlu'):
                stats[key] += getattr(solver,key)
            return expand(duration,solver.y),ledger,quad

        def one(z,duration):
            g0 = self.slow(z)
            stage2,l1,q1 = flow(z,duration/3,g0)
            g1 = self.slow(stage2)
            stage3,l2,q2 = flow(stage2,duration/3,-g0+2*g1)
            g2 = self.slow(stage3)
            end,l3,q3 = flow(stage3,duration/3,-2*g1+3*g2,1.5*(g0-g2))
            return end,l1+l2+l3,q1+q2+q3
        try:
            fine,ledger,quadrature = one(y,h)
            pieces = [Hermite(y,fine,self.full(y),self.full(fine),h)]
            minimum = pieces[0].minima()
            stats['continuous_nonnegative'] = bool(np.all(minimum >= -1e-12*self.scale))
            stats['min_hermite_scaled'] = float(np.min(minimum/self.scale))
            if self.weights is not None:
                before = time.perf_counter()
                denominator = max(1.,abs(float(self.weights@y)))
                change = float(self.weights@(fine-y))
                stats.update(modified_ledger_residual=abs(change-ledger)/denominator,
                             modified_ledger_quadrature=quadrature/denominator)
                if stats['continuous_nonnegative']:
                    integrals=[]
                    for n in (8,16):
                        nodes,w=GAUSS[n]
                        total=0.
                        for piece in pieces:
                            states=piece((nodes+1)/2)
                            vals=[self.weights@self.full(z) for z in states.T]
                            total += piece.h/2*float(w@vals)
                        integrals.append(total)
                    stats.update(physical_ledger_residual=abs(change-integrals[1])/denominator,
                                 physical_ledger_quadrature=abs(integrals[1]-integrals[0])/denominator)
                stats['ledger_seconds'] += time.perf_counter()-before
            stats['seconds']=time.perf_counter()-tick
            return dict(fine=fine,pieces=pieces,stats=stats)
        except Exception as exc:
            stats['seconds']=time.perf_counter()-tick
            exc.stats=stats
            raise

