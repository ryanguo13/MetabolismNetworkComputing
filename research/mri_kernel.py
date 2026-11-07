"""Auditable MRI2 research kernel with exact evolution of inactive coordinates."""
import time
from types import SimpleNamespace

import numpy as np
from scipy.integrate import Radau
from scipy.sparse import csc_matrix

from metabolism.adaptive import GroupScreen
from research.mri_native import NativeSession

GAUSS = {n:np.polynomial.legendre.leggauss(n) for n in (4,8,16)}


class Hermite:
    def __init__(self,a,b,fa,fb,h):
        self.h = h
        self.d = a.copy()
        self.c = h*fa
        self.b = 3*(b-a)-h*(2*fa+fb)
        self.a = 2*(a-b)+h*(fa+fb)

    def __call__(self,s):
        s = np.asarray(s)
        return ((self.a[:,None]*s+self.b[:,None])*s+self.c[:,None])*s+self.d[:,None]

    def derivative(self,s):
        s = np.asarray(s)
        return ((3*self.a[:,None]*s+2*self.b[:,None])*s+self.c[:,None])/self.h

    def minima(self):
        """Component minima at endpoints and all real interior stationary points."""
        out = np.minimum(self.d,self.a+self.b+self.c+self.d)
        aa, bb, cc = 3*self.a,2*self.b,self.c
        roots = []
        linear = (aa == 0) & (bb != 0)
        root = np.full(len(aa),np.nan)
        root[linear] = -cc[linear]/bb[linear]
        roots.append(root)
        disc = bb*bb-4*aa*cc
        valid = (aa != 0) & (disc >= 0)
        # Stable quadratic formula, with a repeated-zero-root special case.
        q = np.full(len(aa),np.nan)
        q[valid] = -.5*(bb[valid]+np.copysign(np.sqrt(disc[valid]),bb[valid]))
        r1, r2 = np.full(len(aa),np.nan),np.full(len(aa),np.nan)
        r1[valid] = q[valid]/aa[valid]
        nonzero = valid & (q != 0)
        r2[nonzero] = cc[nonzero]/q[nonzero]
        r2[valid & (q == 0)] = 0.
        roots.extend((r1,r2))
        for r in roots:
            use = np.isfinite(r) & (r > 0) & (r < 1)
            s = r[use]
            out[use] = np.minimum(out[use],((self.a[use]*s+self.b[use])*s+self.c[use])*s+self.d[use])
        return out


class MRI:
    def __init__(self,case,groups,backend='radau',reduced=True,reuse=True):
        tick = time.perf_counter()
        self.case,self.system,self.scale = case,case['system'],case['scale']
        self.screen = GroupScreen(self.system,self.scale)
        self.groups = np.asarray(groups,dtype=int)
        if (len(np.unique(self.groups)) != len(self.groups) or np.any(self.groups < 0)
                or np.any(self.groups >= len(self.system.groups)) or self.system.time_dependent):
            raise ValueError('Invalid groups or nonautonomous system')
        self.cols = (np.concatenate([self.system.groups[g] for g in self.groups])
                     if len(self.groups) else np.array([],dtype=int))
        self.sf = csc_matrix(self.system.S)[:,self.cols]
        self.active = (np.flatnonzero(np.asarray(abs(self.sf).sum(axis=1)).ravel())
                       if reduced else np.arange(len(self.scale)))
        self.inactive = np.setdiff1d(np.arange(len(self.scale)),self.active)
        self.sfa = self.sf[self.active].tocsc()
        self.pattern = (abs(self.sfa)@csc_matrix(self.system.dependencies)[self.cols][:,self.active]).astype(bool).tocsc()
        self.backend,self.reuse = backend,reuse
        self.weights = case['options'].get('ledger_weights')
        self.pools = csc_matrix(case['options']['pools'])
        if backend not in ('radau','native') or (backend == 'native' and self.weights is not None):
            raise ValueError('Native ledger is not implemented; use Radau for open systems')
        self.native = NativeSession(self.pattern) if backend == 'native' and len(self.active) and reuse else None
        self.setup_seconds = time.perf_counter()-tick

    def full(self,y):
        return self.system.S@self.system.flux(0.,y)

    def slow(self,y):
        rates = self.system.flux(0.,y)
        return self.system.S@rates-self.sf@rates[self.cols]

    def close(self):
        if self.native is not None:
            self.native.close()
            self.native = None

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

        def flow(z,duration,q):
            stats['subflows'] += 1
            def expand(t,u):
                full = z+t*q
                full[self.active] = u
                return full
            def rhs(t,u):
                before = time.perf_counter()
                full = expand(t,u)
                stats['min_rhs'] = min(stats['min_rhs'],float(full.min()))
                check(full,'RHS argument')
                answer = self.sfa@self.system.flux(0.,full)[self.cols]+q[self.active]
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
                answer = z+duration*q
                accepted(duration,answer)
                return answer,(float(self.weights@q)*duration if self.weights is not None else 0.),0.
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
                            vals.append(self.weights@(self.sf@self.system.flux(0.,full)[self.cols]+q))
                        integral.append(float((solver.t-left)/2*(w@vals)))
                    ledger += integral[1]
                    quad += abs(integral[1]-integral[0])
                    stats['ledger_seconds'] += time.perf_counter()-before
            for key in ('nfev','njev','nlu'):
                stats[key] += getattr(solver,key)
            return expand(duration,solver.y),ledger,quad

        def one(z,duration):
            g0 = self.slow(z)
            mid,l1,q1 = flow(z,duration/2,g0)
            end,l2,q2 = flow(mid,duration/2,-g0+2*self.slow(mid))
            return end,l1+l2,q1+q2
        try:
            coarse,_,_ = one(y,h)
            mid,l1,q1 = one(y,h/2)
            fine,l2,q2 = one(mid,h/2)
            deriv = [self.full(z) for z in (y,mid,fine)]
            pieces = [Hermite(y,mid,deriv[0],deriv[1],h/2),Hermite(mid,fine,deriv[1],deriv[2],h/2)]
            coarse_piece = Hermite(y,coarse,deriv[0],self.full(coarse),h)
            minimum = np.minimum(pieces[0].minima(),pieces[1].minima())
            stats['continuous_nonnegative'] = bool(np.all(minimum >= -1e-12*self.scale))
            stats['min_hermite_scaled'] = float(np.min(minimum/self.scale))
            if self.weights is not None:
                before = time.perf_counter()
                denominator = max(1.,abs(float(self.weights@y)))
                change = float(self.weights@(fine-y))
                stats.update(modified_ledger_residual=abs(change-l1-l2)/denominator,
                             modified_ledger_quadrature=(q1+q2)/denominator)
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
            return dict(coarse=coarse,mid=mid,fine=fine,pieces=pieces,coarse_piece=coarse_piece,stats=stats)
        except Exception as exc:
            stats['seconds']=time.perf_counter()-tick
            exc.stats=stats
            raise


def output(trial,fractions):
    fractions=np.asarray(fractions)
    if np.any((fractions < 0)|(fractions > 1)):
        raise ValueError('Outside macro interval')
    result=np.empty((len(trial['fine']),len(fractions)))
    for i,piece in enumerate(trial['pieces']):
        mask=(fractions <= .5) if i == 0 else (fractions > .5)
        result[:,mask]=piece(2*fractions[mask]-i)
    return result
