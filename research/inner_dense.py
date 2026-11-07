"""Piecewise polynomials from Radau's modified-fast-system dense output.

This is a diagnostic curve, not a claimed MRI continuous extension.
The SciPy Q representation is internal and covered by exact-curve tests.
"""
import numpy as np
from research.mri_kernel import Hermite


class Polynomial(Hermite):
    def __init__(self,a,b,c,d,h):
        self.a,self.b,self.c,self.d=[np.asarray(v).copy() for v in (a,b,c,d)]
        self.h=float(h)

    @classmethod
    def from_radau(cls,dense,z,q,slope,duration,active):
        left=dense.t_old;dt=dense.h
        a=np.zeros_like(z);b=.5*dt*dt/duration*slope
        c=dt*(q+left/duration*slope)
        d=z+left*q+.5*left*left/duration*slope
        a[active]=dense.Q[:,2];b[active]=dense.Q[:,1]
        c[active]=dense.Q[:,0];d[active]=dense.y_old
        return cls(a,b,c,d,dt)


class InnerCurve:
    def __init__(self,pieces):
        if not pieces or any(p.h<=0 for p in pieces):raise ValueError('Positive piece lengths required')
        self.pieces=pieces;self.edges=np.r_[0.,np.cumsum([p.h for p in pieces])];self.h=self.edges[-1]
        self.d=pieces[0].d

    def evaluate(self,query,derivative=False):
        query=np.asarray(query,dtype=float)
        if query.ndim!=1 or np.any((query<0)|(query>1)):raise ValueError('Query outside curve')
        t=query*self.h;indices=np.clip(np.searchsorted(self.edges,t,side='right')-1,0,len(self.pieces)-1)
        result=np.empty((len(self.d),len(query)))
        for index in np.unique(indices):
            mask=indices==index;p=self.pieces[index];local=np.clip((t[mask]-self.edges[index])/p.h,0.,1.)
            result[:,mask]=p.derivative(local) if derivative else p(local)
        return result

    def __call__(self,query):return self.evaluate(query)
    def derivative(self,query):return self.evaluate(query,True)
    def minima(self):return np.minimum.reduce([p.minima() for p in self.pieces])

    def join_error(self,scale):
        return max((float(np.max(abs(a(np.array([1.]))[:,0]-b.d)/scale))
                    for a,b in zip(self.pieces[:-1],self.pieces[1:])),default=0.)

    def pack(self):
        return dict(lengths=np.array([p.h for p in self.pieces]),
                    coefficients=np.stack([np.stack([p.a,p.b,p.c,p.d]) for p in self.pieces]))

    @classmethod
    def unpack(cls,lengths,coefficients):
        return cls([Polynomial(*c,h) for h,c in zip(lengths,coefficients)])


def physical_ledger(case,curve,initial,final):
    from research.mri_kernel import GAUSS
    weights=case['options'].get('ledger_weights')
    if weights is None:return {}
    totals=[]
    for n in (8,16):
        nodes,w=GAUSS[n];total=0.
        for piece in curve.pieces:
            values=[weights@(case['system'].S@case['system'].flux(0.,z)) for z in piece((nodes+1)/2).T]
            total+=piece.h/2*float(w@values)
        totals.append(total)
    den=max(1.,abs(float(weights@initial)))
    return dict(physical_ledger_residual=abs(float(weights@(final-initial))-totals[1])/den,
                physical_ledger_quadrature=abs(totals[1]-totals[0])/den)


def full_curve(case,y,h,tol,screen):
    import time
    from research.ordered_radau import OrderedRadau
    start=time.perf_counter();sys=case['system'];scale=case['scale'];pool=case['options']['pools']
    initial_pool=pool@y;den=max(1e-12,float(np.max(abs(initial_pool))));drift=0.
    def rhs(t,z):
        if time.perf_counter()-start>20:raise TimeoutError('Full dense budget')
        if not np.isfinite(z).all() or np.any(z < -1e-12*scale):raise ValueError('Invalid full RHS state')
        return sys.S@sys.flux(0.,z)
    solver=OrderedRadau(rhs,0.,y,h,rtol=tol,atol=.01*tol*scale,jac=lambda t,z:sys.S@screen.gradient(z))
    pieces=[];zeros=np.zeros_like(y);active=np.arange(len(y))
    while solver.status=='running':
        message=solver.step()
        if solver.status=='failed':raise RuntimeError(message)
        if not np.isfinite(solver.y).all() or np.any(solver.y < -1e-12*scale):raise ValueError('Invalid full accepted state')
        drift=max(drift,float(np.max(abs(pool@solver.y-initial_pool)))/den)
        pieces.append(Polynomial.from_radau(solver.dense_output(),y,zeros,zeros,h,active))
    curve=InnerCurve(pieces)
    stats=dict(nfev=solver.nfev,njev=solver.njev,nlu=solver.nlu,accepted_steps=len(pieces),pool_error=drift,
        continuous_nonnegative=bool(np.all(curve.minima()>=-1e-12*scale)),dense_join_error=curve.join_error(scale))
    if stats['dense_join_error']>1e-10:raise ValueError('Discontinuous full curve')
    stats.update(physical_ledger(case,curve,y,solver.y));stats['seconds']=time.perf_counter()-start
    return dict(fine=solver.y.copy(),pieces=[curve],stats=stats)
