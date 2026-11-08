"""Shared inverse Arnoldi basis with direct and inverse-Ritz projection.

K=(I-sigma*H*A)^-1, Q=V.T*K*V. Inverse-Ritz projection uses
T=(I-Q^-1)/(sigma*H), instead of T=V.T*A*V. Neither projection
is unconditionally stable for nonnormal A. Residual and enrichment changes
are heuristic proposals only; independent full-space audit remains required.
"""
import time
import numpy as np
from scipy.linalg import expm,lu_factor,lu_solve
from scipy.sparse import diags,eye
from scipy.sparse.linalg import splu


class RationalBasis:
    def __init__(self,jacobian,rhs,initial,scale,h,shift=.25):
        tick=time.perf_counter();self.y=np.asarray(initial).copy();self.scale=np.asarray(scale).copy();self.h=h;self.shift=shift
        if not np.isfinite(h) or h<=0 or not np.isfinite(shift) or shift<=0 or np.any(~np.isfinite(self.scale)) or np.any(self.scale<=0):raise ValueError('Positive finite scales,h,shift required')
        self.a=(diags(1/self.scale)@jacobian@diags(self.scale)).tocsc();self.b=np.asarray(rhs)/self.scale;self.n=len(initial)
        if not np.isfinite(self.a.data).all() or not np.isfinite(self.b).all():raise ValueError('Finite linear problem required')
        self.matrix=eye(self.n,format='csc')-shift*h*self.a
        if self.n<=32:
            lu=lu_factor(self.matrix.toarray());self.solve=lambda x:lu_solve(lu,x)
        else:lu=splu(self.matrix,permc_spec='MMD_AT_PLUS_A');self.solve=lu.solve
        beta=float(np.linalg.norm(self.b));self.vectors=[] if beta==0 else [self.b/beta];self.images=[];self.av=[];self.pending=None;self.breakdown=beta==0
        self.stats=dict(nlu=1,solves=0,matvecs=0,solve_residual=0.,basis_seconds=time.perf_counter()-tick)

    def extend(self,dimension):
        if not isinstance(dimension,int) or isinstance(dimension,bool) or dimension<1:raise ValueError('Positive integer dimension required')
        tick=time.perf_counter();requested=min(dimension,self.n)
        # Every included vector has its K image, including the final vector;
        # this costs m solves, not m-1, and supports Q without a new factor.
        while self.vectors:
            if len(self.images)<len(self.vectors):
                v=self.vectors[-1];w=self.solve(v);self.stats['solves']+=1
                self.stats['solve_residual']=max(self.stats['solve_residual'],float(np.linalg.norm(self.matrix@w-v,np.inf)))
                self.images.append(w.copy());self.av.append(self.a@v);self.stats['matvecs']+=1
                length0=float(np.linalg.norm(w))
                for _ in range(2):
                    for col in self.vectors:w-=col*np.dot(col,w)
                length=float(np.linalg.norm(w));self.breakdown=length<=100*np.finfo(float).eps*max(length0,1.)
                self.pending=None if self.breakdown else w/length
            if len(self.vectors)>=requested or self.breakdown:break
            self.vectors.append(self.pending.copy())
        self.stats['basis_seconds']+=time.perf_counter()-tick

    def evaluate(self,query,projection='inverse'):
        tick=time.perf_counter();q=np.asarray(query,dtype=float)
        if projection not in ('inverse','direct'):raise ValueError('Unknown projection')
        if q.ndim!=1 or not len(q) or np.any(~np.isfinite(q)) or np.any((q<0)|(q>1)) or np.any(np.diff(q)<0):raise ValueError('Ordered query in[0,1] required')
        m=len(self.vectors)
        if not m:
            return dict(values=np.repeat(self.y[:,None],len(q),axis=1),displacement=np.zeros((self.n,len(q))),coefficients=np.empty((0,len(q))),basis=np.empty((self.n,0)),projection=np.empty((0,0)),drive=np.empty(0),residual=np.zeros((self.n,len(q))),stats=dict(**self.stats,dimension=0,breakdown=True,orthogonality=0.,inverse_condition=1.,residual_scaled=0.,spectral_abscissa=0.,evaluation_seconds=time.perf_counter()-tick,exponentials=0))
        if len(self.images)!=m:raise RuntimeError('Extend basis before evaluating')
        v=np.column_stack(self.vectors);av=np.column_stack(self.av);kinverse=np.column_stack(self.images);Q=v.T@kinverse
        if projection=='inverse':t=(np.eye(m)-np.linalg.solve(Q,np.eye(m)))/(self.shift*self.h)
        else:t=v.T@av
        drive=v.T@self.b;block=np.zeros((m+1,m+1));block[:m,:m]=self.h*t;block[:m,-1]=self.h*drive
        state=np.r_[np.zeros(m),1.];previous=0.;cache={};coeff=[]
        for x in q:
            dt=float(x-previous)
            if dt:
                if dt not in cache:cache[dt]=expm(dt*block)
                state=cache[dt]@state
            coeff.append(state[:m].copy());previous=x
        coeff=np.column_stack(coeff);displacement=v@coeff;values=self.y[:,None]+self.scale[:,None]*displacement
        residual=(v@t-av)@coeff+(v@drive-self.b)[:,None]
        if not np.isfinite(values).all() or not np.isfinite(residual).all():raise FloatingPointError('Nonfinite reduced action')
        stats=dict(**self.stats,dimension=m,breakdown=bool(self.breakdown),orthogonality=float(np.max(abs(v.T@v-np.eye(m)))),inverse_condition=float(np.linalg.cond(Q)),residual_scaled=float(np.max(abs(self.h*residual))),spectral_abscissa=float(max(np.linalg.eigvals(t).real)),evaluation_seconds=time.perf_counter()-tick,exponentials=len(cache))
        return dict(values=values,displacement=displacement,coefficients=coeff,basis=v,projection=t,drive=drive,residual=residual,stats=stats)


def proposal(current,previous,h,scale,initial,target=1e-6,budget=.05):
    """Conservative sampled indicators, NOT certified forward-error bounds.

    Use fixed initial scale for residual/enrichment indicators. Require two
    dimensions even for a lucky low-dimensional approximation. Full audit is
    deliberately a separate mandatory decision outside this function.
    """
    denominator=target*(.01+np.maximum(abs(initial)/scale,1.))
    residual=float(np.max(abs(h*current['residual'])/denominator[:,None]))
    change=None if previous is None else float(np.max(abs(current['displacement']-previous['displacement'])/denominator[:,None]))
    return dict(residual_indicator=residual,enrichment_indicator=change,proposed=bool(change is not None and max(residual,change)<=budget),certified=False)
