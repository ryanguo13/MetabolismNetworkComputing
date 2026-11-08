"""One-shift inverse Arnoldi diagnostic of a frozen affine flow.

In scaled displacement coordinates x'=A x+b, x(0)=0, build an orthonormal
basis from b and (I-shift*H*A)^-1. Project A directly, then exponentiate the
small augmented system [H*V.T*A*V,H*V.T*b;0,0]. This is a linear diagnostic,
not a nonlinear reaction-split solver or a positivity/continuous-output proof.
"""
import time
import numpy as np
from scipy.linalg import expm,lu_factor,lu_solve
from scipy.sparse import diags,eye
from scipy.sparse.linalg import splu


def action(jacobian,rhs,initial,scale,h,query,dimension=12,shift=.25):
    tick=time.perf_counter();q=np.asarray(query,dtype=float);n=len(initial)
    if q.ndim!=1 or not len(q) or np.any(~np.isfinite(q)) or np.any((q<0)|(q>1)) or np.any(np.diff(q)<0):raise ValueError('Ordered finite queries in[0,1] required')
    if not isinstance(dimension,int) or dimension<1 or not np.isfinite(h) or h<=0 or not np.isfinite(shift) or shift<=0:raise ValueError('Positive dimension,h,shift required')
    if np.any(np.asarray(scale)<=0):raise ValueError('Positive scale required')
    a=(diags(1/scale)@jacobian@diags(scale)).tocsc();b=rhs/scale;beta=float(np.linalg.norm(b));maxdim=min(n,dimension)
    stats=dict(nlu=0,solves=0,matvecs=0,requested_dimension=dimension,actual_dimension=0,breakdown=False,solve_residual=0.,exponentials=0)
    if beta==0:
        stats.update(seconds=time.perf_counter()-tick,basis_seconds=0.,evaluation_seconds=0.,residual_scaled=0.,orthogonality=0.)
        return dict(values=np.repeat(initial[:,None],len(q),axis=1),derivatives=np.zeros((n,len(q))),basis=np.empty((n,0)),projection=np.empty((0,0)),drive=np.empty(0),stats=stats)
    matrix=eye(n,format='csc')-shift*h*a
    if n<=32:
        lu=lu_factor(matrix.toarray());solve=lambda r:lu_solve(lu,r)
    else:lu=splu(matrix,permc_spec='MMD_AT_PLUS_A');solve=lu.solve
    stats['nlu']=1;vectors=[b/beta]
    for k in range(maxdim-1):
        w=solve(vectors[-1]);stats['solves']+=1
        stats['solve_residual']=max(stats['solve_residual'],float(np.linalg.norm(matrix@w-vectors[-1],np.inf)))
        original=float(np.linalg.norm(w));v=np.column_stack(vectors)
        # Two modified Gram-Schmidt passes; do not rely on a single projection
        # when shifted inverse modes cluster near zero or one.
        for _ in range(2):
            for col in v.T:w-=col*np.dot(col,w)
        length=float(np.linalg.norm(w))
        if length<=100*np.finfo(float).eps*max(original,1.):stats['breakdown']=True;break
        vectors.append(w/length)
    v=np.column_stack(vectors);av=a@v;stats['matvecs']=v.shape[1];j=v.T@av;drive=v.T@b
    stats.update(actual_dimension=v.shape[1],orthogonality=float(np.max(abs(v.T@v-np.eye(v.shape[1])))),basis_seconds=time.perf_counter()-tick)
    start=time.perf_counter();m=len(drive);block=np.zeros((m+1,m+1));block[:m,:m]=h*j;block[:m,-1]=h*drive
    # Exact semigroup propagation across requested query intervals. Cache only
    # bitwise-equal increments; do not round time gaps to manufacture reuse.
    state=np.r_[np.zeros(m),1.];last=0.;cache={};coeff=[]
    for x in q:
        delta=float(x-last)
        if delta:
            if delta not in cache:cache[delta]=expm(delta*block);stats['exponentials']+=1
            state=cache[delta]@state
        coeff.append(state[:m].copy());last=x
    coeff=np.column_stack(coeff);displacement=v@coeff;derivatives=v@(j@coeff+drive[:,None])
    residual=derivatives-(av@coeff+b[:,None])
    stats.update(residual_scaled=float(np.max(abs(h*residual))),evaluation_seconds=time.perf_counter()-start,seconds=time.perf_counter()-tick)
    values=initial[:,None]+scale[:,None]*displacement
    if not np.isfinite(values).all():raise FloatingPointError('Nonfinite projected exponential')
    return dict(values=values,derivatives=scale[:,None]*derivatives,basis=v,projection=j,drive=drive,stats=stats)
