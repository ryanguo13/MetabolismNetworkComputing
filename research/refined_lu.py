"""Transaction-local LU reuse as a preconditioner for the CURRENT matrix.

Never treats a changed matrix as an exact factor hit. Residual and correction
checks precede acceptance; bounded refinement falls back to a current LU.
Forked caches share immutable factor handles only; replacement is local.
"""
import hashlib
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from scipy.sparse import issparse
from scipy.sparse.linalg import splu


def matrix_key(a):
    return (issparse(a),a.shape,a.dtype.str)


def same(a,b):
    return (np.array_equal(a.indptr,b.indptr) and np.array_equal(a.indices,b.indices) and np.array_equal(a.data,b.data)) if issparse(a) else np.array_equal(a,b)


def fingerprint(a):
    h=hashlib.sha256();h.update(str(matrix_key(a)).encode())
    for v in ((a.indptr,a.indices,a.data) if issparse(a) else (a,)):h.update(v.tobytes())
    return h.hexdigest()


class RefinedLU:
    def __init__(self,enabled=True,correction_tol=1e-12,residual_tol=1e-10,max_corrections=3):
        self.enabled=enabled;self.correction_tol=correction_tol;self.residual_tol=residual_tol;self.max_corrections=max_corrections;self.entries={}
        self.stats=dict(factorizations=0,requested_matrices=0,exact_hits=0,preconditioned_requests=0,solves=0,corrections=0,refreshes=0,max_residual=0.,max_correction=0.)

    def fork(self):
        new=RefinedLU(self.enabled,self.correction_tol,self.residual_tol,self.max_corrections);new.entries=self.entries.copy();return new

    def signatures(self):return {str(k):fingerprint(v[0]) for k,v in self.entries.items()}

    def factor(self,a):
        saved=a.copy()
        if issparse(a):
            factor=splu(a,permc_spec='MMD_AT_PLUS_A');solve=factor.solve
        else:
            factor=lu_factor(a);solve=lambda r:lu_solve(factor,r)
        self.stats['factorizations']+=1;entry=(saved,solve);self.entries[matrix_key(a)]=entry;return entry

    def operator(self,a,scale,residual_scale=None):
        self.stats['requested_matrices']+=1;key=matrix_key(a);entry=self.entries.get(key) if self.enabled else None
        exact=entry is not None and same(a,entry[0])
        if entry is None:entry=self.factor(a);exact=True
        elif exact:self.stats['exact_hits']+=1
        else:self.stats['preconditioned_requests']+=1
        return RefinedOperator(self,a.copy(),np.asarray(scale),entry,exact,residual_scale)


class RefinedOperator:
    def __init__(self,cache,matrix,scale,entry,exact,residual_scale):
        self.cache=cache;self.matrix=matrix;self.scale=scale;self.entry=entry;self.exact=exact
        self.residual_scale=scale if residual_scale is None else np.asarray(residual_scale)

    def raw_solve(self,r):
        self.cache.stats['solves']+=1;return self.entry[1](r)

    def solve(self,rhs):
        c=self.cache;x=self.raw_solve(rhs)
        if not self.exact:
            for _ in range(c.max_corrections):
                d=self.raw_solve(rhs-self.matrix@x);c.stats['corrections']+=1;x=x+d
                norm=float(np.max(abs(d)/np.maximum(abs(x),1e-14*self.scale)));res=float(np.max(abs(rhs-self.matrix@x)/self.residual_scale))
                if np.isfinite(norm) and np.isfinite(res) and norm<=c.correction_tol and res<=c.residual_tol:
                    c.stats['max_correction']=max(c.stats['max_correction'],norm);c.stats['max_residual']=max(c.stats['max_residual'],res);return x
            c.stats['refreshes']+=1;self.entry=c.factor(self.matrix);self.exact=True;x=self.raw_solve(rhs)
        res=float(np.max(abs(rhs-self.matrix@x)/self.residual_scale));c.stats['max_residual']=max(c.stats['max_residual'],res)
        if not np.isfinite(x).all() or not np.isfinite(res) or res>c.residual_tol:raise FloatingPointError('Current linear equation residual exceeds tolerance')
        return x
