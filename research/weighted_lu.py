"""Weighted-infinity refinement after recorded component-norm stagnation.

Correction norm = ||delta/scale||inf / max(||x/scale||inf,1e-14).
Same1e-12 tolerance and current-matrix residual checks; at most8 corrections.
No macro accuracy/ledger threshold changes. Original policy remains frozen.
"""
import numpy as np
from research.refined_lu import RefinedLU,RefinedOperator,matrix_key,same


class WeightedLU(RefinedLU):
    def __init__(self,enabled=True,correction_tol=1e-12,residual_tol=1e-10,max_corrections=8):
        super().__init__(enabled,correction_tol,residual_tol,max_corrections)

    def fork(self):
        new=WeightedLU(self.enabled,self.correction_tol,self.residual_tol,self.max_corrections);new.entries=self.entries.copy();return new

    def operator(self,a,scale,residual_scale=None):
        self.stats['requested_matrices']+=1;entry=self.entries.get(matrix_key(a)) if self.enabled else None
        exact=entry is not None and same(a,entry[0])
        if entry is None:entry=self.factor(a);exact=True
        elif exact:self.stats['exact_hits']+=1
        else:self.stats['preconditioned_requests']+=1
        return WeightedOperator(self,a.copy(),np.asarray(scale),entry,exact,residual_scale)


class WeightedOperator(RefinedOperator):
    def solve(self,rhs):
        c=self.cache;x=self.raw_solve(rhs)
        if not self.exact:
            for _ in range(c.max_corrections):
                d=self.raw_solve(rhs-self.matrix@x);c.stats['corrections']+=1;x=x+d
                norm=float(np.max(abs(d)/self.scale)/max(np.max(abs(x)/self.scale),1e-14));res=float(np.max(abs(rhs-self.matrix@x)/self.residual_scale))
                if np.isfinite(norm) and np.isfinite(res) and norm<=c.correction_tol and res<=c.residual_tol:
                    c.stats['max_correction']=max(c.stats['max_correction'],norm);c.stats['max_residual']=max(c.stats['max_residual'],res);return x
            c.stats['refreshes']+=1;self.entry=c.factor(self.matrix);self.exact=True;x=self.raw_solve(rhs)
        res=float(np.max(abs(rhs-self.matrix@x)/self.residual_scale));c.stats['max_residual']=max(c.stats['max_residual'],res)
        if not np.isfinite(x).all() or not np.isfinite(res) or res>c.residual_tol:raise FloatingPointError('Current linear equation residual exceeds tolerance')
        return x
