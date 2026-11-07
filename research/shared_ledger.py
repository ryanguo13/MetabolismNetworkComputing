"""Same Gauss4/8 modified and Gauss8/16 physical ledgers, shared rate values.

SBML expression batching changes only array assembly; scalar kinetics remain
untouched. Full controls use the same batch gradient and physical ledger.
"""
from dataclasses import replace
from types import FunctionType
import numpy as np
from scipy.sparse import csr_matrix
from research.mri_kernel import GAUSS


class BatchedRates:
    def __init__(self,system):
        self.system=system;self.compiled=None
        f=system.flux
        if isinstance(f,FunctionType) and f.__name__=='<lambda>' and f.__code__.co_names==('array',) and f.__globals__.get('array') is np.array:
            def assemble(terms):return np.stack(np.broadcast_arrays(*terms))
            self.compiled=FunctionType(f.__code__,dict(f.__globals__,array=assemble),argdefs=f.__defaults__)

    def __call__(self,values):
        if self.compiled is None:return np.column_stack([self.system.flux(0.,z) for z in values.T])
        out=self.compiled(0.,values)
        return np.broadcast_to(out[:,None],(len(out),values.shape[1])).copy() if out.ndim==1 else out

    def gradient(self,t,y):
        if self.compiled is None:raise ValueError('No admitted compiled expression for batched gradient')
        z=np.broadcast_to(y[:,None],(len(y),len(y))).astype(complex).copy();z[np.diag_indices(len(y))]+=1j*1e-24
        return csr_matrix(self(z).imag/1e-24)


def optimized_case(case):
    batch=BatchedRates(case['system']);out=case.copy()
    if batch.compiled is not None and case['system'].flux_jacobian is None:
        out['system']=replace(case['system'],flux_jacobian=batch.gradient)
    out['batch_rates']=batch
    return out


def ledgers(case,curve,initial,final,kernel=None,forcing=None):
    weights=case['options'].get('ledger_weights')
    if weights is None:return {}
    batch=case.get('batch_rates') or BatchedRates(case['system']);sys=case['system'];full_weights=np.asarray(weights@sys.S).ravel()
    modified=forcing is not None;fast_weights=np.asarray(weights@kernel.sf).ravel() if modified else None
    physical=[0.,0.];modified_sum=0.;modified_quad=0.;left=0.;evaluations=0
    for piece in curve.pieces:
        mi=[]
        for n in ((4,8,16) if modified else (8,16)):
            nodes,w=GAUSS[n];q=(nodes+1)/2;rates=batch(piece(q));evaluations+=n
            if n>=8:physical[0 if n==8 else 1]+=piece.h/2*float(w@(full_weights@rates))
            if modified and n<=8:
                source=np.array([weights@forcing.value(left+piece.h*s) for s in q])
                mi.append(piece.h/2*float(w@(fast_weights@rates[kernel.cols]+source)))
        if modified:modified_sum+=mi[1];modified_quad+=abs(mi[1]-mi[0])
        left+=piece.h
    den=max(1.,abs(float(weights@initial)));change=float(weights@(final-initial))
    result=dict(physical_ledger_residual=abs(change-physical[1])/den,physical_ledger_quadrature=abs(physical[1]-physical[0])/den,ledger_rate_states=evaluations)
    if modified:result.update(modified_ledger_residual=abs(change-modified_sum)/den,modified_ledger_quadrature=modified_quad/den)
    return result
