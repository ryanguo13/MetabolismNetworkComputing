"""Endpoint-interpolated additive forcing and its frozen linear diagnostic.

Research prototype, not an implementation of a published MRI-GARK tableau.
With zero fast part this reduces to Adams--Moulton (orders 2, 3, 4).
"""
import math
import numpy as np
from scipy.linalg import expm
from research.extrapolated_flow import SlowPolynomial


class EndpointPolynomial(SlowPolynomial):
    def __init__(self,times,values,start,h):
        times=np.asarray(times,dtype=float);values=np.asarray(values,dtype=float)
        if (len(times) not in (2,3,4) or values.ndim!=2 or values.shape[1]!=len(times)
                or not np.isfinite(times).all() or not np.isfinite(values).all()
                or not np.isfinite([start,h]).all() or h<=0 or np.any(np.diff(times)<=0)
                or abs(times[-1]-(start+h))>1e-12*h or abs(times[-2]-start)>1e-12*h):
            raise ValueError('Ordered finite history ending at current and future times required')
        nodes=(times-start)/h
        self.coefficients=np.linalg.solve(np.vander(nodes,len(times),increasing=True),values.T).T
        self.h=h;self.degree=len(times)-1
        self.integrated=np.column_stack([np.zeros(values.shape[0]),h*self.coefficients/np.arange(1,len(times)+1)])


def response(fast,h,order):
    """Return exp(hA), and exact forcing responses at nodes 1,0,-1,... ."""
    fast=np.asarray(fast,dtype=float)
    if (order not in (2,3,4) or fast.ndim!=2 or fast.shape[0]!=fast.shape[1]
            or not np.isfinite(fast).all() or not np.isfinite(h) or h<=0):
        raise ValueError('Finite square matrix, order 2/3/4 and positive h required')
    n=len(fast);aug=np.zeros((n*(order+1),n*(order+1)));aug[:n,:n]=h*fast
    for i in range(order):aug[i*n:(i+1)*n,(i+1)*n:(i+2)*n]=np.eye(n)
    evolved=expm(aug)
    coefficients=np.linalg.solve(np.vander(1-np.arange(order),order,increasing=True),np.eye(order))
    kernels=[sum(h*math.factorial(k)*coefficients[k,j]*evolved[:n,(k+1)*n:(k+2)*n]
                 for k in range(order)) for j in range(order)]
    return evolved[:n,:n],kernels


def amplification(fast,slow,h,order=4):
    """Implicit companion, endpoint Jacobian, and fixed-point iteration matrix."""
    slow=np.asarray(slow,dtype=float);e,k=response(fast,h,order);n=len(e)
    if slow.shape!=(n,n) or not np.isfinite(slow).all():raise ValueError('Invalid slow Jacobian')
    iteration=k[0]@slow;endpoint=np.eye(n)-iteration
    blocks=[v@slow for v in k[1:]];blocks[0]+=e
    matrix=np.zeros((n*(order-1),n*(order-1)))
    matrix[:n]=np.linalg.solve(endpoint,np.hstack(blocks))
    if order>2:matrix[n:,:-n]=np.eye(n*(order-2))
    return matrix,endpoint,iteration
