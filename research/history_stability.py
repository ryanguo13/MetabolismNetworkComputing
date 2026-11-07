"""Frozen linear constant-step amplification of extrapolated additive forcing.

This is a local diagnostic, not a nonlinear or variable-step stability guarantee.
The fast inner solve is exact in this test, so instability cannot be blamed on
Radau tolerances. State ordering is current, previous, ..., oldest.
"""
import math
import numpy as np
from scipy.linalg import expm


def amplification(fast,slow,h,order=4):
    fast=np.asarray(fast,dtype=float);slow=np.asarray(slow,dtype=float)
    if (order not in (2,3,4) or fast.ndim!=2 or fast.shape[0]!=fast.shape[1] or slow.shape!=fast.shape
            or not np.isfinite(fast).all() or not np.isfinite(slow).all() or not np.isfinite(h) or h<=0):
        raise ValueError('Finite square split, order2/3/4 and positive step required')
    n=len(fast);aug=np.zeros((n*(order+1),n*(order+1)));aug[:n,:n]=h*fast
    for i in range(order):aug[i*n:(i+1)*n,(i+1)*n:(i+2)*n]=np.eye(n)
    evolved=expm(aug)
    coefficient=np.linalg.solve(np.vander(-np.arange(order),order,increasing=True),np.eye(order))
    blocks=[]
    for j in range(order):
        block=sum(h*math.factorial(k)*coefficient[k,j]*evolved[:n,(k+1)*n:(k+2)*n]@slow for k in range(order))
        if j==0:block+=evolved[:n,:n]
        blocks.append(block)
    companion=np.zeros((n*order,n*order));companion[:n]=np.hstack(blocks)
    companion[n:,:-n]=np.eye(n*(order-1))
    return companion
