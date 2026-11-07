"""Cubic endpoint-history interpolation; diagnostic, not accepted output."""
import numpy as np
from research.inner_dense import Polynomial


def cubic_history(times,states):
    times=np.asarray(times,dtype=float);states=np.asarray(states)
    if times.shape!=(4,) or states.ndim!=2 or states.shape[1]!=4 or np.any(np.diff(times)<=0):
        raise ValueError('Four increasing endpoint times required')
    h=times[-1]-times[-2];nodes=(times-times[-2])/h
    # Solve only a 4x4 normalized Vandermonde; each state component shares it.
    coeff=np.linalg.solve(np.vander(nodes,4,increasing=True),states.T).T
    return Polynomial(coeff[:,3],coeff[:,2],coeff[:,1],coeff[:,0],h)
