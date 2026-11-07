"""Degree 3-5 endpoint-history curves with conservative polynomial checks.

Diagnostic only. History must contain computed endpoints, never reference data.
"""
import math
import numpy as np


class HistoryPolynomial:
    def __init__(self,times,states):
        times=np.asarray(times,dtype=float);states=np.asarray(states,dtype=float)
        degree=len(times)-1
        if degree not in (3,4,5) or states.ndim!=2 or states.shape[1]!=degree+1:
            raise ValueError('Four to six endpoint states required')
        if not np.isfinite(times).all() or not np.isfinite(states).all() or np.any(np.diff(times)<=0):
            raise ValueError('Finite states and strictly increasing times required')
        self.h=times[-1]-times[-2];nodes=(times-times[-2])/self.h
        self.coefficients=np.linalg.solve(np.vander(nodes,degree+1,increasing=True),states.T).T
        self.d=self.coefficients[:,0];self.degree=degree

    def __call__(self,s):
        s=np.asarray(s,dtype=float)
        if s.ndim!=1 or np.any((s<0)|(s>1)):raise ValueError('Outside current interval')
        return np.polynomial.polynomial.polyval(s,self.coefficients.T)

    def derivative(self,s):
        c=self.coefficients[:,1:]*np.arange(1,self.degree+1)
        return np.polynomial.polynomial.polyval(np.asarray(s),c.T)/self.h

    def minima(self):
        # Bernstein convex-hull lower bound is conservative: it can reject extra
        # curves but cannot certify a negative polynomial as nonnegative.
        transform=np.zeros((self.degree+1,self.degree+1))
        for k in range(self.degree+1):
            for j in range(k+1):transform[k,j]=math.comb(k,j)/math.comb(self.degree,j)
        bernstein=self.coefficients@transform.T
        lower=bernstein.min(axis=1)
        # Exact stationary-point evaluation refines only a negative hull bound.
        for i in np.flatnonzero(lower<0):
            derivative=self.coefficients[i,1:]*np.arange(1,self.degree+1)
            roots=np.polynomial.polynomial.polyroots(derivative)
            interior=[r.real for r in roots if abs(r.imag)<=1e-10 and 0<r.real<1]
            lower[i]=np.min(np.polynomial.polynomial.polyval(np.r_[0.,1.,interior],self.coefficients[i]))
        return lower
