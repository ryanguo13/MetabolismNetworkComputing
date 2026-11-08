"""Rodas4 transformed coefficients plus a derived continuous slow extension.

Source coefficients: SciML OrdinaryDiffEqRosenbrockTableaus, Git blob
f1229035a7a4fb21049c6e7329b3e99fdeceeebd. MIT notice is in
research/licenses/OrdinaryDiffEq-Rodas.md. The nine-stage extension is our
research construction, not the published six-stage Rodas dense output.
"""
from functools import lru_cache
import numpy as np
from research.extrapolated_flow import SlowPolynomial


def base():
    a0=np.array([[0,0,0,0,0,0],[1.544,0,0,0,0,0],
        [.9466785280815826,.2557011698983284,0,0,0,0],
        [3.314825187068521,2.896124015972201,.9986419139977817,0,0,0],
        [1.221224509226641,6.019134481288629,12.53708332932087,-.687886036105895,0,0],
        [1.221224509226641,6.019134481288629,12.53708332932087,-.687886036105895,1,0]])
    c0=np.array([[0,0,0,0,0,0],[-5.6688,0,0,0,0,0],
        [-2.430093356833875,-.2063599157091915,0,0,0,0],
        [-.1073529058151375,-9.594562251023355,-20.47028614809616,0,0,0],
        [7.496443313967647,-10.24680431464352,-33.99990352819905,11.7089089320616,0,0],
        [8.083246795921522,-7.981132988064893,-31.52159432874371,16.31930543123136,-6.058818238834054,0]])
    g=np.linalg.inv(4*np.eye(6)-c0);a=a0@g;b=g.T@np.r_[a0[-1,:-1],1.]
    return a,g,b,a0,c0


def moments(a,g):
    B=a+g;c=a.sum(1);e=B.sum(1)
    return np.vstack([np.ones(len(c)),e,c*c,B@e,c**3,c*(a@e),B@(c*c),B@B@e])


def targets():
    t=np.zeros((8,4));t[0,0]=1;t[1,1]=.5;t[2:4,2]=[1/3,1/6];t[4:,3]=[1/4,1/8,1/12,1/24]
    return t


@lru_cache(None)
def coupling(augmented=True):
    a,g,b,_,_=base();m=moments(a,g);t=targets()
    cubic=np.linalg.lstsq(m[:4],t[:4,:3],rcond=None)[0]
    cubic[:,-1]+=b-cubic.sum(1)
    if not augmented:return a,g,b,cubic
    A=np.zeros((9,9));G=np.eye(9)*.25;A[:6,:6]=a;G[:6,:6]=g
    # Chosen algebraically from112 fixed candidates, before model experiments.
    # These extra stages retain the same diagonal; no extra full factorization.
    for k,x in enumerate((.25,.875,1.)):A[6+k,:6]=cubic@np.array([x,x*x,x*x*x])
    m=moments(A,G);mu=np.linalg.lstsq(m,t,rcond=None)[0];weights=np.r_[b,np.zeros(3)]
    mu[:,-1]+=weights-mu.sum(1)
    if np.max(abs(m@mu-t))>1e-11:raise ValueError('Quartic moment solve failed')
    return A,G,weights,mu


class RodasForcing(SlowPolynomial):
    def __init__(self,coefficients,h):
        self.coefficients=np.asarray(coefficients).copy();self.h=h
        if self.coefficients.ndim!=2 or self.coefficients.shape[1] not in (3,4) or not np.isfinite(self.coefficients).all():raise ValueError('Finite quadratic/cubic forcing required')
        self.degree=self.coefficients.shape[1]-1
        self.integrated=np.column_stack([np.zeros(len(coefficients)),h*self.coefficients/np.arange(1,self.degree+2)])
