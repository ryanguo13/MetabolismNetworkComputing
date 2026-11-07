"""Cubic slow-driving extension derived from the SPC-MRI framework.

Not the published Example8 linear coupling. With B=alpha+gamma, e=B1,
c=alpha1, choose mu(t)=M[t,t²,t³] and enforce
  M moment rows (1,e,c²,B e) = (t,t²/2,t³/3,t³/6).
These match the slow Taylor expansion through cubic order for an exact
full Jacobian and imply mu(1)=b plus integral(mu) e=1/6.
The endpoint ROW coupling moments integral(mu)c=1/6 and integral(mu)g=0
are also checked separately. This is a research construction, not a
stiff-order or positivity theorem. Independent full ODE/ledger audits remain.
"""
import time
import numpy as np
from scipy.linalg import lu_factor,lu_solve,expm
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from research.mri_kernel import MRI,GAUSS
from research.extrapolated_flow import DensePower,SlowPolynomial
from research.inner_dense import InnerCurve,physical_ledger
from research.shared_radau import SharedRadau
from research.row_spc import coefficients


def coupling():
    a,g,b,_=coefficients();beta=a+g;e=beta.sum(axis=1);c=a.sum(axis=1)
    moments=np.vstack([np.ones(4),e,c*c,beta@e])
    mu=np.linalg.solve(moments,np.array([[1,0,0],[0,.5,0],[0,0,1/3],[0,0,1/6]]))
    return a,g,b,mu


class PolynomialForcing(SlowPolynomial):
    def __init__(self,coefficients,h):
        self.coefficients=np.asarray(coefficients).copy();self.h=h;self.degree=2
        if self.coefficients.ndim!=2 or self.coefficients.shape[1]!=3 or not np.isfinite(self.coefficients).all():
            raise ValueError('Finite quadratic forcing required')
        self.integrated=np.column_stack([np.zeros(len(coefficients)),h*self.coefficients/np.arange(1,4)])


def linear_curve(fast,slow,h,fractions):
    """Exact frozen-linear maps at arbitrary fractions; independent block expm."""
    a,g,b,mu=coupling();j=fast+slow;n=len(j);identity=np.eye(n);increments=[];ks=[]
    for i in range(4):
        state=identity+sum((a[i,k]*increments[k] for k in range(i)),start=np.zeros((n,n)))
        previous=sum((g[i,k]*increments[k] for k in range(i)),start=np.zeros((n,n)))
        ki=np.linalg.solve(identity-h*g[i,i]*j,h*j@(state+previous));increments.append(ki)
        ks.append(h*slow@(state+previous+g[i,i]*ki))
    # In normalized time, z_s = h F z + q0 + s q1 + s² q2.
    q=[(i+1)*sum((mu[k,i]*ks[k] for k in range(4)),start=np.zeros((n,n))) for i in range(3)]
    aug=np.zeros((4*n,4*n));aug[:n,:n]=h*fast;aug[:n,n:2*n]=identity
    aug[n:2*n,2*n:3*n]=identity;aug[2*n:3*n,3*n:]=identity
    initial=np.vstack([identity,q[0],q[1],2*q[2]])
    return [expm(s*aug)[:n]@initial for s in fractions]


class ROWContinuous(MRI):
    def trial(self,y,h,tol=1e-8,base_only=False,timeout=20.):
        tick=time.perf_counter();a,g,b,mu=coupling();y=np.asarray(y,dtype=float).copy()
        if not np.isfinite(h) or h<=0 or not np.isfinite(tol) or tol<=0:raise ValueError('Positive finite step and tolerance required')
        stats=dict(predictor_nfev=0,predictor_njev=0,predictor_nlu=0,predictor_solves=0,nfev=0,njev=0,nlu=0,
            subflows=0,accepted_steps=0,output_nfev=0,pool_error=0.,min_internal=float(y.min()),linear_residuals=[])
        pool0=self.pools@y;poolden=max(1e-12,float(np.max(abs(pool0))))
        def check(z):
            if time.perf_counter()-tick>timeout:raise TimeoutError('Continuous ROW SPC budget')
            if not np.isfinite(z).all() or np.any(z < -1e-12*self.scale):raise ValueError('Invalid ROW state; no clipping')
        def accepted(z):
            check(z);stats['min_internal']=min(stats['min_internal'],float(z.min()))
            stats['pool_error']=max(stats['pool_error'],float(np.max(abs(self.pools@z-pool0)))/poolden)
        try:
            check(y);gradient=self.screen.gradient(y);j=(self.system.S@gradient).tocsc();fastj=(self.sf@gradient[self.cols]).tocsc();slowj=j-fastj
            matrix=eye(len(y),format='csc')-h*g[0,0]*j
            if len(y)<=32:
                lu=lu_factor(matrix.toarray());solve=lambda r:lu_solve(lu,r)
            else:lu=splu(matrix,permc_spec='MMD_AT_PLUS_A');solve=lu.solve
            stats.update(predictor_njev=1,predictor_nlu=1)
            increments=[];ks=[];stages=[];rates=[];full=[]
            for i in range(4):
                state=y+sum((a[i,k]*increments[k] for k in range(i)),start=np.zeros_like(y));accepted(state)
                rate=self.system.flux(0.,state);f=self.system.S@rate;stats['predictor_nfev']+=1
                prev=sum((g[i,k]*increments[k] for k in range(i)),start=np.zeros_like(y))
                rhs=h*f+h*(j@prev);ki=solve(rhs);stats['predictor_solves']+=1
                stats['linear_residuals'].append(float(np.max(abs(matrix@ki-rhs)/self.scale)))
                increments.append(ki);ks.append(h*(f-self.sf@rate[self.cols])+h*(slowj@(prev+g[i,i]*ki)))
                stages.append(state);rates.append(rate);full.append(f)
            increments=np.column_stack(increments);ks=np.column_stack(ks)
            stats['predictor_seconds']=time.perf_counter()-tick
            raw=None;forcing=None
            if base_only:end=y+increments@b;accepted(end)
            else:
                forcing=PolynomialForcing((ks@mu)*np.arange(1,4)/h,h);stats['subflows']=1
                def expand(t,u):
                    z=y+forcing.integral(t);z[self.active]=u;return z
                def rhs(t,u):
                    z=expand(t,u);check(z);return self.sfa@self.system.flux(0.,z)[self.cols]+forcing.value(t)[self.active]
                def jac(t,u):
                    z=expand(t,u);check(z);return (self.sfa@self.screen.gradient(z)[self.cols][:,self.active]).tocsc()
                pieces=[]
                if len(self.active):
                    solver=SharedRadau(rhs,0.,y[self.active],h,rtol=tol,atol=.01*tol*self.scale[self.active],jac=jac,seed=fastj[self.active][:,self.active].tocsc(),first_step=h)
                    while solver.status=='running':
                        left=solver.t;message=solver.step()
                        if solver.status=='failed':raise RuntimeError(message)
                        accepted(expand(solver.t,solver.y));pieces.append(forcing.dense(y,left,solver.t-left,self.active,solver.dense_output()))
                    rawend=expand(h,solver.y)
                    for k in ('nfev','njev','nlu'):stats[k]=getattr(solver,k)
                else:
                    rawend=y+forcing.integral(h);accepted(rawend);pieces=[forcing.dense(y,0.,h,self.active)]
                raw=InnerCurve(pieces);stats['accepted_steps']=len(pieces)
                stats['raw_nonnegative']=bool(np.all(raw.minima()>=-1e-12*self.scale));stats['raw_join_error']=raw.join_error(self.scale)
                if not stats['raw_nonnegative'] or stats['raw_join_error']>1e-10:raise ValueError('Invalid ROW raw curve')
                end=rawend;accepted(end)
                if self.weights is not None:
                    total=0.;quad=0.;left=0.
                    for piece in pieces:
                        integrals=[]
                        for n in (4,8):
                            q=(GAUSS[n][0]+1)/2
                            vals=[self.weights@(self.sf@self.system.flux(0.,z)[self.cols]+forcing.value(left+piece.h*s)) for s,z in zip(q,piece(q).T)]
                            integrals.append(piece.h/2*float(GAUSS[n][1]@vals))
                        total+=integrals[1];quad+=abs(integrals[1]-integrals[0]);left+=piece.h
                    den=max(1.,abs(float(self.weights@y)))
                    stats.update(modified_ledger_residual=abs(float(self.weights@(rawend-y))-total)/den,modified_ledger_quadrature=quad/den)
                stats['endpoint_correction_scaled']=0.
            # Cubic stage extension for standalone ROW; modified fast curve
            # for SPC. This is an independently audited candidate physical output.
            curve=(InnerCurve([DensePower(np.column_stack([y,increments@mu]),h)]) if base_only else raw)
            stats.update(continuous_nonnegative=bool(np.all(curve.minima()>=-1e-12*self.scale)),dense_join_error=curve.join_error(self.scale))
            if stats['continuous_nonnegative']:stats.update(physical_ledger(self.case,curve,y,end))
            stats['seconds']=time.perf_counter()-tick
            return dict(fine=end,pieces=[curve],raw=raw,raw_endpoint=None if base_only else rawend,forcing_coefficients=None if base_only else forcing.coefficients,
                increments=increments,slow_increments=ks,stages=np.column_stack(stages),rates=np.column_stack(rates),full=np.column_stack(full),stats=stats)
        except Exception as exc:
            stats['seconds']=time.perf_counter()-tick;exc.stats=stats;raise
