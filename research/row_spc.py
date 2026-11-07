"""SPC MRI-GARK-ROW, Guenther/Sandu arXiv2102.10203v1 eq8.1/10.1-2,
Example8 ROS34PW2 with linear coupling satisfying eq10.9.

z=v+(theta/H)*K_s*mu converts the published fast IVP to constant forcing.
The endpoint is z(H)+K_s*(b-mu). Raw z is NOT the returned physical output;
the returned Hermite curve has full RHS endpoint derivatives and independent audit.
"""
import time
import numpy as np
from scipy.linalg import lu_factor,lu_solve,expm
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from research.mri_kernel import MRI,Hermite,GAUSS
from research.extrapolated_flow import DensePower
from research.spc_mri import AffineForcing
from research.inner_dense import InnerCurve,physical_ledger
from research.shared_radau import SharedRadau


def coefficients(theta=.5):
    if not np.isfinite(theta):raise ValueError('Finite coupling parameter required')
    a=np.zeros((4,4));g=np.eye(4)*.4358665215084597
    a[1,0]=.87173304301691801;g[1,0]=-.87173304301691801
    a[2,:2]=[.84457060015369423,-.11299064236484185];g[2,:2]=[-.90338057013044082,.054180672388095326]
    a[3,2]=1.;g[3,:3]=[.24212380706095346,-1.2232505839045147,.54526025533510214]
    b=np.array([.24212380706095346,-1.2232505839045147,1.5452602553351020,.43586652150845900])
    matrix=np.vstack([np.ones(4),a.sum(axis=1),g.sum(axis=1)])
    mu=np.r_[theta,np.linalg.solve(matrix[:,1:],np.array([1.,1/3,0.])-theta*matrix[:,0])]
    return a,g,b,mu


def linear_map(fast,slow,h,theta=.5):
    """Exact frozen-linear map; full predictor and exact fast correction."""
    a,g,b,mu=coefficients(theta);j=fast+slow;n=len(j);identity=np.eye(n);increments=[];ks=[]
    for i in range(4):
        state=identity+sum((a[i,k]*increments[k] for k in range(i)),start=np.zeros((n,n)))
        previous=sum((g[i,k]*increments[k] for k in range(i)),start=np.zeros((n,n)))
        ki=np.linalg.solve(identity-h*g[i,i]*j,h*j@(state+previous));increments.append(ki)
        ks.append(h*slow@(state+previous+g[i,i]*ki))
    aug=np.zeros((2*n,2*n));aug[:n,:n]=h*fast;aug[:n,n:]=identity;e=expm(aug)
    kmu=sum((mu[i]*ks[i] for i in range(4)),start=np.zeros((n,n)))
    return e[:n,:n]+e[:n,n:]@kmu+sum(((b[i]-mu[i])*ks[i] for i in range(4)),start=np.zeros((n,n)))


class ROWSPC(MRI):
    def trial(self,y,h,tol=1e-8,theta=.5,base_only=False,timeout=20.):
        tick=time.perf_counter();a,g,b,mu=coefficients(theta);y=np.asarray(y,dtype=float).copy()
        if not np.isfinite(h) or h<=0 or not np.isfinite(tol) or tol<=0:raise ValueError('Positive finite step and tolerance required')
        stats=dict(predictor_nfev=0,predictor_njev=0,predictor_nlu=0,predictor_solves=0,nfev=0,njev=0,nlu=0,
            subflows=0,accepted_steps=0,output_nfev=0,pool_error=0.,min_internal=float(y.min()),linear_residuals=[])
        pool0=self.pools@y;poolden=max(1e-12,float(np.max(abs(pool0))))
        def check(z):
            if time.perf_counter()-tick>timeout:raise TimeoutError('ROW SPC budget')
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
                forcing=AffineForcing(np.column_stack([ks@mu/h,np.zeros(len(y))]),h);stats['subflows']=1
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
                end=rawend+ks@(b-mu);accepted(end)
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
                stats['endpoint_correction_scaled']=float(np.max(abs(ks@(b-mu))/self.scale))
            # Full-RHS Hermite is a separately audited physical output.
            fend=self.full(end);stats['output_nfev']=1
            p=Hermite(y,end,full[0],fend,h);curve=InnerCurve([DensePower(np.column_stack([p.d,p.c,p.b,p.a]),h)])
            stats.update(continuous_nonnegative=bool(np.all(curve.minima()>=-1e-12*self.scale)),dense_join_error=curve.join_error(self.scale))
            if stats['continuous_nonnegative']:stats.update(physical_ledger(self.case,curve,y,end))
            stats['seconds']=time.perf_counter()-tick
            return dict(fine=end,pieces=[curve],raw=raw,raw_endpoint=None if base_only else rawend,forcing_coefficients=None if base_only else forcing.coefficients,
                increments=increments,slow_increments=ks,stages=np.column_stack(stages),rates=np.column_stack(rates),full=np.column_stack(full),stats=stats)
        except Exception as exc:
            stats['seconds']=time.perf_counter()-tick;exc.stats=stats;raise
