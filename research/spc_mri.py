"""Published SPC-MRI-GARK SDIRK2(1)2 and SDIRK3(2)4 research kernels.

Roberts/Sarshar/Sandu, arXiv1812.00808v5, Definition2.1 and AppendixA.1.1/A.1.3.
Primary gamma rows only; embedded estimators are not implemented. Raw fast
curves and full-RHS Hermite output require independent accuracy/ledger audits.
"""
import time
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from research.mri_kernel import MRI,Hermite,GAUSS
from research.extrapolated_flow import SlowPolynomial,DensePower
from research.inner_dense import InnerCurve,physical_ledger
from research.shared_radau import SharedRadau


def tableau(order):
    if order==2:
        r=np.sqrt(2.);d=1-1/r
        a=np.array([[d,0.],[1/r,d]])
        gamma=np.array([[5*r-6,7-5*r],[12-9*r,9*r-12]])
    elif order==3:
        d=9/40
        a=np.array([[d,0,0,0],[163/520,d,0,0],[-6481433/8838675,87795409/70709400,d,0],
                    [4032/9943,6929/15485,-723/9272,d]])
        gamma=np.array([[3/2,-46850957023/152236344800,-2336165553/30447268960,-231399837/2003109800],
                        [-21765/9943,18740344238109/12407262101200,-2318739807/928641703280,341049771/500777450]])
    else:raise ValueError('Published orders 2/3 only')
    return a,gamma


class AffineForcing(SlowPolynomial):
    def __init__(self,coefficients,h):
        coefficients=np.asarray(coefficients,dtype=float)
        if coefficients.ndim!=2 or coefficients.shape[1]!=2 or not np.isfinite(coefficients).all() or not np.isfinite(h) or h<=0:
            raise ValueError('Finite affine forcing and positive h required')
        self.coefficients=coefficients.copy();self.h=h;self.degree=1
        self.integrated=np.column_stack([np.zeros(len(coefficients)),h*coefficients/np.array([1.,2.])])


class SPC(MRI):
    def predictor(self,y,h,order,residual_tol=1e-11,maxiter=12,timeout=20.):
        tick=time.perf_counter();a,gamma=tableau(order);y=np.asarray(y,dtype=float).copy()
        if not np.isfinite(h) or h<=0 or residual_tol<=0 or not np.isfinite(residual_tol):raise ValueError('Positive finite step/tolerance required')
        stats=dict(predictor_nfev=0,predictor_njev=0,predictor_nlu=0,predictor_solves=0,pool_error=0.,
                   min_internal=float(y.min()),predictor_residuals=[],predictor_iterations=[])
        pool0=self.pools@y;den=max(1e-12,float(np.max(abs(pool0))))
        def check(z):
            if time.perf_counter()-tick>timeout:raise TimeoutError('SPC predictor budget')
            if not np.isfinite(z).all() or np.any(z < -1e-12*self.scale):raise ValueError('Invalid SPC predictor state; no clipping')
        def rhs(z):
            check(z);stats['predictor_nfev']+=1;rates=self.system.flux(0.,z)
            return self.system.S@rates,rates
        gradient=None;factor=None
        def refresh(z):
            nonlocal gradient,factor
            check(z);gradient=self.screen.gradient(z);jac=(self.system.S@gradient).tocsc();stats['predictor_njev']+=1
            matrix=eye(len(y),format='csc')-h*a[0,0]*jac
            if len(y)<=32:
                lu=lu_factor(matrix.toarray());factor=lambda r:lu_solve(lu,r)
            else:
                lu=splu(matrix,permc_spec='MMD_AT_PLUS_A');factor=lu.solve
            stats['predictor_nlu']+=1
        try:
            check(y);f0,_=rhs(y);refresh(y);stages=[];fulls=[];rates=[]
            for i in range(len(a)):
                base=y+h*sum((a[i,j]*fulls[j] for j in range(i)),start=np.zeros_like(y))
                z=y+h*a[i].sum()*f0
                if not np.isfinite(z).all() or np.any(z < -1e-12*self.scale):z=y.copy()
                for iteration in range(maxiter):
                    f,rate=rhs(z);residual=base+h*a[i,i]*f-z
                    norm=float(np.max(abs(residual)/self.scale))
                    if norm<=residual_tol:break
                    if iteration in (4,8):refresh(z)
                    delta=factor(residual);stats['predictor_solves']+=1;alpha=1.
                    while np.any(z+alpha*delta < -1e-12*self.scale) and alpha>2**-12:alpha*=.5
                    z=z+alpha*delta
                else:raise RuntimeError('SPC predictor stage did not converge')
                check(z);stats['min_internal']=min(stats['min_internal'],float(z.min()))
                stats['pool_error']=max(stats['pool_error'],float(np.max(abs(self.pools@z-pool0)))/den)
                stages.append(z.copy());fulls.append(f);rates.append(rate)
                stats['predictor_residuals'].append(norm);stats['predictor_iterations'].append(iteration+1)
            stats['predictor_seconds']=time.perf_counter()-tick
            return dict(endpoint=stages[-1],stages=np.column_stack(stages),full=np.column_stack(fulls),
                        rates=np.column_stack(rates),initial_rhs=f0,gradient=gradient,stats=stats,gamma=gamma)
        except Exception as exc:
            stats['predictor_seconds']=time.perf_counter()-tick;exc.stats=stats;raise

    def trial(self,y,h,order=3,tol=1e-8,output='inner',base_only=False,residual_tol=1e-11,timeout=20.):
        tick=time.perf_counter();y=np.asarray(y,dtype=float).copy()
        if output not in ('inner','hermite') or not np.isfinite(tol) or tol<=0:raise ValueError('Invalid SPC output/tolerance')
        pred=self.predictor(y,h,order,residual_tol=residual_tol,timeout=timeout);stats=pred['stats'].copy()
        stats.update(subflows=0,nfev=0,njev=0,nlu=0,accepted_steps=0,output_nfev=0)
        pool0=self.pools@y;poolden=max(1e-12,float(np.max(abs(pool0))))
        def check(z):
            if time.perf_counter()-tick>timeout:raise TimeoutError('SPC trial budget')
            if not np.isfinite(z).all() or np.any(z < -1e-12*self.scale):raise ValueError('Invalid SPC fast state; no clipping')
        def accepted(z):
            check(z);stats['min_internal']=min(stats['min_internal'],float(z.min()))
            stats['pool_error']=max(stats['pool_error'],float(np.max(abs(self.pools@z-pool0)))/poolden)
        try:
            if base_only:
                end=pred['endpoint'];p=Hermite(y,end,pred['initial_rhs'],pred['full'][:,-1],h)
                curve=InnerCurve([DensePower(np.column_stack([p.d,p.c,p.b,p.a]),h)])
                forcing=None;raw=None
            else:
                slow=pred['full']-self.sf@pred['rates'][self.cols]
                forcing=AffineForcing(slow@pred['gamma'].T,h);stats['subflows']=1
                def expand(t,u):
                    z=y+forcing.integral(t);z[self.active]=u;return z
                def rhs(t,u):
                    z=expand(t,u);check(z)
                    return self.sfa@self.system.flux(0.,z)[self.cols]+forcing.value(t)[self.active]
                def jac(t,u):
                    z=expand(t,u);check(z)
                    return (self.sfa@self.screen.gradient(z)[self.cols][:,self.active]).tocsc()
                pieces=[]
                if len(self.active):
                    # Borrowed seed is stale: SharedRadau retains its normal refresh path.
                    seed=(self.sfa@pred['gradient'][self.cols][:,self.active]).tocsc()
                    solver=SharedRadau(rhs,0.,y[self.active],h,rtol=tol,atol=.01*tol*self.scale[self.active],jac=jac,seed=seed,first_step=h)
                    while solver.status=='running':
                        left=solver.t;message=solver.step()
                        if solver.status=='failed':raise RuntimeError(message)
                        accepted(expand(solver.t,solver.y));pieces.append(forcing.dense(y,left,solver.t-left,self.active,solver.dense_output()))
                    end=expand(h,solver.y)
                    for k in ('nfev','njev','nlu'):stats[k]=getattr(solver,k)
                else:
                    end=y+forcing.integral(h);accepted(end);pieces=[forcing.dense(y,0.,h,self.active)]
                stats['accepted_steps']=len(pieces);raw=InnerCurve(pieces)
                stats['raw_nonnegative']=bool(np.all(raw.minima()>=-1e-12*self.scale));stats['raw_join_error']=raw.join_error(self.scale)
                if not stats['raw_nonnegative'] or stats['raw_join_error']>1e-10:raise ValueError('Invalid raw SPC curve')
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
                    stats.update(modified_ledger_residual=abs(float(self.weights@(end-y))-total)/den,modified_ledger_quadrature=quad/den)
                if output=='inner':curve=raw
                else:
                    check(end);fend=self.full(end);stats['output_nfev']=1
                    p=Hermite(y,end,pred['initial_rhs'],fend,h)
                    curve=InnerCurve([DensePower(np.column_stack([p.d,p.c,p.b,p.a]),h)])
            stats['continuous_nonnegative']=bool(np.all(curve.minima()>=-1e-12*self.scale));stats['dense_join_error']=curve.join_error(self.scale)
            if stats['dense_join_error']>1e-10:raise ValueError('Discontinuous SPC output')
            if stats['continuous_nonnegative']:stats.update(physical_ledger(self.case,curve,y,end))
            stats['seconds']=time.perf_counter()-tick
            return dict(fine=end,pieces=[curve],raw=raw,predictor=pred,forcing_coefficients=None if forcing is None else forcing.coefficients,stats=stats)
        except Exception as exc:
            stats['seconds']=time.perf_counter()-tick;exc.stats=stats;raise
