"""Affine-centered fast variable retaining full slow polynomial on all species.

Use u=z_active-Q_active(t)+t*q0_active, initially y_active. Then
u prime=f_fast(z)+q0_active, so its initial derivative equals the original
modified ODE derivative. Q+t affine terms are reconstructed exactly.
"""
import numpy as np
from research.rodas_coupling import RodasForcing


class CenteredForcing(RodasForcing):
    def dense(self,initial,left,dt,active,dense=None):
        result=super().dense(initial,left,dt,np.array([],dtype=int))
        if dense is not None:
            result.coefficients[active,0]+=dense.y_old-initial[active]-left*self.coefficients[active,0]
            result.coefficients[active,1:4]+=dense.Q
            result.coefficients[active,1]-=dt*self.coefficients[active,0]
        return result

import time
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from research.mri_kernel import MRI
from research.extrapolated_flow import DensePower
from research.inner_dense import InnerCurve,physical_ledger
from research.shared_radau import SharedRadau
from research.rodas_coupling import coupling
from research.mri_kernel import GAUSS


class CenteredRodas(MRI):
    def trial(self,y,h,tol=1e-8,base_only=False,timeout=20.,augmented=True):
        tick=time.perf_counter();a,g,b,mu=coupling(augmented);y=np.asarray(y,dtype=float).copy()
        if not np.isfinite(h) or h<=0 or not np.isfinite(tol) or tol<=0:raise ValueError('Positive finite step and tolerance required')
        stats=dict(predictor_nfev=0,predictor_njev=0,predictor_nlu=0,predictor_solves=0,nfev=0,njev=0,nlu=0,
            subflows=0,accepted_steps=0,output_nfev=0,pool_error=0.,min_internal=float(y.min()),linear_residuals=[])
        pool0=self.pools@y;poolden=max(1e-12,float(np.max(abs(pool0))))
        def check(z):
            if time.perf_counter()-tick>timeout:raise TimeoutError('Rodas continuous budget')
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
            for i in range(len(b)):
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
                forcing=CenteredForcing((ks@mu)*np.arange(1,mu.shape[1]+1)/h,h);stats['subflows']=1
                def expand(t,u):
                    z=y+forcing.integral(t);z[self.active]+=u-y[self.active]-t*forcing.coefficients[self.active,0];return z
                def rhs(t,u):
                    z=expand(t,u);check(z);return self.sfa@self.system.flux(0.,z)[self.cols]+forcing.coefficients[self.active,0]
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
            # Derived cubic/quartic extension for standalone Rodas; modified fast curve
            # for SPC. This is an independently audited candidate physical output.
            curve=(InnerCurve([DensePower(np.column_stack([y,increments@mu]),h)]) if base_only else raw)
            stats.update(continuous_nonnegative=bool(np.all(curve.minima()>=-1e-12*self.scale)),dense_join_error=curve.join_error(self.scale))
            if stats['continuous_nonnegative']:stats.update(physical_ledger(self.case,curve,y,end))
            stats['seconds']=time.perf_counter()-tick
            return dict(rodas_linearization=dict(jacobian=j,solve=solve,matrix=matrix,h=h,gamma=g[0,0],initial=y.copy()),fine=end,pieces=[curve],raw=raw,raw_endpoint=None if base_only else rawend,forcing_coefficients=None if base_only else forcing.coefficients,
                increments=increments,slow_increments=ks,stages=np.column_stack(stages),rates=np.column_stack(rates),full=np.column_stack(full),stats=stats)
        except Exception as exc:
            stats['seconds']=time.perf_counter()-tick;exc.stats=stats;raise
