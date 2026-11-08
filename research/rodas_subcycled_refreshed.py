"""Per-predictor-substep Jacobian refresh control for frozen-LU subcycling.

Separate immutable experiment: same prediction and fast correction structure,
with m Jacobians and m full LU factors, all charged. No reused stale derivative.
"""
import time
import numpy as np
from scipy.linalg import lu_factor,lu_solve
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from research.mri_kernel import MRI,GAUSS
from research.rodas_coupling import coupling
from research.rodas_centered import CenteredForcing
from research.extrapolated_flow import DensePower
from research.inner_dense import InnerCurve,physical_ledger
from research.shared_radau import SharedRadau


class RefreshedSubcycledRodas(MRI):
    def trial(self,y,h,tol=1e-10,substeps=1,base_only=False,timeout=20.):
        tick=time.perf_counter();y=np.asarray(y,dtype=float).copy()
        if not isinstance(substeps,int) or isinstance(substeps,bool) or substeps<1:raise ValueError('Positive integer substeps required')
        if not np.isfinite(h) or h<=0 or not np.isfinite(tol) or tol<=0:raise ValueError('Positive finite step and tolerance required')
        dt=h/substeps;a,g,b,mu=coupling(True)
        stats=dict(predictor_nfev=0,predictor_njev=0,predictor_nlu=0,predictor_solves=0,nfev=0,njev=0,nlu=0,
            subflows=0,accepted_steps=0,pool_error=0.,min_internal=float(y.min()),linear_residuals=[])
        pool0=self.pools@y;poolden=max(1e-12,float(np.max(abs(pool0))))
        def check(z):
            if time.perf_counter()-tick>timeout:raise TimeoutError('Subcycled Rodas budget')
            if not np.isfinite(z).all() or np.any(z < -1e-12*self.scale):raise ValueError('Invalid subcycled state; no clipping')
        def accepted(z):
            check(z);stats['min_internal']=min(stats['min_internal'],float(z.min()))
            stats['pool_error']=max(stats['pool_error'],float(np.max(abs(self.pools@z-pool0)))/poolden)
        try:
            check(y);gradient=self.screen.gradient(y);j=(self.system.S@gradient).tocsc();fastj=(self.sf@gradient[self.cols]).tocsc();slowj=j-fastj
            matrix=eye(len(y),format='csc')-dt*g[0,0]*j
            if len(y)<=32:
                lu=lu_factor(matrix.toarray());solve=lambda r:lu_solve(lu,r)
            else:lu=splu(matrix,permc_spec='MMD_AT_PLUS_A');solve=lu.solve
            stats.update(predictor_njev=1,predictor_nlu=1)
            predicted=y.copy();endpoints=[y.copy()];fullpieces=[];forcings=[];increments=[];slowincrements=[]
            for step in range(substeps):
                if step:
                    gradient=self.screen.gradient(predicted);j=(self.system.S@gradient).tocsc()
                    slowj=j-(self.sf@gradient[self.cols]).tocsc()
                    matrix=eye(len(y),format='csc')-dt*g[0,0]*j
                    if len(y)<=32:
                        lu=lu_factor(matrix.toarray());solve=lambda r:lu_solve(lu,r)
                    else:lu=splu(matrix,permc_spec='MMD_AT_PLUS_A');solve=lu.solve
                    stats['predictor_njev']+=1;stats['predictor_nlu']+=1
                inc=[];ks=[]
                for i in range(len(b)):
                    state=predicted+sum((a[i,k]*inc[k] for k in range(i)),start=np.zeros_like(y));accepted(state)
                    rate=self.system.flux(0.,state);f=self.system.S@rate;stats['predictor_nfev']+=1
                    prev=sum((g[i,k]*inc[k] for k in range(i)),start=np.zeros_like(y))
                    rhs=dt*f+dt*(j@prev);ki=solve(rhs);stats['predictor_solves']+=1
                    stats['linear_residuals'].append(float(np.max(abs(matrix@ki-rhs)/self.scale)))
                    inc.append(ki);ks.append(dt*(f-self.sf@rate[self.cols])+dt*(slowj@(prev+g[i,i]*ki)))
                inc=np.column_stack(inc);ks=np.column_stack(ks)
                increments.append(inc);slowincrements.append(ks)
                fullpieces.append(DensePower(np.column_stack([predicted,inc@mu]),dt))
                forcings.append(CenteredForcing((ks@mu)*np.arange(1,mu.shape[1]+1)/dt,dt))
                predicted=predicted+inc@b;accepted(predicted);endpoints.append(predicted.copy())
            stats['predictor_seconds']=time.perf_counter()-tick
            segment_counts=[];correction_endpoints=[y.copy()]
            if base_only:
                end=predicted;pieces=fullpieces
            else:
                end=y.copy();pieces=[];total=0.;quad=0.
                for forcing in forcings:
                    origin=end.copy();stats['subflows']+=1;segment=[]
                    def expand(t,u):
                        z=origin+forcing.integral(t);z[self.active]+=u-origin[self.active]-t*forcing.coefficients[self.active,0];return z
                    def rhs(t,u):
                        z=expand(t,u);check(z);return self.sfa@self.system.flux(0.,z)[self.cols]+forcing.coefficients[self.active,0]
                    def jac(t,u):
                        z=expand(t,u);check(z);return (self.sfa@self.screen.gradient(z)[self.cols][:,self.active]).tocsc()
                    if len(self.active):
                        solver=SharedRadau(rhs,0.,origin[self.active],dt,rtol=tol,atol=.01*tol*self.scale[self.active],jac=jac,seed=fastj[self.active][:,self.active].tocsc(),first_step=dt)
                        while solver.status=='running':
                            left=solver.t;message=solver.step()
                            if solver.status=='failed':raise RuntimeError(message)
                            accepted(expand(solver.t,solver.y));segment.append(forcing.dense(origin,left,solver.t-left,self.active,solver.dense_output()))
                        end=expand(dt,solver.y)
                        for k in ('nfev','njev','nlu'):stats[k]+=getattr(solver,k)
                    else:
                        end=origin+forcing.integral(dt);segment=[forcing.dense(origin,0.,dt,self.active)]
                    accepted(end);correction_endpoints.append(end.copy());segment_counts.append(len(segment));pieces.extend(segment)
                    if self.weights is not None:
                        left=0.
                        for piece in segment:
                            integrals=[]
                            for n in (4,8):
                                q=(GAUSS[n][0]+1)/2
                                vals=[self.weights@(self.sf@self.system.flux(0.,z)[self.cols]+forcing.value(left+piece.h*s)) for s,z in zip(q,piece(q).T)]
                                integrals.append(piece.h/2*float(GAUSS[n][1]@vals))
                            total+=integrals[1];quad+=abs(integrals[1]-integrals[0]);left+=piece.h
                stats['accepted_steps']=len(pieces)
                if self.weights is not None:
                    den=max(1.,abs(float(self.weights@y)))
                    stats.update(modified_ledger_residual=abs(float(self.weights@(end-y))-total)/den,modified_ledger_quadrature=quad/den)
            curve=InnerCurve(pieces)
            stats.update(continuous_nonnegative=bool(np.all(curve.minima()>=-1e-12*self.scale)),dense_join_error=curve.join_error(self.scale))
            if stats['dense_join_error']>1e-10:raise ValueError('Discontinuous subcycled output')
            if stats['continuous_nonnegative']:stats.update(physical_ledger(self.case,curve,y,end))
            stats['seconds']=time.perf_counter()-tick
            return dict(fine=end,pieces=[curve],stats=stats,forcing_coefficients=np.stack([f.coefficients for f in forcings]),
                increments=np.stack(increments),slow_increments=np.stack(slowincrements),predictor_endpoints=np.column_stack(endpoints),
                correction_endpoints=np.column_stack(correction_endpoints),segment_counts=np.array(segment_counts,dtype=int),
                rodas_linearization=dict(jacobian=j,solve=solve,matrix=matrix,h=dt,gamma=g[0,0],initial=endpoints[-2].copy()))
        except Exception as exc:
            stats['seconds']=time.perf_counter()-tick;exc.stats=stats;raise
