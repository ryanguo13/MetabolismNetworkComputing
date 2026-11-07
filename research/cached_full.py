"""Uninterrupted mature Radau controls with the same batching/refinement options."""
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from research.cached_row import RefinedRadau
from research.refined_lu import RefinedLU
from research.shared_ledger import optimized_case,ledgers
from research.extrapolated_flow import DensePower
from research.inner_dense import InnerCurve


def full_curve(case,y,h,tol,whole=False,reuse=False):
    case=optimized_case(case);screen=GroupScreen(case['system'],case['scale']);tick=time.perf_counter();sys=case['system'];scale=case['scale'];pool=case['options']['pools']
    pool0=pool@y;den=max(1e-12,float(np.max(abs(pool0))));drift=0.;cache=RefinedLU(reuse)
    def rhs(t,z):
        if time.perf_counter()-tick>30:raise TimeoutError('Refined full Radau budget')
        if not np.isfinite(z).all() or np.any(z < -1e-12*scale):raise ValueError('Invalid full RHS state')
        return sys.S@sys.flux(0.,z)
    kwargs=dict(first_step=h) if whole else {}
    solver=RefinedRadau(rhs,0.,y,h,rtol=tol,atol=.01*tol*scale,jac=lambda t,z:sys.S@screen.gradient(z),refined_cache=cache,linear_scale=scale,**kwargs)
    pieces=[]
    while solver.status=='running':
        left=solver.t;message=solver.step()
        if solver.status=='failed':raise RuntimeError(message)
        if not np.isfinite(solver.y).all() or np.any(solver.y < -1e-12*scale):raise ValueError('Invalid full accepted state')
        drift=max(drift,float(np.max(abs(pool@solver.y-pool0)))/den);dense=solver.dense_output()
        pieces.append(DensePower(np.column_stack([dense.y_old,dense.Q]),solver.t-left))
    curve=InnerCurve(pieces);stats=dict(nfev=solver.nfev,njev=solver.njev,nlu=solver.nlu,accepted_steps=len(pieces),pool_error=drift,
        continuous_nonnegative=bool(np.all(curve.minima()>=-1e-12*scale)),dense_join_error=curve.join_error(scale),linear=cache.stats.copy())
    if stats['dense_join_error']>1e-10:raise ValueError('Discontinuous full curve')
    if stats['continuous_nonnegative']:stats.update(ledgers(case,curve,y,solver.y))
    stats['seconds']=time.perf_counter()-tick
    return dict(fine=solver.y.copy(),pieces=[curve],stats=stats)
