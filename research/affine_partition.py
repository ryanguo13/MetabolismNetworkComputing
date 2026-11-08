"""Local affine reaction model and displacement-form reference oracles.

A frozen model is diagnostic data, never a replacement for nonlinear truth.
Keep v(y0), Dv(y0), reaction columns and groups; use displacement references to
avoid subtracting nearly cancelling affine offsets from large concentrations.
"""
import time
from copy import copy
import numpy as np
from scipy.integrate import solve_ivp
from metabolism.splitting import ReactionSystem
from metabolism.adaptive import GroupScreen
from research.rodas_coupling import coupling


def freeze(case,initial):
    sys=case['system'];origin=np.asarray(initial).copy();gradient=GroupScreen(sys,case['scale']).gradient(origin).tocsc();rates=sys.flux(0.,origin).copy()
    def flux(t,z):return rates+gradient@(z-origin)
    frozen=ReactionSystem(sys.S,flux,origin.copy(),sys.groups,sys.dependencies,lambda t,z:gradient)
    result={k:v for k,v in case.items() if k!='model'};result['system']=frozen
    return result,dict(origin=origin,rates=rates,gradient=gradient,jacobian=(sys.S@gradient).tocsc(),rhs=sys.S@rates)


def displacement_reference(jacobian,drive,initial,h,query,scale,tol):
    """Adaptive full Radau for d'=Jd+drive(t), d(0)=0, no positivity bypass.

    This oracle reports diagnostic reference values without claiming that the
    affine system has positive chemistry. Candidate guards remain independent.
    """
    tick=time.perf_counter();n=len(initial)
    r=solve_ivp(lambda t,d:jacobian@d+drive(t),(0.,h),np.zeros(n),method='Radau',jac=jacobian,
        rtol=tol,atol=.01*tol*scale,t_eval=h*np.asarray(query))
    if not r.success or not np.isfinite(r.y).all():raise RuntimeError('Affine displacement reference failed')
    return initial[:,None]+r.y,dict(seconds=time.perf_counter()-tick,nfev=r.nfev,njev=r.njev,nlu=r.nlu)


def affine_predictor(context,sf,cols,h,augmented):
    """Independent linear algebra predictor in increments, no state differencing."""
    from scipy.sparse import eye
    from scipy.sparse.linalg import splu
    a,g,b,mu=coupling(augmented);j=context['jacobian'];fastj=(sf@context['gradient'][cols]).tocsc();slowj=j-fastj
    f=context['rhs'];fast=sf@context['rates'][cols];n=len(f);solve=splu(eye(n,format='csc')-h*g[0,0]*j).solve;inc=[];slow=[]
    for i in range(len(b)):
        stage=sum((a[i,k]*inc[k] for k in range(i)),start=np.zeros(n));prev=sum((g[i,k]*inc[k] for k in range(i)),start=np.zeros(n))
        ki=solve(h*(f+j@(stage+prev)));inc.append(ki)
        slow.append(h*(f-fast+slowj@(stage+prev+g[i,i]*ki)))
    inc=np.column_stack(inc);slow=np.column_stack(slow);coeff=slow@mu*np.arange(1,mu.shape[1]+1)/h
    return dict(increments=inc,slow_increments=slow,forcing=coeff,fastj=fastj,fast=fast)


def decomposition(nonlinear_candidate,nonlinear_truth,affine_candidate,affine_truth,scale,target=1e-6,active=None):
    """Signed identity E_N=E_L+(E_N-E_L), one shared nonlinear denominator.

    The remainder includes nonlinear and numerical differences; norms are not
    additive or causal variance fractions. Record alignment and cancellation.
    """
    den=target*(.01*scale[:,None]+np.maximum(abs(nonlinear_truth),scale[:,None]))
    en=(nonlinear_candidate-nonlinear_truth)/den;el=(affine_candidate-affine_truth)/den;diff=en-el
    def norm(a):return float(np.max(abs(a)))
    index=np.unravel_index(np.argmax(abs(en)),en.shape);result=dict(nonlinear=norm(en),frozen=norm(el),difference=norm(diff),endpoint=dict(nonlinear=norm(en[:,-1]),frozen=norm(el[:,-1]),difference=norm(diff[:,-1])),worst=dict(species=int(index[0]),query=int(index[1]),nonlinear=float(en[index]),frozen=float(el[index]),difference=float(diff[index])),identity_roundoff=norm(en-el-diff),truth_linearization_gap=norm((nonlinear_truth-affine_truth)/den))
    if active is not None:
        inactive=np.setdiff1d(np.arange(len(scale)),active)
        for key,inds in [('active',active),('inactive',inactive)]:result[key]={k:(norm(a[inds]) if len(inds) else 0.) for k,a in [('nonlinear',en),('frozen',el),('difference',diff)]}
    length=np.linalg.norm(en)*np.linalg.norm(el);result['cosine']=None if length==0 else float(np.sum(en*el)/length)
    return result
