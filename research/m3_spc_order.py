"""Preserve coarse cancellation and asymptotic SPC endpoint/output diagnostics."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from metabolism.splitting import ReactionSystem
from metabolism.m2_experiments import save
from research.spc_mri import SPC
from research.test_spc_mri import linear_case,exact_linear_step
from research.m3_shadow_study import ROOT,digest


def evaluate():
    case,y,fast,slow=linear_case();linear=[];query=np.linspace(0,1,65)
    for order in (2,3):
        for h in (.1,.05,.025,.0125,.00625):
            exact=np.column_stack([expm(h*s*(fast+slow))@y for s in query]);end,_=exact_linear_step(fast,slow,y,h,order)
            for output in ('inner','hermite'):
                result=SPC(case,[0]).trial(y,h,order,1e-12,output,residual_tol=1e-13)
                linear.append(dict(order=order,h=h,output=output,endpoint_error=float(np.linalg.norm(result['fine']-exact[:,-1])),
                    formula_error=float(np.linalg.norm(result['fine']-end)),curve_error=float(np.max(abs(result['pieces'][0](query)-exact)))))
    y=np.array([1.,.5]);s=csc_matrix([[-1,1,-1,1],[1,-1,1,-1]])
    rates=lambda t,z:np.array([3*z[0],z[1],.2*z[0]**2,.3*z[1]**2])
    grad=lambda t,z:csc_matrix([[3.,0.],[0.,1.],[.4*z[0],0.],[0.,.6*z[1]]])
    system=ReactionSystem(s,rates,y,[np.array([0,1]),np.array([2,3])],csc_matrix([[1,0],[0,1],[1,0],[0,1]]),grad)
    case=dict(system=system,scale=np.ones(2),options=dict(pools=csc_matrix([[1.,1.]])))
    refs=[solve_ivp(lambda t,z:s@rates(t,z),(0,.4),y,method='DOP853',rtol=t,atol=2e-14).y[:,-1] for t in (3e-14,1e-12)]
    nonlinear=[]
    for order in (2,3):
        previous=None
        for count in (4,8,16,32,64):
            kernel=SPC(case,[0]);z=y.copy()
            for _ in range(count):z=kernel.trial(z,.4/count,order,1e-12,residual_tol=1e-13)['fine']
            error=float(np.linalg.norm(z-refs[0]))
            nonlinear.append(dict(order=order,steps=count,error=error,observed_order=None if previous is None else float(np.log2(previous/error))))
            previous=error
    return dict(linear=linear,nonlinear=nonlinear,nonlinear_reference_difference=float(np.linalg.norm(refs[0]-refs[1])))


if __name__=='__main__':
    path=ROOT/'results/m3-spc/order.json'
    if path.exists():raise FileExistsError('Preserve diagnostic')
    save(path,dict(source_sha256={p:digest(ROOT/p) for p in ('research/spc_mri.py','research/test_spc_mri.py','research/m3_spc_order.py')},results=evaluate()))
