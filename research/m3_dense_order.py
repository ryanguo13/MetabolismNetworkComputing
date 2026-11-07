"""Exact linear stage-order diagnosis for raw inner MRI3 curves."""
import json
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix
from metabolism.splitting import ReactionSystem
from metabolism.m2_experiments import save
from research.mri_dense import DenseMRI
from research.m3_shadow_study import ROOT,digest
OUT=ROOT/'results/m3-inner-dense'


def run():
    source={p:digest(ROOT/p) for p in ['research/m3_dense_order.py','research/mri_dense.py','research/inner_dense.py']}
    reg=dict(date='2026-11-08',steps=[.1,.05,.025,.0125,.00625],tolerance=1e-13,
        scope='Linear positive conservative three-state problem, exact expm reference. One-step errors; no extrapolation to all networks.',
        matrix=[[-10.,3.,0.],[10.,-4.,2.],[0.,1.,-2.]],source_sha256=source)
    path=OUT/'order-registration.json'
    if path.exists():assert json.loads(path.read_text())==reg
    else:save(path,reg)
    s=csc_matrix([[-1,1,0,0],[1,-1,-1,1],[0,0,1,-1]])
    j=csc_matrix([[10,0,0],[0,3,0],[0,1,0],[0,0,2]]);y=np.array([.8,.1,.1]);a=(s@j).toarray()
    sys=ReactionSystem(s,lambda t,z:np.asarray(j@z).ravel(),y,[np.array([0,1]),np.array([2,3])],j.astype(bool),lambda t,z:j)
    case=dict(system=sys,scale=np.ones(3),options=dict(pools=csc_matrix([[1,1,1]])))
    solver=DenseMRI(case,[0]);records=[];arrays={}
    for i,h in enumerate(reg['steps']):
        trial=solver.trial(y,h,reg['tolerance']);curve=trial['pieces'][0]
        grid=np.unique(np.r_[np.linspace(0,1,193),1/3,2/3,curve.edges/h])
        grid=np.clip(grid,0,1);truth=np.column_stack([expm(h*t*a)@y for t in grid]);values=curve(grid);hermite=trial['hermite'](grid)
        row=dict(h=h,endpoint=float(np.max(abs(values[:,-1]-truth[:,-1]))),
            inner=float(np.max(abs(values-truth))),hermite=float(np.max(abs(hermite-truth))),
            stage2=float(np.max(abs(curve(np.array([1/3]))[:,0]-expm(h*a/3)@y))),
            stage3=float(np.max(abs(curve(np.array([2/3]))[:,0]-expm(2*h*a/3)@y))))
        records.append(row)
        arrays.update({f'{i}-query':grid,f'{i}-reference':truth,f'{i}-inner':values,f'{i}-hermite':hermite})
    orders={k:[float(np.log2(x[k]/z[k])) for x,z in zip(records[:-1],records[1:])] for k in ('endpoint','inner','hermite','stage2','stage3')}
    assert orders['endpoint'][-1]>3.7 and 1.7<orders['inner'][-1]<2.3
    save(OUT/'order.json',dict(runs=records,observed_one_step_orders=orders,source_sha256=source))
    np.savez_compressed(OUT/'order.npz',**arrays);print(json.dumps(orders,indent=2))

if __name__=='__main__':run()
