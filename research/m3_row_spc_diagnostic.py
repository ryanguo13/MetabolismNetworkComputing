"""Registered post-screen order, frozen-linear, and endpoint/output diagnosis."""
import json
import numpy as np
from scipy.linalg import expm
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.row_spc import linear_map
from research.test_row_spc import order_diagnostic
from research.m3_priority import metrics
from research.m3_shadow_study import ROOT,digest
from research.m3_row_spc_study import OUT


def evaluate(reg):
    old=json.loads((ROOT/'results/m3-growth-history/stability.json').read_text())['runs'];rows=[]
    with np.load(ROOT/'results/m3-growth-history/stability.npz') as arrays:
        for state in reg['states']:
            j=arrays[state+'-jacobian']
            for threshold in reg['thresholds']:
                fast=arrays[f'{state}-{threshold}-fast'];slow=arrays[f'{state}-{threshold}-slow']
                active=next(r['active'] for r in old if r['state']==state and r['threshold']==threshold)
                for h in reg['steps']:
                    truth=expm(h*j)
                    for theta in reg['theta']:
                        m=linear_map(fast,slow,h,theta);rho=float(max(abs(np.linalg.eigvals(m))))
                        rows.append(dict(state=state,threshold=threshold,h=h,theta=theta,active=active,spectral_radius=rho,
                            exact_spectral_radius=float(max(abs(np.linalg.eigvals(truth)))),map_error=float(np.linalg.norm(m-truth,2)),
                            passes_frozen_linear_screen=rho<=1+1e-9))
    cases={n:prepare(n) for n in ('teusink','kecoli')};errors=[]
    for r in json.loads((OUT/'runs.json').read_text())['runs']:
        if r['status']!='success' or r['method']=='radau':continue
        with np.load(OUT/f"{r['key']}.npz") as a,np.load(OUT/f"reference-{r['model']}-{r['h']}.npz") as ref:
            case=cases[r['model']];error=metrics(case,a['endpoint'][:,None],ref['reference'][:,-1:],1e-6)
            errors.append(dict(key=r['key'],endpoint=error,output=r['targets']['1e-06']['actual'],
                endpoint_scaled_error=float(np.max(abs(a['endpoint']-ref['reference'][:,-1])/case['scale']))))
    return dict(order=order_diagnostic(),stability=rows,errors=errors)


def run():
    paths=['research/row_spc.py','research/test_row_spc.py','research/m3_row_spc_diagnostic.py']
    reg=dict(date='2026-11-08',states=['early','late'],thresholds=[.01,.03,.1,.3,1.,3.,10.,30.,100.,300.],
        steps=[1e-4,3e-4,1e-3,3e-3,1e-2],theta=[0.,.5,1.],
        motivation='No qualified initial ROW-SPC candidate; distinguish endpoint/coupling error from dense output. Theta=.5 coarse nonlinear order cancellation requires a finer asymptotic range.',
        scope='300 frozen-linear exact-fast maps, two stored actual small-model Jacobians. Spectral radius is diagnostic only, not a nonlinear/variable-step claim. Nonlinear order extended through256 with original coarse points preserved.',
        source_sha256={p:digest(ROOT/p) for p in paths},input_sha256={p:digest(ROOT/p) for p in
          ('results/m3-row-spc/runs.json','results/m3-row-spc/initial-order.json','results/m3-growth-history/stability.json','results/m3-growth-history/stability.npz')})
    if (OUT/'diagnostic-registration.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'diagnostic-registration.json',reg);save(OUT/'diagnostic.json',evaluate(reg))


if __name__=='__main__':run()
