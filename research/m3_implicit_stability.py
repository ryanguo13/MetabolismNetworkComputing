"""Registered implicit endpoint-history screen on the prior frozen Jacobians."""
import json
import numpy as np
from metabolism.m2_experiments import save
from research.implicit_history import amplification
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-implicit-history';OLD=ROOT/'results/m3-growth-history'


def evaluate():
    prior=json.loads((OLD/'stability.json').read_text())['runs'];rows=[];arrays={}
    with np.load(OLD/'stability.npz') as a:
        for old in prior:
            key=f"{old['state']}-{old['threshold']}";fast=a[key+'-fast'];slow=a[key+'-slow']
            for order in (2,3,4):
                matrix,endpoint,iteration=amplification(fast,slow,old['h'],order)
                rho=float(max(abs(np.linalg.eigvals(matrix))));rho_iter=float(max(abs(np.linalg.eigvals(iteration))))
                name=f"{key}-{old['h']}-{order}"
                rows.append(dict(name=name,state=old['state'],threshold=old['threshold'],h=old['h'],order=order,
                    fast_active=old['active'],slow_active=int(np.count_nonzero(np.any(slow!=0,axis=1))),
                    slow_rank=int(np.linalg.matrix_rank(slow)),spectral_radius=rho,explicit4_radius=old['spectral_radius'],
                    fixed_point_radius=rho_iter,endpoint_condition=float(np.linalg.cond(endpoint)),
                    passes_frozen_linear_screen=rho<=1+1e-9))
                arrays[name+'-companion']=matrix;arrays[name+'-endpoint']=endpoint
    return rows,arrays


def run():
    OUT.mkdir(exist_ok=True)
    sources=['research/implicit_history.py','research/test_implicit_history.py','research/m3_implicit_stability.py']
    reg=dict(date='2026-11-08',orders=[2,3,4],configurations=300,
        forcing='Lagrange slow RHS through future endpoint and current/past states; exact fast flow; endpoint equation solved exactly for frozen linear model.',
        screen='rho(companion)<=1+1e-9 only; no nonlinear, variable-step, positivity, accuracy or speed certification.',
        fixed_point='rho of future forcing response times slow Jacobian diagnoses naive fixed-point convergence.',
        source_sha256={p:digest(ROOT/p) for p in sources},
        input_sha256={f'results/m3-growth-history/{p}':digest(OLD/p) for p in ('manifest.json','stability.json','stability.npz')})
    if (OUT/'stability-registration.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'stability-registration.json',reg)
    rows,arrays=evaluate();save(OUT/'stability.json',dict(runs=rows));np.savez_compressed(OUT/'stability.npz',**arrays)
    for state in ('early','late'):
        for h in (1e-4,3e-4,1e-3,3e-3,1e-2):
            print(state,h,[(q,min((r['fast_active'] for r in rows if r['state']==state and r['h']==h and r['order']==q and r['passes_frozen_linear_screen']),default=None)) for q in (2,3,4)])


if __name__=='__main__':run()
