"""Replay current predictor equations to diagnose overstrict correction scaling."""
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor,lu_solve
from metabolism.m3_continuation import prepare
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from research.row_continuous import coupling
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-cached-row'


def evaluate():
    rows=[];gamma=coupling()[1][0,0]
    for model,index in (('teusink',0),('kecoli',2)):
        case=prepare(model);screen=GroupScreen(case['system'],case['scale']);scale=case['scale'];h=.003
        with np.load(OUT/f'{index}-original.npz') as a:y0=a['0-initial'];y=a['1-initial']
        matrices=[eye(len(y),format='csc')-h*gamma*(case['system'].S@screen.gradient(z)) for z in (y0,y)]
        def factor(m):
            if len(y)>32:return splu(m.tocsc(),permc_spec='MMD_AT_PLUS_A').solve
            lu=lu_factor(m.toarray());return lambda b:lu_solve(lu,b)
        old=factor(matrices[0]);fresh=factor(matrices[1]);rhs=h*(case['system'].S@case['system'].flux(0.,y));truth=fresh(rhs);x=old(rhs)
        for iteration in range(1,9):
            d=old(rhs-matrices[1]@x);x+=d
            component=abs(d)/np.maximum(abs(x),1e-14*scale);index_worst=int(np.argmax(component))
            rows.append(dict(model=model,iteration=iteration,component_relative=float(max(component)),
                weighted_infinity_relative=float(np.max(abs(d)/scale)/max(np.max(abs(x)/scale),1e-14)),
                residual_scaled=float(np.max(abs(rhs-matrices[1]@x)/scale)),forward_difference_scaled=float(np.max(abs(x-truth)/scale)),
                worst_component=index_worst,worst_increment_scaled=float(abs(x[index_worst])/scale[index_worst])))
    return rows


if __name__=='__main__':
    reg=dict(source_sha256={p:digest(ROOT/p) for p in ('research/cache_refinement_diagnostic.py','research/refined_lu.py')},
        input_sha256={p:digest(ROOT/p) for p in ('results/m3-cached-row/0-original.npz','results/m3-cached-row/2-original.npz')},
        motivation='Initial component-relative refinement refreshes every subsequent predictor despite unchanged H. Compare original norm, weighted-infinity norm, current residual and fresh-LU forward discrepancy; eight iterations retained.')
    if (OUT/'refinement-diagnostic.json').exists():raise FileExistsError('Preserve diagnostic')
    save(OUT/'refinement-registration.json',reg);save(OUT/'refinement-diagnostic.json',dict(runs=evaluate()))
    for r in evaluate():print(r)
