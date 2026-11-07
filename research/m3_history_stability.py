"""Small-model frozen-linear diagnostic of whether cheaper partitions are viable."""
import json
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.mri_kernel import MRI
from research.history_stability import amplification
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-growth-history'


def run():
    sources=['research/m3_history_stability.py','research/history_stability.py','research/test_history_stability.py']
    reg=dict(date='2026-11-08',states={'early':2,'late':3},thresholds=[.01,.03,.1,.3,1.,3.,10.,30.,100.,300.],
        steps=[1e-4,3e-4,1e-3,3e-3,1e-2],order=4,
        scope='Post-growth exploratory small-model frozen-linear constant-step test. Exact fast solve; spectral radius alone does not prove nonlinear/variable-step stability or accuracy.',
        group_rule='Existing1e-3*kappa >= threshold; eigenanalysis of complete additive Jacobians at two known states.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={'results/m3-priority/p0.npz':digest(ROOT/'results/m3-priority/p0.npz'),'results/m3-growth-history/runs.json':digest(OUT/'runs.json')})
    path=OUT/'stability-registration.json'
    if path.exists():
        assert json.loads(path.read_text())==reg
        if (OUT/'stability.json').exists():raise FileExistsError('Preserve evidence')
    else:save(path,reg)
    case=prepare('teusink');prefix=np.load(ROOT/'results/m3-priority/p0.npz')['teusink-prefix'];screen=GroupScreen(case['system'],case['scale'])
    rows=[];arrays={}
    for state,column in reg['states'].items():
        y=prefix[:,column];kappa,gradient=screen.evaluate(y);jac=(case['system'].S@gradient).toarray()
        arrays[state+'-state']=y;arrays[state+'-jacobian']=jac
        for threshold in reg['thresholds']:
            groups=np.flatnonzero(1e-3*kappa>=threshold);kernel=MRI(case,groups)
            fast=(kernel.sf@gradient[kernel.cols]).toarray();slow=jac-fast;key=f'{state}-{threshold}'
            arrays[key+'-fast']=fast;arrays[key+'-slow']=slow
            for h in reg['steps']:
                eigen=np.linalg.eigvals(amplification(fast,slow,h,4));rho=float(max(abs(eigen)))
                rows.append(dict(state=state,threshold=threshold,h=h,groups=groups.tolist(),active=len(kernel.active),
                    fast_reactions=len(kernel.cols),spectral_radius=rho,passes_frozen_linear_screen=rho<=1+1e-9))
            kernel.close()
    save(OUT/'stability.json',dict(runs=rows));np.savez_compressed(OUT/'stability.npz',**arrays)
    for state in reg['states']:
        print(state,[(r['threshold'],r['active'],round(r['spectral_radius'],5)) for r in rows if r['state']==state and r['h']==.001],flush=True)


if __name__=='__main__':run()
