"""P1 step-growth check before stopping for lack of throughput prospect."""
import numpy as np

from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.mri_kernel import MRI
from research.m3_priority import OUT,GRID,full,metrics,assess,hashes


def run():
    rows,arrays=[],{}
    for name,start,steps in [('teusink',1.,(.1,1.,9.)),('kecoli',.1,(.01,.1,.9))]:
        case=prepare(name)
        y=full(case,case['model'].y0,np.array([0.,start]),1e-12)[:,-1]
        kappa,_=GroupScreen(case['system'],case['scale']).evaluate(y)
        groups=np.flatnonzero(1e-3*kappa >= .01)
        solver=MRI(case,groups,backend='radau' if name=='teusink' else 'native')
        try:
            for h in steps:
                observed=case['times'][(case['times']>start)&(case['times']<start+h)]
                fractions=np.unique(np.r_[GRID,(observed-start)/h])
                ref=full(case,y,h*fractions,1e-12)
                loose=full(case,y,h*fractions,1e-11)
                for tol in (1e-8,1e-10):
                    key=f'{name}-{h}-{tol}'
                    row=dict(key=key,model=name,start=start,h=h,tolerance=tol,
                        reference_pass=max(metrics(case,loose,ref,1e-6).values())<=.05)
                    try:
                        trial=solver.trial(y,h,tol,timeout=15.)
                        assessed,values=assess(case,trial,fractions,ref)
                        row.update(assessed)
                        arrays[key]=values
                        arrays[key+'-reference']=ref
                        arrays[key+'-fractions']=fractions
                        arrays[key+'-coarse']=trial['coarse_piece'](fractions)
                        arrays[key+'-nodes']=np.column_stack([y,trial['mid'],trial['fine']])
                    except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:
                        row.update(status='failed',reason=str(exc),stats=getattr(exc,'stats',{}))
                    rows.append(row)
                    save(OUT/'growth.json',dict(scope='Post-P1 development growth scan, both tolerances. Tests whether larger H can rescue throughput beyond initial P0 grid. Same frozen groups, targets, independent audits and 15s budget.',runs=rows,source_sha256=hashes()))
                    np.savez_compressed(OUT/'growth.npz',**arrays)
                    print(name,h,tol,row['status'],row.get('reason'),row.get('targets',{}).get('0.0001'),flush=True)
        finally:solver.close()


if __name__=='__main__':
    run()
