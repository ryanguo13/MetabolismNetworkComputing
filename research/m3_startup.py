"""Explicit follow-up to the failed smallest Teusink t=0 P0 probe."""
import json
import argparse
import hashlib
from pathlib import Path
import time

import numpy as np

from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.mri_kernel import MRI
from research.m3_priority import OUT,GRID,full,metrics,assess,hashes


def run(start=0.):
    case=prepare('teusink')
    y=case['model'].y0 if start==0 else full(case,case['model'].y0,np.array([0.,start]),1e-12)[:,-1]
    kappa,_=GroupScreen(case['system'],case['scale']).evaluate(y)
    groups=np.flatnonzero(1e-3*kappa >= .01)
    solver=MRI(case,groups)
    rows,arrays=[],{}
    try:
        for h in (1e-7,1e-8):
            observed=case['times'][(case['times']>start)&(case['times']<start+h)]
            fractions=np.unique(np.r_[GRID,(observed-start)/h])
            ref=full(case,y,h*fractions,1e-12)
            loose=full(case,y,h*fractions,1e-11)
            for tol in (1e-8,1e-10):
                key=f'{h}-{tol}'
                row=dict(key=key,model='teusink',start=start,h=h,tolerance=tol,
                    reference_pass=max(metrics(case,loose,ref,1e-6).values())<=.05)
                try:
                    trial=solver.trial(y,h,tol)
                    result,values=assess(case,trial,fractions,ref)
                    row.update(result)
                    arrays[key]=values
                    arrays[key+'-reference']=ref
                    arrays[key+'-fractions']=fractions
                    arrays[key+'-coarse']=trial['coarse_piece'](fractions)
                    arrays[key+'-nodes']=np.column_stack([y,trial['mid'],trial['fine']])
                except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:
                    row.update(status='failed',reason=str(exc),stats=getattr(exc,'stats',{}))
                rows.append(row)
                print(h,tol,row['status'],row.get('targets'),flush=True)
    finally:solver.close()
    label='startup' if start==0 else 'startup-early'
    source=hashes()
    source['research/m3_startup.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(OUT/f'{label}.json',dict(scope='Post-P0 early-layer follow-up, not part of the initial frozen 48 cases. Decrease H to 1e-7 and 1e-8 at the recorded start; identical groups, same targets and ledger criteria.',runs=rows,source_sha256=source))
    np.savez_compressed(OUT/f'{label}.npz',**arrays)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',type=float,choices=[0.,1e-6],default=0.)
    run(parser.parse_args().start)
