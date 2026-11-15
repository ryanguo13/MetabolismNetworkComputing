"""Independent correction of the small-model current-backend dispatch error.

Original32 TypeError attempts remain immutable and excluded from comparison.
"""
import argparse
import math
from pathlib import Path
import time
import numpy as np
from scipy.integrate import solve_ivp
from metabolism.m2_experiments import save
from research.m3_shadow_study import ROOT,digest
from research.m3_ledger_flow_report import registration,close
from research.m3_ordered_cost import structural
from research import m3_audit_output_cost as study
from research.m3_audit_output_integrity import nonnegative

ORIGINAL=study.OUT
OUT=ROOT/'results/m3-audit-output-corrected'
ORIGINAL_GUARDED=study.guarded_audit


def guarded_audit(case,screen,y,h,q,tol,method):
    if method!='current' or case['system'].flux_jacobian is not None:
        return ORIGINAL_GUARDED(case,screen,y,h,q,tol,method)
    start=time.perf_counter();minimum=float(y.min())
    def rhs(t,z):
        nonlocal minimum
        if time.perf_counter()-start>20:raise TimeoutError('Audit solve budget')
        minimum=min(minimum,float(z.min()))
        if not np.isfinite(z).all() or np.any(z < -1e-12*case['scale']):raise RuntimeError('Invalid full ODE RHS domain; no clipping')
        return case['system'].S@case['system'].flux(0.,z)
    # Preserve the actual original full/current fallback: default SciPy Radau,
    # with no supplied analytic Jacobian and no whole-interval first step.
    r=solve_ivp(rhs,(0.,h),y,t_eval=h*q,rtol=tol,atol=.01*tol*case['scale'],method='Radau')
    if not r.success:raise RuntimeError(r.message)
    guard=structural(case,r.y)
    if not guard['nonnegative'] or not np.isfinite(guard['pool']) or guard['pool']>1e-10:raise RuntimeError('Invalid audit output domain/pool')
    return r.y,dict(backend='SciPy Radau; no supplied Jacobian',nfev=r.nfev,njev=r.njev,nlu=r.nlu,min_rhs_argument=minimum,seconds=time.perf_counter()-start),guard


def register():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve corrected evidence')
    reg=study.read(ORIGINAL/'registration.json')
    reg['cases']=[c for c in reg['cases'] if c['reference_key']=='0-2-1'];assert len(reg['cases'])==1
    reg['correction']='Independent32-audit+96-control rerun of only Teusink early state. Original current branch incorrectly assumed analytic CVODE and produced32 NoneType-Jacobian failures. Original full/current actually falls back to default SciPy Radau when flux_jacobian isNone. Preserve those implementation-error attempts and exclude them scientifically; restore original backend with same internal domain protection, exact selected configs/candidate/queries and all6x2 mature controls. This registration is before corrected new solves; original results already observed.'
    reg['source_sha256']['research/m3_audit_output_corrected.py']=digest(Path(__file__))
    reg['source_sha256']['research/m3_audit_output_integrity.py']=digest(ROOT/'research/m3_audit_output_integrity.py')
    reg['input_sha256'].update({str(p.relative_to(ROOT)):digest(p) for p in [ORIGINAL/'registration.json',ORIGINAL/'runs.json',ORIGINAL/'manifest.json']})
    save(OUT/'registration.json',reg)


def checks():
    reg=registration(OUT/'registration.json');data=study.read(OUT/'runs.json');cfg=reg['cases'][0];expected={'audits':{},'controls':{}}
    for ai,ac in enumerate(cfg['audit_configs']):
        for mode in reg['query_modes']:
            for rep in reg['repeats']:
                expected['audits'][f"audit-{cfg['reference_key']}-{ai}-{mode}",rep]=dict(target=ac['target'],method=ac['method'],tolerance=ac['tolerance'],query_mode=mode)
    for method in reg['control_methods'][cfg['model']]:
        for tol in reg['control_tolerances']:
            for mode in reg['query_modes']:
                for rep in reg['repeats']:
                    expected['controls'][f"control-{cfg['reference_key']}-{method}-{tol}-{mode}",rep]=dict(method=method,tolerance=tol,query_mode=mode)
    assert len(expected['audits'])==32 and len(expected['controls'])==96
    for kind in expected:
        rows=data[kind];identities=[(r['key'],r['repeat']) for r in rows]
        assert len(identities)==len(set(identities)) and set(identities)==set(expected[kind])
        for row in rows:
            spec=expected[kind][row['key'],row['repeat']];close({k:row[k] for k in spec},spec)
            nonnegative(row['seconds'],'seconds',positive=True)
            first=next(r for r in rows if r['key']==row['key'] and r['repeat']==-1)
            assert row['status']==first['status']
            if row['status']=='failed':assert row['reason']==first['reason'];continue
            for k,v in row['stats'].items():
                if isinstance(v,(int,float)) and not isinstance(v,bool):
                    if not math.isfinite(v):raise ValueError('Nonfinite stat')
                    if k.endswith('seconds'):nonnegative(v,k)
            nonnegative(row['guard']['pool'],'pool')
            if kind=='audits':
                for k in ('runtime_score_seconds','audit_plus_scoring_seconds','offline_qualification_seconds'):nonnegative(row[k],k)
                study.finite_metrics(row['budget']);study.finite_metrics(row['runtime_candidate_error'])
            else:
                for target in row['targets'].values():study.finite_metrics(target['actual'])
    return dict(configuration_repeat_sets_exact=True,finite_metrics_and_timing=True,expected_audits=32,expected_controls=96)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','write','verify']);args=p.parse_args()
    if args.action=='register':register();return
    study.OUT=OUT;study.guarded_audit=guarded_audit
    if args.action=='run':study.run();return
    integrity=checks();result=study.verify();result['corrected_batch_integrity']=integrity
    result['original_implementation_errors_excluded']=32
    if args.action=='write':
        if (OUT/'manifest.json').exists():raise FileExistsError('Preserve corrected manifest')
        save(OUT/'summary.json',result)
        save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    else:
        close(result,study.read(OUT/'summary.json'))
        for path,sha in study.read(OUT/'manifest.json').items():assert digest(ROOT/path)==sha,path
    print(result)


if __name__=='__main__':main()
