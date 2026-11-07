"""Registered large-model predictor residual sweep after stagnation diagnostics."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.spc_mri import SPC
from research.extrapolated_flow import pack
from research.inner_dense import full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_growth_study import audit
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-spc-solver-tolerance'


def run():
    OUT.mkdir(exist_ok=True);old=ROOT/'results/m3-spc'
    sources=['research/spc_mri.py','research/test_spc_mri.py','research/m3_spc_solver_tolerance.py','research/spc_residual_diagnostic.py',
             'research/shared_radau.py','research/fullstep_radau.py','research/inner_dense.py','research/extrapolated_flow.py','research/m3_growth_study.py','research/m3_extrapolated_paths.py']
    reg=dict(date='2026-11-08',model='kecoli',start=.1,steps=[.001,.003,.01],thresholds=[.01,.1,1.],orders=[2,3],outputs=['inner','hermite'],
        inner_tolerances=[1e-6,1e-8],predictor_residuals=[1e-9,1e-8],
        motivation='Original1e-11 predictor residual stalls after large decrease; 3 traces record persistent residual oscillations. Test internal solver tolerances, retaining all original output and independent full audit gates.',
        scope='144 SPC candidates,12 identical standalone SDIRK controls,18 mature Radau controls. Single-run optimistic local timing; not formal speed or full-window evidence.',
        method='Same published coefficients, frozen predictor implementation, normal Jacobian refresh, fast seed stale, raw/physical ledger protection and query grids as initial SPC study.',
        timing='Complete predictor and fast correction plus guards; grouping and independent audit/scoring excluded. Same residual tolerance available to base SDIRK controls.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={p:digest(ROOT/p) for p in
            ('results/m3-priority/p0.npz','results/m3-spc/registration.json','results/m3-spc/runs.json','results/m3-spc/predictor-trace.json')})
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'registration.json',reg);case=prepare('kecoli');screen=GroupScreen(case['system'],case['scale'])
    y=np.load(ROOT/'results/m3-priority/p0.npz')['kecoli-prefix'][:,3];kappa,_=screen.evaluate(y);rows=[];refs=[]
    for h in reg['steps']:
        query=np.load(old/f'reference-kecoli-{h}.npz')['query'];reference,budget,audit_seconds=audit(case,y,query,h);ok=max(budget.values())<=.05
        refs.append(dict(h=h,budget=budget,reference_ok=ok,audit_pair_seconds=audit_seconds))
        np.savez_compressed(OUT/f'reference-{h}.npz',initial=y,query=query,reference=reference)
        configs=[dict(method='radau',tolerance=t,whole=w) for t in (1e-6,1e-8,1e-10) for w in (False,True)]
        configs += [dict(method='sdirk',order=q,output='hermite',tolerance=1e-8,threshold=0.,predictor_residual=p) for p in reg['predictor_residuals'] for q in reg['orders']]
        configs += [dict(method='spc',order=q,output=o,tolerance=t,threshold=g,predictor_residual=p) for p in reg['predictor_residuals'] for g in reg['thresholds'] for q in reg['orders'] for o in reg['outputs'] for t in reg['inner_tolerances']]
        for index,config in enumerate(configs):
            key=f'{h}-{index}';row=dict(key=key,model='kecoli',h=h,**config);kernel=None;arrays={}
            if config['method']!='radau':
                groups=np.flatnonzero(1e-3*kappa>=config['threshold']) if config['method']=='spc' else np.array([],dtype=int)
                kernel=SPC(case,groups);row.update(groups=groups.tolist(),active=len(kernel.active),setup_seconds=kernel.setup_seconds)
            tick=time.perf_counter()
            try:
                if config['method']=='radau':trial=(fullstep_curve if config['whole'] else full_curve)(case,y,h,config['tolerance'],screen)
                else:trial=kernel.trial(y,h,config['order'],config['tolerance'],config['output'],base_only=config['method']=='sdirk',residual_tol=config['predictor_residual'])
                curve=trial['pieces'][0];values=curve(query);structural(case,values);elapsed=time.perf_counter()-tick
                guard,targets=qualified(case,curve,values,reference,trial['stats'])
                for v in targets.values():v['passes'] &= ok
                row.update(status='success',seconds=elapsed,stats=trial['stats'],guard=guard,targets=targets);arrays['endpoint']=trial['fine']
                for k,v in (curve.pack() if config['method']=='radau' else pack(curve)).items():arrays[k]=v
                if config['method']!='radau':
                    pred=trial['predictor'];arrays.update(predictor_stages=pred['stages'],predictor_full=pred['full'],predictor_rates=pred['rates'],initial_rhs=pred['initial_rhs'])
                    if config['method']=='spc':
                        arrays['forcing']=trial['forcing_coefficients']
                        for k,v in pack(trial['raw']).items():arrays['raw-'+k]=v
            except (ValueError,RuntimeError,TimeoutError,FloatingPointError) as exc:
                row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
            finally:
                if kernel is not None:kernel.close()
            rows.append(row)
            if arrays:np.savez_compressed(OUT/f'{key}.npz',**arrays)
            save(OUT/'runs.json',dict(runs=rows,references=refs))
            print(key,config['method'],row['status'],[t for t,v in row.get('targets',{}).items() if v['passes']],flush=True)


if __name__=='__main__':run()
