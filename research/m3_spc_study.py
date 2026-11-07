"""Registered two-model SPC-MRI local screen with equal SDIRK/Radau controls."""
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

OUT=ROOT/'results/m3-spc'


def run():
    OUT.mkdir(exist_ok=True)
    sources=['research/spc_mri.py','research/test_spc_mri.py','research/m3_spc_study.py',
             'research/shared_radau.py','research/fullstep_radau.py','research/inner_dense.py',
             'research/extrapolated_flow.py','research/m3_growth_study.py','research/m3_extrapolated_paths.py']
    reg=dict(date='2026-11-08',models={'teusink':dict(start=1.,thresholds=[.3,3.,10.]),'kecoli':dict(start=.1,thresholds=[.01,.1,1.])},
        steps=[.001,.003,.01],orders=[2,3],outputs=['inner','hermite'],tolerances=[1e-6,1e-8],predictor_residual=1e-11,
        method='Published SPC-MRI-GARK SDIRK2(1)2 / SDIRK3(2)4, primary gamma rows, arXiv1812.00808v5 Definition2.1 AppendixA.1.1/A.1.3. Embedded formulas not used.',
        scope='144 candidate,12 standalone SDIRK and36 mature full Radau local trials; single development timings, no full-window evidence.',
        predictor='Complete system stages, shared frozen analytic Jacobian/LU; refresh at iterations4/8 if needed, at most12, positivity-damped updates; original stage equation residual required.',
        corrector='Exactly one modified fast solve, exact inactive forcing integral; borrowed predictor reaction gradient seeds a stale/refreshable fast Jacobian. Rejected internal states retained in failure diagnostics, never clipped.',
        output_policy='Raw fast curve or full-RHS Hermite. Raw curve positivity and modified ledger checked in both. Actual physical ledger and actual query errors tested on selected output.',
        baselines='Same SDIRK predictor alone with full RHS Hermite; mature full Radau tolerances1e-6/1e-8/1e-10, automatic/whole first step, same state/grid and guards.',
        audits='Independent full1e-12/1e-11,65 points plus public observations; strict budget<=.05 required. Failed budget excludes qualification, retains diagnostics.',
        timing='All complete predictor stages/factors, one fast solve, final output/guards counted. Cached grouping and independent reference/accuracy scoring outside timer. Optimistic local costs; failed inner work counters may be partial.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={p:digest(ROOT/p) for p in
            ('results/m3-priority/p0.npz','results/m3-implicit-history/manifest.json')})
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'registration.json',reg);prefix=np.load(ROOT/'results/m3-priority/p0.npz');rows=[];refs=[]
    for name,spec in reg['models'].items():
        case=prepare(name);screen=GroupScreen(case['system'],case['scale']);y=prefix[name+'-prefix'][:,3]
        kappa,_=screen.evaluate(y)
        for h in reg['steps']:
            start=spec['start'];query=np.unique(np.r_[np.linspace(0,1,65),(case['times'][(case['times']>start)&(case['times']<start+h)]-start)/h])
            reference,budget,audit_seconds=audit(case,y,query,h);ok=max(budget.values())<=.05
            refs.append(dict(model=name,h=h,budget=budget,reference_ok=ok,audit_pair_seconds=audit_seconds))
            np.savez_compressed(OUT/f'reference-{name}-{h}.npz',initial=y,query=query,reference=reference)
            configs=[dict(method='radau',tolerance=t,whole=w) for t in (1e-6,1e-8,1e-10) for w in (False,True)]
            configs += [dict(method='sdirk',order=q,output='hermite',tolerance=1e-8,threshold=0.) for q in reg['orders']]
            configs += [dict(method='spc',order=q,output=o,tolerance=t,threshold=g) for g in spec['thresholds'] for q in reg['orders'] for o in reg['outputs'] for t in reg['tolerances']]
            for index,config in enumerate(configs):
                key=f'{name}-{h}-{index}';row=dict(key=key,model=name,h=h,**config);kernel=None;arrays={}
                if config['method']!='radau':
                    groups=np.flatnonzero(1e-3*kappa>=config['threshold']) if config['method']=='spc' else np.array([],dtype=int)
                    kernel=SPC(case,groups);row.update(groups=groups.tolist(),active=len(kernel.active),setup_seconds=kernel.setup_seconds)
                tick=time.perf_counter()
                try:
                    if config['method']=='radau':
                        trial=(fullstep_curve if config['whole'] else full_curve)(case,y,h,config['tolerance'],screen)
                    else:
                        trial=kernel.trial(y,h,config['order'],config['tolerance'],config['output'],base_only=config['method']=='sdirk',residual_tol=reg['predictor_residual'])
                    curve=trial['pieces'][0];values=curve(query);structural(case,values);elapsed=time.perf_counter()-tick
                    guard,targets=qualified(case,curve,values,reference,trial['stats'])
                    for v in targets.values():v['passes'] &= ok
                    row.update(status='success',seconds=elapsed,stats=trial['stats'],guard=guard,targets=targets)
                    arrays['endpoint']=trial['fine']
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
