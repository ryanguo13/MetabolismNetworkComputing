"""Registered two-model ROW-SPC development screen with equal full controls."""
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.row_spc import ROWSPC
from research.extrapolated_flow import pack
from research.inner_dense import full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_growth_study import audit
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-row-spc'


def run():
    OUT.mkdir(exist_ok=True)
    sources=['research/'+p for p in ('row_spc.py','test_row_spc.py','m3_row_spc_study.py','spc_mri.py',
        'shared_radau.py','fullstep_radau.py','inner_dense.py','extrapolated_flow.py','m3_growth_study.py','m3_extrapolated_paths.py')]
    reg=dict(date='2026-11-08',models={'teusink':dict(start=1.,thresholds=[.3,3.,10.]),'kecoli':dict(start=.1,thresholds=[.01,.1,1.])},
        steps=[.001,.003,.01],theta=[0.,.5,1.],tolerances=[1e-6,1e-8],
        method='Guenther/Sandu arXiv2102.10203v1 eq8.1/10.1/10.2/10.9, Example8 ROS34PW2, nu=0. mu constraints solved from printed alpha/gamma. No embedded estimator.',
        scope='108 candidate,6 standalone full ROS34PW2,36 mature full Radau local trials. Single optimistic development timings; not full-window or formal M4 evidence.',
        predictor='Full analytic frozen Jacobian; one factorization, four RHS, four solves; original physical stages checked. No nonlinear iterations or clipping.',
        output='Raw transformed fast state z satisfies z prime=F(z)+Ks mu/H. Endpoint=z(H)+Ks(b-mu). Modified ledger uses z(H); independent physical output is full-RHS endpoint Hermite. Raw z is not physical dense output.',
        controls='Standalone full ROS34PW2 same predictor and Hermite; mature full Radau tolerances1e-6/1e-8/1e-10 automatic/whole first step.',
        audit='Independent complete ODE1e-12/1e-11,65 points plus public observations; strict reference budget<=.05. All original local accuracy/pool/ledger/net-flux gates unchanged.',
        timing='Predictor, fast correction, output guards and observation structural checks included. Cached grouping, independent reference and accuracy scoring excluded; setup separately reported. Partial failed inner counters possible.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={p:digest(ROOT/p) for p in
            ('results/m3-priority/p0.npz','results/m3-spc/manifest.json','results/m3-implicit-history/literature-provenance.json')})
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'registration.json',reg);prefix=np.load(ROOT/'results/m3-priority/p0.npz');rows=[];refs=[]
    for name,spec in reg['models'].items():
        case=prepare(name);screen=GroupScreen(case['system'],case['scale']);y=prefix[name+'-prefix'][:,3];kappa,_=screen.evaluate(y)
        for h in reg['steps']:
            start=spec['start'];query=np.unique(np.r_[np.linspace(0,1,65),(case['times'][(case['times']>start)&(case['times']<start+h)]-start)/h])
            reference,budget,audit_seconds=audit(case,y,query,h);ok=max(budget.values())<=.05
            refs.append(dict(model=name,h=h,budget=budget,reference_ok=ok,audit_pair_seconds=audit_seconds))
            np.savez_compressed(OUT/f'reference-{name}-{h}.npz',initial=y,query=query,reference=reference)
            configs=[dict(method='radau',tolerance=t,whole=w) for t in (1e-6,1e-8,1e-10) for w in (False,True)]
            configs += [dict(method='row',tolerance=1e-8,threshold=0.,theta=.5)]
            configs += [dict(method='row-spc',theta=q,tolerance=t,threshold=g) for g in spec['thresholds'] for q in reg['theta'] for t in reg['tolerances']]
            for index,config in enumerate(configs):
                key=f'{name}-{h}-{index}';row=dict(key=key,model=name,h=h,**config);kernel=None;arrays={}
                if config['method']!='radau':
                    groups=np.flatnonzero(1e-3*kappa>=config['threshold']) if config['method']=='row-spc' else np.array([],dtype=int)
                    kernel=ROWSPC(case,groups);row.update(groups=groups.tolist(),active=len(kernel.active),setup_seconds=kernel.setup_seconds)
                tick=time.perf_counter()
                try:
                    if config['method']=='radau':trial=(fullstep_curve if config['whole'] else full_curve)(case,y,h,config['tolerance'],screen)
                    else:trial=kernel.trial(y,h,config['tolerance'],config['theta'],base_only=config['method']=='row')
                    curve=trial['pieces'][0];values=curve(query);structural(case,values);elapsed=time.perf_counter()-tick
                    guard,targets=qualified(case,curve,values,reference,trial['stats'])
                    for v in targets.values():v['passes'] &= ok
                    row.update(status='success',seconds=elapsed,stats=trial['stats'],guard=guard,targets=targets)
                    arrays['endpoint']=trial['fine']
                    for k,v in (curve.pack() if config['method']=='radau' else pack(curve)).items():arrays[k]=v
                    if config['method']!='radau':
                        for k in ('increments','slow_increments','stages','rates','full'):arrays[k]=trial[k]
                        if config['method']=='row-spc':
                            arrays.update(forcing=trial['forcing_coefficients'],raw_endpoint=trial['raw_endpoint'])
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
