"""Prospective output-request cost challenge under original runtime audit duties."""
import argparse
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
import zipfile
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from metabolism.adaptive import GroupScreen,normalized_error
from metabolism.cvode import solve
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.ordered_radau import OrderedRadau
from research.inner_dense import InnerCurve,full_curve,physical_ledger
from research.fullstep_radau import fullstep_curve
from research.extrapolated_flow import unpack
from research.remainder_storage import load_arrays
from research.m3_ledger_flow_report import registration,close
from research.m3_formal_audit_reanalysis import formal_metrics,rates,read
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-audit-output-cost'
TRUTH=ROOT/'results/m3-remainder-curve'
FORMAL=ROOT/'results/m3-formal-audit-reanalysis'


def finite_metrics(values):
    if not values or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<0 for v in values.values()):
        raise ValueError('Nonfinite/invalid metric')
    return values


def register():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    prior=read(FORMAL/'summary.json');refs={r['key']:r for r in read(TRUTH/'runs.json')['references']}
    cases=[]
    paths=[FORMAL/'manifest.json',FORMAL/'summary.json',TRUTH/'manifest.json',TRUTH/'runs.json']
    for key in dict.fromkeys(r['reference_key'] for r in prior['comparisons']):
        rr=refs[key]
        source=ROOT/('results/m1/BIOMD0000000064-reference.npz' if rr['model']=='teusink' else 'results/m1-completion/kecoli-reference.npz')
        with np.load(source) as a:times=a['times']
        public=(times[(times>rr['start'])&(times<rr['start']+rr['h'])]-rr['start'])/rr['h']
        required=np.unique(np.r_[0.,.25,.5,.75,1.,public])
        with np.load(TRUTH/f'reference-{key}.npz') as a:enhanced=a['query']
        indices=np.searchsorted(enhanced,required)
        if not np.array_equal(enhanced[indices],required):raise ValueError('Required original outputs missing from truth grid')
        configs=[]
        for row in prior['comparisons']:
            if row['reference_key']!=key:continue
            best=row['best_single']
            for factor in (1.,.1):configs.append(dict(target=row['target'],method=best['method'],tolerance=best['tolerance']*factor))
        candidate=f'candidate-{key}-merb-1e-08'
        paths.extend([source,TRUTH/f'reference-{key}.npz'])
        candidatepath=TRUTH/f'{candidate}.npz'
        if candidatepath.exists():paths.append(candidatepath)
        else:paths.append(TRUTH/'packaging.json')
        cases.append(dict(reference_key=key,model=rr['model'],state_index=rr['state_index'],start=rr['start'],h=rr['h'],
            required_queries=required.tolist(),enhanced_queries=enhanced.tolist(),required_truth_indices=indices.tolist(),
            audit_configs=configs,scoring_candidate=candidate))
    sources=['research/m3_audit_output_cost.py','metabolism/continuation.py','metabolism/adaptive.py','metabolism/cvode.py',
        'research/ordered_radau.py','research/inner_dense.py','research/fullstep_radau.py','research/m3_formal_audit_reanalysis.py',
        'research/extrapolated_flow.py','research/remainder_storage.py']
    save(OUT/'registration.json',dict(date='2026-11-15',prospective=True,cases=cases,repeats=[-1,0,1,2],
        query_modes=['required','enhanced'],control_tolerances=[1e-4,1e-5,1e-6,1e-7,1e-8,1e-10],
        control_methods=dict(teusink=['ordered-auto','ordered-whole'],kecoli=['current','ordered-auto','ordered-whole']),
        hypothesis='MacroSplit requires every public output plus quarter/midpoint/three-quarter/end, not65 uniform queries. CVODE calls its native output once per requested time. Test this exact redundant request cost while retaining all original audit obligations.',
        audit='Independent original nonlinear ODE, same initial prefix; original internal domain finite and negative threshold-1e-12*case.scale. Required grid includes every original public point in interval and quarters/end; zero only initializes solve and is excluded from runtime scoring. Enhanced65 grid is a separately measured control. No audit is removed.',
        calibration='Freeze per state/formal target the posthoc fastest qualified single method/tolerance and its tenfold tightening before any new solves. External budget .05 in original formal concentration/one-way-flux targets1e-3/1e-5, net1e-3. Independent immutable65/public CVODE truth used only in offline qualification at the requested coordinates, never injected into runtime decisions. Minimal-grid result has no65-point continuous guarantee.',
        runtime_scoring='Time output of a fixed previously computed MERB1e-8 candidate, its structural sampling, and the exact MacroSplit observable_error arithmetic: local_fraction .1 times concentration/flux formal target and net1e-3. Candidate may fail; preserve result. Its generation is not retimed or credited as zero cost; reported audit+score budget is an optimistic residual for future candidate work. No truth used in scoring.',
        controls='Same query reduction available to all mature controls. OrderedRadau auto/whole retain continuous positivity/pools/physical ledger. k-ecoli457 CVODE has original native full-only protections: internal RHS domain and requested output/pool; accepted internal states not exposed. Teusink native mature path excluded because original full-only CVODE explicitly forbids quadrature ledger; no ledger waived. Select fastest qualified across all measured mature methods. Audit references themselves have no continuous/physical-ledger certification, so timing is optimistic for any stronger policy.',
        timing='Coordinate exclusive numerical workload; single-thread BLAS/OpenMP. Prepare models/GroupScreen/scoring candidate outside solve timer. Audits time native allocation/J/solve/output and sampling guards, then fixed candidate evaluation+observable comparison. Offline truth scoring timed separately and excluded from runtime. Mature solve/output/required structure checks timed; truth comparison separate. No new controller, reaction split coverage, full-window or formal seven-repeat claim.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths}))


def guarded_audit(case,screen,y,h,q,tol,method):
    start=time.perf_counter();system=case['system'];minimum=float(y.min())
    def rhs(t,z):
        nonlocal minimum
        if time.perf_counter()-start>20:raise TimeoutError('Audit solve budget')
        minimum=min(minimum,float(z.min()))
        if not np.isfinite(z).all() or np.any(z < -1e-12*case['scale']):raise RuntimeError('Invalid full ODE RHS domain; no clipping')
        return system.S@system.flux(0.,z)
    if method=='current':
        model=SimpleNamespace(y0=y,rhs=rhs,jacobian=lambda t,z:(system.S@system.flux_jacobian(0.,z)).tocsc(),
            sparsity=(abs(system.S)@system.dependencies).astype(bool).tocsc())
        values,stats=solve(model,h*q,tol,.01*tol*case['scale'])
        stats['native_accepted_internal_states_exposed']=False
    else:
        r=solve_ivp(rhs,(0.,h),y,t_eval=h*q,rtol=tol,atol=.01*tol*case['scale'],method=OrderedRadau,
            jac=lambda t,z:system.S@screen.gradient(z),first_step=h if method=='ordered-whole' else None)
        if not r.success:raise RuntimeError(r.message)
        values=r.y;stats=dict(nfev=r.nfev,njev=r.njev,nlu=r.nlu)
    guard=structural(case,values)
    if not guard['nonnegative'] or not np.isfinite(guard['pool']) or guard['pool']>1e-10:raise RuntimeError('Invalid audit output domain/pool')
    stats.update(min_rhs_argument=minimum,seconds=time.perf_counter()-start)
    return values,stats,guard


def runtime_score(case,candidate,q,reference,target):
    start=time.perf_counter();values=candidate(q[1:]);ref=reference[:,1:];guard=structural(case,np.column_stack([candidate.d,values]))
    if not guard['nonnegative'] or guard['pool']>1e-10:raise RuntimeError('Invalid fixed scoring candidate')
    vf,vr=rates(case,values),rates(case,ref)
    result=dict(concentration=normalized_error(values,ref,case['scale'][:,None],.1*target)[1],
        flux=normalized_error(vf,vr,case['options']['flux_scale'][:,None],.1*target)[1])
    if 'net_map' in case['options']:
        result['net_flux']=normalized_error(case['options']['net_map']@vf,case['options']['net_map']@vr,
            case['options']['net_scale'][:,None],.1e-3)[1]
    finite_metrics(result)
    return result,time.perf_counter()-start


def run():
    reg=registration(OUT/'registration.json');rows=[];controls=[]
    if (OUT/'runs.json').exists():raise FileExistsError('Preserve measured data')
    def persist():save(OUT/'runs.json',dict(audits=rows,controls=controls))
    for ci,cfg in enumerate(reg['cases']):
        case=prepare(cfg['model']);screen=GroupScreen(case['system'],case['scale']);h=cfg['h'];key=cfg['reference_key']
        with np.load(TRUTH/f'reference-{key}.npz') as a:y=a['initial'];truth=a['reference']
        with load_arrays(TRUTH/f"{cfg['scoring_candidate']}.npz") as a:candidate=unpack(a['lengths'],a['coefficients'])
        queries={mode:np.array(cfg[mode+'_queries']) for mode in reg['query_modes']}
        truths={'enhanced':truth,'required':truth[:,cfg['required_truth_indices']]}
        truth_rates={mode:rates(case,t) for mode,t in truths.items()}
        audit_configs=list(itertools.product(range(len(cfg['audit_configs'])),reg['query_modes']))
        mature_configs=list(itertools.product(reg['control_methods'][cfg['model']],reg['control_tolerances'],reg['query_modes']))
        for rep in reg['repeats']:
            shift=(rep+1)%len(audit_configs)
            for ai,mode in audit_configs[shift:]+audit_configs[:shift]:
                ac=cfg['audit_configs'][ai];name=f'audit-{key}-{ai}-{mode}'
                row=dict(key=name,reference_key=key,model=cfg['model'],target=ac['target'],method=ac['method'],tolerance=ac['tolerance'],query_mode=mode,repeat=rep);tick=time.perf_counter()
                try:
                    values,stats,guard=guarded_audit(case,screen,y,h,queries[mode],ac['tolerance'],ac['method'])
                    observed,scoreseconds=runtime_score(case,candidate,queries[mode],values,ac['target'])
                    scoretick=time.perf_counter();budget=finite_metrics(formal_metrics(case,values,truths[mode],ac['target'],rates(case,values),truth_rates[mode]));offline=time.perf_counter()-scoretick
                    row.update(status='success',stats=stats,guard=guard,budget=budget,qualified=max(budget.values())<=.05,
                        runtime_candidate_error=observed,runtime_candidate_pass=max(observed.values())<=1,runtime_score_seconds=scoreseconds,
                        seconds=stats['seconds'],audit_plus_scoring_seconds=stats['seconds']+scoreseconds,offline_qualification_seconds=offline)
                    if rep==-1:np.savez_compressed(OUT/f'{name}.npz',values=values)
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                    row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),qualified=False)
                rows.append(row);persist()
            shift=(rep+1)%len(mature_configs)
            for method,tol,mode in mature_configs[shift:]+mature_configs[:shift]:
                name=f'control-{key}-{method}-{tol}-{mode}';row=dict(key=name,reference_key=key,model=cfg['model'],method=method,tolerance=tol,query_mode=mode,repeat=rep);tick=time.perf_counter()
                try:
                    if method=='current':
                        assert 'ledger_weights' not in case['options']
                        values,stats,guard=guarded_audit(case,screen,y,h,queries[mode],tol,method);curve=None;valid=True
                    else:
                        r=(fullstep_curve if method=='ordered-whole' else full_curve)(case,y,h,tol,screen);curve=r['pieces'][0];values=curve(queries[mode]);stats=r['stats'];guard=structural(case,values)
                        valid=stats['continuous_nonnegative'] and guard['nonnegative'] and max(stats['pool_error'],guard['pool'])<=1e-10
                        valid &= all(stats.get(k,0.)<=1e-10 for k in ('physical_ledger_residual','physical_ledger_quadrature'))
                    elapsed=time.perf_counter()-tick;vf=rates(case,values);targets={}
                    for target in (1e-3,1e-5):
                        actual=finite_metrics(formal_metrics(case,values,truths[mode],target,vf,truth_rates[mode]));targets[str(target)]=dict(actual=actual,passes=bool(valid and max(actual.values())<=1))
                    row.update(status='success',seconds=elapsed,stats=stats,guard=guard,targets=targets)
                    if rep==-1:np.savez_compressed(OUT/f'{name}.npz',values=values,**(curve.pack() if curve else {}))
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:
                    row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc))
                controls.append(row);persist()
            print('output cost measured',key,rep,flush=True)


def verify():
    reg=registration(OUT/'registration.json');data=read(OUT/'runs.json');expected_audits=sum(len(c['audit_configs'])*2*4 for c in reg['cases'])
    expected_controls=sum(len(reg['control_methods'][c['model']])*len(reg['control_tolerances'])*2*4 for c in reg['cases'])
    assert len(data['audits'])==expected_audits and len(data['controls'])==expected_controls
    comparisons=[]
    for cfg in reg['cases']:
        key=cfg['reference_key'];case=prepare(cfg['model'])
        with np.load(TRUTH/f'reference-{key}.npz') as a:y=a['initial'];truth=a['reference']
        with load_arrays(TRUTH/f"{cfg['scoring_candidate']}.npz") as a:candidate=unpack(a['lengths'],a['coefficients'])
        queries={mode:np.array(cfg[mode+'_queries']) for mode in reg['query_modes']};truths={'enhanced':truth,'required':truth[:,cfg['required_truth_indices']]}
        for kind in ('audits','controls'):
            subset=[r for r in data[kind] if r['reference_key']==key]
            for name in dict.fromkeys(r['key'] for r in subset):
                rows=[r for r in subset if r['key']==name];first=next(r for r in rows if r['repeat']==-1)
                assert len(rows)==4 and {r['repeat'] for r in rows}==set(reg['repeats'])
                for row in rows:
                    if not math.isfinite(row['seconds']) or row['seconds']<=0:raise ValueError('Invalid elapsed time')
                    assert row['status']==first['status']
                    if row['status']=='failed':continue
                    with np.load(OUT/f'{name}.npz') as a:
                        values=a['values'];curve=InnerCurve.unpack(a['lengths'],a['coefficients']) if 'lengths' in a else None
                    if not np.isfinite(values).all():raise ValueError('Invalid stored output')
                    close(structural(case,values),row['guard']);mode=row['query_mode'];vf=rates(case,values);vr=rates(case,truths[mode])
                    if kind=='audits':
                        budget=finite_metrics(formal_metrics(case,values,truths[mode],row['target'],vf,vr));close(budget,row['budget']);assert row['qualified']==(max(budget.values())<=.05)
                        assert math.isfinite(row['runtime_score_seconds']) and row['runtime_score_seconds']>=0
                        close(row['audit_plus_scoring_seconds'],row['seconds']+row['runtime_score_seconds']);finite_metrics(row['runtime_candidate_error'])
                        observed,_=runtime_score(case,candidate,queries[mode],values,row['target']);close(observed,row['runtime_candidate_error']);assert row['runtime_candidate_pass']==(max(observed.values())<=1)
                    else:
                        valid=row['guard']['nonnegative'] and row['guard']['pool']<=1e-10
                        if curve is not None:
                            np.testing.assert_allclose(curve(queries[mode]),values,rtol=1e-13,atol=1e-14)
                            positive=bool(np.all(curve.minima()>=-1e-12*case['scale']))
                            assert positive==row['stats']['continuous_nonnegative']
                            assert curve.join_error(case['scale'])<=1e-10
                            for ledgerkey,ledger in physical_ledger(case,curve,y,values[:,-1]).items():close(ledger,row['stats'][ledgerkey])
                            valid &= positive and row['stats']['pool_error']<=1e-10 and all(row['stats'].get(k,0.)<=1e-10 for k in ('physical_ledger_residual','physical_ledger_quadrature'))
                        for target in (1e-3,1e-5):
                            actual=finite_metrics(formal_metrics(case,values,truths[mode],target,vf,vr));close(actual,row['targets'][str(target)]['actual']);assert row['targets'][str(target)]['passes']==(valid and max(actual.values())<=1)
        for target,mode in itertools.product((1e-3,1e-5),reg['query_modes']):
            audit_configs=[];baseline_configs=[]
            for ai,ac in enumerate(cfg['audit_configs']):
                if ac['target']!=target:continue
                name=f'audit-{key}-{ai}-{mode}';rows=[r for r in data['audits'] if r['key']==name and r['repeat']>=0]
                good=all(r['status']=='success' and r['qualified'] for r in rows)
                audit_configs.append(dict(method=ac['method'],tolerance=ac['tolerance'],qualified=good,median_seconds=float(np.median([r['seconds'] for r in rows])),
                    median_with_scoring_seconds=float(np.median([r.get('audit_plus_scoring_seconds',r['seconds']) for r in rows]))))
            for method,tol in itertools.product(reg['control_methods'][cfg['model']],reg['control_tolerances']):
                name=f'control-{key}-{method}-{tol}-{mode}';rows=[r for r in data['controls'] if r['key']==name and r['repeat']>=0]
                baseline_configs.append(dict(method=method,tolerance=tol,qualified=all(r['status']=='success' and r['targets'][str(target)]['passes'] for r in rows),median_seconds=float(np.median([r['seconds'] for r in rows]))))
            best=min((r for r in baseline_configs if r['qualified']),key=lambda r:r['median_seconds'],default=None)
            auditbest=min((r for r in audit_configs if r['qualified']),key=lambda r:r['median_with_scoring_seconds'],default=None)
            comparisons.append(dict(reference_key=key,model=cfg['model'],target=target,query_mode=mode,query_points=len(queries[mode])-1,audits=audit_configs,baselines=baseline_configs,best_baseline=best,best_audit=auditbest,
                residual_1p5_budget=None if best is None or auditbest is None else best['median_seconds']/1.5-auditbest['median_with_scoring_seconds']))
        print('verified output cost',key,flush=True)
    crc=0
    for path in OUT.glob('*.npz'):
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
        crc+=1
    return dict(audits=expected_audits,mature_controls=expected_controls,npz_crc_checked=crc,comparisons=comparisons,M4_admission=False,
        limitation='Known local states, original required-grid audit responsibility retained. Runtime candidate scoring measured using fixed saved MERB curve, generation cost excluded and no coverage. Native accepted internal states unexposed; no continuous reference certification. No formal speed claim.')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','write','verify']);args=p.parse_args()
    if args.action=='register':register();return
    if args.action=='run':run();return
    result=verify()
    if args.action=='write':
        if (OUT/'manifest.json').exists():raise FileExistsError('Preserve manifest')
        save(OUT/'summary.json',result);save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP, coordinated exclusive numerical workload'))
        save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'})
    elif (OUT/'summary.json').exists():
        close(result,read(OUT/'summary.json'))
        for path,sha in read(OUT/'manifest.json').items():assert digest(ROOT/path)==sha,path
    print(json.dumps(dict(audits=result['audits'],mature_controls=result['mature_controls'],npz_crc_checked=result['npz_crc_checked'],comparisons=[{k:r[k] for k in ('reference_key','target','query_mode','query_points','best_baseline','best_audit','residual_1p5_budget')} for r in result['comparisons']]),indent=2))


if __name__=='__main__':main()
