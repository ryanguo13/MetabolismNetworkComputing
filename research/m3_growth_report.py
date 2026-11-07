"""Verify saved growth curves against freshly integrated independent references."""
import argparse
import json
import platform
import subprocess
import sys
import numpy as np
import scipy
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.extrapolated_flow import ExtrapolatedFlow,SlowPolynomial,unpack
from research.inner_dense import InnerCurve,physical_ledger
from research.m3_extrapolated_paths import qualified
from research.m3_growth_study import audit,final_assessment,query_grid
from research.growth_history import schedule
from research.history_stability import amplification
from research.m3_ledger_flow_report import close,modified
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-growth-history';OLD=ROOT/'results/m3-inner-dense'
def read(path):return json.loads(path.read_text())
def check_registration(path):
    reg=read(path)
    for p,sha in (reg['source_sha256']|reg['input_sha256']).items():assert digest(ROOT/p)==sha,p
    return reg


def check_curve(case,curve,y,end,query,reference,row):
    np.testing.assert_allclose(curve(np.array([0.,1.])),np.column_stack([y,end]),rtol=2e-12,atol=1e-12)
    assert curve.join_error(case['scale'])<=1e-10
    positive=bool(np.all(curve.minima()>=-1e-12*case['scale']));assert positive==row['stats']['continuous_nonnegative']
    if positive:
        for k,v in physical_ledger(case,curve,y,end).items():close(v,row['stats'][k])
    guard,targets=qualified(case,curve,curve(query),reference,row['stats'])
    close(guard,row['guard']);close(targets,row['targets'])


def verify_growth(cases):
    reg=check_registration(OUT/'registration.json');data=read(OUT/'runs.json')
    assert len(data['trajectories'])==36 and len(data['runs'])==294 and len(data['baselines'])==18
    expected={(i,q,g,t) for i in reg['paths'] for q in reg['orders'] for g in reg['growth'] for t in reg['tolerances']}
    assert {(s['path'],s['order'],s['ratio'],s['tolerance']) for s in data['trajectories']}==expected
    old=read(OLD/'history.json')['runs'];decisions=[];failures=[]
    for index in reg['paths']:
        original=next(r for r in old if r['shard']==f'history-{index}' and r['step']==0)
        case=cases[original['model']];h0=original['h'];duration=16*h0
        with np.load(OUT/f'reference-{index}.npz') as base:
            origin=base['initial'];grid=base['global_times']
            expected_grid=np.unique(np.concatenate([query_grid(case,original['start'],schedule(h0,duration,q,r)) for q in reg['orders'] for r in reg['growth']]))
            np.testing.assert_array_equal(grid,expected_grid)
            reference,budget,_=audit(case,origin,grid/duration,duration);assert max(budget.values())<=.05
            close(budget,next(r['budget'] for r in data['references'] if r['path']==index))
            for row in [r for r in data['baselines'] if r['path']==index]:
                if row['status']!='success':failures.append(row);continue
                k=row['key'];curve=InnerCurve.unpack(base[k+'-lengths'],base[k+'-coefficients'])
                check_curve(case,curve,origin,base[k+'-endpoint'],grid/duration,reference,row)
            for summary in [r for r in data['trajectories'] if r['path']==index]:
                name=summary['name'];q=summary['order'];local=[r for r in data['runs'] if r['name']==name]
                planned=schedule(h0,duration,q,summary['ratio']);assert len(local)==len(planned)-1 and summary['complete']
                assert [r['step'] for r in local]==list(range(len(local)))
                assert all(r['status']=='success' for r in local)
                kernel=ExtrapolatedFlow(case,base['groups']);pieces=[]
                with np.load(OUT/f'{name}.npz') as a:
                    np.testing.assert_array_equal(a['schedule'],planned)
                    ht=list(a['bootstrap_times']);hs=a['bootstrap_states'];slow=list(a['bootstrap_slow'].T)
                    np.testing.assert_array_equal(ht,np.arange(q)*h0)
                    np.testing.assert_array_equal(np.column_stack(slow),np.column_stack([kernel.slow(z) for z in hs[:,:-1].T]))
                    previous=origin
                    for row in local:
                        step=row['step'];k=str(step);h=row['h'];left=planned[step];right=planned[step+1]
                        assert h==right-left and row['left']==left and row['right']==right
                        y=a[k+'-initial'];end=a[k+'-endpoint'];np.testing.assert_array_equal(y,previous)
                        curve=(InnerCurve.unpack if step==0 else unpack)(a[k+'-lengths'],a[k+'-coefficients'])
                        query=a[k+'-query'];pos=np.flatnonzero((grid>=left-1e-13*h0)&(grid<=right+1e-13*h0))
                        np.testing.assert_array_equal(query,np.clip((grid[pos]-left)/h,0,1))
                        ref,budget,_=audit(case,y,query,h);assert max(budget.values())<=.05;close(budget,row['reference_budget'])
                        check_curve(case,curve,y,end,query,ref,row)
                        if step==0:
                            startup=curve(np.array(ht)/ht[-1]);startup[:,-1]=end
                            np.testing.assert_array_equal(startup,hs)
                        else:
                            forcing=SlowPolynomial(ht[-q:],np.column_stack([*slow[-(q-1):],kernel.slow(y)]),h)
                            np.testing.assert_array_equal(forcing.coefficients,a[k+'-forcing'])
                            modified(case,kernel,curve,y,end,forcing,row['stats'])
                            assert row['stats']['subflows']==1
                            slow.append(forcing.coefficients[:,0].copy());ht.append(right)
                        pieces.extend(curve.pieces);previous=end
                    stats,guard,targets=final_assessment(case,pieces,origin,previous,local,grid,reference)
                    close(stats,summary['stats']);close(guard,summary['guard']);close(targets,summary['targets'])
                    close(sum(r['seconds'] for r in local),summary['seconds'])
                    for t in ('0.0001','1e-06'):assert summary['local_passes'][t]==sum(r['targets'][t]['passes'] for r in local)
                kernel.close()
                print('verified',name,flush=True)
            for target in ('0.0001','1e-06'):
                candidates=[r for r in data['trajectories'] if r['path']==index and r['complete'] and r['targets'][target]['passes']]
                controls=[r for r in data['baselines'] if r['path']==index and r['status']=='success' and r['targets'][target]['passes']]
                candidate=min(candidates,key=lambda r:r['seconds']) if candidates else None
                baseline=min(controls,key=lambda r:r['seconds']) if controls else None
                selected=[r for r in data['runs'] if candidate and r['name']==candidate['name']]
                decisions.append(dict(path=index,target=target,qualified_candidates=len(candidates),
                    candidate=None if candidate is None else dict(name=candidate['name'],seconds=candidate['seconds'],steps=candidate['steps'],actual=candidate['targets'][target]['actual'],
                        physical_ledger=candidate['stats'].get('physical_ledger_residual'),startup_seconds=selected[0]['seconds'],audit_pair_seconds=sum(r['audit_seconds'] for r in selected),
                        work={k:sum(r['stats'].get(k,0) for r in selected) for k in ('nfev','njev','nlu','accepted_steps')}),
                    baseline=None if baseline is None else dict(key=baseline['key'],seconds=baseline['seconds'],actual=baseline['targets'][target]['actual'],
                        work={k:baseline['stats'][k] for k in ('nfev','njev','nlu','accepted_steps')}),
                    optimistic_single_run_ratio=baseline['seconds']/candidate['seconds'] if candidate and baseline else None))
    return dict(trajectories=36,macrosteps=294,uninterrupted_baselines=18,decisions=decisions,integration_failures=failures,
        strict_passes=sum(r['complete'] and r['targets']['1e-06']['passes'] for r in data['trajectories']),
        loose_passes=sum(r['complete'] and r['targets']['0.0001']['passes'] for r in data['trajectories']))


def verify_stability(case):
    reg=check_registration(OUT/'stability-registration.json');rows=read(OUT/'stability.json')['runs'];assert len(rows)==100
    prefix=np.load(ROOT/'results/m3-priority/p0.npz')['teusink-prefix'];screen=GroupScreen(case['system'],case['scale']);bounds=[]
    with np.load(OUT/'stability.npz') as a:
        for state,column in reg['states'].items():
            y=prefix[:,column];kappa,gradient=screen.evaluate(y);fulljac=(case['system'].S@gradient).toarray()
            np.testing.assert_array_equal(y,a[state+'-state']);np.testing.assert_array_equal(fulljac,a[state+'-jacobian'])
            for threshold in reg['thresholds']:
                groups=np.flatnonzero(1e-3*kappa>=threshold);kernel=ExtrapolatedFlow(case,groups)
                fast=(kernel.sf@gradient[kernel.cols]).toarray();slow=fulljac-fast;key=f'{state}-{threshold}'
                np.testing.assert_array_equal(fast,a[key+'-fast']);np.testing.assert_array_equal(slow,a[key+'-slow'])
                for h in reg['steps']:
                    row=next(r for r in rows if r['state']==state and r['threshold']==threshold and r['h']==h)
                    assert row['groups']==groups.tolist() and row['active']==len(kernel.active) and row['fast_reactions']==len(kernel.cols)
                    rho=float(max(abs(np.linalg.eigvals(amplification(fast,slow,h,4)))))
                    close(rho,row['spectral_radius']);assert row['passes_frozen_linear_screen']==(rho<=1+1e-9)
                kernel.close()
            for h in reg['steps']:
                viable=[r for r in rows if r['state']==state and r['h']==h and r['passes_frozen_linear_screen']]
                bounds.append(dict(state=state,h=h,minimum_active_among_tested_linear_passes=min((r['active'] for r in viable),default=None),
                    largest_passing_threshold=max((r['threshold'] for r in viable),default=None)))
    return dict(configurations=100,minimum_dimension=bounds,scope='Frozen local constant-step linear diagnostic with exact fast flow; not nonlinear or variable-step certification.')


def verify():
    cases={n:prepare(n) for n in ('teusink','kecoli')}
    result=dict(date='2026-11-08',growth=verify_growth(cases),stability=verify_stability(cases['teusink']),M4_admission=False)
    for file in ('results/m3-extrapolated-flow/manifest.json','results/m3-inner-dense/manifest.json'):
        for p,sha in read(ROOT/file).items():assert digest(ROOT/p)==sha,p
    result.update(historical_manifests_unchanged=True,scope='Single-run development, short windows, optimistic cost without audits or grouping. Original M4 gates remain unmet.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args();result=verify()
    if args.write:
        save(OUT/'summary.json',result)
        if not (OUT/'environment.json').exists():save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
        files=[p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']
        files += [ROOT/'research'/p for p in ('growth_history.py','test_growth_history.py','m3_growth_study.py','history_stability.py','test_history_stability.py','m3_history_stability.py','m3_growth_report.py')]
        files += [ROOT/'docs/m3-growth-history.md']
        save(OUT/'manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
