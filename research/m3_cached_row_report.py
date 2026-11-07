"""Independent reference/ledger reconstruction and transaction replay."""
import argparse
import collections
import json
import platform
import subprocess
import sys
import zipfile
import numpy as np
import scipy
from scipy.sparse import eye
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.cached_row import CachedROW
from research.weighted_row import WeightedROW
from research.row_continuous import ROWContinuous,coupling,PolynomialForcing
from research.inner_dense import InnerCurve
from research.extrapolated_flow import unpack
from research.m3_ledger_flow_report import registration,close,check_curve,modified
from research.m3_extrapolated_paths import qualified
from research.m3_growth_study import audit
from research.m3_cached_row_study import aggregate
from research.cache_refinement_diagnostic import evaluate as refinement_trace
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-cached-row';EXTRA=ROOT/'results/m3-weighted-row'
def read(p):return json.loads(p.read_text())


def verify_study(directory,cases):
    reg=registration(directory/'registration.json');data=read(directory/'runs.json');weighted=directory==EXTRA
    assert len(data['paths'])==4*len(reg['variants']);assert len(data['controls'])==(48 if weighted else 72)
    decisions=[];failures=[];drift=[];references={}
    for index,spec in enumerate(reg['paths']):
        case=cases[spec['model']];h=spec['h'];duration=reg['steps']*h
        with np.load(directory/f'path-{index}.npz') as src:origin=src['origin'];grid=src['grid'];groups=src['groups']
        prefix=np.load(ROOT/'results/m3-priority/p0.npz')[spec['model']+'-prefix'][:,3];np.testing.assert_array_equal(origin,prefix)
        start=spec['start'];np.testing.assert_array_equal(grid,np.unique(np.r_[np.arange(reg['steps']*64+1)*h/64,case['times'][(case['times']>start)&(case['times']<start+duration)]-start]))
        screen=GroupScreen(case['system'],case['scale']);kappa,_=screen.evaluate(origin);np.testing.assert_array_equal(groups,np.flatnonzero(1e-3*kappa>=spec['threshold']))
        reference,budget,_=audit(case,origin,grid/duration,duration);rr=next(r for r in data['references'] if r['path']==index);close(budget,rr['budget']);assert max(budget.values())<=.05
        for row in [r for r in data['controls'] if r['path']==index]:
            assert row['status']=='success'
            with np.load(directory/f"{row['key']}.npz") as a:
                curve=(InnerCurve.unpack if row['method']=='original' else unpack)(a['lengths'],a['coefficients'])
                check_curve(case,curve,origin,a['endpoint'],reference,grid/duration,row['stats'],row['guard'],row['targets'])
        for variant in reg['variants']:
            key=f'{index}-{variant}';summary=next(p for p in data['paths'] if p['key']==key);rows=[r for r in data['steps'] if r['key']==key]
            assert [r['step'] for r in rows]==list(range(len(rows)))
            base=variant.startswith('row-');chosen=[] if base else groups;kernel=(ROWContinuous(case,chosen) if variant=='original' else (WeightedROW if weighted else CachedROW)(case,chosen,reuse=variant.endswith('cached')))
            pieces=[];accepted=[];y=origin.copy();max_replay=0.;nlu=dict(predictor=0,fast=0);corrections=0
            with np.load(directory/f'{key}.npz') as a:
                for row in rows:
                    step=row['step'];left=step*h;right=(step+1)*h;indices=np.flatnonzero((grid>=left-1e-14)&(grid<=right+1e-14));q=np.clip((grid[indices]-left)/h,0,1);np.testing.assert_array_equal(q,row['query'])
                    # All integrations in this batch completed. Failed qualification
                    # is independently reconstructed and must stop the path.
                    assert row['status']=='success';np.testing.assert_array_equal(y,a[f'{step}-initial']);end=a[f'{step}-endpoint'];curve=unpack(a[f'{step}-lengths'],a[f'{step}-coefficients'])
                    refkey=(spec['model'],h,y.tobytes(),q.tobytes())
                    if refkey not in references:references[refkey]=audit(case,y,q,h)[:2]
                    ref,local_budget=references[refkey];close(local_budget,row['reference_budget']);assert max(local_budget.values())<=.05
                    check_curve(case,curve,y,end,ref,q,row['stats'],row['guard'],row['targets'])
                    aa,gg,b,mu=coupling();ki=a[f'{step}-increments'];ks=a[f'{step}-slow_increments'];stages=a[f'{step}-stages'];rates=a[f'{step}-rates'];full=a[f'{step}-full']
                    # Use original scalar gradients, never a cached numerical Jacobian.
                    grad=screen.gradient(y);j=case['system'].S@grad;slowj=j-kernel.sf@grad[kernel.cols];residuals=[]
                    for i in range(4):
                        state=y+sum((aa[i,k]*ki[:,k] for k in range(i)),start=np.zeros_like(y));np.testing.assert_array_equal(state,stages[:,i])
                        np.testing.assert_array_equal(case['system'].flux(0.,state),rates[:,i]);np.testing.assert_array_equal(case['system'].S@rates[:,i],full[:,i]);assert np.all(state>=-1e-12*case['scale'])
                        prev=sum((gg[i,k]*ki[:,k] for k in range(i)),start=np.zeros_like(y));rhs=h*full[:,i]+h*(j@prev)
                        residual=float(np.max(abs((eye(len(y),format='csc')-h*gg[i,i]*j)@ki[:,i]-rhs)/case['scale']));residuals.append(residual);assert residual<=1e-10
                        expected=h*(full[:,i]-kernel.sf@rates[kernel.cols,i])+h*(slowj@(prev+gg[i,i]*ki[:,i]))
                        assert np.max(abs(expected-ks[:,i])/case['scale'])<1e-11
                    if not base:
                        forcing=PolynomialForcing((ks@mu)*np.arange(1,4)/h,h);np.testing.assert_array_equal(forcing.coefficients,a[f'{step}-forcing'])
                        modified(case,kernel,curve,y,end,forcing,row['stats'])
                    else:
                        np.testing.assert_allclose(end,y+ki@b,rtol=3e-12,atol=3e-13)
                        np.testing.assert_allclose(a[f'{step}-coefficients'][0],np.column_stack([y,ki@mu]),rtol=3e-12,atol=3e-13)
                    if variant!='original':
                        assert kernel.cache_state()==row['cache_before'];trial=kernel.trial(y,h,reg['tolerance'],base_only=base)
                        delta=float(np.max(abs(trial['fine']-end)/case['scale']));max_replay=max(max_replay,delta);assert delta<1e-12
                        np.testing.assert_allclose(trial['pieces'][0](q),curve(q),rtol=2e-12,atol=2e-12)
                        for field in ('predictor_linear','fast_linear'):
                            close(trial['stats'][field],row['stats'][field]);corrections+=row['stats'][field]['corrections']
                        nlu['predictor']+=row['stats']['predictor_linear']['factorizations'];nlu['fast']+=row['stats']['fast_linear']['factorizations']
                        if row['accepted']:kernel.commit(trial)
                        else:kernel.discard(trial);assert row['cache_before']==row['cache_after']
                        assert kernel.cache_state()==row['cache_after']
                    assert row['accepted']==row['targets']['1e-06']['passes']
                    if row['accepted']:accepted.append(row);pieces.extend(curve.pieces);y=end.copy()
                    else:
                        assert row is rows[-1];failures.append(dict(key=key,step=step,stats=row['stats'],targets=row['targets']))
                np.testing.assert_array_equal(y,a['accepted_endpoint'])
            assert len(accepted)==summary['accepted_steps'];assert summary['complete']==(len(accepted)==reg['steps'])
            close(sum(r['seconds'] for r in rows),summary['optimistic_seconds']);close(sum(r.get('audit_seconds',0.) for r in rows),summary['audit_seconds'])
            close(summary['audited_seconds'],summary['optimistic_seconds']+summary['audit_seconds']+summary['setup_seconds']+summary['group_seconds'])
            if pieces:
                endtime=len(accepted)*h;indices=np.flatnonzero(grid<=endtime+1e-14);q=np.clip(grid[indices]/endtime,0,1)
                stats,guard,targets=aggregate(case,accepted,pieces,origin,y,q,reference[:,indices],summary['complete']);close(stats,summary['stats']);close(guard,summary['guard']);close(targets,summary['targets'])
            kernel.close();drift.append(dict(key=key,max_replay_scaled=max_replay,actual_factors=nlu,refinement_corrections=corrections))
            for target in ('0.0001','1e-06'):
                qualified_path=summary.get('targets',{}).get(target,{}).get('passes',False)
                good=[r for r in data['controls'] if r['path']==index and r['status']=='success' and r['targets'][target]['passes']];best=min(good,key=lambda r:r['seconds']) if good else None
                decisions.append(dict(key=key,model=spec['model'],h=h,target=target,complete=summary['complete'],passes=qualified_path,accepted_steps=len(accepted),
                    optimistic_seconds=summary['optimistic_seconds'],audited_seconds=summary['audited_seconds'],control_key=best['key'] if best else None,control_seconds=best['seconds'] if best else None,
                    optimistic_ratio=best['seconds']/summary['optimistic_seconds'] if best and qualified_path else None,
                    audited_ratio=best['seconds']/summary['audited_seconds'] if best and qualified_path else None))
            print('verified',directory.name,key,flush=True)
    return dict(paths=len(data['paths']),attempts=len(data['steps']),controls=len(data['controls']),
        locally_rejected=sum(not r['accepted'] for r in data['steps']),complete_paths=sum(p['complete'] for p in data['paths']),
        globally_qualified={t:sum(p.get('targets',{}).get(t,{}).get('passes',False) for p in data['paths']) for t in ('0.0001','1e-06')},
        decisions=decisions,work=drift,local_rejections=failures)


def verify_shadow():
    from research.m3_row_cumulative_shadow import evaluate
    reg=registration(OUT/'shadow-registration.json');recorded=read(OUT/'shadow.json')['runs'];replayed=evaluate(reg)
    assert len(recorded)==len(replayed)==8
    for saved,current in zip(recorded,replayed):
        for key in ('path','model','steps','duration','full_path','grids','reference_budget','comparison','scope'):close(saved[key],current[key])
        for key in ('status','targets','counts'):close(saved['estimate'][key],current['estimate'][key])
        from research.defect_compact import estimate_compact
        index=saved['path'];pieces=[];case=prepare(saved['model'])
        with np.load(OUT/f'{index}-original.npz') as source:
            for step in range(saved['steps']):pieces.extend(unpack(source[f'{step}-lengths'],source[f'{step}-coefficients']).pieces)
        curve=InnerCurve(pieces)
        with np.load(OUT/f'path-{index}.npz') as src:grid=src['grid'];grid=grid[grid<=saved['duration']+1e-14]
        query=np.clip(grid/saved['duration'],0,1)
        _,arrays=estimate_compact(case,pieces,curve(query),query,tuple(saved['grids']))
        with np.load(OUT/f"shadow-{saved['path']}-{saved['grids'][0]}-{saved['grids'][1]}.npz") as a:
            for name,value in arrays.items():np.testing.assert_allclose(value,a[name],rtol=2e-8,atol=1e-12)
            for grid in saved['grids']:
                assert np.isfinite(a[f'error-{grid}']).all()
                np.testing.assert_array_equal(a[f'error-{grid}'][:,0],np.zeros(a[f'error-{grid}'].shape[0]))
                assert np.all(np.diff(a[f'nodes-{grid}'])>0)
    return dict(records=8,false_accepts={t:sum(r['comparison'][t]['false_accept'] for r in recorded) for t in ('0.0001','1e-06')},
        false_rejections={t:sum(r['comparison'][t]['actual_error_pass'] and not r['comparison'][t]['estimator_would_accept'] for r in recorded) for t in ('0.0001','1e-06')},
        detected_cumulative_strict_failure=all(not r['comparison']['1e-06']['estimator_would_accept'] for r in recorded if r['path']==2),
        scope='Post-hoc diagnosis only; all full audits retained. Four existing paths, two grids, no safety guarantee.')


def verify():
    registration(OUT/'refinement-registration.json');close(refinement_trace(),read(OUT/'refinement-diagnostic.json')['runs'])
    cases={m:prepare(m) for m in ('teusink','kecoli')};result={p.name:verify_study(p,cases) for p in (OUT,EXTRA)}
    for p,sha in read(ROOT/'results/m3-row-spc/manifest.json').items():assert digest(ROOT/p)==sha,p
    crc=0
    for directory in (OUT,EXTRA):
        for p in directory.glob('*.npz'):
            with zipfile.ZipFile(p) as z:assert z.testzip() is None,p
            crc+=1
    return dict(studies=result,cumulative_shadow=verify_shadow(),npz_crc_checked=crc,historical_manifest_unchanged=True,M4_admission=False,
        scope='Development16-step own-history trajectories; original formal windows/targets,50% split coverage and7-repeat1.5x gate remain unchanged. No formal speedup claim.')


def write_report(result):
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve frozen report')
    save(OUT/'summary.json',result);save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh, single-thread BLAS/OpenMP'))
    files=[p for d in (OUT,EXTRA) for p in d.iterdir() if p.is_file() and p.name!='manifest.json']
    files += [ROOT/'research'/p for p in ('cached_row.py','cached_full.py','refined_lu.py','shared_ledger.py','test_cached_row.py','m3_cached_row_study.py','cache_refinement_diagnostic.py','weighted_lu.py','weighted_row.py','test_weighted_row.py','m3_weighted_row_study.py','m3_cached_row_report.py','m3_row_cumulative_shadow.py')]
    files += [ROOT/'docs/m3-cached-row.md']
    save(OUT/'manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args();result=verify()
    if args.write:write_report(result)
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
