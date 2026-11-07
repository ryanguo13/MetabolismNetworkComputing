"""Reproduce streaming states, independent truth and prospective decisions."""
import argparse
import json
import platform
import subprocess
import sys
import zipfile
import numpy as np
import scipy
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from metabolism.adaptive import GroupScreen
from research.streaming_defect import StreamingDefect
from research.linear_error_output import LinearStreaming
from research.linear_context_row import LinearContextROW
from research.extrapolated_flow import unpack
from research.inner_dense import InnerCurve,physical_ledger
from research.m3_growth_study import audit
from research.m3_priority import full,metrics
from research.m3_extrapolated_paths import qualified
from research.m3_ledger_flow_report import registration,close,check_curve
from research.m3_streaming_controller import state_signature,proposal_budget,LEDGERS
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-streaming-defect';LINEAR=ROOT/'results/m3-streaming-linear';CONTROL=ROOT/'results/m3-streaming-controller';BUDGET=ROOT/'results/m3-streaming-budget';OLD=ROOT/'results/m3-cached-row'
def read(p):return json.loads(p.read_text())


def verify_offline(directory,cases,refcache):
    reg=registration(directory/'registration.json');data=read(directory/'runs.json');old=read(OLD/'runs.json');specs=read(OLD/'registration.json')['paths'];cls=LinearStreaming if directory==LINEAR else StreamingDefect
    assert len(data['steps'])==59 and len(data['paths'])==4 and len(data['controls'])==24
    counts={t:dict(false_accept=0,false_reject=0) for t in ('0.0001','1e-06')};decisions=[];work=[]
    for index in reg['paths']:
        spec=specs[index];case=cases[spec['model']];h=spec['h'];summary=next(r for r in data['paths'] if r['path']==index);n=summary['steps'];duration=n*h
        assert n==next(r['accepted_steps'] for r in old['paths'] if r['key']==f'{index}-original')
        with np.load(directory/f'reference-{index}.npz') as a:origin=a['initial'];groups=a['groups'];grid=a['grid'];stored_ref=a['reference']
        if index not in refcache:refcache[index]=audit(case,origin,np.clip(grid/duration,0,1),duration)[:2]
        reference,budget=refcache[index];np.testing.assert_allclose(reference,stored_ref,rtol=2e-12,atol=2e-12);close(budget,next(r['budget'] for r in data['references'] if r['path']==index));assert max(budget.values())<=.05
        for row in [r for r in data['controls'] if r['path']==index]:
            assert row['status']=='success'
            with np.load(directory/f"{row['key']}.npz") as a:
                c=InnerCurve.unpack(a['lengths'],a['coefficients']);check_curve(case,c,origin,a['endpoint'],reference,np.clip(grid/duration,0,1),row['stats'],row['guard'],row['targets'])
        estimators=[cls(case,origin,g) for g in reg['grids']];kernel=LinearContextROW(case,groups);seen=[];truth=[];y=origin.copy();ledger={k:0. for k in LEDGERS};weights=case['options'].get('ledger_weights');den0=1. if weights is None else max(1.,abs(float(weights@origin)))
        rows=[r for r in data['steps'] if r['path']==index];assert [r['step'] for r in rows]==list(range(n));totals=[dict(factorizations=0,predictor_factor_hits=0,linear_solves=0,defect_rate_states=0) for _ in reg['grids']]
        with np.load(directory/f'errors-{index}.npz') as err,np.load(OLD/f'{index}-original.npz') as src:
            for row in rows:
                step=row['step'];mask=np.flatnonzero((grid>=step*h-1e-14)&(grid<=(step+1)*h+1e-14));q=np.clip((grid[mask]-step*h)/h,0,1);np.testing.assert_array_equal(q,row['query'])
                trial=kernel.trial(y,h,reg['inner_tolerance']);np.testing.assert_array_equal(trial['fine'],src[f'{step}-endpoint']);np.testing.assert_array_equal(y,src[f'{step}-initial'])
                curve=trial['pieces'][0];values=curve(q);refkey=(index,step)
                if refkey not in refcache:refcache[refkey]=audit(case,y,q,h)[:2]
                local_ref,lb=refcache[refkey];close(lb,row['local_budget']);assert max(lb.values())<=.05
                check_curve(case,curve,y,trial['fine'],local_ref,q,row['candidate_stats'],row['local_guard'],row['local_targets'])
                seen.append(values);truth.append(reference[:,mask]);actual={str(t):metrics(case,np.column_stack(seen),np.column_stack(truth),t) for t in (1e-4,1e-6)};close(actual,row['actual_accumulated'])
                den=1. if weights is None else max(1.,abs(float(weights@y)))
                for k in ledger:ledger[k]+=trial['stats'].get(k,0.)*den/den0
                close(ledger,row['absolute_ledger_budgets'])
                for gi,est in enumerate(estimators):
                    result=est.trial(trial,q);record=row['estimates'][gi];assert result['status']==record['status']=='success';close(result['targets'],record['targets']);close(result['stats'],record['stats'])
                    for branch,c in enumerate(result['error_curves']):
                        prefix=f'{step}-{gi}-{branch}-';np.testing.assert_allclose(result['errors'][branch],err[prefix+'endpoint'],rtol=2e-10,atol=1e-13)
                        np.testing.assert_allclose(np.stack([p.coefficients for p in c.pieces]),err[prefix+'coefficients'],rtol=2e-10,atol=1e-13)
                        np.testing.assert_array_equal([p.h for p in c.pieces],err[prefix+'lengths'])
                    for target,error in actual.items():
                        predicted=result['targets'][target]['would_accept'];valid=max(error.values())<=1
                        comparison=dict(actual_error_pass=valid,would_accept=predicted,false_accept=bool(predicted and not valid),false_reject=bool(not predicted and valid),combined_would_accept=bool(predicted and max(ledger.values())<=1e-10))
                        close(comparison,record['comparison'][target]);counts[target]['false_accept']+=comparison['false_accept'];counts[target]['false_reject']+=comparison['false_reject']
                    for stats in result['stats']:
                        assert stats['max_linear_residual']<=1e-10
                        for k in totals[gi]:totals[gi][k]+=stats[k]
                    est.commit(result)
                y=trial['fine'].copy()
        kernel.close();close(summary['actual_accumulated'],actual);close(summary['absolute_ledger_budgets'],ledger);close(summary['final_estimates'],rows[-1]['estimates'])
        close(summary['candidate_seconds'],sum(r['candidate_seconds'] for r in rows));close(summary['local_audit_seconds'],sum(r['local_audit_seconds'] for r in rows))
        for gi,g in enumerate(reg['grids']):
            seconds=sum(r['estimates'][gi]['seconds'] for r in rows);close(seconds,summary['estimator_seconds'][gi]);work.append(dict(path=index,grids=g,seconds=seconds,**totals[gi]))
            for target in ('0.0001','1e-06'):
                good=[r for r in data['controls'] if r['path']==index and r['targets'][target]['passes']];best=min(good,key=lambda r:r['seconds']);final=summary['final_estimates'][gi]['comparison'][target]
                decisions.append(dict(path=index,model=spec['model'],grids=g,target=target,actual_error=max(actual[target].values()),**final,
                    estimator_seconds=seconds,candidate_seconds=summary['candidate_seconds'],control_seconds=best['seconds'],control_key=best['key'],
                    optimistic_candidate_plus_estimator_ratio=best['seconds']/(summary['candidate_seconds']+seconds),
                    scope='Diagnostic cost only, not qualified speedup; final score/ledger and original completeness gates may reject.'))
        print('verified',directory.name,index,flush=True)
    return dict(steps=59,score_decisions=177,controls=24,counts=counts,decisions=decisions,work=work)


def verify_controller(cases,directory=CONTROL):
    total_budget=directory==BUDGET
    reg=registration(directory/'registration.json');data=read(directory/'runs.json');assert len(data['paths'])==2 and len(data['controls'])==12
    results=[]
    for index,spec in enumerate(reg['paths']):
        case=cases[spec['model']];summary=next(p for p in data['paths'] if p['path']==index);rows=[r for r in data['attempts'] if r['path']==index];h=spec['h0'];duration=spec['duration']
        with np.load(directory/f'path-{index}.npz') as arrays:
            origin=arrays['origin'];groups=arrays['groups'];prefix=np.load(ROOT/'results/m3-priority/p0.npz')[spec['model']+'-prefix'][:,3].copy();component=int(np.flatnonzero(prefix>0)[0]);prefix[component]*=1.001;np.testing.assert_array_equal(prefix,origin)
            screen=GroupScreen(case['system'],case['scale']);kappa,_=screen.evaluate(origin);np.testing.assert_array_equal(groups,np.flatnonzero(1e-3*kappa>=spec['threshold']))
            kernel=LinearContextROW(case,groups);est=LinearStreaming(case,origin,reg['grids']);y=origin.copy();left=0.;ledger={k:0. for k in LEDGERS};weights=case['options'].get('ledger_weights');den0=1. if weights is None else max(1.,abs(float(weights@origin)));pieces=[];accepted=[];max_actual={k:0. for k in ('concentration','flux')+(('net_flux',) if 'net_map' in case['options'] else ())};false_accepts=0;rejections=0
            for row in rows:
                attempt=row['attempt'];h=min(h,duration-left);right=left+h;close([left,h,right],[row['left'],row['h'],row['right']]);before=state_signature(est,y,left,ledger);assert before==row['before']
                q=np.array(row['query']);assert row['status']=='success'
                trial=kernel.trial(y,h,reg['inner_tolerance']);np.testing.assert_array_equal(y,arrays[f'{attempt}-initial']);np.testing.assert_array_equal(trial['fine'],arrays[f'{attempt}-endpoint']);curve=trial['pieces'][0]
                np.testing.assert_array_equal(np.stack([p.coefficients for p in curve.pieces]),arrays[f'{attempt}-coefficients'])
                estimate=est.trial(trial,q);close({k:estimate[k] for k in ('status','targets','stats')},row['estimate'])
                for branch,c in enumerate(estimate['error_curves']):np.testing.assert_allclose(np.stack([p.coefficients for p in c.pieces]),arrays[f'{attempt}-error-{branch}-coefficients'],rtol=2e-10,atol=1e-13)
                den=1. if weights is None else max(1.,abs(float(weights@y)));pending={k:ledger[k]+trial['stats'].get(k,0.)*den/den0 for k in LEDGERS};close(pending,row['pending_ledger'])
                score=estimate['targets'].get(reg['target'],{}).get('running_score',{});budget=max(max(score.values(),default=np.inf),max(pending.values())/1e-10) if total_budget else proposal_budget(score,pending,right,duration);close(budget,row['budget_ratio'])
                if total_budget:
                    growth=max([max(0.,score.get(k,0.)-est.scores[reg['target']].get(k,0.))/max(1e-12,1-est.scores[reg['target']].get(k,0.)) for k in score]+[(pending[k]-ledger[k])/max(1e-22,1e-10-ledger[k]) for k in LEDGERS]);close(growth,row['growth_ratio'])
                structural_ok=trial['stats']['continuous_nonnegative'] and trial['stats']['pool_error']<=1e-10 and all(trial['stats'].get(k,0.)<=1e-10 for k in LEDGERS)
                proposed=bool(estimate['status']=='success' and structural_ok and budget<=1);assert proposed==row['proposed']
                local_ref,lb,_=audit(case,y,q,h);close(lb,row['local_budget']);check_curve(case,curve,y,trial['fine'],local_ref,q,row['stats'],row['guard'],row['local_targets'])
                times=np.unique(np.r_[0.,left+h*q]);positions=np.searchsorted(times,left+h*q);ref=full(case,origin,times,1e-12)[:,positions];loose=full(case,origin,times,1e-11)[:,positions]
                gb=metrics(case,loose,ref,1e-6);close(gb,row['global_budget']);actual=metrics(case,curve(q),ref,1e-6);close(actual,row['actual']);running={k:max(max_actual[k],actual[k]) for k in actual};close(running,row['running_actual'])
                ref_ok=max([*lb.values(),*gb.values()])<=.05;assert ref_ok==row['reference_ok'];actual_ok=ref_ok and row['local_targets'][reg['target']]['passes'] and max(running.values())<=1
                false=bool(proposed and not actual_ok);accept=bool(proposed and actual_ok);assert false==row['false_accept'] and accept==row['accepted'];false_accepts+=false
                if accept:
                    est.commit(estimate);y=trial['fine'].copy();left=right;ledger=pending;max_actual=running;pieces.extend(curve.pieces);accepted.append(row)
                    if (growth<.1 if total_budget else budget<.25):h=min(1.5*h,2*spec['h0'])
                else:
                    est.discard(estimate);assert state_signature(est,y,left,ledger)==before;h*=.5;rejections+=1
                assert state_signature(est,y,left,ledger)==row['after']
            close(left,summary['duration']);assert len(accepted)==summary['accepted_steps'];assert summary['complete']==(left>=duration);close(ledger,summary['ledger']);np.testing.assert_array_equal(y,arrays['final'])
            if pieces:
                grid=arrays['grid'];reference,budget,_=audit(case,origin,grid/left,left);close(budget,summary['final_reference_budget']);curve=InnerCurve(pieces)
                check_curve_without_complete(case,curve,origin,y,reference,grid/left,summary)
            kernel.close()
        with np.load(directory/f'control-reference-{index}.npz') as a:grid=a['grid'];stored=a['reference']
        ref,budget,_=audit(case,origin,grid/duration,duration);np.testing.assert_allclose(ref,stored,rtol=2e-12,atol=2e-12);assert max(budget.values())<=.05
        for row in [r for r in data['controls'] if r['path']==index]:
            with np.load(directory/f"{row['key']}.npz") as a:c=InnerCurve.unpack(a['lengths'],a['coefficients']);check_curve(case,c,origin,a['endpoint'],ref,grid/duration,row['stats'],row['guard'],row['targets'])
        optimistic=summary['setup_seconds']+sum(r.get('candidate_seconds',r.get('seconds',0.))+r.get('estimate_seconds',0.)+r.get('decision_seconds',0.) for r in rows);close(optimistic,summary['optimistic_seconds'])
        close(optimistic+sum(r.get('local_audit_seconds',0.)+r.get('global_audit_seconds',0.) for r in rows)+summary.get('final_audit_seconds',0.),summary['audited_seconds'])
        results.append(dict(path=index,model=spec['model'],attempts=len(rows),accepted=len(accepted),rejected=rejections,false_accepts=false_accepts,complete=summary['complete'],reason=summary['reason'],fraction=left/duration,optimistic_seconds=optimistic,audited_seconds=summary['audited_seconds']))
        print('verified',directory.name,index,flush=True)
    return results


def check_curve_without_complete(case,curve,y,end,ref,q,summary):
    # Completion is a separate requirement; direct numerical checks still apply
    # to an incomplete prefix and must not turn it into an accepted full path.
    stats=summary['stats'];values=curve(q);guard,targets=qualified(case,curve,values,ref,stats)
    for k,v in physical_ledger(case,curve,y,end).items():close(v,stats[k])
    close(guard,summary['guard'])
    for v in targets.values():v['passes'] &= summary['complete'] and max(summary['final_reference_budget'].values())<=.05
    close(targets,summary['targets'])


def verify_controller_structure():
    checks=0
    for directory in (CONTROL,BUDGET):
        reg=registration(directory/'registration.json');data=read(directory/'runs.json')
        for index,spec in enumerate(reg['paths']):
            case=prepare(spec['model']);rows=[r for r in data['attempts'] if r['path']==index];summary=next(s for s in data['paths'] if s['path']==index)
            assert len(rows)<=reg['max_attempts'] and [r['attempt'] for r in rows]==list(range(len(rows)))
            accepted=[];expected_grid=[np.array([0.])]
            with np.load(directory/f'path-{index}.npz') as arrays:
                for r in rows:
                    left,h,right=r['left'],r['h'],r['right'];public=case['times'][(case['times']>spec['start']+left)&(case['times']<spec['start']+right)]-spec['start']
                    q=np.unique(np.r_[np.linspace(0,1,65),(public-left)/h]);np.testing.assert_array_equal(q,r['query'])
                    if r['accepted']:
                        accepted.append(r);expected_grid.append(left+h*q)
                    elif r['status']=='success':assert r['before']==r['after']
                    if r.get('false_accept'):assert r is rows[-1]
                    c=unpack(arrays[f"{r['attempt']}-lengths"],arrays[f"{r['attempt']}-coefficients"])
                    assert abs(c.h-h)<=1e-13*h
                    np.testing.assert_allclose(c(np.array([0.,1.])),np.column_stack([arrays[f"{r['attempt']}-initial"],arrays[f"{r['attempt']}-endpoint"]]),rtol=2e-12,atol=2e-12)
                    checks+=1
                np.testing.assert_array_equal(np.unique(np.concatenate(expected_grid)),arrays['grid'])
            if accepted:
                close(summary['stats']['pool_error'],max(r['stats']['pool_error'] for r in accepted))
                assert summary['stats']['continuous_nonnegative']==all(r['stats']['continuous_nonnegative'] for r in accepted)
                for k in ('modified_ledger_residual','modified_ledger_quadrature'):close(summary['stats'][k],summary['ledger'][k])
            if summary['reason']=='attempt budget exhausted':assert len(rows)==reg['max_attempts'] and not summary['complete']
            elif summary['reason']=='minimum step reached':assert not rows[-1]['accepted'] and rows[-1]['h']/2<spec['h0']*reg['min_step_ratio']
            elif summary['reason']=='independent audit vetoed controller proposal':assert rows[-1]['false_accept']
            else:assert summary['complete'] or summary['reason']=='reference budget failed'
            with np.load(directory/f'control-reference-{index}.npz') as a:
                np.testing.assert_array_equal(a['grid'],np.unique(np.r_[np.linspace(0,spec['duration'],513),np.concatenate(expected_grid)]))
    return checks


def verify():
    cases={n:prepare(n) for n in ('teusink','kecoli')};cache={}
    result=dict(hermite=verify_offline(OUT,cases,cache),linear=verify_offline(LINEAR,cases,cache),controller=verify_controller(cases),total_budget=verify_controller(cases,BUDGET))
    for p,sha in read(OLD/'manifest.json').items():assert digest(ROOT/p)==sha,p
    crc=0
    for folder in (OUT,LINEAR,CONTROL,BUDGET):
        for p in folder.glob('*.npz'):
            with zipfile.ZipFile(p) as z:assert z.testzip() is None,p
            crc+=1
    result.update(controller_query_and_rollback_checks=verify_controller_structure(),npz_crc_checked=crc,historical_manifest_unchanged=True,M4_admission=False,scope='Streaming estimator diagnostics and four short perturbed controller trajectories under two policies only. Original M4 gates unchanged and unfulfilled.')
    return result


def write_report(result):
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve report')
    save(OUT/'summary.json',result);save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
    files=[p for d in (OUT,LINEAR,CONTROL,BUDGET) for p in d.iterdir() if p.is_file() and p.name!='manifest.json']
    files += [ROOT/'research'/p for p in ('streaming_defect.py','linear_context_row.py','test_streaming_defect.py','m3_streaming_defect.py','linear_error_output.py','test_linear_error_output.py','m3_streaming_linear.py','m3_streaming_controller.py','m3_streaming_budget.py','m3_streaming_report.py')]
    files += [ROOT/'docs/m3-streaming-defect.md']
    save(OUT/'manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args();result=verify()
    if args.write:write_report(result)
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
