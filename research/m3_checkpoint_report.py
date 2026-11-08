"""Independent replay of attribution and checkpoint recovery, including audits."""
import argparse
from copy import deepcopy
import json
import platform
import subprocess
import sys
import zipfile
import numpy as np
import scipy
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.linear_context_row import LinearContextROW
from research.linear_error_output import LinearStreaming
from research.extrapolated_flow import unpack
from research.inner_dense import InnerCurve,physical_ledger
from research.m3_growth_study import audit
from research.m3_priority import full,metrics
from research.m3_extrapolated_paths import qualified
from research.m3_ledger_flow_report import registration,close,check_curve,modified
from research.row_continuous import PolynomialForcing
from research.m3_checkpoint_controller import state_signature,LEDGERS
from research.budget_checkpoint import fingerprint
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-checkpoint-controller';DIAG=ROOT/'results/m3-budget-attribution'
def read(p):return json.loads(p.read_text())
def occupancy(scores,ledger):return max(max(scores.values(),default=np.inf),max(ledger.values())/1e-10)


def attribution(cases):
    reg=registration(DIAG/'registration.json');rows=read(DIAG/'runs.json')['runs'];assert len(rows)==24
    details=[]
    for row in rows:
        mi=reg['models'].index(row['model']);si=reg['states'].index(row['state']);case=cases[row['model']]
        with np.load(ROOT/f'results/m3-streaming-budget/path-{mi}.npz') as old:y=old[row['state']];groups=old['groups']
        assert row['h'] in [reg['base_steps'][mi][si]/d for d in reg['divisors']]
        assert row['tolerance'] in reg['tolerances'];q=np.linspace(0,1,65);kernel=LinearContextROW(case,groups)
        trial=kernel.trial(y,row['h'],row['tolerance']);curve=trial['pieces'][0]
        with np.load(DIAG/f"{row['key']}.npz") as a:
            np.testing.assert_array_equal(a['initial'],y);np.testing.assert_array_equal(a['groups'],groups)
            np.testing.assert_array_equal(a['endpoint'],trial['fine']);np.testing.assert_array_equal(a['coefficients'],np.stack([p.coefficients for p in curve.pieces]));np.testing.assert_array_equal(a['lengths'],[p.h for p in curve.pieces])
            ref,budget,_=audit(case,y,q,row['h']);np.testing.assert_allclose(ref,a['reference'],rtol=2e-12,atol=2e-12)
        close(budget,row['reference_budget']);assert max(budget.values())<=.05
        check_curve(case,curve,y,trial['fine'],ref,q,row['stats'],row['guard'],row['targets'])
        modified(case,kernel,curve,y,trial['fine'],PolynomialForcing(trial['forcing_coefficients'],row['h']),row['stats'])
        est=LinearStreaming(case,y,reg['grids']).trial(trial,q)
        close({k:est[k] for k in ('status','targets','stats')},{k:row['estimate'][k] for k in ('status','targets','stats')})
        details.append(dict(model=row['model'],state=row['state'],h=row['h'],tolerance=row['tolerance'],physical=row['stats'].get('physical_ledger_residual',0.),actual=row['targets']['1e-06']['actual'],score=max(est['targets']['1e-06']['running_score'].values()),inner_steps=row['stats']['accepted_steps']))
        kernel.close()
    assert len({r['key'] for r in rows})==24
    print('verified 24 attribution trials',flush=True)
    return details


def controller(index,case):
    reg=registration(OUT/'registration.json');data=read(OUT/'runs.json');spec=reg['paths'][index];summary=next(r for r in data['paths'] if r['path']==index)
    rows=[r for r in data['attempts'] if r['path']==index];events={r['after_attempt']:r for r in data['recoveries'] if r['path']==index}
    assert [r['attempt'] for r in rows]==list(range(len(rows))) and len(rows)<=spec['max_attempts']
    with np.load(OUT/f'path-{index}.npz') as a:origin=a['origin'];groups=a['groups'];final=a['final'];stored_grid=a['grid']
    expected=np.load(ROOT/'results/m3-priority/p0.npz')[spec['model']+'-prefix'][:,3].copy();component=int(np.flatnonzero(expected>0)[0]);expected[component]*=1.001
    np.testing.assert_array_equal(origin,expected);assert component==summary['perturbed_component']
    screen=GroupScreen(case['system'],case['scale']);kappa,_=screen.evaluate(origin);np.testing.assert_array_equal(groups,np.flatnonzero(1e-3*kappa>=spec['threshold']))
    kernel=LinearContextROW(case,groups);est=LinearStreaming(case,origin,reg['grids']);y=origin.copy();left=0.;h=spec['h0'];duration=spec['duration'];cap=2*h
    ledger={k:0. for k in LEDGERS};weights=case['options'].get('ledger_weights');den0=1. if weights is None else max(1.,abs(float(weights@origin)))
    actual_max={k:0. for k in ('concentration','flux')+(('net_flux',) if 'net_map' in case['options'] else ())};accepted=[];curves={};consecutive=0;recoveries=0;false_count=0;reason=None
    # Independent value snapshots, deliberately not using Checkpoint.restore.
    def snapshot():
        return dict(y=y.copy(),left=left,ledger=ledger.copy(),actual=actual_max.copy(),accepted=accepted.copy(),h=h,errors=[e.copy() for e in est.errors],scores=deepcopy(est.scores),signature=fingerprint(est,y,left,ledger,actual_max,groups))
    checkpoints=[snapshot()]
    for row in rows:
        attempt=row['attempt'];h=min(h,duration-left);right=left+h
        close([left,h,right,cap],[row['left'],row['h'],row['right'],row['cap']]);assert est.generation==row['generation']
        public=case['times'][(case['times']>spec['start']+left)&(case['times']<spec['start']+right)]-spec['start']
        q=np.unique(np.r_[np.linspace(0,1,65),(public-left)/h]);np.testing.assert_array_equal(q,row['query'])
        before=state_signature(est,y,left,ledger);assert before==row['before']
        # This batch has no integration exceptions; require all saved trials to
        # be explicitly replayable rather than silently skip unsupported rows.
        assert row['status']=='success'
        trial=kernel.trial(y,h,spec['inner_tolerance']);curve=trial['pieces'][0];estimate=est.trial(trial,q);curves[attempt]=curve
        with np.load(OUT/f'attempt-{index}-{attempt}.npz') as a:
            np.testing.assert_array_equal(y,a['initial']);np.testing.assert_array_equal(trial['fine'],a['endpoint']);np.testing.assert_array_equal(trial['forcing_coefficients'],a['forcing'])
            np.testing.assert_array_equal(a['coefficients'],np.stack([p.coefficients for p in curve.pieces]));np.testing.assert_array_equal(a['lengths'],[p.h for p in curve.pieces])
            for j,e in enumerate(estimate['errors']):np.testing.assert_array_equal(e,a[f'error-{j}'])
        close({k:estimate[k] for k in ('status','targets','stats')},row['estimate'])
        for stats in estimate['stats']:assert stats['max_linear_residual']<=1e-10
        den=1. if weights is None else max(1.,abs(float(weights@y)));pending={k:ledger[k]+trial['stats'].get(k,0.)*den/den0 for k in ledger};close(pending,row['pending_ledger'])
        score=estimate['targets'].get(reg['target'],{}).get('running_score',{});budget=occupancy(score,pending);close(budget,row['budget_ratio'])
        growth=max([max(0.,score[k]-est.scores[reg['target']].get(k,0.))/max(1e-12,1-est.scores[reg['target']].get(k,0.)) for k in score]+[(pending[k]-ledger[k])/max(1e-22,1e-10-ledger[k]) for k in ledger]);close(growth,row['growth_ratio'])
        structural_ok=trial['stats']['continuous_nonnegative'] and trial['stats']['pool_error']<=1e-10 and all(trial['stats'].get(k,0.)<=1e-10 for k in ledger)
        proposed=bool(estimate['status']=='success' and structural_ok and budget<=1);assert proposed==row['proposed']
        ref,lb,_=audit(case,y,q,h);close(lb,row['local_budget']);check_curve(case,curve,y,trial['fine'],ref,q,row['stats'],row['guard'],row['local_targets'])
        modified(case,kernel,curve,y,trial['fine'],PolynomialForcing(trial['forcing_coefficients'],h),row['stats'])
        times=np.unique(np.r_[0.,left+h*q]);positions=np.searchsorted(times,left+h*q)
        ref=full(case,origin,times,1e-12)[:,positions];loose=full(case,origin,times,1e-11)[:,positions];gb=metrics(case,loose,ref,1e-6);close(gb,row['global_budget'])
        actual=metrics(case,curve(q),ref,1e-6);close(actual,row['actual']);running={k:max(actual_max[k],actual[k]) for k in actual};close(running,row['running_actual'])
        reference_ok=max([*lb.values(),*gb.values()])<=.05;assert reference_ok==row['reference_ok']
        actual_ok=reference_ok and row['local_targets'][reg['target']]['passes'] and max(running.values())<=1
        accept=bool(proposed and actual_ok);false=bool(proposed and not actual_ok);assert accept==row['accepted'] and false==row['false_accept'];false_count+=false
        if accept:
            est.commit(estimate);y=trial['fine'].copy();left=right;ledger=pending;actual_max=running;accepted.append(attempt);consecutive=0
            if growth<.1:h=min(1.5*h,cap)
            checkpoints.append(snapshot())
        else:
            est.discard(estimate);assert before==state_signature(est,y,left,ledger);h*=.5;consecutive+=1
        assert state_signature(est,y,left,ledger)==row['after']
        if false:reason='independent audit vetoed controller proposal'
        elif not reference_ok:reason='reference budget failed'
        recover=not reason and left<duration and consecutive>=3 and bool(accepted) and recoveries<reg['max_recoveries']
        assert recover==(attempt in events)
        if recover:
            event=events[attempt];eligible=[c for c in checkpoints if c['left']<=left-duration/8 and occupancy(c['scores'][reg['target']],c['ledger'])<=.7];cp=eligible[-1] if eligible else checkpoints[0]
            removed=[a for a in accepted if a not in cp['accepted']];assert removed==event['removed'] and removed
            cap=min(cap,max(rows[a]['h'] for a in removed)/2);close(cap,event['new_cap']);close([left,cp['left']],[event['from_time'],event['to_time']]);assert event['before']==fingerprint(est,y,left,ledger,actual_max,groups)
            assert event['checkpoint']==cp['signature'] and event['generation_before']==est.generation
            y=cp['y'].copy();left=cp['left'];ledger=cp['ledger'].copy();actual_max=cp['actual'].copy();accepted=cp['accepted'].copy()
            est.expected=y.copy();est.errors=[e.copy() for e in cp['errors']];est.scores=deepcopy(cp['scores']);est.generation+=1
            assert event['after']==fingerprint(est,y,left,ledger,actual_max,groups)==cp['signature'];assert event['generation_after']==est.generation
            ci=next(i for i,c in enumerate(checkpoints) if c is cp);checkpoints=checkpoints[:ci+1];h=min(cp['h'],cap);close(h,event['next_h']);consecutive=0;recoveries+=1
        if reason or left>=duration:assert row is rows[-1]
        elif h<spec['h0']*reg['min_step_ratio']:reason='minimum step reached';assert row is rows[-1]
        if (attempt+1)%64==0:print('replayed',index,attempt+1,flush=True)
    complete=left>=duration
    if not complete and reason is None:reason='attempt budget exhausted';assert len(rows)==spec['max_attempts']
    assert reason==summary['reason'] and complete==summary['complete'];assert accepted==summary['retained_attempts'];assert len(accepted)==summary['accepted_steps'] and recoveries==summary['recoveries']
    close(left,summary['duration']);close(ledger,summary['ledger']);np.testing.assert_array_equal(y,final)
    grid=np.unique(np.concatenate([np.array([0.])]+[rows[a]['left']+rows[a]['h']*np.array(rows[a]['query']) for a in accepted]));np.testing.assert_array_equal(grid,stored_grid)
    curve=InnerCurve([p for a in accepted for p in curves[a].pieces]);ref,budget,_=audit(case,origin,grid/left,left);close(budget,summary['final_reference_budget'])
    stats=dict(pool_error=max(rows[a]['stats']['pool_error'] for a in accepted),continuous_nonnegative=all(rows[a]['stats']['continuous_nonnegative'] for a in accepted));stats.update(physical_ledger(case,curve,origin,y));stats.update({k:ledger[k] for k in ledger if k.startswith('modified')});close(stats,summary['stats'])
    guard,targets=qualified(case,curve,curve(grid/left),ref,stats)
    for v in targets.values():v['passes'] &= complete and max(budget.values())<=.05
    close(guard,summary['guard']);close(targets,summary['targets'])
    control_grid=np.unique(np.r_[np.linspace(0,duration,513),grid]);ref,budget,_=audit(case,origin,control_grid/duration,duration);assert max(budget.values())<=.05
    with np.load(OUT/f'control-reference-{index}.npz') as a:
        np.testing.assert_array_equal(control_grid,a['grid']);np.testing.assert_allclose(ref,a['reference'],rtol=2e-12,atol=2e-12)
    controls=[r for r in data['controls'] if r['path']==index];assert len(controls)==6
    for row in controls:
        with np.load(OUT/f"{row['key']}.npz") as a:
            c=InnerCurve.unpack(a['lengths'],a['coefficients']);check_curve(case,c,origin,a['endpoint'],ref,control_grid/duration,row['stats'],row['guard'],row['targets'])
    optimistic=summary['setup_seconds']+sum(r.get('candidate_seconds',r.get('seconds',0.))+r.get('estimate_seconds',0.)+r.get('decision_seconds',0.) for r in rows)+sum(e['seconds'] for e in events.values());close(optimistic,summary['optimistic_seconds'])
    audited=optimistic+sum(r.get('local_audit_seconds',0.)+r.get('global_audit_seconds',0.) for r in rows)+summary.get('final_audit_seconds',0.);close(audited,summary['audited_seconds'])
    best=min((r for r in controls if r['targets'][reg['target']]['passes']),key=lambda r:r['seconds'])
    result=dict(model=spec['model'],attempts=len(rows),accepted_total=sum(r['accepted'] for r in rows),retained_steps=len(accepted),abandoned_steps=sum(len(e['removed']) for e in events.values()),rejected=sum(not r['accepted'] for r in rows),recoveries=recoveries,false_accepts=false_count,complete=complete,reason=reason,fraction=left/duration,targets=targets,ledger=ledger,optimistic_seconds=optimistic,audited_seconds=audited,best_control_seconds=best['seconds'],best_control_key=best['key'],optimistic_ratio=best['seconds']/optimistic if targets[reg['target']]['passes'] else None,audited_ratio=best['seconds']/audited if targets[reg['target']]['passes'] else None)
    kernel.close();print('verified checkpoint path',index,flush=True);return result


def verify():
    cases={n:prepare(n) for n in ('teusink','kecoli')}
    result=dict(attribution=attribution(cases),paths=[controller(i,cases[n]) for i,n in enumerate(cases)])
    for p,sha in read(ROOT/'results/m3-streaming-defect/manifest.json').items():assert digest(ROOT/p)==sha,p
    count=0
    for d in (OUT,DIAG):
        for p in d.glob('*.npz'):
            with zipfile.ZipFile(p) as z:assert z.testzip() is None,p
            count+=1
    result.update(npz_crc_checked=count,historical_manifest_unchanged=True,M4_admission=False,scope='Two repeated perturbed short windows, original formal gates unchanged. All discarded work and independent audits retained; exploratory timing only.')
    return result


def write(result):
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve frozen evidence')
    save(OUT/'summary.json',result);save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
    files=[p for d in (OUT,DIAG) for p in d.iterdir() if p.is_file() and p.name!='manifest.json']
    files += [ROOT/'research'/p for p in ('m3_budget_attribution.py','budget_checkpoint.py','test_budget_checkpoint.py','m3_checkpoint_controller.py','m3_checkpoint_report.py')]
    files += [ROOT/'docs/m3-checkpoint-recovery.md']
    save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args();result=verify()
    if args.write:write(result)
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
