"""Read-only verification by default; --write records this bounded shadow study."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy

from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.defect_fast import BatchRates,interpolate_fast
from research.defect_shadow import interpolate,observables
from research.m3_priority import metrics
from research.m3_shadow_study import ROOT,digest,build_pieces,compare

DIRS=[ROOT/'results'/p for p in ('m3-shadow','m3-shadow-fast')]


def read(path):return json.loads(path.read_text())


def verify():
    cases={n:prepare(n) for n in ('teusink','kecoli')}
    checks=Counter()
    summary={}
    for directory in DIRS:
        reg=read(directory/'registration.json')
        for path,sha in (reg['source_sha256']|reg['development_sha256']).items():
            assert digest(ROOT/path)==sha,path
            checks['registered_hashes']+=1
        result={}
        for stage in ('development','new-states'):
            data=read(directory/f'{stage}.json')
            assert data['source_sha256']==reg['source_sha256']
            rows=data['runs']
            assert len(rows)==(68 if stage=='development' else 36)
            arrays=np.load(directory/f'{stage}.npz')
            for row in rows:
                case=cases[row['model']]
                if stage=='new-states':
                    key=row['key']; refkey=key.rsplit('-',2)[0]
                    truth=row['truth']; saved=arrays
                    query=arrays[refkey+'-fractions']
                    budget=metrics(case,arrays[refkey+'-loose'],arrays[refkey+'-reference'],1e-6)
                    np.testing.assert_allclose(list(budget.values()),list(row['reference_tightening'].values()),rtol=1e-12,atol=1e-12)
                    assert row['reference_pass'] and max(budget.values())<=.05
                    for target,entry in truth.get('targets',{}).items():
                        actual=metrics(case,arrays[key],arrays[refkey+'-reference'],float(target))
                        np.testing.assert_allclose(list(actual.values()),list(entry['actual'].values()),rtol=1e-12,atol=1e-12)
                    checks['independent_reference_rechecks']+=1
                else:
                    dataset=row['source_dataset'];key=row['source_key']
                    saved=np.load(ROOT/f'results/m3-priority/{dataset}.npz')
                    truth=next(r for r in read(ROOT/f'results/m3-priority/{dataset}.json')['runs'] if r['key']==key)
                    # All historical query arrays are named by their reference key.
                    refkey=key.rsplit('-',2)[0] if dataset=='p0' else key
                    query=saved[refkey+'-fractions']
                pieces=build_pieces(case,saved[key+'-nodes'],row['h'])
                values=saved[key]
                shadow=row['shadow']
                assert compare(truth,shadow)==row['comparison']
                if shadow['status']=='success':
                    histories=[interpolate(arrays[row['key']+f'-nodes-{s}'],arrays[row['key']+f'-error-{s}'],query) for s in (8,16)]
                    low,high=[values+e for e in histories]
                    assert all(np.isfinite(v).all() and np.all(v>=-1e-12*case['scale'][:,None]) for v in (low,high))
                    for target,entry in shadow['targets'].items():
                        central=observables(case,values,high,float(target))
                        refinement=observables(case,low,high,float(target))
                        score={k:4*(central[k]+refinement[k]) for k in central}
                        # Independent scalar rates and interpolation; allow roundoff in net cancellation.
                        for k in score:
                            np.testing.assert_allclose(score[k],entry['score'][k],rtol=2e-7,atol=2e-7)
                        assert (max(score.values())<=1)==entry['would_accept']
                    checks['recomputed_shadow_scores']+=1
                elif shadow['reason']=='negative_candidate_curve':
                    assert np.any(np.minimum.reduce([p.minima() for p in pieces]) < -1e-12*case['scale'])
                    checks['negative_curves']+=1
                else:
                    assert shadow['reason']=='negative_or_nonfinite_linearized_correction'
                    corrected=[values+interpolate(arrays[row['key']+f'-nodes-{s}'],arrays[row['key']+f'-error-{s}'],query) for s in (8,16)]
                    assert any(not np.isfinite(v).all() or np.any(v < -1e-12*case['scale'][:,None]) for v in corrected)
                    checks['negative_corrections']+=1
            result[stage]={}
            for name in cases:
                rr=[r for r in rows if r['model']==name]
                good=[r for r in rr if r['shadow']['status']=='success']
                model=dict(configurations=len(rr),success=len(good),guard_rejects=len(rr)-len(good),
                           median_seconds=float(np.median([r['shadow']['seconds'] for r in good])),targets={})
                if stage=='new-states':
                    model['median_audit_seconds']=float(np.median([r['audit_seconds'] for r in good]))
                    model['median_paired_cost_ratio']=float(np.median([r['shadow']['seconds']/r['audit_seconds'] for r in good]))
                for target in ('0.0001','1e-06'):
                    cc=[r['comparison'][target] for r in rr if target in r['comparison']]
                    actual=sum(c['actual_all_checks_pass'] for c in cc)
                    accepted=sum(c['policy_would_accept'] and c['actual_all_checks_pass'] for c in cc)
                    item=dict(evaluated=len(cc),actual_pass=actual,true_accept=accepted,recall=accepted/actual,
                        false_accepts=sum(c['false_accept'] for c in cc),policy_false_accepts=sum(c['policy_false_accept'] for c in cc))
                    if stage=='new-states':
                        item['local_screen_pass']=(not item['false_accepts'] and not item['policy_false_accepts']
                            and item['recall']>=.5 and model['median_paired_cost_ratio']<=.25)
                    model['targets'][target]=item
                result[stage][name]=model
        summary[directory.name]=result
    # Compare unchanged mathematics on identical development candidates.
    old,new=[read(d/'development.json')['runs'] for d in DIRS]
    a,b=[np.load(d/'development.npz') for d in DIRS]
    equivalence={}
    for name,case in cases.items():
        error_delta=[];score_delta=[];net_delta=[];batch_delta=[];jac_delta=[]
        batch=BatchRates(case);screen=GroupScreen(case['system'],case['scale'])
        for x,y in zip(old,new):
            if x['model']!=name:continue
            assert x['key']==y['key'] and x['shadow']['status']==y['shadow']['status']
            source=np.load(ROOT/f"results/m3-priority/{x['source_dataset']}.npz")
            states=source[x['source_key']+'-nodes']
            scalar=np.column_stack([case['system'].flux(0.,z) for z in states.T])
            batch_delta.append(float(np.max(abs(batch.flux(states)-scalar)/np.maximum(1,abs(scalar)))))
            j1=batch.jacobian(states[:,0]);j2=case['system'].S@screen.gradient(states[:,0])
            jac_delta.append(float(np.max(abs((j1-j2).toarray()))/max(1,float(abs(j2).max()))))
            for s in (8,16):
                key=x['key']+f'-error-{s}'
                if key in a:
                    error_delta.append(float(np.max(abs(a[key]-b[key])/case['scale'][:,None])))
                    nodes=b[x['key']+f'-nodes-{s}'];query=np.r_[nodes,(nodes[:-1]+nodes[1:])/2]
                    np.testing.assert_allclose(interpolate_fast(nodes,b[key],query),interpolate(nodes,b[key],query),rtol=1e-12,atol=1e-14)
            for target,v in x['shadow'].get('targets',{}).items():
                w=y['shadow']['targets'][target]
                assert v['would_accept']==w['would_accept']
                score_delta.extend(abs(v['score'][k]-w['score'][k])/max(1,abs(v['score'][k])) for k in v['score'])
                if 'net_flux' in v['score']:net_delta.append(abs(v['score']['net_flux']-w['score']['net_flux']))
        equivalence[name]=dict(max_scale_normalized_history_delta=max(error_delta),
            max_normalized_score_delta=max(score_delta),max_absolute_net_score_delta=max(net_delta,default=0),
            max_normalized_batch_rate_delta=max(batch_delta),max_normalized_jacobian_delta=max(jac_delta),same_decisions=True)
        assert max(error_delta)<1e-9 and max(score_delta)<1e-8
        assert max(batch_delta)<1e-12 and max(jac_delta)<1e-12
    historical=[read(ROOT/'results/m3-priority/manifest.json'),
                read(ROOT/'results/m3-research/output-probe.json')['evidence_sha256']]
    for manifest in historical:
        for path,sha in manifest.items():
            assert digest(ROOT/path)==sha,path
            checks['historical_manifest_entries']+=1
    return dict(date='2026-11-07',scope='Local shadow only; no controller or M4 admission',
                studies=summary,equivalence=equivalence,checks=dict(checks),M4_admission=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true')
    args=parser.parse_args();result=verify()
    if args.write:
        save(DIRS[1]/'summary.json',result)
        save(DIRS[1]/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
            platform=platform.platform(),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            execution='bash scripts/run-macos.sh; single-thread BLAS/OpenMP; SUNDIALS 6.4.1'))
        paths=[p for d in DIRS for p in d.iterdir() if p.is_file() and p.name!='manifest.json']
        paths += [p for p in (ROOT/'research').glob('*shadow*.py')]+[ROOT/'research/defect_fast.py']
        paths += [ROOT/'docs/m3-shadow-evaluation.md']
        save(DIRS[1]/'manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in paths})
    else:
        assert result==read(DIRS[1]/'summary.json')
        for path,sha in read(DIRS[1]/'manifest.json').items():assert digest(ROOT/path)==sha,path
    print(json.dumps(result,indent=2))
