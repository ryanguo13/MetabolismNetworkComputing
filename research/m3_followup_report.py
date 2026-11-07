"""Package and independently check the stress and ordered-kernel evidence."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.m3_priority import metrics
from research.mri_kernel import output
from research.m3_shadow_study import ROOT,digest,build_pieces,compare
from research.defect_shadow import interpolate,observables

STRESS=ROOT/'results/m3-shadow-stress'
COST=ROOT/'results/m3-ordered-cost'

def read(p):return json.loads(p.read_text())

def package():
    path=STRESS/'curves.npz'
    if not path.exists():return
    source=np.load(path)
    recovered=np.load(STRESS/'recovered-reference.npz')
    recovery=read(STRESS/'reference-recovery.json')
    assert digest(path)==recovery['original_archive_sha256']
    def get(key):return recovered[key] if key==recovery['key'] else source[key]
    files={}
    for model in ('teusink','kecoli'):
        for scenario in ('fixed','switch','perturbed'):
            for hi in (0,1):
                key=f'{model}-{scenario}-{hi}';destination=STRESS/(key+'.npz')
                np.savez_compressed(destination,**{k:get(k) for k in source.files if k.startswith(key+'-')})
                files[destination.name]=digest(destination)
    save(STRESS/'packaging.json',dict(original_sha256=digest(path),arrays=len(source.files),shards=files,
        reason='Path shards keep files below hosting limits. One corrupted reference independently recovered as documented in reference-recovery.json; other members preserved. Damaged original retained in ignored scratch.'))
    source.close();path.rename(ROOT/'results/scratch/m3-shadow-stress-original.npz')

def verify():
    cases={n:prepare(n) for n in ('teusink','kecoli')};checks=defaultdict(int)
    for directory in (STRESS,COST):
        reg=read(directory/'registration.json')
        for p,sha in reg['source_sha256'].items():assert digest(ROOT/p)==sha,p
        assert read(directory/'runs.json')['source_sha256']==reg['source_sha256']
    shards=read(STRESS/'packaging.json');count=0
    for p,sha in shards['shards'].items():
        assert digest(STRESS/p)==sha
        with np.load(STRESS/p) as arrays:count+=len(arrays.files)
    assert count==shards['arrays'];checks['packaged_arrays']=count
    stress={};verified_paths=set()
    for row in read(STRESS/'runs.json')['runs']:
        path=row['key'].rsplit('-',1)[0];key=row['key'];case=cases[row['model']]
        if row['status']!='success':continue
        arrays=np.load(STRESS/(path+'.npz'))
        if path not in verified_paths:
            global_budget=metrics(case,arrays[path+'-global-loose'],arrays[path+'-global-ref'],1e-6)
            np.testing.assert_allclose(list(global_budget.values()),list(row['global_reference_budget'].values()),rtol=1e-12,atol=1e-12)
            assert max(global_budget.values())<=.05
            verified_paths.add(path)
        query=arrays[key+'-dense_query'];values=arrays[key+'-dense'];ref=arrays[key+'-reference']
        pieces=build_pieces(case,arrays[key+'-nodes'],row['h'])
        reconstructed=output(dict(fine=arrays[key+'-nodes'][:,-1],pieces=pieces),query)
        np.testing.assert_allclose(reconstructed,values,rtol=1e-12,atol=1e-13)
        budget=metrics(case,arrays[key+'-loose'],ref,1e-6)
        assert max(budget.values())<=.05
        np.testing.assert_allclose(list(budget.values()),list(row['local_reference_budget'].values()),rtol=1e-12,atol=1e-12)
        assert compare(row['truth'],row['shadow'])==row['comparison']
        for target,entry in row['truth'].get('targets',{}).items():
            actual=metrics(case,values,ref,float(target))
            np.testing.assert_allclose(list(actual.values()),list(entry['actual'].values()),rtol=1e-12,atol=1e-12)
            if row['shadow']['status']=='success':
                q=arrays[key+'-query'];vv=arrays[key+'-values']
                corrected=[vv+interpolate(arrays[key+f'-nodes-{s}'],arrays[key+f'-error-{s}'],q) for s in (8,16)]
                central=observables(case,vv,corrected[1],float(target));refinement=observables(case,*corrected,float(target))
                scores={k:4*(central[k]+refinement[k]) for k in central}
                recorded=row['shadow']['targets'][target]
                np.testing.assert_allclose(list(scores.values()),list(recorded['score'].values()),rtol=2e-7,atol=2e-7)
                assert (max(scores.values())<=1)==recorded['would_accept']
            global_actual=metrics(case,values,arrays[path+'-global-ref'][:,arrays[key+'-global_positions']],float(target))
            np.testing.assert_allclose(list(global_actual.values()),list(row['global_errors'][target].values()),rtol=1e-12,atol=1e-12)
            name=f"{row['model']}/{row['scenario']}/{row['h']}/{target}"
            s=stress.setdefault(name,dict(steps=0,local_pass=0,policy_accept=0,numerical_false_accept=0,policy_false_accept=0,global_exceed=0,accepted_path_global_exceed=0))
            c=row['comparison'][target];s['steps']+=1;s['local_pass']+=c['actual_all_checks_pass'];s['policy_accept']+=c['policy_would_accept']
            s['numerical_false_accept']+=c['false_accept'];s['policy_false_accept']+=c['policy_false_accept']
            s['global_exceed']+=max(global_actual.values())>1;s['accepted_path_global_exceed']+=row['accumulated_exceedance'][target]
        checks['stress_steps_recomputed']+=1
    rows=read(COST/'runs.json')['runs'];arrays=np.load(COST/'curves.npz');cost=defaultdict(list)
    for row in rows:
        if row['status']!='success':continue
        case=cases[row['model']];index=row['key'].split('-',1)[0]
        for target,stored in row['actual'].items():
            actual=metrics(case,arrays[row['key']],arrays[index+'-reference'],float(target))
            np.testing.assert_allclose(list(actual.values()),list(stored.values()),rtol=1e-12,atol=1e-12)
            passed=max(actual.values())<=1 and row['guard']['nonnegative'] and row['guard']['pool']<=1e-10
            if row['method'].startswith('mri'):
                passed &= row['truth']['targets'][target]['passes']
            else:
                passed &= all(row['counts'].get(k,0.)<=1e-10 for k in ('pool_error','ledger_residual','ledger_quadrature'))
            if row['repeat']>=0:
                key=f"{row['model']}/{row['h']}/{target}/{row['method']}/{row['tolerance']}"
                cost[key].append(dict(seconds=row['seconds'],passed=bool(passed),audited_seconds=row.get('audited_seconds')))
        checks['cost_runs_recomputed']+=1
    cost_summary={k:dict(repeats=len(v),all_pass=all(r['passed'] for r in v),
        median_seconds=float(np.median([r['seconds'] for r in v])),
        median_audited_seconds=float(np.median([r['audited_seconds'] for r in v])) if v[0]['audited_seconds'] is not None else None) for k,v in cost.items()}
    for p,sha in read(ROOT/'results/m3-shadow-fast/manifest.json').items():assert digest(ROOT/p)==sha,p
    checks['prior_shadow_manifest_unchanged']=1
    audit_data=read(COST/'optimized-audit.json');audit_arrays=np.load(COST/'optimized-audit.npz')
    audit_reg=read(COST/'audit-registration.json')
    assert audit_data['source_sha256']==audit_reg['source_sha256']
    for p,sha in audit_reg['source_sha256'].items():assert digest(ROOT/p)==sha
    audits=defaultdict(list)
    for row in audit_data['runs']:
        index=row['key'].split('-',1)[0];case=cases[row['model']]
        delta=metrics(case,audit_arrays[row['key']],arrays[index+'-reference'],1e-6)
        np.testing.assert_allclose(list(delta.values()),list(row['reference_difference'].values()),rtol=1e-12,atol=1e-12)
        assert max(delta.values())<=.05
        if row['repeat']>=0:audits[f"{row['model']}/{row['h']}"].append(row)
    audit_summary={k:dict(median_audit_seconds=float(np.median([r['seconds'] for r in v])),
        median_estimator_seconds=v[0]['median_estimator_seconds'],
        ratio=v[0]['median_estimator_seconds']/float(np.median([r['seconds'] for r in v]))) for k,v in audits.items()}
    return dict(date='2026-11-07',stress=stress,cost=cost_summary,optimized_audit=audit_summary,checks=dict(checks),M4_admission=False,
        limitations=['Short forced diagnostic paths, not an online controller or full-window qualification.',
                     'Development repeats only; CVODE baseline does not expose all internal-state checks.',
                     'Local full-system auditing remains mandatory.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    if args.write:package()
    result=verify()
    if args.write:
        save(COST/'summary.json',result)
        files=[p for d in (STRESS,COST) for p in d.iterdir() if p.is_file() and p.name!='manifest.json']
        files += [ROOT/'research'/p for p in ('m3_shadow_stress.py','ordered_radau.py','test_ordered_radau.py','m3_ordered_cost.py','m3_followup_report.py','m3_ordered_audit.py','recover_stress_reference.py')]
        files += [ROOT/'docs/m3-followup-evaluation.md']
        save(COST/'manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})
    else:
        assert result==read(COST/'summary.json')
        for p,sha in read(COST/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
