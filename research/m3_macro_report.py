"""Independent saved-trajectory checks for single-macro and third-order studies."""
import argparse
from collections import defaultdict
import json
import platform
import subprocess
import sys
import numpy as np
import scipy
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from metabolism.adaptive import GroupScreen
from scipy.sparse import csc_matrix
from research.mri_kernel import Hermite,GAUSS
from research.m3_priority import metrics
from research.m3_shadow_study import ROOT,digest,compare
from research.defect_compact import estimate_compact
from research.m3_ordered_cost import structural

SINGLE=ROOT/'results/m3-single';THIRD=ROOT/'results/m3-third'

def read(p):return json.loads(p.read_text())

def close_mapping(a,b):
    assert a.keys()==b.keys()
    for k in a:np.testing.assert_allclose(a[k],b[k],rtol=2e-7,atol=2e-7)

def verify():
    cases={n:prepare(n) for n in ('teusink','kecoli')};checks=defaultdict(int);summary={};timings={};diagnostics=[]
    for directory,registration in ((SINGLE,'registration.json'),(SINGLE,'partition-registration.json'),(THIRD,'registration.json')):
        reg=read(directory/registration)
        for p,sha in (reg['source_sha256']|reg.get('evidence_sha256',{})).items():assert digest(ROOT/p)==sha,p
    for phase,directory,file in [('development',SINGLE,'development.json'),('cost',SINGLE,'cost.json'),('partitions',SINGLE,'partitions.json'),('third',THIRD,'runs.json')]:
        data=read(directory/file);rows=data['runs'];reg=read(directory/('partition-registration.json' if phase=='partitions' else 'registration.json'))
        assert data['source_sha256']==reg['source_sha256']
        assert len(rows)=={'development':48,'cost':48,'partitions':36,'third':36}[phase]
        groups=defaultdict(lambda:dict(trials=0,actual_pass=0,policy_accept=0,false_accept=0,policy_false_accept=0))
        timegroups=defaultdict(list);rechecked_refs=set()
        for row in rows:
            if row['status']=='failed':checks[phase+'_failed']+=1;continue
            case=cases[row['model']];key=row['key'];arrays=np.load(directory/(row['shard']+'.npz'))
            refs=np.load(SINGLE/(row['reference_shard']+'.npz')) if phase=='partitions' else arrays
            prefix=key.split('-',1)[0]+'-' if phase=='development' else ''
            ref=refs[prefix+'reference'];query=refs[prefix+'query'];eq=refs[prefix+'estimate_query']
            ref_id=row.get('reference_shard',row['shard'])+'/'+prefix
            if ref_id not in rechecked_refs:
                for label in ('loose','optimized'):
                    budget=metrics(case,refs[prefix+label],ref,1e-6)
                    assert max(budget.values())<=.05
                    if phase!='partitions':close_mapping(budget,row['reference_budgets'][0 if label=='loose' else 1])
                rechecked_refs.add(ref_id);checks['references_recomputed']+=1
            method=row.get('method','single')
            if method=='full':
                values=arrays[key if phase=='cost' else key+'-values'];guard=structural(case,values)
                assert guard==row['guard']
                for target,record in row['actual'].items():
                    actual=metrics(case,values,ref,float(target));close_mapping(actual,record)
                    passed=(max(actual.values())<=1 and guard['nonnegative'] and guard['pool']<=1e-10
                        and all(row['counts'].get(k,0.)<=1e-10 for k in ('pool_error','ledger_residual','ledger_quadrature')))
                    if phase!='cost' or row['repeat']>=0:
                        name=f"{row['model']}/{row['h']}/{target}/full/{row['tolerance']}"
                        timegroups[name].append(dict(seconds=row['seconds'],passes=bool(passed)))
                checks['full_curves_recomputed']+=1;continue
            values=arrays[key] if phase=='cost' else arrays[key+'-values']
            endpoints=np.column_stack([values[:,0],values[:,-1]]) if phase=='cost' else arrays[key+'-endpoints']
            rhs=lambda z:case['system'].S@case['system'].flux(0.,z)
            p=Hermite(endpoints[:,0],endpoints[:,1],rhs(endpoints[:,0]),rhs(endpoints[:,1]),row['h'])
            reconstructed=p(query);np.testing.assert_allclose(reconstructed,values,rtol=1e-12,atol=1e-13)
            truth=row['truth'];stats=truth['stats'];assert stats['subflows']==(3 if phase=='third' else 2)
            positive=bool(np.all(p.minima()>=-1e-12*case['scale']));assert positive==stats['continuous_nonnegative']
            if positive:
                guard=structural(case,values);np.testing.assert_allclose(guard['pool'],truth['output_pool_error'],rtol=1e-12,atol=1e-14)
                if 'ledger_weights' in case['options']:
                    weights=case['options']['ledger_weights'];integrals=[]
                    for n in (8,16):
                        nodes,w=GAUSS[n];integrals.append(row['h']/2*float(w@np.array([weights@rhs(z) for z in p((nodes+1)/2).T])))
                    den=max(1.,abs(float(weights@endpoints[:,0])))
                    np.testing.assert_allclose(stats['physical_ledger_residual'],abs(float(weights@(endpoints[:,1]-endpoints[:,0]))-integrals[1])/den,atol=1e-13)
                    np.testing.assert_allclose(stats['physical_ledger_quadrature'],abs(integrals[1]-integrals[0])/den,atol=1e-13)
                for target,record in truth['targets'].items():close_mapping(metrics(case,values,ref,float(target)),record['actual'])
            else:assert truth['status']=='rejected'
            shadows=row['shadows'] if 'shadows' in row else {'4-8':row['shadow']}
            comparisons=row['comparisons'] if 'comparisons' in row else {'4-8':row['comparison']}
            for label,shadow in shadows.items():
                grid=tuple(map(int,label.split('-')));fresh,recomputed_arrays=estimate_compact(case,[p],p(eq),eq,grid)
                assert fresh['status']==shadow['status']
                if phase!='cost':
                    for label2,array in recomputed_arrays.items():
                        stored_key=key+'-'+(label+'-' if phase=='development' else '')+label2
                        if label2.startswith('error-'):
                            assert np.max(abs(array-arrays[stored_key])/case['scale'][:,None])<1e-9
                        else:np.testing.assert_allclose(array,arrays[stored_key],rtol=1e-12,atol=1e-14)
                        checks['saved_estimator_arrays_recomputed']+=1
                if fresh['status']=='success':
                    for target,entry in fresh['targets'].items():
                        stored=shadow['targets'][target]
                        for field in ('score','central','refinement'):close_mapping(entry[field],stored[field])
                        assert entry['would_accept']==stored['would_accept']
                assert compare(truth,shadow)==comparisons[label]
                for target,c in comparisons[label].items():
                    name=f"{row['model']}/{label}/{target}";g=groups[name];g['trials']+=1
                    for a,b in [('actual_pass','actual_all_checks_pass'),('policy_accept','policy_would_accept'),('false_accept','false_accept'),('policy_false_accept','policy_false_accept')]:g[a]+=c[b]
                    if phase in ('cost','third') and (phase=='third' or row['repeat']>=0):
                        name=f"{row['model']}/{row['h']}/{target}/{method}-{label}/{row['tolerance']}"
                        timegroups[name].append(dict(seconds=row['base_seconds']+shadow['seconds'],passes=c['actual_all_checks_pass'],policy_accept=c['policy_would_accept'],base_seconds=row['base_seconds'],audited_seconds=row['base_seconds']+shadow['seconds']+row['audit_seconds']))
            if phase=='third':
                diagnostics.append(dict(model=row['model'],h=row['h'],tolerance=row['tolerance'],
                    endpoint=metrics(case,values[:,-1:],ref[:,-1:],1e-6),dense=truth.get('targets',{}).get('1e-06',{}).get('actual'),
                    physical_ledger_residual=stats.get('physical_ledger_residual')))
            checks[phase+'_candidate_curves_recomputed']+=1
        summary[phase]=dict(groups)
        timings[phase]={k:dict(measurements=len(v),all_pass=all(x['passes'] for x in v),median_seconds=float(np.median([x['seconds'] for x in v])),
            policy_all_accept=all(x.get('policy_accept',True) for x in v),median_audited_seconds=float(np.median([x['audited_seconds'] for x in v])) if 'audited_seconds' in v[0] else None) for k,v in timegroups.items()}
    for manifest in ('results/m3-ordered-cost/manifest.json','results/m3-shadow-fast/manifest.json'):
        for path,sha in read(ROOT/manifest).items():assert digest(ROOT/path)==sha,path
    checks['historical_manifests_unchanged']=2
    small=cases['teusink'];state=np.load(SINGLE/'dev-teusink-3.npz')['initial']
    screen=GroupScreen(small['system'],small['scale']);kappa,_=screen.evaluate(state)
    fast_groups=np.flatnonzero(1e-3*kappa>=.01)
    fast=np.concatenate([small['system'].groups[i] for i in fast_groups])
    slow=np.setdiff1d(np.arange(small['system'].S.shape[1]),fast)
    structure=dict(model='teusink',start=1.,fast_groups=fast_groups.tolist(),slow_columns_zero_based=slow.tolist(),
        slow_dependency_nonzeros=csc_matrix(small['system'].dependencies)[slow].nnz,
        slow_gradient_nonzeros=screen.gradient(state)[slow].nnz,time_dependent=small['system'].time_dependent,
        consequence='For this autonomous model and default grouping, slow forcing is constant; all MRI stages solve the same full vector field. A merged solve is a full-system control, not evidence of multirate acceleration.')
    assert structure['slow_dependency_nonzeros']==0 and not structure['time_dependent']
    return dict(date='2026-11-08',checks=dict(checks),counts=summary,timings=timings,third_output_diagnostics=diagnostics,
        default_slow_structure=structure,M4_admission=False,scope='Local development only; no fresh-state promotion, online controller, full-window speed claim or reduced audit.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args();result=verify()
    if args.write:
        save(SINGLE/'summary.json',result)
        if not (SINGLE/'environment.json').exists():save(SINGLE/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
        files=[p for directory in (SINGLE,THIRD) for p in directory.iterdir() if p.is_file() and p.name!='manifest.json']
        files += [ROOT/'research'/p for p in ('mri_single.py','mri_third.py','defect_compact.py','test_mri_single.py','test_mri_third.py','m3_single_study.py','m3_single_partitions.py','m3_third_study.py','m3_macro_report.py')]
        files += [ROOT/'docs/m3-macro-redesign.md']
        save(SINGLE/'manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})
    else:
        assert result==read(SINGLE/'summary.json')
        for path,sha in read(SINGLE/'manifest.json').items():assert digest(ROOT/path)==sha,path
    print(json.dumps(result,indent=2))
