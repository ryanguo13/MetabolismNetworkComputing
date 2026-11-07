"""Read-only polynomial, reference and stage-order verification."""
import argparse
from collections import defaultdict
import json
import platform
import subprocess
import sys
import numpy as np
import scipy
from scipy.linalg import expm
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.inner_dense import InnerCurve,physical_ledger
from research.history_output import cubic_history
from research.mri_kernel import Hermite
from research.m3_priority import metrics
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest
OUT=ROOT/'results/m3-inner-dense'


def read(p):return json.loads(p.read_text())

def close(a,b):
    assert a.keys()==b.keys()
    for k in a:np.testing.assert_allclose(a[k],b[k],rtol=2e-8,atol=1e-10)


def verify():
    reg=read(OUT/'registration.json');data=read(OUT/'runs.json');orderreg=read(OUT/'order-registration.json')
    assert data['source_sha256']==reg['source_sha256']
    for p,sha in (reg['source_sha256']|orderreg['source_sha256']).items():assert digest(ROOT/p)==sha,p
    rows=data['runs'];assert len(rows)==48
    cases={name:prepare(name) for name in ('teusink','kecoli')};refs=set();counts=defaultdict(lambda:dict(configurations=0,passed=0))
    details=[];failures=[];checked=0
    for row in rows:
        if row['status']=='failed':failures.append(dict(key=row['key'],shard=row['shard'],reason=row['reason']));continue
        case=cases[row['model']];arrays=np.load(OUT/(row['shard']+'.npz'));key=row['key'];query=arrays['query'];ref=arrays['reference'];y=arrays['initial']
        if row['shard'] not in refs:
            for label,budget in zip(('loose','optimized'),row['reference_budgets']):
                actual=metrics(case,arrays[label],ref,1e-6);close(actual,budget);assert max(actual.values())<=.05
            screen=GroupScreen(case['system'],case['scale']);kappa,_=screen.evaluate(y)
            np.testing.assert_array_equal(arrays['groups'],np.flatnonzero(1e-3*kappa>=row['threshold']))
            cols=np.concatenate([case['system'].groups[g] for g in arrays['groups']]);slow=np.setdiff1d(np.arange(case['system'].S.shape[1]),cols)
            constant=case['system'].dependencies[slow].nnz==0 and not case['system'].time_dependent
            assert bool(constant)==row['constant_slow'];refs.add(row['shard'])
        curve=InnerCurve.unpack(arrays[key+'-lengths'],arrays[key+'-coefficients'])
        assert len(curve.pieces)==row['intervals'];np.testing.assert_allclose(curve.h,row['h'],rtol=1e-12,atol=0)
        values=curve(query);np.testing.assert_allclose(values,arrays[key+'-values'],rtol=1e-12,atol=1e-13)
        endpoint=arrays[key+'-endpoint'];np.testing.assert_allclose(curve(np.array([1.]))[:,0],endpoint,rtol=1e-12,atol=1e-13)
        truth=row['truth'];stats=truth['stats'];join=curve.join_error(case['scale'])
        np.testing.assert_allclose(join,stats['dense_join_error'],atol=1e-13);assert join<=1e-10
        positive=bool(np.all(curve.minima()>=-1e-12*case['scale']));assert positive==stats['continuous_nonnegative']
        guard=structural(case,values);assert guard==row['guard']
        ledger=physical_ledger(case,curve,y,endpoint)
        for k,v in ledger.items():np.testing.assert_allclose(v,stats[k],rtol=1e-8,atol=1e-13)
        endpoint_error=metrics(case,endpoint[:,None],ref[:,-1:],1e-6);close(endpoint_error,row['endpoint'])
        if positive:
            np.testing.assert_allclose(guard['pool'],truth['output_pool_error'],rtol=1e-12,atol=1e-14)
            for target,entry in truth['targets'].items():
                actual=metrics(case,values,ref,float(target));close(actual,entry['actual'])
                numerical=max(actual.values())<=1
                valid=(guard['nonnegative'] and max(guard['pool'],stats['pool_error'])<=1e-10 and
                    all(stats.get(k,0.)<=1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature')))
                assert bool(numerical)==entry['numerical_pass'] and bool(numerical and valid)==entry['passes']
                name=f"{row['model']}/{row['method']}/constant={row['constant_slow']}/{target}"
                counts[name]['configurations']+=1;counts[name]['passed']+=entry['passes']
        else:assert truth['status']=='rejected'
        if row['method']=='inner':
            full=lambda z:case['system'].S@case['system'].flux(0.,z)
            hermite=Hermite(y,endpoint,full(y),full(endpoint),row['h']);hv=hermite(query)
            for target,actual in row['hermite']['actual'].items():close(metrics(case,hv,ref,float(target)),actual)
            close(physical_ledger(case,InnerCurve([hermite]),y,endpoint),row['hermite']['ledger'])
            # Actual internal ends/midpoints are in the independently audited grid.
            required=np.unique(np.clip(np.round(np.r_[curve.edges/curve.h,(curve.edges[1:]+curve.edges[:-1])/(2*curve.h)],14),0,1))
            assert all(np.min(abs(query-t))<=1e-14 for t in required)
        details.append(dict(model=row['model'],start=row['start'],h=row['h'],threshold=row['threshold'],method=row['method'],
            tolerance=row['tolerance'],constant_slow=row['constant_slow'],seconds=row['seconds'],intervals=row['intervals'],
            strict_error=max(truth.get('targets',{}).get('1e-06',{}).get('actual',{'missing':float('inf')}).values()),
            strict_pass=truth.get('targets',{}).get('1e-06',{}).get('passes',False),physical_ledger_residual=ledger.get('physical_ledger_residual')))
        checked+=1
    order=read(OUT/'order.json');assert order['source_sha256']==orderreg['source_sha256']
    oa=np.load(OUT/'order.npz');a=np.asarray(orderreg['matrix']);y=np.array([.8,.1,.1])
    for i,row in enumerate(order['runs']):
        q=oa[f'{i}-query'];ref=np.column_stack([expm(row['h']*t*a)@y for t in q]);np.testing.assert_allclose(ref,oa[f'{i}-reference'],rtol=1e-12,atol=1e-14)
        for label in ('inner','hermite'):np.testing.assert_allclose(np.max(abs(oa[f'{i}-'+label]-ref)),row[label],rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(np.max(abs(oa[f'{i}-inner'][:,-1]-ref[:,-1])),row['endpoint'],rtol=1e-12,atol=1e-14)
    for field,recorded in order['observed_one_step_orders'].items():
        actual=[float(np.log2(x[field]/z[field])) for x,z in zip(order['runs'][:-1],order['runs'][1:])]
        np.testing.assert_allclose(actual,recorded,rtol=1e-12,atol=1e-14)
    history=read(OUT/'history.json');hreg=read(OUT/'history-registration.json')
    assert history['source_sha256']==hreg['source_sha256']
    for p,sha in hreg['source_sha256'].items():assert digest(ROOT/p)==sha,p
    assert len(history['runs'])==48
    history_summary=defaultdict(lambda:dict(intervals=0,local_pass=0,global_numerical_pass=0))
    history_checked=0
    for row in history['runs']:
        assert row['status']=='success'
        case=cases[row['model']];arrays=np.load(OUT/(row['shard']+'.npz'));key=row['key'];step=row['step']
        q=arrays[key+'-query'];ref=arrays[key+'-reference'];initial=arrays['endpoints'][:,step];final=arrays['endpoints'][:,step+1]
        close(metrics(case,arrays[key+'-loose'],ref,1e-6),row['reference_budget']);assert max(row['reference_budget'].values())<=.05
        if step==0:
            close(metrics(case,arrays['global_loose'],arrays['global_reference'],1e-6),row['global_budget'])
            assert max(row['global_budget'].values())<=.05
        assert row['history_available']==(step>=2)
        for label,record in row['outputs'].items():
            curve=InnerCurve.unpack(arrays[key+'-'+label+'-lengths'],arrays[key+'-'+label+'-coefficients'])
            values=curve(q);np.testing.assert_allclose(values,arrays[key+'-'+label+'-values'],rtol=1e-12,atol=1e-13)
            if label=='history':
                reconstructed=cubic_history(arrays['endpoint_times'][step-2:step+2],arrays['endpoints'][:,step-2:step+2])
                np.testing.assert_allclose(reconstructed(q),values,rtol=1e-12,atol=1e-13)
            positive=bool(np.all(curve.minima()>=-1e-12*case['scale']));assert positive==record['nonnegative']
            guard=structural(case,values);assert guard==record['guard']
            ledger=physical_ledger(case,curve,initial,final) if positive else {};close(ledger,record['ledger'])
            for target,result in record['targets'].items():
                actual=metrics(case,values,ref,float(target)) if positive else None
                global_error=metrics(case,values,arrays['global_reference'][:,arrays[key+'-global_positions']],float(target)) if positive else None
                if positive:close(actual,result['actual']);close(global_error,result['global_error'])
                valid=(positive and guard['nonnegative'] and max(guard['pool'],row['stats']['pool_error'])<=1e-10
                    and all(v<=1e-10 for v in ledger.values()) and all(row['stats'].get(k,0.)<=1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature')))
                passed=bool(valid and max(actual.values())<=1);assert passed==result['passes']
                name=f"{row['shard']}/{label}/{target}";c=history_summary[name];c['intervals']+=1;c['local_pass']+=passed
                c['global_numerical_pass']+=bool(positive and max(global_error.values())<=1)
            history_checked+=1
    for manifest in ('results/m3-single/manifest.json','results/m3-ordered-cost/manifest.json'):
        for p,sha in read(ROOT/manifest).items():assert digest(ROOT/p)==sha,p
    return dict(date='2026-11-08',checks=dict(curves=checked,reference_intervals=len(refs),linear_steps=5,historical_manifests=2,history_curves=history_checked),
        history=dict(history_summary),counts=dict(counts),details=details,failures=failures,observed_one_step_orders=order['observed_one_step_orders'],
        M4_admission=False,conclusion='Raw inner dense output resolves constant-forcing interpolation artifacts, but has only second-order one-step interior error on the tested coupled linear system. It is not a qualified third-order continuous extension.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args();result=verify()
    if args.write:
        save(OUT/'summary.json',result)
        if not (OUT/'environment.json').exists():save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
        files=[p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']
        files += [ROOT/'research'/p for p in ('inner_dense.py','mri_dense.py','test_inner_dense.py','m3_dense_study.py','m3_dense_order.py','m3_dense_report.py','history_output.py','test_history_output.py','m3_history_study.py')]
        files += [ROOT/'docs/m3-inner-output.md']
        save(OUT/'manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})
    else:
        assert result==read(OUT/'summary.json')
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
