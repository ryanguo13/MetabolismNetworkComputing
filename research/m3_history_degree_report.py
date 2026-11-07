"""Independent verification of higher-degree history and guarded cost."""
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
from research.history_polynomial import HistoryPolynomial
from research.inner_dense import InnerCurve,physical_ledger
from research.m3_priority import metrics
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest
OUT=ROOT/'results/m3-history-degree';OLD=ROOT/'results/m3-inner-dense'


def read(p):return json.loads(p.read_text())

def close(a,b):
    assert a.keys()==b.keys()
    for k in a:np.testing.assert_allclose(a[k],b[k],rtol=2e-8,atol=1e-10)


def verify():
    for file in ('registration.json','cost-registration.json'):
        reg=read(OUT/file)
        for p,sha in (reg['source_sha256']|reg.get('evidence_sha256',{})|reg.get('input_sha256',{})).items():assert digest(ROOT/p)==sha,p
    cases={n:prepare(n) for n in ('teusink','kecoli')};old={r['shard']+'/'+r['key']:r for r in read(OLD/'history.json')['runs']}
    rows=read(OUT/'runs.json')['runs'];assert len(rows)==90
    counts=defaultdict(lambda:dict(intervals=0,passes=0,global_numerical_pass=0,max_physical_ledger=0.))
    for row in rows:
        index=row['source_shard'].rsplit('-',1)[1];a=np.load(OUT/(index+'.npz'));src=np.load(OLD/(row['source_shard']+'.npz'))
        case=cases[row['model']];step=row['step'];degree=row['degree'];key=row['key'];original=old[row['source_shard']+'/'+str(step)]
        p=HistoryPolynomial(src['endpoint_times'][step-degree+1:step+2],src['endpoints'][:,step-degree+1:step+2])
        np.testing.assert_allclose(p.coefficients,a[key+'-coefficients'],rtol=1e-12,atol=1e-14)
        q=src[f'{step}-query'];values=p(q);np.testing.assert_allclose(values,a[key+'-values'],rtol=1e-12,atol=1e-13)
        positive=bool(np.all(p.minima()>=-1e-12*case['scale']));assert positive==row['positive']
        guard=structural(case,values);assert guard==row['guard']
        ledger=physical_ledger(case,InnerCurve([p]),src['endpoints'][:,step],src['endpoints'][:,step+1]) if positive else {}
        close(ledger,row['ledger'])
        valid=positive and guard['nonnegative'] and max(guard['pool'],original['stats']['pool_error'])<=1e-10
        valid &= all(v<=1e-10 for v in ledger.values())
        valid &= all(original['stats'].get(k,0.)<=1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature'))
        for target,record in row['targets'].items():
            actual=metrics(case,values,src[f'{step}-reference'],float(target)) if positive else None
            global_error=metrics(case,values,src['global_reference'][:,src[f'{step}-global_positions']],float(target)) if positive else None
            if positive:close(actual,record['actual']);close(global_error,record['global_error'])
            assert record['passes']==bool(valid and max(actual.values())<=1)
            if row['common_interval']:
                name=f"{row['source_shard']}/degree{degree}/{target}";c=counts[name];c['intervals']+=1;c['passes']+=record['passes']
                c['global_numerical_pass']+=bool(positive and max(global_error.values())<=1)
                c['max_physical_ledger']=max(c['max_physical_ledger'],ledger.get('physical_ledger_residual',0))
    cost=read(OUT/'cost.json')['runs'];assert len(cost)==96;groups=defaultdict(list);failures=[]
    for row in cost:
        if row['status']=='failed':failures.append(row);continue
        index=row['source_shard'].rsplit('-',1)[1];a=np.load(OUT/f'cost-{index}.npz');src=np.load(OLD/(row['source_shard']+'.npz'))
        case=cases[row['model']];key=row['key'];q=src['4-query'];y=src['endpoints'][:,4];end=a[key+'-endpoint']
        if row['method']=='endpoint':
            p=HistoryPolynomial(src['endpoint_times'][:6],np.column_stack([src['endpoints'][:,:5],end]));curve=InnerCurve([p])
            np.testing.assert_allclose(p.coefficients,a[key+'-coefficients'],rtol=1e-12,atol=1e-14)
            assert row['stats']['subflows']==3
        else:curve=InnerCurve.unpack(a[key+'-lengths'],a[key+'-coefficients'])
        values=curve(q);np.testing.assert_allclose(values,a[key+'-values'],rtol=1e-12,atol=1e-13)
        guard=structural(case,values);assert guard==row['guard']
        positive=bool(np.all(curve.minima()>=-1e-12*case['scale']));assert positive==row['stats']['continuous_nonnegative']
        ledger=physical_ledger(case,curve,y,end)
        for k,v in ledger.items():np.testing.assert_allclose(v,row['stats'][k],rtol=1e-8,atol=1e-13)
        valid=positive and guard['nonnegative'] and max(guard['pool'],row['stats']['pool_error'])<=1e-10
        valid &= all(row['stats'].get(k,0.)<=1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature'))
        for target,record in row['targets'].items():
            actual=metrics(case,values,src['4-reference'],float(target)) if positive else None
            if positive:close(actual,record['actual'])
            assert record['passes']==bool(valid and max(actual.values())<=1)
            if row['repeat']>=0:groups[f"{row['source_shard']}/{target}/{row['method']}/{row['tolerance']}"].append(row)
    timing={k:dict(repeats=len(v),all_pass=all(r['targets'][k.split('/')[1]]['passes'] for r in v),
        median_seconds=float(np.median([r['seconds'] for r in v])),iqr_seconds=float(np.subtract(*np.percentile([r['seconds'] for r in v],[75,25])))) for k,v in groups.items()}
    decision=[]
    for path in range(6):
        for target in ('0.0001','1e-06'):
            prefix=f'history-{path}/{target}/'
            selected={}
            for method in ('endpoint','full'):
                options=[(k,v) for k,v in timing.items() if k.startswith(prefix+method+'/') and v['all_pass'] and v['repeats']==3]
                selected[method]=list(min(options,key=lambda kv:kv[1]['median_seconds'])) if options else None
            speedup=selected['full'][1]['median_seconds']/selected['endpoint'][1]['median_seconds'] if all(selected.values()) else None
            decision.append(dict(path=path,target=target,selected=selected,unaudited_local_speedup=speedup,
                reaches_1p5_development_screen=bool(speedup is not None and speedup>=1.5)))
    for p,sha in read(OLD/'manifest.json').items():assert digest(ROOT/p)==sha,p
    return dict(date='2026-11-08',checks=dict(reanalysis_curves=90,cost_records=96,historical_manifest_unchanged=True),
        common_interval_degree_comparison=dict(counts),timing=timing,decision=decision,failures=failures,M4_admission=False,
        scope='Development only: free computed history and cached setup; neither mandatory audit nor startup cost included. No new independent trajectory validation.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args();result=verify()
    if args.write:
        save(OUT/'summary.json',result)
        if not (OUT/'environment.json').exists():save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
        files=[p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']
        files += [ROOT/'research'/p for p in ('history_polynomial.py','test_history_polynomial.py','mri_endpoint.py','test_mri_endpoint.py','m3_history_degree.py','m3_history_cost.py','m3_history_degree_report.py')]
        files += [ROOT/'docs/m3-history-cost.md']
        save(OUT/'manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})
    else:
        assert result==read(OUT/'summary.json')
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
