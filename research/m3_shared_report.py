"""Read-only reconstruction of stage-sharing and baseline-challenge evidence."""
import argparse
from collections import defaultdict
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import scipy
from scipy.integrate import Radau
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.history_polynomial import HistoryPolynomial
from research.inner_dense import InnerCurve, physical_ledger
from research.m3_ordered_cost import structural
from research.m3_priority import metrics
from research.m3_shadow_study import ROOT, digest

OUT=ROOT/'results/m3-shared-radau'
OLD=ROOT/'results/m3-inner-dense'


def read(p): return json.loads(p.read_text())


def verify_study(folder,cases):
    reg=read(folder/'registration.json')
    for p,sha in (reg['source_sha256']|reg['input_sha256']).items(): assert digest(ROOT/p)==sha,p
    rows=read(folder/'runs.json')['runs']
    expected={(f'history-{i}',m,t,r) for i in reg['paths'] for m in reg['methods']
              for t in reg['tolerances'] for r in (-1,0,1,2)}
    actual={(r['source_shard'],r['method'],r['tolerance'],r['repeat']) for r in rows}
    assert len(rows)==len(expected) and actual==expected
    groups=defaultdict(list);failures=[];passes=defaultdict(int);reconstructed=0
    sources={i:np.load(OLD/f'history-{i}.npz') for i in reg['paths']}
    arrays={i:np.load(folder/f'cost-{i}.npz') for i in reg['paths']}
    for row in rows:
        index=int(row['source_shard'].rsplit('-',1)[1]);a,src=arrays[index],sources[index]
        if row['repeat']>=0:
            for target in ('0.0001','1e-06'):
                groups[f"{index}/{target}/{row['method']}/{row['tolerance']}"].append(row)
        assert np.isfinite(row['seconds']) and row['seconds']>0
        if row['status']=='failed': failures.append(row);continue
        case=cases[row['model']];key=row['key'];y=src['endpoints'][:,4];end=a[key+'-endpoint'];stats=row['stats']
        if row['method'].startswith('full'):
            curve=InnerCurve.unpack(a[key+'-lengths'],a[key+'-coefficients'])
            assert curve.join_error(case['scale'])==stats['dense_join_error']
        else:
            poly=HistoryPolynomial(src['endpoint_times'][:6],np.column_stack([src['endpoints'][:,:5],end]))
            np.testing.assert_array_equal(poly.coefficients,a[key+'-coefficients'])
            curve=InnerCurve([poly]);assert stats['subflows']==3
        np.testing.assert_allclose(curve(np.array([0.,1.])),np.column_stack([y,end]),rtol=2e-13,atol=2e-14)
        values=curve(src['4-query']);np.testing.assert_array_equal(values,a[key+'-values'])
        positive=bool(np.all(curve.minima()>=-1e-12*case['scale']));assert positive==stats['continuous_nonnegative']
        guard=structural(case,values);assert guard==row['guard']
        ledger=physical_ledger(case,curve,y,end)
        for k,v in ledger.items(): np.testing.assert_allclose(v,stats[k],rtol=1e-8,atol=1e-13)
        valid=positive and guard['nonnegative'] and max(guard['pool'],stats['pool_error'])<=1e-10
        valid &= all(stats.get(k,0.)<=1e-10 for k in ('modified_ledger_residual','modified_ledger_quadrature','physical_ledger_residual','physical_ledger_quadrature'))
        for target,record in row['targets'].items():
            error=metrics(case,values,src['4-reference'],float(target)) if positive else None
            if positive:
                assert error.keys()==record['actual'].keys()
                for k,v in error.items(): np.testing.assert_allclose(v,record['actual'][k],rtol=1e-10,atol=1e-12)
            assert record['passes']==bool(valid and max(error.values())<=1)
            passes[target]+=record['passes']
        if 'actual_factorizations' in stats:
            assert stats['actual_factorizations']==stats['nlu']
            assert stats['actual_jacobians']==stats['njev']
            if row['method']!='jacobian': assert stats['lu_cache_misses']==stats['nlu']
        # Cache lookup must not change the solved equation or its numerical output.
        other='jacobian' if row['method']=='shared' and 'jacobian' in reg['methods'] else ('full' if row['method']=='full_shared' else None)
        if other:
            otherkey=f"{other}-{row['tolerance']}-{row['repeat']}"
            if otherkey+'-values' in a:
                np.testing.assert_array_equal(values,a[otherkey+'-values'])
                np.testing.assert_array_equal(end,a[otherkey+'-endpoint'])
        reconstructed+=1
    timing={}
    for key,records in groups.items():
        target=key.split('/')[1]
        seconds=[r['seconds'] for r in records]
        success=[r for r in records if r['status']=='success']
        timing[key]=dict(repeats=len(records),all_pass=all(r['status']=='success' and r['targets'][target]['passes'] for r in records),
                         median_seconds=float(np.median(seconds)),iqr_seconds=float(np.subtract(*np.percentile(seconds,[75,25]))),
                         work={k:sorted(set(r['stats'].get(k,0) for r in success))
                               for k in ('nfev','njev','nlu','accepted_steps','jacobian_seeds','lu_cache_hits')})
    decisions=[]
    for index in reg['paths']:
        for target in ('0.0001','1e-06'):
            selected={}
            for family in ('candidate','full'):
                options=[(k,v) for k,v in timing.items() if k.startswith(f'{index}/{target}/') and v['all_pass']
                         and v['repeats']==3 and k.split('/')[2].startswith('full')==(family=='full')]
                selected[family]=list(min(options,key=lambda pair:pair[1]['median_seconds'])) if options else None
            speedup=selected['full'][1]['median_seconds']/selected['candidate'][1]['median_seconds'] if all(selected.values()) else None
            cost_diagnostic=None
            if all(selected.values()):
                chosen=groups[selected['candidate'][0]]
                baseline=selected['full'][1]['median_seconds']
                cost_diagnostic=dict(
                    rhs_seconds_median=float(np.median([r['stats']['rhs_seconds'] for r in chosen])),
                    jacobian_seconds_median=float(np.median([r['stats']['jacobian_seconds'] for r in chosen])),
                    required_saving_seconds_for_1p5=selected['candidate'][1]['median_seconds']-baseline/1.5,
                    hypothetical_speedup_if_fast_rhs_timer_were_zero=baseline/float(np.median([r['seconds']-r['stats']['rhs_seconds'] for r in chosen])))
            decisions.append(dict(path=index,target=target,selected=selected,speedup_without_estimator_audit_startup=speedup,
                                  local_1p5_screen=bool(speedup is not None and speedup>=1.5),cost_diagnostic=cost_diagnostic))
    for a in list(sources.values())+list(arrays.values()): a.close()
    return dict(records=len(rows),reconstructed_curves=reconstructed,warmups=sum(r['repeat']==-1 for r in rows),
                passes_including_warmups=dict(passes),failures=failures,timing=timing,decisions=decisions)


def verify():
    cases={name:prepare(name) for name in ('teusink','kecoli')}
    result=dict(date='2026-11-08',stage_sharing=verify_study(OUT,cases),baseline_challenge=verify_study(OUT/'fairness',cases),
                M4_admission=False,scope='Development known-history states; three timed repeats, no estimator/audit/startup. No full-window or split-coverage qualification.')
    for p,sha in read(ROOT/'results/m3-history-degree/manifest.json').items(): assert digest(ROOT/p)==sha,p
    for p,sha in read(OLD/'manifest.json').items(): assert digest(ROOT/p)==sha,p
    result['historical_manifests_unchanged']=True
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    result=verify()
    if args.write:
        save(OUT/'summary.json',result)
        if not (OUT/'environment.json').exists():
            save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
                 parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP',
                 radau_source_sha256=digest(Path(inspect.getsourcefile(Radau)))))
        files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json']
        files += [ROOT/'research'/p for p in ('shared_radau.py','test_shared_radau.py','fullstep_radau.py','test_fullstep_radau.py',
                                             'm3_shared_cost.py','m3_shared_fairness.py','m3_shared_report.py')]
        files += [ROOT/'docs/m3-stage-sharing.md']
        save(OUT/'manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})
    else:
        assert result==read(OUT/'summary.json')
        for p,sha in read(OUT/'manifest.json').items(): assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
