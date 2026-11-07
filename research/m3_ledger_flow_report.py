"""Reconstruct ledger diagnosis, extrapolated-flow screens and self-history paths."""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
import scipy
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.mri_kernel import Hermite,GAUSS
from research.inner_dense import InnerCurve,Polynomial,physical_ledger
from research.ledger_diagnostic import split_ledger,flux_integral
from research.extrapolated_flow import ExtrapolatedFlow,SlowPolynomial,unpack
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_priority import metrics
from research.m3_shadow_study import ROOT,digest

DIAG=ROOT/'results/m3-ledger-order';OUT=ROOT/'results/m3-extrapolated-flow';OLD=ROOT/'results/m3-inner-dense'
def read(p):return json.loads(p.read_text())
def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:close(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,(bool,str)) or a is None:assert a==b,(a,b)
    else:np.testing.assert_allclose(a,b,rtol=2e-8,atol=1e-12)
def registration(path):
    reg=read(path)
    for p,sha in (reg['source_sha256']|reg['input_sha256']).items():assert digest(ROOT/p)==sha,p
    return reg


def check_curve(case,curve,y,end,reference,query,stats,guard,targets):
    np.testing.assert_allclose(curve(np.array([0.,1.])),np.column_stack([y,end]),rtol=2e-12,atol=1e-12)
    assert curve.join_error(case['scale'])<=1e-10
    positive=bool(np.all(curve.minima()>=-1e-12*case['scale']))
    assert positive==stats['continuous_nonnegative']
    if positive:
        for k,v in physical_ledger(case,curve,y,end).items():close(v,stats[k])
    values=curve(query);newguard,newtargets=qualified(case,curve,values,reference,stats)
    close(newguard,guard);close(newtargets,targets)
    return values


def modified(case,kernel,curve,y,end,forcing,stats):
    if kernel.weights is None:return
    sums=[0.,0.];delta=0.;left=0.
    for p in curve.pieces:
        ints=[]
        for n in (4,8):
            q=(GAUSS[n][0]+1)/2;vals=[]
            for t,z in zip(left+p.h*q,p(q).T):
                vals.append(kernel.weights@(kernel.sf@kernel.system.flux(0.,z)[kernel.cols]+forcing.value(t)))
            ints.append(float(p.h/2*(GAUSS[n][1]@vals)))
        sums[0]+=ints[0];sums[1]+=ints[1];delta+=abs(ints[1]-ints[0]);left+=p.h
    den=max(1.,abs(float(kernel.weights@y)))
    close(abs(float(kernel.weights@(end-y))-sums[1])/den,stats['modified_ledger_residual'])
    close(delta/den,stats['modified_ledger_quadrature'])


def diagnostic(case):
    reg=registration(DIAG/'registration.json');data=read(DIAG/'runs.json');rows=data['runs'];assert len(rows)==144
    refs={(r['state'],r['h_index']):r for r in data['references']};assert len(refs)==18
    series={};failures=[]
    for state in reg['states']:
        with np.load(DIAG/f'{state}.npz') as a:
            y=a['initial'];query=a['query'];weights=case['options']['ledger_weights'];den=max(1.,abs(float(weights@y)))
            for hi,h in enumerate(reg['h_values']):
                ref=a[f'{hi}-reference'];loose=a[f'{hi}-loose'];r=refs[state,hi]
                close(metrics(case,loose,ref,1e-6),r['budget'])
                ri={n:flux_integral(case,ref[:,np.searchsorted(query,(GAUSS[n][0]+1)/2)],h,n) for n in (8,16)}
                rr=abs(float(weights@(ref[:,-1]-y))-ri[16])/den;rq=abs(ri[16]-ri[8])/den
                close(rr,r['ledger_residual']);close(rq,r['ledger_quadrature'])
                assert r['reference_ok']==(max(r['budget'].values())<=.05 and max(rr,rq)<=1e-12)
                oracle=Hermite(y,ref[:,-1],case['system'].S@case['system'].flux(0.,y),case['system'].S@case['system'].flux(0.,ref[:,-1]),h)
                for row in [v for v in rows if v['state']==state and v['h_index']==hi]:
                    if row['status']!='success':failures.append(row);continue
                    k=row['key'];end=a[k+'-endpoint'];curve=InnerCurve([Polynomial(*a[k+'-coefficients'],h)])
                    close(physical_ledger(case,InnerCurve([oracle]),y,ref[:,-1]),row['oracle_hermite_ledger'])
                    values=curve.pieces[0](query);np.testing.assert_allclose(values,a[k+'-values'],rtol=2e-13,atol=2e-14)
                    close(structural(case,values),row['guard'])
                    assert bool(np.all(curve.minima()>=-1e-12*case['scale']))==row['stats']['continuous_nonnegative']
                    for name,val in physical_ledger(case,curve,y,end).items():close(val,row['stats'][name])
                    ci=flux_integral(case,values[:,np.searchsorted(query,(GAUSS[16][0]+1)/2)],h,16)
                    d=split_ledger(weights,y,end,ref[:,-1],ci,ri[16]);close(d,row['decomposition'])
                    assert abs(d['decomposition_roundoff'])<1e-13
                    _,target=qualified(case,curve,values,ref,row['stats'])
                    for t,v in target.items():
                        v['passes'] &= r['reference_ok']
                        v['endpoint']=metrics(case,end[:,None],ref[:,-1:],float(t))
                    close(target,row['targets'])
                    if row['tolerance']==1e-12:
                        other=next(v for v in rows if v['state']==state and v['h_index']==hi and v['partition']==row['partition'] and v['method']==row['method'] and v['tolerance']==1e-10)
                        inner=float(np.max(abs(a[other['key']+'-endpoint']-end)/case['scale'])) if other['status']=='success' else None
                        referr=float(np.max(abs(loose[:,-1]-ref[:,-1])/case['scale']))
                        floor=None if inner is None else inner+referr+10*np.finfo(float).eps*float(np.max(abs(end)/case['scale']))
                        series.setdefault(f"{state}/{row['partition']}/{row['method']}",[]).append(dict(h=h,
                            endpoint_scaled_error=row['endpoint_scaled_error'],endpoint_numerical_floor=floor,
                            **d,oracle_ledger=row['oracle_hermite_ledger']['physical_ledger_residual'],reference_ok=r['reference_ok'],
                            strict_pass=row['targets']['1e-06']['passes']))
    for records in series.values():
        for i,r in enumerate(records):
            r['observed_endpoint_order']=None
            if i and all(x['reference_ok'] and x['endpoint_numerical_floor'] is not None and x['endpoint_scaled_error']>10*x['endpoint_numerical_floor'] for x in (records[i-1],r)):
                r['observed_endpoint_order']=float(np.log2(records[i-1]['endpoint_scaled_error']/r['endpoint_scaled_error']))
    return dict(records=144,references=18,unresolved_reference_quadrature=[r for r in refs.values() if not r['reference_ok']],series=series,failures=failures)


def local_screen(cases):
    reg=registration(OUT/'registration.json');rows=read(OUT/'runs.json')['runs'];assert len(rows)==144
    counts={};failures=[]
    for index in reg['paths']:
        with np.load(OLD/f'history-{index}.npz') as src,np.load(OUT/f'path-{index}.npz') as a:
            subset=[r for r in rows if r['path']==index];case=cases[subset[0]['model']];y=src['endpoints'][:,4]
            kernel=ExtrapolatedFlow(case,src['groups']);slow=np.column_stack([kernel.slow(z) for z in src['endpoints'][:,:4].T])
            np.testing.assert_array_equal(slow,a['past_slow'])
            for row in subset:
                if row['status']!='success':failures.append(row);continue
                key=row['key'];end=a[key+'-endpoint'];extra=row['method']=='extrapolated'
                curve=(unpack if extra else InnerCurve.unpack)(a[key+'-lengths'],a[key+'-coefficients'])
                values=check_curve(case,curve,y,end,src['4-reference'],src['4-query'],row['stats'],row['guard'],row['targets'])
                np.testing.assert_array_equal(values,a[key+'-values'])
                if extra:
                    q=row['order'];forcing=SlowPolynomial(src['endpoint_times'][5-q:5],np.column_stack([slow[:,5-q:4],kernel.slow(y)]),row['h'])
                    np.testing.assert_array_equal(forcing.coefficients,a[key+'-forcing']);assert row['stats']['subflows']==1
                    modified(case,kernel,curve,y,end,forcing,row['stats'])
                name=f"{index}/{row['method']}/{row['order']}"
                c=counts.setdefault(name,dict(records=0,strict_passes=0));c['records']+=1;c['strict_passes']+=row['targets']['1e-06']['passes']
            kernel.close()
    return dict(records=144,counts=counts,failures=failures)


def paths(cases):
    reg=registration(OUT/'path-registration.json');data=read(OUT/'paths.json');rows=data['runs'];assert len(rows)==144
    decisions=[]
    for index in reg['paths']:
        source=next(r for r in rows if r['path']==index);case=cases[source['model']];h=source['h'];duration=16*h
        with np.load(OUT/f'path-reference-{index}.npz') as base:
            origin=base['initial'];grid=base['times'];ref=base['reference'];budget=metrics(case,base['loose'],ref,1e-6)
            assert max(budget.values())<=.05
            for row in [v for v in data['baselines'] if v['path']==index]:
                if row['status']!='success':continue
                k=row['key'];curve=InnerCurve.unpack(base[k+'-lengths'],base[k+'-coefficients'])
                check_curve(case,curve,origin,base[k+'-endpoint'],ref,grid/duration,row['stats'],row['guard'],row['targets'])
            for order in reg['orders']:
                subset=[v for v in rows if v['path']==index and v['order']==order]
                summary=next(v for v in data['trajectories'] if v['path']==index and v['order']==order)
                kernel=ExtrapolatedFlow(case,base['groups']);slow=[];pieces=[]
                with np.load(OUT/f'trajectory-{index}-{order}.npz') as a:
                    for row in subset:
                        assert row['status']=='success'
                        step=row['step'];k=str(step);y=a['endpoints'][:,step];end=a[k+'-endpoint']
                        np.testing.assert_array_equal(end,a['endpoints'][:,step+1])
                        curve=(InnerCurve.unpack if row['startup'] else unpack)(a[k+'-lengths'],a[k+'-coefficients'])
                        budget=metrics(case,a[k+'-loose'],a[k+'-reference'],1e-6);close(budget,row['reference_budget']);assert max(budget.values())<=.05
                        check_curve(case,curve,y,end,a[k+'-reference'],a[k+'-query'],row['stats'],row['guard'],row['targets'])
                        if not row['startup']:
                            forcing=SlowPolynomial(a['endpoint_times'][step-order+1:step+1],np.column_stack([*slow[-(order-1):],kernel.slow(y)]),h)
                            np.testing.assert_array_equal(forcing.coefficients,a[k+'-forcing'])
                            modified(case,kernel,curve,y,end,forcing,row['stats']);slow.append(forcing.coefficients[:,0])
                        else:slow.append(kernel.slow(y))
                        pieces.extend(curve.pieces)
                    curve=InnerCurve(pieces);values=curve(grid/duration)
                    for k,v in physical_ledger(case,curve,origin,a['endpoints'][:,-1]).items():close(v,summary['stats'][k])
                    guard,targets=qualified(case,curve,values,ref,summary['stats']);close(guard,summary['guard'])
                    for t,v in targets.items():v['passes'] &= all(r['targets'][t]['passes'] for r in subset)
                    close(targets,summary['targets']);close(sum(r['seconds'] for r in subset),summary['seconds'])
                kernel.close()
                for t in ('0.0001','1e-06'):
                    valid=[r for r in data['baselines'] if r['path']==index and r['status']=='success' and r['targets'][t]['passes']]
                    best=min(valid,key=lambda r:r['seconds']) if valid else None
                    decisions.append(dict(path=index,order=order,target=t,passes=summary['targets'][t]['passes'],local_passes=summary['local_passes'][t],
                        actual=summary['targets'][t]['actual'],physical_ledger=summary['stats'].get('physical_ledger_residual'),
                        candidate_seconds=summary['seconds'],baseline_key=best['key'] if best else None,baseline_seconds=best['seconds'] if best else None,
                        unaudited_single_run_ratio=best['seconds']/summary['seconds'] if best else None))
    return dict(steps=144,trajectories=9,uninterrupted_controls=len(data['baselines']),decisions=decisions)


def verify():
    cases={n:prepare(n) for n in ('teusink','kecoli')}
    result=dict(date='2026-11-08',ledger_diagnosis=diagnostic(cases['teusink']),local_screen=local_screen(cases),
                self_history=paths(cases),M4_admission=False,scope='Development only; correctness progress with all original guards, no full registered-window or formal cost qualification.')
    for p,sha in read(ROOT/'results/m3-shared-radau/manifest.json').items():assert digest(ROOT/p)==sha,p
    for p,sha in read(OLD/'manifest.json').items():assert digest(ROOT/p)==sha,p
    result['historical_manifests_unchanged']=True
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args();result=verify()
    if args.write:
        save(OUT/'summary.json',result)
        if not (OUT/'environment.json').exists():save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
            parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
        files=[p for folder in (OUT,DIAG) for p in folder.iterdir() if p.is_file() and p.name!='manifest.json']
        files += [ROOT/'research'/p for p in ('ledger_diagnostic.py','test_ledger_diagnostic.py','m3_ledger_order.py','extrapolated_flow.py',
            'test_extrapolated_flow.py','m3_extrapolated_study.py','m3_extrapolated_paths.py','m3_ledger_flow_report.py')]
        files += [ROOT/'docs/m3-extrapolated-flow.md']
        save(OUT/'manifest.json',{p.relative_to(ROOT).as_posix():digest(p) for p in files})
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
