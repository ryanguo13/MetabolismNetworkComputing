"""Replay inverse projections, adaptive proposals and their mandatory audits."""
import argparse
import json
import platform
import subprocess
import sys
import zipfile
import numpy as np
import scipy
from scipy.linalg import expm
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.affine_partition import freeze,displacement_reference
from research.controlled_rational import RationalBasis,proposal
from research.rational_filter import filtered_proposal
from research.m3_growth_study import audit
from research.m3_priority import STATES,metrics
from research.m3_ordered_cost import structural
from research.m3_ledger_flow_report import registration,close
from research.m3_collocation_report import timeless
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-controlled-rational';FILTER=ROOT/'results/m3-filtered-rational'
def read(p):return json.loads(p.read_text())


def verify_arrays(directory,key,result):
    with np.load(directory/f'{key}.npz') as a:
        for k in ('projection','drive','coefficients'):np.testing.assert_array_equal(a[k],result[k])


def verify_basis(directory,key,basis):
    with np.load(directory/f'basis-{key}.npz') as a:
        for k,v in [('vectors',basis.vectors),('images',basis.images),('av',basis.av)]:np.testing.assert_array_equal(a[k],np.column_stack(v))


def projected_check(basis,result,query,method):
    """Independent pointwise expm and K solve, no semigroup propagation."""
    v=result['basis'];m=v.shape[1];Q=v.T@np.column_stack(basis.images)
    expected=v.T@(basis.a@v) if method=='direct' else (np.eye(m)-np.linalg.inv(Q))/(basis.shift*basis.h)
    np.testing.assert_allclose(result['projection'],expected,rtol=1e-10,atol=1e-7)
    block=np.zeros((m+1,m+1));block[:m,:m]=basis.h*result['projection'];block[:m,-1]=basis.h*result['drive']
    coef=np.column_stack([(expm(t*block)@np.r_[np.zeros(m),1.])[:m] for t in query])
    np.testing.assert_allclose(coef,result['coefficients'],rtol=2e-9,atol=3e-11)
    # Full-space residual from differentiated displacement, no concentration
    # offset subtraction, and independent shifted inverse evaluation.
    x=v@result['coefficients'];dx=v@(result['projection']@result['coefficients']+result['drive'][:,None])
    residual=dx-basis.a@x-basis.b[:,None]
    # Direct full-space and factored residuals sum strongly cancelling terms
    # in different orders. Bound that arithmetic discrepancy componentwise
    # and separately require it to consume <.01 of the strict state scale.
    k=basis.n+3*m+10;eps=np.finfo(float).eps;gamma=k*eps/(1-k*eps)
    magnitude=abs(v)@(abs(result['projection'])@abs(result['coefficients'])+abs(result['drive'])[:,None])+abs(basis.a)@(abs(v)@abs(result['coefficients']))+abs(basis.b)[:,None]
    difference=abs(residual-result['residual'])
    assert np.all(difference<=8*gamma*magnitude+1e-14)
    denominator=1e-6*(.01+np.maximum(abs(basis.y)/basis.scale,1.))
    arithmetic_score=float(np.max(basis.h*difference/denominator[:,None]))
    assert arithmetic_score<.01
    return arithmetic_score


def verify():
    for p,sha in read(ROOT/'results/m3-merb-predictor/manifest.json').items():assert digest(ROOT/p)==sha,p
    reg=registration(OUT/'registration.json');fr=registration(FILTER/'registration.json');data=read(OUT/'runs.json');fd=read(FILTER/'runs.json')
    assert len(data['runs'])==176 and len(data['references'])==8 and len(data['decisions'])==32
    assert len(fd['paths'])==32 and len(fd['runs'])==150
    conclusions=[];references=[];max_filter_difference=0.;max_residual_arithmetic=0.
    for rr in data['references']:
        name=rr['model'];case=prepare(name);h=rr['h'];key=rr['key'];si=rr['state_index'];start=STATES[name][si]
        with np.load(OUT/f'reference-{key}.npz') as a:
            y=a['initial'];q=a['query'];stored={k:a[k] for k in ('reference','loose','independent')}
        np.testing.assert_array_equal(y,np.load(ROOT/'results/m3-priority/p0.npz')[name+'-prefix'][:,si])
        public=(case['times'][(case['times']>start)&(case['times']<start+h)]-start)/h
        np.testing.assert_array_equal(q,np.unique(np.r_[np.linspace(0,1,65),public]))
        frozen,ctx=freeze(case,y)
        ref,work=displacement_reference(ctx['jacobian'],lambda t:ctx['rhs'],y,h,q,case['scale'],1e-12)
        loose,lo_work=displacement_reference(ctx['jacobian'],lambda t:ctx['rhs'],y,h,q,case['scale'],1e-11)
        independent,ib,_=audit(frozen,y,q,h);budget=metrics(frozen,loose,ref,1e-6);cross=metrics(frozen,independent,ref,1e-6)
        for k,v in [('reference',ref),('loose',loose),('independent',independent)]:np.testing.assert_array_equal(v,stored[k])
        close(budget,rr['budget']);close(ib,rr['independent_budget']);close(cross,rr['cross_budget']);close(timeless(work),timeless(rr['affine_work']));close(timeless(lo_work),timeless(rr['affine_loose_work']))
        assert max([*budget.values(),*ib.values(),*cross.values()])<=.05 and rr['reference_ok']
        a=ctx['jacobian'].multiply(case['scale'][None,:]).multiply(1/case['scale'][:,None]).tocsr();mu=float(np.max(a.diagonal()+np.asarray(abs(a).sum(axis=1)).ravel()-abs(a.diagonal())))
        references.append(dict(key=key,model=name,state_index=si,h=h,reference_minimum=float(ref.min()),reference_sampled_guard=structural(frozen,ref),h_log_norm=h*mu))
        dimensions=sorted(set(min(len(y),d) for d in reg['dimensions']))
        for shift in reg['shifts']:
            seq=f'{key}-{shift}';basis=RationalBasis(ctx['jacobian'],ctx['rhs'],y,case['scale'],h,shift);previous={m:None for m in reg['projections']};selected={m:None for m in reg['projections']}
            subset=[r for r in data['runs'] if r['sequence']==seq]
            assert {(r['dimension_requested'],r['method']) for r in subset}=={(d,m) for d in dimensions for m in reg['projections']}
            for dim in dimensions:
                basis.extend(dim)
                for method in reg['projections']:
                    ck=f'{seq}-{dim}-{method}';row=next(r for r in subset if r['key']==ck);assert row['status']=='success'
                    r=basis.evaluate(q,method);close(timeless(r['stats']),timeless(row['stats']));verify_arrays(OUT,ck,r)
                    indicator=proposal(r,previous[method],h,case['scale'],y,budget=reg['budget']);close(indicator,row['indicator'])
                    guard=structural(frozen,r['values']);actual=metrics(frozen,r['values'],ref,1e-6);close(guard,row['guard']);close(actual,row['actual'])
                    offered=bool(indicator['proposed'] and guard['nonnegative'] and guard['pool']<=1e-10);auditok=bool(max(actual.values())<=1 and guard['nonnegative'] and guard['pool']<=1e-10)
                    assert offered==row['offered'] and auditok==row['audit_ok'] and row['unsafe_proposal']==(offered and not auditok)
                    close(h*mu,row['h_log_norm']);close(row['optimistic_seconds'],rr['formation_seconds']+row['stats']['basis_seconds']+row['evaluation_cumulative_seconds'])
                    assert r['stats']['nlu']==1 and r['stats']['solves']==r['stats']['dimension']
                    if offered and selected[method] is None:selected[method]=ck
                    previous[method]=r
            verify_basis(OUT,seq,basis)
            for method in reg['projections']:
                decision=next(d for d in data['decisions'] if d['sequence']==seq and d['method']==method);assert decision['selected_key']==selected[method]
                expected='fallback_required' if selected[method] is None else ('audited_linear_samples_only' if next(r for r in subset if r['key']==selected[method])['audit_ok'] else 'audit_veto')
                assert decision['outcome']==expected and not decision['M4_eligible']
        for shift in fr['shifts']:
            for method in fr['projections']:
                pathkey=f'{key}-{shift}-{method}';path=next(p for p in fd['paths'] if p['key']==pathkey);rows=[r for r in fd['runs'] if r['path']==pathkey]
                basis=RationalBasis(ctx['jacobian'],ctx['rhs'],y,case['scale'],h,shift);previous=None;chosen=None
                assert len(rows)==path['attempts'] and [r['dimension_requested'] for r in rows]==dimensions[:len(rows)]
                for row in rows:
                    assert chosen is None and row['status']=='success';basis.extend(row['dimension_requested']);r=basis.evaluate(q,method);close(timeless(r['stats']),timeless(row['stats']));verify_arrays(FILTER,row['key'],r)
                    raw=proposal(r,previous,h,case['scale'],y,budget=fr['budget']);close(raw,row['raw_indicator'])
                    filtered,values=filtered_proposal(r,np.column_stack(basis.images),ctx['rhs']/case['scale'],h,shift,case['scale'],y,raw['enrichment_indicator'],budget=fr['budget'],safety=fr['safety']);close(filtered,row['filtered_indicator'])
                    explicit=h*basis.solve(r['residual']);difference=float(np.max(abs(values-explicit)));max_filter_difference=max(max_filter_difference,difference)
                    np.testing.assert_allclose(values,explicit,rtol=2e-6,atol=2e-10)
                    guard=structural(frozen,r['values']);actual=metrics(frozen,r['values'],ref,1e-6);close(guard,row['guard']);close(actual,row['actual'])
                    offered=bool(filtered['proposed'] and guard['nonnegative'] and guard['pool']<=1e-10);auditok=bool(max(actual.values())<=1 and guard['nonnegative'] and guard['pool']<=1e-10)
                    assert row['offered']==offered and row['audit_ok']==auditok
                    if offered:chosen=row['key']
                    previous=r
                # Pointwise independent exponentiation at final selected or
                # rejected dimension, covering every final path output.
                max_residual_arithmetic=max(max_residual_arithmetic,projected_check(basis,r,q,method));verify_basis(FILTER,pathkey,basis)
                ar,ab,_=audit(frozen,y,q,h);ac=metrics(frozen,ar,ref,1e-6);close(ab,path['reference_budget']);close(ac,path['cross_budget']);assert max([*ab.values(),*ac.values()])<=.05 and path['reference_ok']
                assert chosen==path['selected_key'] and not path['M4_eligible']
                selected=next((v for v in rows if v['key']==chosen),None);outcome='fallback_required' if selected is None else ('audited_linear_samples_only' if selected['audit_ok'] else 'audit_veto');assert path['outcome']==outcome
                close(path['candidate_seconds']+path['audit_seconds'],path['total_candidate_and_audit_seconds'])
                conclusions.append(dict(key=pathkey,model=name,state_index=si,h=h,shift=shift,method=method,outcome=outcome,dimension=None if selected is None else selected['stats']['dimension'],actual=None if selected is None else selected['actual'],candidate_seconds=path['candidate_seconds'],audit_seconds=path['audit_seconds'],total_seconds=path['total_candidate_and_audit_seconds']))
        print('verified',name,si,h,flush=True)
    # Counterexample is independent of models and makes the certification limit
    # explicit even when all observed model proposals happen to pass.
    n=32;a=-32*np.eye(n)+96*np.diag(np.ones(n-1),1);r=np.zeros(n);r[-1]=1.;block=np.zeros((n+1,n+1));block[:n,:n]=a;block[:n,-1]=r
    actual=(expm(block)@np.r_[np.zeros(n),1.])[:n];filtered=np.linalg.solve(np.eye(n)-.25*a,r);ratio=float(np.max(abs(actual))/np.max(abs(filtered)));assert ratio>fr['safety']
    crc=0
    for directory in (OUT,FILTER):
        for p in directory.glob('*.npz'):
            with zipfile.ZipFile(p) as z:assert z.testzip() is None,p
            crc+=1
    return dict(raw_records=len(data['runs']),raw_decisions={k:sum(d['outcome']==k for d in data['decisions']) for k in ('audited_linear_samples_only','fallback_required','audit_veto')},raw_unsafe_proposals=sum(r['unsafe_proposal'] for r in data['runs']),filtered_attempts=len(fd['runs']),filtered_paths=conclusions,filtered_decisions={k:sum(p['outcome']==k for p in fd['paths']) for k in ('audited_linear_samples_only','fallback_required','audit_veto')},references=references,filtered_formula_max_difference=max_filter_difference,residual_arithmetic_strict_max=max_residual_arithmetic,stable_nonnormal_counterexample_ratio=ratio,npz_crc_checked=crc,historical_manifest_unchanged=True,M4_admission=False,
        conclusion='Inverse-Ritz and shift-filtered residual reduce some overconservative expansions, but remain heuristic: stable nonnormal example defeats safety4. Earlier small-model affine truth is negative. Retain full audits and explicit fallback-required outcomes. No nonlinear/continuous-output/coverage/formal cost gate satisfied.')


def write(result):
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve evidence')
    save(OUT/'summary.json',result);save(OUT/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
    files=[p for directory in (OUT,FILTER) for p in directory.iterdir() if p.is_file() and p.name!='manifest.json']
    files += [ROOT/'research'/p for p in ('controlled_rational.py','test_controlled_rational.py','rational_filter.py','test_rational_filter.py','m3_controlled_rational_study.py','m3_filtered_rational_study.py','m3_controlled_rational_report.py')]
    files += [ROOT/'docs/m3-controlled-rational.md']
    save(OUT/'manifest.json',{str(p.relative_to(ROOT)):digest(p) for p in files})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args();result=verify()
    if args.write:write(result)
    else:
        close(result,read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items():assert digest(ROOT/p)==sha,p
    print(json.dumps(result,indent=2))
