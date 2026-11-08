"""Registered one-factor Rodas continuous-extension correctness/cost screen."""
import itertools
import time
import numpy as np
from scipy.linalg import expm
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.rodas_coupling import base,coupling,moments,targets
from research.rodas_centered import CenteredRodas as RodasContinuous
from research.test_rodas_continuous import order_diagnostic
from research.extrapolated_flow import pack
from research.inner_dense import full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_growth_study import audit
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-rodas-centered'


def algebra_diagnostic():
    a,g,b,*_=base();target=targets();m=moments(a,g);cubic=coupling(False)[3]
    rows=[];two=[]
    for count in (2,3):
        for nodes in itertools.combinations([.125,.25,.375,.5,.625,.75,.875,1.],count):
            for mode in (('cubic',) if count==2 else ('cubic','simple')):
                A=np.zeros((6+count,6+count));G=np.eye(6+count)*.25;A[:6,:6]=a;G[:6,:6]=g
                for k,x in enumerate(nodes):
                    if mode=='cubic':A[6+k,:6]=cubic@np.array([x,x*x,x*x*x])
                    else:A[6+k,k]=x
                M=moments(A,G);mu=np.linalg.lstsq(M,target,rcond=None)[0];mu[:,-1]+=np.r_[b,np.zeros(count)]-mu.sum(1)
                residual=float(np.max(abs(M@mu-target)));rank=int(np.linalg.matrix_rank(M));valid=residual<=1e-10
                row=dict(nodes=nodes,mode=mode,rank=rank,residual=residual,valid=valid,score=float(np.linalg.norm(mu,ord=np.inf)),condition=float(np.linalg.cond(M)))
                (two if count==2 else rows).append(row)
    best=min([r for r in rows if r['valid'] and r['rank']==8],key=lambda r:r['score'])
    a9,g9,b9,mu=coupling();assert list(best['nodes'])==[.25,.875,1.] and best['mode']=='cubic'
    return dict(base_rank=int(np.linalg.matrix_rank(m)),base_target_rank=int(np.linalg.matrix_rank(np.c_[m,target])),base_least_squares_residual=float(np.max(abs(m@np.linalg.lstsq(m,target,rcond=None)[0]-target))),two_stage_candidates=two,three_stage_candidates=rows,selected=best,alpha=a9.tolist(),gamma=g9.tolist(),weights=b9.tolist(),mu=mu.tolist(),residual=float(np.max(abs(moments(a9,g9)@mu-target))))


def linear_maps(fast,slow,h,query,augmented,whole):
    a,g,b,mu=coupling(augmented);n=len(fast);j=fast+slow;identity=np.eye(n);inc=[];ks=[]
    for i in range(len(b)):
        state=identity+sum((a[i,k]*inc[k] for k in range(i)),start=np.zeros((n,n)));prev=sum((g[i,k]*inc[k] for k in range(i)),start=np.zeros((n,n)))
        ki=np.linalg.solve(identity-h*g[i,i]*j,h*j@(state+prev));inc.append(ki);ks.append(h*slow@(state+prev+g[i,i]*ki))
    if whole:return [identity+sum(((mu@np.array([x**(k+1) for k in range(mu.shape[1])]))[i]*inc[i] for i in range(len(b))),start=np.zeros((n,n))) for x in query]
    degree=mu.shape[1];block=np.zeros(((degree+1)*n,(degree+1)*n));block[:n,:n]=h*fast
    for k in range(degree):block[k*n:(k+1)*n,(k+1)*n:(k+2)*n]=identity
    import math
    polys=[(k+1)*sum((mu[i,k]*ks[i] for i in range(len(b))),start=np.zeros((n,n))) for k in range(degree)]
    initial=np.vstack([identity,*[math.factorial(k)*polys[k] for k in range(degree)]])
    return [expm(x*block)[:n]@initial for x in query]


def linear_diagnostic():
    rows=[];q=np.linspace(0,1,33);y=np.array([1.,.5])
    for fast_rate,slow_rate,h,augmented,whole in itertools.product((1e2,1e4,1e6),(1.,100.),(1e-4,1e-3,.01,.1),(False,True),(False,True)):
        fast=fast_rate*np.array([[-1.,1.],[1.,-1.]]);slow=slow_rate*np.array([[-1.,0.],[1.,0.]])
        maps=linear_maps(fast,slow,h,q,augmented,whole);values=np.column_stack([m@y for m in maps]);truth=np.column_stack([expm(x*h*(fast+slow))@y for x in q])
        rows.append(dict(fast_rate=fast_rate,slow_rate=slow_rate,h=h,augmented=augmented,whole=whole,spectral_radius=float(max(abs(np.linalg.eigvals(maps[-1])))),minimum_sampled=float(values.min()),max_error=float(np.max(abs(values-truth))),endpoint_error=float(np.max(abs(values[:,-1]-truth[:,-1])))))
    return rows


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    sources=['research/'+p for p in ('m3_rodas_centered.py','m3_rodas_study.py','rodas_coupling.py','rodas_centered.py','test_rodas_continuous.py','test_rodas_increment.py','test_rodas_centered.py','inner_dense.py','shared_radau.py','m3_extrapolated_paths.py','m3_growth_study.py','m3_priority.py')]
    configs=[dict(augmented=a,whole=w,tolerance=t) for a in (False,True) for w in (False,True) for t in ((1e-6,1e-10) if not w else (1e-6,))]
    reg=dict(date='2026-11-08',models=[dict(name='teusink',start=1.,threshold=3.),dict(name='kecoli',start=.1,threshold=1.)],steps=[.001,.003,.01,.03],configurations=configs,queries=65,
        algebra='Base8x6 moment matrix rank5 and augmented target rank7. Enumerate28 two-cubic-stage combinations and112 three-stage combinations on fixed eighth-grid nodes; select full-row-rank feasible3-stage extension minimizing infinity norm of moment weights. No model data used in this selection.',
        change='Affine-centered fast variable u=z_active-Q_active(t)+t*q0_active begins at y_active, derivative fast(z)+q0_active retains cancellation at t=0. Same predictor and slow polynomial; no projection. Prior direct-state and raw-increment failures preserved.',
        scope='Six/nine-stage continuous extensions, same current full J and diagonal. Fixed local states; no estimator/controller/global claim. Extra stages share actual factor. Original M4 gates unchanged.',
        controls=dict(tolerances=[1e-6,1e-7,1e-8,1e-10],initial_steps=['auto','whole']),
        reference='Independent full ODE1e-12/1e-11 pair, budget<=.05 at strict development target;65 uniform points plus original public observations.',
        timing='All kernel stages, inner solves and full continuous/physical/modified guards plus query guard timed. Setup and independent audit pair separately recorded. Single exploratory timing; no formal speedup.',
        source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={p:digest(ROOT/p) for p in ('results/m3-priority/p0.npz','results/m3-collocation-correction/summary.json','results/m3-collocation-correction/manifest.json','results/m3-rodas-continuous/registration.json','results/m3-rodas-continuous/runs.json','results/m3-rodas-increment/registration.json','results/m3-rodas-increment/runs.json')})
    save(OUT/'registration.json',reg)
    rows=[];controls=[];references=[]
    def persist():save(OUT/'runs.json',dict(candidates=rows,controls=controls,references=references))
    for mi,spec in enumerate(reg['models']):
        case=prepare(spec['name']);y=np.load(ROOT/'results/m3-priority/p0.npz')[spec['name']+'-prefix'][:,3].copy();tick=time.perf_counter();screen=GroupScreen(case['system'],case['scale']);kappa,_=screen.evaluate(y);groups=np.flatnonzero(1e-3*kappa>=spec['threshold']);kernel=RodasContinuous(case,groups);setup=time.perf_counter()-tick
        for hi,h in enumerate(reg['steps']):
            key=f'{mi}-{hi}';public=case['times'][(case['times']>spec['start'])&(case['times']<spec['start']+h)]-spec['start'];q=np.unique(np.r_[np.linspace(0,1,65),public/h]);ref,budget,seconds=audit(case,y,q,h)
            references.append(dict(key=key,model=spec['name'],h=h,budget=budget,seconds=seconds,setup_seconds=setup,groups=len(groups),active=len(kernel.active)));np.savez_compressed(OUT/f'reference-{key}.npz',initial=y,groups=groups,query=q,reference=ref)
            for tol,whole in itertools.product(reg['controls']['tolerances'],(False,True)):
                ck=f'control-{key}-{tol}-{whole}';row=dict(key=ck,reference_key=key,tolerance=tol,whole=whole);tick=time.perf_counter()
                try:
                    r=(fullstep_curve if whole else full_curve)(case,y,h,tol,screen);c=r['pieces'][0];values=c(q);structural(case,values);seconds=time.perf_counter()-tick;guard,targets=qualified(case,c,values,ref,r['stats'])
                    for t in targets.values():t['passes'] &= max(budget.values())<=.05
                    row.update(status='success',seconds=seconds,stats=r['stats'],guard=guard,targets=targets);np.savez_compressed(OUT/f'{ck}.npz',endpoint=r['fine'],**c.pack())
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
                controls.append(row);persist()
            for ci,config in enumerate(configs):
                ck=f'candidate-{key}-{ci}';row=dict(key=ck,reference_key=key,model=spec['name'],h=h,**config);tick=time.perf_counter()
                try:
                    r=kernel.trial(y,h,config['tolerance'],base_only=config['whole'],augmented=config['augmented']);c=r['pieces'][0];values=c(q);structural(case,values);seconds=time.perf_counter()-tick;guard,targets=qualified(case,c,values,ref,r['stats'])
                    for t in targets.values():t['passes'] &= max(budget.values())<=.05
                    row.update(status='success',seconds=seconds,stats=r['stats'],guard=guard,targets=targets)
                    arrays=dict(endpoint=r['fine'],increments=r['increments'],slow_increments=r['slow_increments'],stages=r['stages'],rates=r['rates'],full=r['full'],**pack(c))
                    if r['forcing_coefficients'] is not None:arrays['forcing']=r['forcing_coefficients']
                    np.savez_compressed(OUT/f'{ck}.npz',**arrays)
                except (ValueError,RuntimeError,FloatingPointError,TimeoutError) as exc:row.update(status='failed',seconds=time.perf_counter()-tick,reason=str(exc),stats=getattr(exc,'stats',{}))
                rows.append(row);persist()
            print(spec['name'],h,[(r['augmented'],r['whole'],r['tolerance'],r['status'],r.get('targets',{}).get('1e-06',{}).get('passes'),round(r['seconds'],5)) for r in rows[-6:]],flush=True)
        kernel.close()


if __name__=='__main__':run()
