"""Fresh adaptive action paths using registered filtered indicators, full audits."""
import json
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.controlled_rational import RationalBasis,proposal
from research.rational_filter import filtered_proposal
from research.affine_partition import freeze
from research.m3_growth_study import audit
from research.m3_priority import metrics
from research.m3_ordered_cost import structural
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-filtered-rational';OLD=ROOT/'results/m3-controlled-rational'


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    old=json.loads((OLD/'registration.json').read_text());references=json.loads((OLD/'runs.json').read_text())['references']
    reg=dict(date='2026-11-08',dimensions=old['dimensions'],shifts=old['shifts'],projections=old['projections'],budget=.05,safety=4.,
        selection='Prospectively use4*||H*K*residual|| and previous-dimension change<=.05 strict initial scale, plus sampled positivity/pool. Stop on first proposal; then independently audit. Audit veto terminates path, requires external fallback; never silently accept. No parameter tuning from new outcomes.',
        filter='K=(I-sigma*H*A)^-1. Saved W=KV; KAV=(W-V)/(sigma H). Filter requires no new factors or solves. Safety4 is heuristic and a32-state stable Jordan-chain counterexample disproves universal certification.',
        scope='32 new autonomous affine action paths across two models,two source states,two H,two shifts,two projections. No nonlinear method, no coverage, no M4 claim. Every rejected dimension retained.',
        reference='Independent CVODE1e-12/1e-11 pair recomputed after final decision for every path; cross-checked to previously independently audited displacement-Radau truth. Budget/cross<=.05 strict. No reference used by selector.',
        timing='Fresh per-path gradient/rates/J formation, one inverse factor and all extension/evaluation/indicator/guard time. Excludes one-time GroupScreen setup and diagnostic ReactionSystem clone; these separately measured. Reference pair time charged separately and combined cost reported. Not repeated/formal cost evidence.',
        source_sha256={p:digest(ROOT/p) for p in ('research/controlled_rational.py','research/rational_filter.py','research/test_rational_filter.py','research/m3_filtered_rational_study.py','research/affine_partition.py','research/m3_growth_study.py','research/m3_priority.py','research/m3_ordered_cost.py')},
        input_sha256={p:digest(ROOT/p) for p in ('results/m3-controlled-rational/registration.json','results/m3-controlled-rational/runs.json','results/m3-merb-predictor/manifest.json')})
    save(OUT/'registration.json',reg);rows=[];paths=[]
    def persist():save(OUT/'runs.json',dict(runs=rows,paths=paths))
    for rr in references:
        name=rr['model'];h=rr['h'];key=rr['key'];case=prepare(name);tick=time.perf_counter();screen=GroupScreen(case['system'],case['scale']);screen_seconds=time.perf_counter()-tick
        with np.load(OLD/f'reference-{key}.npz') as src:y=src['initial'];q=src['query'];truth=src['reference']
        tick=time.perf_counter();frozen,_=freeze(case,y);diagnostic_setup=time.perf_counter()-tick
        for shift in reg['shifts']:
            for method in reg['projections']:
                pathkey=f'{key}-{shift}-{method}';tick=time.perf_counter();gradient=screen.gradient(y);j=(case['system'].S@gradient).tocsc();rhs=case['system'].S@case['system'].flux(0.,y);formation=time.perf_counter()-tick
                basis=RationalBasis(j,rhs,y,case['scale'],h,shift);previous=None;selected=None;states=[]
                for dim in sorted(set(min(len(y),d) for d in reg['dimensions'])):
                    ck=f'{pathkey}-{dim}';basis.extend(dim);row=dict(key=ck,path=pathkey,reference_key=key,dimension_requested=dim)
                    try:
                        r=basis.evaluate(q,method);raw=proposal(r,previous,h,case['scale'],y,budget=reg['budget'])
                        filtered,values=filtered_proposal(r,np.column_stack(basis.images),rhs/case['scale'],h,shift,case['scale'],y,raw['enrichment_indicator'],budget=reg['budget'],safety=reg['safety'])
                        guard=structural(frozen,r['values']);offered=bool(filtered['proposed'] and guard['nonnegative'] and guard['pool']<=1e-10)
                        row.update(status='success',stats=r['stats'],raw_indicator=raw,filtered_indicator=filtered,guard=guard,offered=offered)
                        np.savez_compressed(OUT/f'{ck}.npz',**{k:r[k] for k in ('coefficients','projection','drive')})
                        previous=r;states.append((row,r))
                        if offered:selected=ck
                    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                        row.update(status='failed',reason=str(exc));previous=None
                    rows.append(row)
                    if selected is not None:break
                seconds=time.perf_counter()-tick
                # Snapshot before audit: it is impossible for reference scores
                # to select an earlier/later dimension in this completed path.
                np.savez_compressed(OUT/f'basis-{pathkey}.npz',vectors=np.column_stack(basis.vectors),images=np.column_stack(basis.images),av=np.column_stack(basis.av))
                ref,budget,audit_seconds=audit(frozen,y,q,h);cross=metrics(frozen,ref,truth,1e-6);refok=max([*budget.values(),*cross.values()])<=.05 and rr['reference_ok']
                for row,result in states:
                    actual=metrics(frozen,result['values'],truth,1e-6);row.update(actual=actual,audit_ok=bool(refok and max(actual.values())<=1 and row['guard']['nonnegative'] and row['guard']['pool']<=1e-10))
                chosen=next((r for r in rows if r['key']==selected),None)
                paths.append(dict(key=pathkey,reference_key=key,model=name,state_index=rr['state_index'],h=h,shift=shift,method=method,selected_key=selected,attempts=len([r for r in rows if r['path']==pathkey]),outcome='fallback_required' if chosen is None else ('audited_linear_samples_only' if chosen['audit_ok'] else 'audit_veto'),candidate_seconds=seconds,formation_seconds=formation,screen_setup_seconds=screen_seconds,diagnostic_clone_seconds=diagnostic_setup,audit_seconds=audit_seconds,total_candidate_and_audit_seconds=seconds+audit_seconds,reference_budget=budget,cross_budget=cross,reference_ok=refok,M4_eligible=False))
                persist()
        print(key,[(p['method'],p['shift'],p['outcome'],p['selected_key'],p['candidate_seconds']) for p in paths[-4:]],flush=True)


if __name__=='__main__':run()
