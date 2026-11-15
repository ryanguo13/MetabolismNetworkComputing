"""Independent raw-record audit for growth-safety screen v2."""
import hashlib
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/m4-growth-safety-screen-v2'
MODELS=('teusink','kecoli'); TARGETS=(.001,1e-5); SAFETIES=(.9,1.8); WINDOW=1e-6

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def main():
    failures=[]
    def check(ok,msg):
        if not ok: failures.append(msg)
    reg=json.loads((OUT/'registration.json').read_text()); rows=json.loads((OUT/'runs.json').read_text()).get('runs',[])
    check(len(rows)==36,f'expected 36 rows got {len(rows)}')
    check(reg.get('warmups')==1 and reg.get('repeats')==2,'registration repeat contract mismatch')
    source_checks={}
    for p,h in reg['source_sha256'].items():
        actual=digest(ROOT/p) if (ROOT/p).is_file() else None
        source_checks[p]=actual==h; check(actual==h,f'source hash mismatch {p}')
    summaries=[]
    for model in MODELS:
      for target in TARGETS:
        combo=[r for r in rows if r.get('model')==model and r.get('target')==target]
        expected=[('baseline',None,True),('candidate',.9,True),('candidate',1.8,True)]
        for _ in range(2): expected += [('baseline',None,False),('candidate',.9,False),('candidate',1.8,False)]
        actual=[(r.get('method'),r.get('growth_safety'),r.get('warmup')) for r in combo]
        check(actual==expected,f'{model}/{target}: raw order mismatch')
        for method,safety,warmup in expected:
          group=[r for r in combo if r.get('method')==method and r.get('growth_safety')==safety and r.get('warmup')==warmup]
          expected_count=1 if warmup else 2
          check(len(group)==expected_count,f'{model}/{target}/{method}/{safety}/{warmup}: row count')
        for method,safety,repeat_ids in [('baseline',None,[0,1]),('candidate',.9,[0,1]),('candidate',1.8,[1,2])]:
          group=[r for r in combo if r.get('method')==method and r.get('growth_safety')==safety and not r.get('warmup')]
          check(sorted(r.get('repeat') for r in group)==repeat_ids,f'{model}/{target}/{method}/{safety}: repeat ids')
        baselines=[r for r in combo if r.get('method')=='baseline' and not r.get('warmup')]
        check(len(baselines)==2 and all(r.get('status')=='success' and finite(r.get('seconds')) for r in baselines),f'{model}/{target}: baseline rows')
        bmed=sorted(r['seconds'] for r in baselines)[len(baselines)//2] if baselines else math.nan
        for safety in SAFETIES:
          candidates=[r for r in combo if r.get('method')=='candidate' and r.get('growth_safety')==safety and not r.get('warmup')]
          for r in candidates:
            p=r.get('parameters') or {}
            check(p.get('growth_safety')==safety and p.get('audit_control_only') is True and p.get('local_fraction')==1. and p.get('group_every')==1,f'{model}/{target}/{safety}: candidate config mismatch')
            check(r.get('status')=='success' and finite(r.get('seconds')) and r['seconds']>0,f'{model}/{target}/{safety}: failed row')
            st=r.get('stats') or {}; attempts=st.get('attempts') or []
            check(bool(attempts) and st.get('full_calls')==len(attempts),f'{model}/{target}/{safety}: incomplete full auditing')
            check(r.get('every_attempt_audited') is True,f'{model}/{target}/{safety}: audit marker false')
            check(all(a.get('rollback_verified') is True for a in attempts if a.get('status')=='rejected'),f'{model}/{target}/{safety}: rollback missing')
            accepted=[a for a in attempts if a.get('status')=='accepted']; intervals=sorted((a.get('time'),a.get('h')) for a in accepted)
            covered=0.; end=0.
            for start,h in intervals:
              check(finite(start) and finite(h) and h>0 and start>=end-1e-14,f'{model}/{target}/{safety}: overlap or bad interval')
              if finite(start) and finite(h) and h>0 and start>=end-1e-14: covered+=h; end=start+h
            cov=covered/WINDOW
            check(end<=WINDOW+1e-14,f'{model}/{target}/{safety}: interval beyond registered window')
            check(finite(r.get('split_coverage_recomputed')) and math.isclose(cov,r['split_coverage_recomputed'],rel_tol=1e-9,abs_tol=1e-12),f'{model}/{target}/{safety}: coverage mismatch')
            m=r.get('metrics') or {}; ctrl=max(m.get('concentration_error',math.inf),m.get('flux_error',math.inf),m.get('net_flux_error',0.))
            check(m.get('finite') is True and m.get('significant_negative_output') is False and m.get('invariant_error',math.inf)<=1e-10 and m.get('relative_ledger_error',math.inf)<=1e-10 and m.get('passes') is True,f'{model}/{target}/{safety}: physical or accuracy gate fail')
            summaries.append(dict(model=model,target=target,safety=safety,seconds=r.get('seconds'),baseline_median=bmed,speedup=bmed/r['seconds'] if finite(r.get('seconds')) else None,coverage=cov,accuracy_control=ctrl,accuracy_pass=bool(ctrl<=1),attempts=len(attempts),accepted=len(accepted),rejected=len(attempts)-len(accepted),full_calls=st.get('full_calls')))
    grouped={}
    for s in summaries: grouped.setdefault((s['model'],s['target'],s['safety']),[]).append(s)
    advances=True
    for vals in grouped.values():
      speeds=sorted(x['speedup'] for x in vals if finite(x.get('speedup')))
      advances &= len(vals)==2 and len(speeds)==2 and speeds[0]>=1.5 and min(x['coverage'] for x in vals)>=.5 and all(x['accuracy_pass'] for x in vals)
    result=dict(evidence_integrity='pass' if not failures else 'fail',source_hash_checks=source_checks,summaries=summaries,candidate_worth_full_window=bool(advances),failures=failures,conclusion='candidate merits a full-window screen' if advances else 'candidate does not merit a full-window or formal screen')
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); raise SystemExit(1 if failures else 0)
if __name__=='__main__': main()
