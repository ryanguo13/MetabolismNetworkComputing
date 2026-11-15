"""Independent raw-data audit for growth-factor screen v3."""
import hashlib
import json
import math
from dataclasses import asdict

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.adaptive import AdaptiveConfig
from metabolism.continuation import ContinuationConfig
from metabolism.m3_continuation import ROOT, digest, metrics, prepare

OUT=ROOT/"results/m4-growth-factor-screen-v3"
TARGETS=(1e-3,1e-5)
SAFETIES=(.9,1.1,1.3,1.5)
REPEATS=(-1,0,1)
WINDOW=1e-6
STATE_INDEX=61


def close(a,b,rtol=1e-9,atol=1e-12): np.testing.assert_allclose(a,b,rtol=rtol,atol=atol)

def main():
    reg=json.loads((OUT/"registration.json").read_text()); rows=json.loads((OUT/"runs.json").read_text())["rows"]
    assert len(rows)==30, f"expected 30 rows, got {len(rows)}"
    for p,h in reg["source_sha256"].items(): assert digest(ROOT/p)==h,f"source hash mismatch {p}"
    case=prepare("kecoli")
    with np.load(case["reference_path"]) as src:
        y0=src["concentrations"][:,STATE_INDEX].copy(); archive_time=float(src["times"][STATE_INDEX])
    assert archive_time==reg["archive_time"]
    grid=WINDOW*np.array([0.,.25,.5,.75,1.])
    tr=solve_ivp(lambda t,y:case["system"].S@case["model"].flux(t,y),(0.,WINDOW),y0,
        method="Radau",t_eval=grid[1:],rtol=1e-13,atol=np.maximum(case["scale"]*1e-15,1e-18),
        jac=lambda t,y:case["system"].S@case["model"].flux_jacobian(t,y))
    assert tr.success
    truth=np.column_stack((y0,tr.y))
    with np.load(OUT/"reference.npz") as saved: close(saved["initial"],y0); close(saved["values"],truth)
    flux=np.column_stack([case["model"].flux(t,y) for t,y in zip(grid,truth.T)])
    local=dict(case,times=grid,reference=truth,flux_reference=flux)
    expected=[]
    for target in TARGETS:
        seq=[("baseline",None,-1)]+[("candidate",s,-1) for s in SAFETIES]
        for repeat in (0,1): seq += [("baseline",None,repeat)]+[("candidate",s,repeat) for s in SAFETIES]
        expected += [(target,*item) for item in seq]
    observed=[(r["target"],r["kind"],r["growth_safety"],r["repeat"]) for r in rows]
    assert observed==expected,"missing, duplicate or reordered rows"
    for row in rows:
        assert row["status"]=="success",row.get("reason")
        assert math.isfinite(row["seconds"]) and row["seconds"]>0
        target,kind,safety,repeat=row["target"],row["kind"],row["growth_safety"],row["repeat"]
        assert row["warmup"] is (repeat==-1)
        if kind=="baseline":
            assert safety is None
            cfg=AdaptiveConfig(**reg["baseline"],target=target)
        else:
            assert safety in SAFETIES
            candidate_params=dict(reg["candidate"], target=target, growth_safety=safety)
            cfg=ContinuationConfig(**candidate_params)
        assert row["parameters"]==asdict(cfg)
        name=f"{kind}-{target}-{safety or 'all'}-{repeat}.npz"
        with np.load(OUT/name) as z: values=z["values"]
        assert values.shape==truth.shape and np.isfinite(values).all()
        m=metrics(values,local,target,row["stats"])
        for k,v in m.items():
            if isinstance(v,(bool,np.bool_)): assert row["metrics"][k] is bool(v)
            else: close(row["metrics"][k],v)
        if kind=="candidate":
            stats=row["stats"]; attempts=stats["attempts"]
            assert stats["full_calls"]==len(attempts) and row["every_attempt_audited"] is True
            assert stats["clipping_count"]==0 and stats["projection_count"]==0
            assert all(a["rollback_verified"] is True for a in attempts if a["status"]=="rejected")
            coverage=sum(a["h"] for a in attempts if a["status"]=="accepted")/WINDOW
            close(row["coverage_recomputed"],coverage,atol=1e-14)
    summaries=[]
    for target in TARGETS:
        b=[r["seconds"] for r in rows if r["target"]==target and r["kind"]=="baseline" and r["repeat"]>=0]
        for safety in SAFETIES:
            c=[r for r in rows if r["target"]==target and r["kind"]=="candidate" and r["growth_safety"]==safety and r["repeat"]>=0]
            ts=[r["seconds"] for r in c]
            summaries.append(dict(target=target,growth_safety=safety,baseline_median=float(np.median(b)),
                candidate_median=float(np.median(ts)),speedup=float(np.median(b)/np.median(ts)),
                iqr=float(np.percentile(ts,75)-np.percentile(ts,25)),
                coverage=[r["coverage_recomputed"] for r in c],
                all_metrics_pass=all(r["metrics"]["passes"] for r in c),
                all_attempts_audited=all(r["every_attempt_audited"] for r in c)))
    report=dict(evidence_integrity="pass",rows=len(rows),summaries=summaries,
        scope="Short local Kecoli diagnostic only; no full-window or M4 qualification.")
    (OUT/"independent_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
