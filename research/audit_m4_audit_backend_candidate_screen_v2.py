"""Independent raw-record audit of backend candidate screen v2."""
import json
import math
from dataclasses import asdict

import numpy as np
from scipy.integrate import solve_ivp

from metabolism.continuation import ContinuationConfig
from metabolism.m3_continuation import ROOT, digest, metrics, prepare

OUT = ROOT / "results/m4-audit-backend-candidate-screen-v2"
METHODS = ("CVODE-KLU", "BDF")
TARGETS = (1e-3, 1e-5)
WINDOW = 1e-6
STATE_INDEX = 61


def close(a, b, *, rtol=1e-9, atol=1e-12):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def main():
    registration = json.loads((OUT / "registration.json").read_text())
    rows = json.loads((OUT / "runs.json").read_text())["rows"]
    assert registration["state_index"] == STATE_INDEX
    assert registration["targets"] == list(TARGETS)
    assert registration["methods"] == ["AdaptiveSplit", *METHODS]
    for path, expected in registration["source_sha256"].items():
        assert digest(ROOT/path) == expected, f"source hash mismatch: {path}"
    case=prepare("kecoli")
    with np.load(case["reference_path"]) as source:
        y0=source["concentrations"][:,STATE_INDEX].copy()
        archive_time=float(source["times"][STATE_INDEX])
    assert archive_time==registration["archive_time"]
    grid=WINDOW*np.array([0.,.25,.5,.75,1.])
    tr=solve_ivp(lambda t,y:case["system"].S@case["model"].flux(t,y),(0.,WINDOW),y0,
        method="Radau",t_eval=grid[1:],rtol=1e-13,atol=np.maximum(case["scale"]*1e-15,1e-18),
        jac=lambda t,y:case["system"].S@case["model"].flux_jacobian(t,y))
    assert tr.success
    truth=np.column_stack((y0,tr.y))
    with np.load(OUT/"reference.npz") as saved:
        close(saved["initial"],y0); close(saved["values"],truth)
    flux_ref=np.column_stack([case["model"].flux(t,y) for t,y in zip(grid,truth.T)])
    local_case=dict(case,times=grid,reference=truth,flux_reference=flux_ref)
    expected=[]
    for target in TARGETS:
        seq=[("baseline",None,-1),("candidate",METHODS[0],-1),("candidate",METHODS[1],-1)]
        for rep in (0,1): seq.extend([("baseline",None,rep),("candidate",METHODS[0],rep),("candidate",METHODS[1],rep)])
        expected.extend((target,*item) for item in seq)
    observed=[(r["target"],r["kind"],r["audit_method"],r["repeat"]) for r in rows]
    assert observed==expected, "missing/duplicate/out-of-order rows"
    summary=[]
    for row in rows:
        assert row["status"]=="success",row.get("reason")
        assert math.isfinite(row["seconds"]) and row["seconds"]>0
        target,kind,method,repeat=row["target"],row["kind"],row["audit_method"],row["repeat"]
        assert row["warmup"] is (repeat==-1)
        if kind=="baseline":
            assert method is None
            p=dict(target=target,initial_step=1e-8,max_step=WINDOW,group_every=5,
                enter=1.,leave=.3,inner_fraction=.05,local_fraction=.1,max_rejections=16,
                max_attempts=256,slow_bound_limit=2.,fallback_method="CVODE-KLU",
                audit_rtol=1e-11,fallback_rtol=1e-11,estimator_ratio_limit=2.,timeout=240.)
        else:
            assert method in METHODS
            p=dict(registration["candidate"],target=target,audit_method=method,timeout=240.)
            p=asdict(ContinuationConfig(**p))
        assert row["parameters"]==p
        file=f"{kind}-{target}-{method or 'all'}-{repeat}.npz"
        with np.load(OUT/file) as saved: values=saved["values"]
        assert values.shape==truth.shape and np.isfinite(values).all()
        actual=metrics(values,local_case,target,row["stats"])
        for k,v in actual.items():
            if isinstance(v,(bool,np.bool_)): assert row["metrics"][k] is bool(v)
            else: close(row["metrics"][k],v)
        st=row["stats"]
        assert st["clipping_count"]==0 and st["projection_count"]==0
        if kind=="candidate":
            assert st["full_calls"]==len(st["attempts"])
            coverage=sum(a["h"] for a in st["attempts"] if a["status"]=="accepted")/WINDOW
            close(row["coverage_recomputed"],coverage,atol=1e-14)
            for a in st["attempts"]:
                if a["status"]=="rejected": assert a["rollback_verified"] is True
    for target in TARGETS:
        base=[r["seconds"] for r in rows if r["target"]==target and r["kind"]=="baseline" and r["repeat"]>=0]
        for method in METHODS:
            cand=[r for r in rows if r["target"]==target and r["kind"]=="candidate" and r["audit_method"]==method and r["repeat"]>=0]
            ts=[r["seconds"] for r in cand]
            summary.append(dict(target=target,method=method,baseline_median=float(np.median(base)),
                candidate_median=float(np.median(ts)),speedup=float(np.median(base)/np.median(ts)),
                candidate_iqr=float(np.percentile(ts,75)-np.percentile(ts,25)),
                coverage=[r["coverage_recomputed"] for r in cand],
                all_metrics_pass=all(r["metrics"]["passes"] for r in cand)))
    report=dict(evidence_integrity="pass",rows=len(rows),summary=summary,
        limitation="One early-state short Kecoli interval; not full-window or M4 qualification.")
    (OUT/"independent_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
