"""Read-only residual tracing of the frozen SPC predictor, not a timing run."""
import inspect
import sys
import numpy as np
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.spc_mri import SPC
from research.m3_shadow_study import ROOT,digest


def trace():
    case=prepare('kecoli');y=np.load(ROOT/'results/m3-priority/p0.npz')['kecoli-prefix'][:,3];kernel=SPC(case,[])
    lines,start=inspect.getsourcelines(SPC.predictor);line=start+next(i for i,s in enumerate(lines) if 'if norm<=residual_tol:break' in s)
    code=SPC.predictor.__code__;rows=[];history=[]
    def tracer(frame,event,arg):
        if frame.f_code!=code:return None
        if event=='line' and frame.f_lineno==line:
            d=frame.f_locals;r=d['residual']/case['scale'];index=int(np.argmax(abs(r)))
            history.append(dict(stage=d['i'],iteration=d['iteration'],norm=d['norm'],worst_component=index,state=float(d['z'][index]),signed_scaled_residual=float(r[index])))
        return tracer
    for h,order in ((.001,3),(.003,2),(.01,2)):
        history=[];previous=sys.gettrace();sys.settrace(tracer)
        try:
            kernel.predictor(y,h,order);status='success';reason=None
        except (RuntimeError,ValueError,TimeoutError) as exc:status='failed';reason=str(exc)
        finally:sys.settrace(previous)
        rows.append(dict(h=h,order=order,status=status,reason=reason,residuals=history))
    kernel.close();return rows


if __name__=='__main__':
    path=ROOT/'results/m3-spc/predictor-trace.json'
    if path.exists():raise FileExistsError('Preserve diagnostic')
    save(path,dict(scope='Post-screen diagnostics; scaled residual stagnation is observed, numerical noise is a hypothesis, not a proved cause.',
        source_sha256={p:digest(ROOT/p) for p in ('research/spc_mri.py','research/spc_residual_diagnostic.py')},runs=trace()))
