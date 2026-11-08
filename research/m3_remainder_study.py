"""Registered full-linear-oracle test of nonlinear continuous correction."""
import itertools
import time
import numpy as np
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.remainder_curve import RemainderCurve
from research.merb_predictor import MERBPredictor
from research.test_remainder_curve import order_diagnostic
from research.extrapolated_flow import pack
from research.inner_dense import full_curve
from research.fullstep_radau import fullstep_curve
from research.m3_extrapolated_paths import qualified
from research.m3_ordered_cost import structural
from research.m3_growth_study import audit
from research.m3_priority import STATES, metrics
from research.m3_shadow_study import ROOT, digest

OUT = ROOT/'results/m3-remainder-curve'
ARRAYS = ('stages', 'rates', 'remainders', 'forcings', 'initial_rates')
MERB_ARRAYS = ('stage', 'initial_rates', 'stage_rates', 'remainder', 'forcing')


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():
        raise FileExistsError('Preserve evidence')
    sources = ['research/'+f for f in ('remainder_curve.py', 'test_remainder_curve.py',
        'm3_remainder_study.py', 'merb_predictor.py', 'ordered_radau.py', 'shared_radau.py',
        'inner_dense.py', 'fullstep_radau.py', 'm3_extrapolated_paths.py', 'm3_growth_study.py',
        'm3_priority.py', 'm3_ordered_cost.py', 'extrapolated_flow.py')]
    reg = dict(date='2026-11-08', models=['teusink', 'kecoli'], state_indices=[3, 2],
        steps=[.003, .01, .03], tolerances=[1e-8, 1e-10], methods=['merb', 'cubic1', 'cubic2'],
        controls=dict(tolerances=[1e-6, 1e-7, 1e-8, 1e-10], whole=[False, True]),
        hypothesis='Two nonlinear remainder samples improve the whole curve: N=f(y+d)-f0-Jd, A=8*N(H/2)-N(H), B=2*N(H)-8*N(H/2); dprime=Jd+f0+A*(t/H)^2+B*(t/H)^3. Start with affine curve; one/two sweeps. Reactionwise subtraction; same current J. No reference values in prediction.',
        scope='72 candidate records (24 matched MERB controls) and96 mature controls,12 independent ODE pairs. Development local mechanism/cost test, not M4 evidence. Classical local O(H^5) for smooth fixed-J only; no stiff order claim. Full linear adaptive Radau oracle, no rational compression or reaction coverage.',
        states='Earlier and later saved own prefix states from prior research; neither is a globally unseen test set. No reset during a solve. Abort negative affine/internal/output state before nonlinear rates; preserve all failures. No fallback solver in this diagnostic.',
        reference='Independent original ODE CVODE pair1e-12/1e-11, budget<=.05 development strict units.65 uniform queries plus public observation grid. All original development positivity/pool/physical+modified ledger/net-flux guards.',
        timing='One exploratory run. Candidate includes current gradient/J, all linear solves, positivity/pools/ledger and sample guards. Setup and independent audit/scoring separately timed. Report candidate+audit cost, and predictor-only optimistic budget; never claim formal speed. Mature controls have same output guards and automatic/whole first step.',
        decision='Do not integrate rational action or fast reaction correction unless nonlinear accuracy/ledger improves at H>=.01. Full-linear timing is not a lower bound on every possible rational implementation. No seven-repeat/full-window escalation without plausible total cost.',
        source_sha256={p:digest(ROOT/p) for p in sources},
        input_sha256={p:digest(ROOT/p) for p in ('results/m3-controlled-rational/manifest.json', 'results/m3-merb-predictor/manifest.json', 'results/m3-priority/p0.npz', 'results/m3-priority/p0.json')})
    save(OUT/'registration.json', reg)
    save(OUT/'order.json', order_diagnostic())
    rows, controls, references, setups = [], [], [], []
    def persist():
        save(OUT/'runs.json', dict(candidates=rows, controls=controls, references=references, setups=setups))
    for mi, name in enumerate(reg['models']):
        case = prepare(name); tick = time.perf_counter()
        kernels = dict(merb=MERBPredictor(case), cubic=RemainderCurve(case))
        screen = GroupScreen(case['system'], case['scale'])
        setups.append(dict(model=name, seconds=time.perf_counter()-tick))
        for si in reg['state_indices']:
            with np.load(ROOT/'results/m3-priority/p0.npz') as src:
                y = src[name+'-prefix'][:, si]
            start = STATES[name][si]
            for hi, h in enumerate(reg['steps']):
                key = f'{mi}-{si}-{hi}'
                public = (case['times'][(case['times']>start)&(case['times']<start+h)]-start)/h
                q = np.unique(np.r_[np.linspace(0, 1, 65), public])
                ref, budget, seconds = audit(case, y, q, h); refok = max(budget.values()) <= .05
                references.append(dict(key=key, model=name, state_index=si, start=start, h=h,
                                       budget=budget, seconds=seconds, reference_ok=refok))
                np.savez_compressed(OUT/f'reference-{key}.npz', initial=y, query=q, reference=ref)
                for method, tol in itertools.product(reg['methods'], reg['tolerances']):
                    ck = f'candidate-{key}-{method}-{tol}'
                    row = dict(key=ck, reference_key=key, model=name, state_index=si, h=h, method=method, tolerance=tol)
                    tick = time.perf_counter()
                    try:
                        r = kernels['merb'].trial(y, h, tol) if method == 'merb' else kernels['cubic'].trial(y, h, tol, int(method[-1]))
                        c = r['pieces'][0]; values = c(q); structural(case, values)
                        elapsed = time.perf_counter()-tick
                        guard, targets = qualified(case, c, values, ref, r['stats'])
                        for t, v in targets.items():
                            v['passes'] &= refok
                            v['endpoint'] = metrics(case, r['fine'][:, None], ref[:, -1:], float(t))
                        row.update(status='success', seconds=elapsed, stats=r['stats'], guard=guard, targets=targets)
                        arrays = dict(endpoint=r['fine'], **pack(c), **{k:r[k] for k in (MERB_ARRAYS if method=='merb' else ARRAYS)})
                        curves = [r['stage_curve']] if method=='merb' else r['curves'][:-1]
                        for i, curve in enumerate(curves):
                            arrays.update({f'previous{i}_'+k:v for k, v in pack(curve).items()})
                        np.savez_compressed(OUT/f'{ck}.npz', **arrays)
                    except (ValueError, RuntimeError, FloatingPointError, TimeoutError) as exc:
                        row.update(status='failed', seconds=time.perf_counter()-tick, reason=str(exc), stats=getattr(exc, 'stats', {}))
                    rows.append(row); persist()
                for tol, whole in itertools.product(reg['controls']['tolerances'], reg['controls']['whole']):
                    ck = f'control-{key}-{tol}-{whole}'
                    row = dict(key=ck, reference_key=key, model=name, state_index=si, h=h, tolerance=tol, whole=whole)
                    tick = time.perf_counter()
                    try:
                        r = (fullstep_curve if whole else full_curve)(case, y, h, tol, screen)
                        c = r['pieces'][0]; values = c(q); structural(case, values)
                        elapsed = time.perf_counter()-tick
                        guard, targets = qualified(case, c, values, ref, r['stats'])
                        for v in targets.values(): v['passes'] &= refok
                        row.update(status='success', seconds=elapsed, stats=r['stats'], guard=guard, targets=targets)
                        np.savez_compressed(OUT/f'{ck}.npz', endpoint=r['fine'], **c.pack())
                    except (ValueError, RuntimeError, FloatingPointError, TimeoutError) as exc:
                        row.update(status='failed', seconds=time.perf_counter()-tick, reason=str(exc), stats=getattr(exc, 'stats', {}))
                    controls.append(row); persist()
                print(name, si, h, [(r['method'], r['tolerance'], r['status'], r.get('targets', {}).get('1e-06', {}).get('passes')) for r in rows[-6:]], flush=True)


if __name__ == '__main__':
    run()
