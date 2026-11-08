"""Reintegrate all registered records, independently reconstruct forcing/guards."""
import argparse
import itertools
import io
import json
import platform
import subprocess
import sys
import zipfile
from collections import Counter
import numpy as np
import scipy
from metabolism.adaptive import GroupScreen
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.remainder_curve import RemainderCurve
from research.merb_predictor import MERBPredictor
from research.test_remainder_curve import order_diagnostic
from research.extrapolated_flow import unpack
from research.inner_dense import InnerCurve, full_curve, physical_ledger
from research.fullstep_radau import fullstep_curve
from research.m3_growth_study import audit
from research.m3_priority import metrics
from research.m3_ledger_flow_report import registration, close, check_curve
from research.m3_collocation_report import timeless
from research.m3_merb_report import ledger as merb_ledger, tight
from research.m3_remainder_study import OUT, ARRAYS, MERB_ARRAYS
from research.m3_shadow_study import ROOT, digest
from research.mri_kernel import GAUSS
from research.remainder_storage import load_arrays, load_bytes


def read(p):
    return json.loads(p.read_text())


def modified_ledger(case, curve, y, end, j, f0, forcing):
    weights = case['options'].get('ledger_weights')
    if weights is None: return {}
    totals = []
    polynomial = np.column_stack([f0, np.zeros_like(y), forcing])
    for n in (8, 16):
        q = (GAUSS[n][0]+1)/2; total = 0.; left = 0.
        for piece in curve.pieces:
            tau = (left+piece.h*q)/curve.h
            rhs = j@(piece(q)-y[:, None])+np.polynomial.polynomial.polyval(tau, polynomial.T)
            total += piece.h/2*float(GAUSS[n][1]@(weights@rhs)); left += piece.h
        totals.append(total)
    den = max(1., abs(float(weights@y)))
    return dict(modified_ledger_residual=abs(float(weights@(end-y))-totals[1])/den,
                modified_ledger_quadrature=abs(totals[1]-totals[0])/den)


def verify():
    reg = registration(OUT/'registration.json'); data = read(OUT/'runs.json')
    for name in ('results/m3-controlled-rational/manifest.json', 'results/m3-merb-predictor/manifest.json'):
        for p, sha in read(ROOT/name).items(): assert digest(ROOT/p) == sha, p
    close(order_diagnostic(), read(OUT/'order.json'))
    assert len(data['candidates']) == 72 and len(data['controls']) == 96 and len(data['references']) == 12
    assert {r['key'] for r in data['references']} == {f'{mi}-{si}-{hi}' for mi in range(2) for si in reg['state_indices'] for hi in range(3)}
    comparisons = []
    for name in reg['models']:
        case = prepare(name); screen = GroupScreen(case['system'], case['scale'])
        kernel = RemainderCurve(case); merb = MERBPredictor(case)
        for rr in [r for r in data['references'] if r['model']==name]:
            key, h = rr['key'], rr['h']
            with np.load(OUT/f'reference-{key}.npz') as a:
                y, q, savedref = a['initial'], a['query'], a['reference']
            ref, budget, _ = audit(case, y, q, h)
            np.testing.assert_array_equal(ref, savedref); close(budget, rr['budget'])
            assert rr['reference_ok'] and max(budget.values()) <= .05
            gradient = screen.gradient(y); j = case['system'].S@gradient
            v0 = case['system'].flux(0, y); f0 = case['system'].S@v0
            subset = [r for r in data['candidates'] if r['reference_key']==key]
            controls = [r for r in data['controls'] if r['reference_key']==key]
            assert len(subset)==6 and {(r['method'], r['tolerance']) for r in subset} == set(itertools.product(reg['methods'], reg['tolerances']))
            assert len(controls)==8 and {(r['tolerance'], r['whole']) for r in controls} == set(itertools.product(reg['controls']['tolerances'], reg['controls']['whole']))
            for row in subset+controls:
                candidate = 'method' in row
                method = row.get('method')
                try:
                    if candidate:
                        result = merb.trial(y, h, row['tolerance']) if method=='merb' else kernel.trial(y, h, row['tolerance'], int(method[-1]))
                    else:
                        result = (fullstep_curve if row['whole'] else full_curve)(case, y, h, row['tolerance'], screen)
                except (ValueError, RuntimeError, FloatingPointError, TimeoutError) as exc:
                    assert row['status']=='failed' and str(exc)==row['reason'], (row['key'], exc)
                    close(timeless(getattr(exc, 'stats', {})), timeless(row['stats']))
                    assert not (OUT/f"{row['key']}.npz").exists()
                    continue
                assert row['status']=='success', row['key']
                close(timeless(result['stats']), timeless(row['stats']))
                with load_arrays(OUT/f"{row['key']}.npz") as a:
                    curve = (unpack if candidate else InnerCurve.unpack)(a['lengths'], a['coefficients'])
                    np.testing.assert_array_equal(result['fine'], a['endpoint'])
                    np.testing.assert_array_equal(curve(q), result['pieces'][0](q))
                    if candidate:
                        for k in (MERB_ARRAYS if method=='merb' else ARRAYS):
                            np.testing.assert_array_equal(result[k], a[k])
                        previous = [result['stage_curve']] if method=='merb' else result['curves'][:-1]
                        for i, c in enumerate(previous):
                            saved = unpack(a[f'previous{i}_lengths'], a[f'previous{i}_coefficients'])
                            np.testing.assert_array_equal(saved(q), c(q))
                            assert np.all(c.minima()>=-1e-12*case['scale']) and c.join_error(case['scale'])<=1e-10
                targets = {t:{k:v for k,v in r.items() if k!='endpoint'} for t,r in row['targets'].items()}
                check_curve(case, curve, y, result['fine'], ref, q, row['stats'], row['guard'], targets)
                tight(physical_ledger(case, curve, y, result['fine']), row['stats'])
                if not candidate: continue
                for t,v in row['targets'].items():
                    close(metrics(case, result['fine'][:, None], ref[:, -1:], float(t)), v['endpoint'])
                assert row['stats']['nonlinear_njev']==1
                assert row['stats']['lu_cache_misses']==row['stats']['nlu']
                if method=='merb':
                    tight(merb_ledger(case, curve, y, result['fine'], j, f0, result['forcing']), row['stats'])
                    continue
                sweeps = int(method[-1])
                assert row['stats']['nonlinear_nfev']==1+2*sweeps and row['stats']['subflows']==1+sweeps
                for i in range(sweeps):
                    np.testing.assert_array_equal(result['curves'][i](np.array([.5, 1.])), result['stages'][i])
                    rates = np.column_stack([case['system'].flux(0, z) for z in result['stages'][i].T])
                    np.testing.assert_array_equal(rates, result['rates'][i])
                    remainder = rates-v0[:, None]-gradient@(result['stages'][i]-y[:, None])
                    np.testing.assert_array_equal(remainder, result['remainders'][i])
                    ab = np.linalg.solve(np.array([[.25, .125], [1., 1.]]), remainder.T).T
                    np.testing.assert_allclose(case['system'].S@ab, result['forcings'][i], rtol=2e-12, atol=1e-14)
                tight(modified_ledger(case, curve, y, result['fine'], j, f0, result['forcings'][-1]), row['stats'])
            for target in ('0.0001', '1e-06'):
                valid = [r for r in controls if r['status']=='success' and r['targets'][target]['passes']]
                best = min(valid, key=lambda r:r['seconds']) if valid else None
                for row in subset:
                    ok = row['status']=='success'
                    comparisons.append(dict(key=row['key'], model=name, state_index=rr['state_index'], h=h,
                        method=row['method'], tolerance=row['tolerance'], target=target,
                        passes=ok and row['targets'][target]['passes'], actual=row['targets'][target]['actual'] if ok else None,
                        physical_ledger=row['stats'].get('physical_ledger_residual'),
                        candidate_seconds=row['seconds'], predictor_seconds=row['stats'].get('predictor_seconds'),
                        audit_seconds=rr['seconds'], total_candidate_and_audit_seconds=row['seconds']+rr['seconds'],
                        baseline_key=best['key'] if best else None, baseline_seconds=best['seconds'] if best else None,
                        optimistic_ratio=best['seconds']/row['seconds'] if best else None,
                        with_audit_ratio=best['seconds']/(row['seconds']+rr['seconds']) if best else None,
                        residual_1p5_budget_after_audit=best['seconds']/1.5-rr['seconds'] if best else None))
            print('verified', name, rr['state_index'], h, flush=True)
    counts = {}
    for method in reg['methods']:
        rows = [r for r in data['candidates'] if r['method']==method]
        counts[method] = dict(records=len(rows), failures=sum(r['status']!='success' for r in rows),
            passes={t:sum(r['status']=='success' and r['targets'][t]['passes'] for r in rows) for t in ('0.0001','1e-06')})
    crc = 0
    archives = list(OUT.glob('*.npz'))
    if (OUT/'packaging.json').exists():
        archives += [OUT/name for name in read(OUT/'packaging.json')['archives']]
    for p in archives:
        with zipfile.ZipFile(p if p.exists() else io.BytesIO(load_bytes(p))) as z: assert z.testzip() is None, p
        crc += 1
    return dict(counts=counts, candidate_records=72, mature_controls=96, reference_pairs=12,
        mature_failures=dict(Counter(r['reason'] for r in data['controls'] if r['status']=='failed')),
        candidate_failures=dict(Counter(r['reason'] for r in data['candidates'] if r['status']=='failed')),
        comparisons=comparisons, npz_crc_checked=crc, historical_manifests_unchanged=True,
        reaction_split_coverage=0., M4_admission=False)


def write(result):
    if (OUT/'manifest.json').exists(): raise FileExistsError('Preserve evidence')
    save(OUT/'summary.json', result)
    save(OUT/'environment.json', dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        platform=platform.platform(), parent_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        execution='scripts/run-macos.sh; single-thread BLAS/OpenMP'))
    paths = [p for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json']
    paths += [ROOT/'research'/f for f in ('remainder_curve.py','test_remainder_curve.py','m3_remainder_study.py','m3_remainder_report.py','remainder_storage.py')]
    paths += [ROOT/'docs/m3-remainder-curve.md']
    save(OUT/'manifest.json', {str(p.relative_to(ROOT)):digest(p) for p in paths})


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--write', action='store_true')
    args = parser.parse_args(); result = verify()
    if args.write: write(result)
    elif (OUT/'summary.json').exists():
        close(result, read(OUT/'summary.json'))
        for p,sha in read(OUT/'manifest.json').items(): assert digest(ROOT/p)==sha,p
    print(json.dumps({k:v for k,v in result.items() if k!='comparisons'}, indent=2))
