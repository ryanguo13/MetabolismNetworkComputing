"""Independent fail-closed evaluator for versioned M4 evidence bundles.

Evidence bundle schema v1 is intentionally separate from historical run files.
No saved pass flag or aggregate metric is trusted.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import median, quantiles

import numpy as np

MODELS = ('teusink', 'kecoli')
TARGETS = (1e-3, 1e-5)
WINDOWS = {'teusink': (0.0, 10.0), 'kecoli': (0.0, 1.0)}
GROUP_COUNTS = {'teusink': 17, 'kecoli': 2666}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite_number(value, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not positive or value > 0))


def _iqr(values):
    q = quantiles(values, n=4, method='inclusive')
    return q[2] - q[0]


def _canonical_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _audit_record_sha256(audit):
    return _canonical_sha256({key: value for key, value in audit.items()
                              if key != 'record_sha256'})


def _recompute_errors(run, model, target):
    required = ('values', 'reference', 'scale', 'flux_values', 'flux_reference',
                'flux_scale', 'times')
    if any(key not in run for key in required):
        raise ValueError('trajectory arrays missing')
    values = np.asarray(run['values'], dtype=np.float64)
    reference = np.asarray(run['reference'], dtype=np.float64)
    scale = np.asarray(run['scale'], dtype=np.float64)
    flux = np.asarray(run['flux_values'], dtype=np.float64)
    flux_reference = np.asarray(run['flux_reference'], dtype=np.float64)
    flux_scale = np.asarray(run['flux_scale'], dtype=np.float64)
    times = np.asarray(run['times'], dtype=np.float64)
    if (values.ndim != 2 or values.shape != reference.shape
            or values.shape[1] != len(times) or scale.shape != (values.shape[0],)
            or flux.ndim != 2 or flux.shape != flux_reference.shape
            or flux.shape[1] != len(times) or flux_scale.shape != (flux.shape[0],)
            or len(times) < 2 or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0) or not np.isfinite(values).all()
            or not np.isfinite(reference).all() or not np.isfinite(flux).all()
            or not np.isfinite(flux_reference).all() or not np.isfinite(scale).all()
            or not np.isfinite(flux_scale).all() or np.any(scale <= 0)
            or np.any(flux_scale <= 0)):
        raise ValueError('invalid trajectory arrays')
    begin, end = WINDOWS[model]
    if times[0] != begin or times[-1] != end:
        raise ValueError('trajectory window mismatch')
    concentration_error = float(np.max(np.abs(values-reference) / (scale[:, None]*target)))
    flux_error = float(np.max(np.abs(flux-flux_reference) / (flux_scale[:, None]*target)))
    if 'pool_matrix' not in run or 'initial_state' not in run:
        raise ValueError('pool matrix or initial state missing')
    pools = np.asarray(run['pool_matrix'], dtype=np.float64)
    initial = np.asarray(run['initial_state'], dtype=np.float64)
    if (pools.ndim != 2 or pools.shape[1] != values.shape[0]
            or initial.shape != (values.shape[0],) or not np.isfinite(pools).all()
            or not np.isfinite(initial).all()):
        raise ValueError('invalid pool matrix or initial state')
    invariant_error = float(np.max(np.abs(pools @ (values-initial[:, None])))) / max(
        1e-12, float(np.max(np.abs(pools @ initial))))
    ledger_error = None
    if model == 'teusink':
        if 'ledger_weights' not in run or 'ledger_increment' not in run:
            raise ValueError('open ledger weights or increments missing')
        weights = np.asarray(run['ledger_weights'], dtype=np.float64)
        increment = np.asarray(run['ledger_increment'], dtype=np.float64)
        if (weights.shape != (values.shape[0],) or increment.shape != (len(times),)
                or not np.isfinite(weights).all() or not np.isfinite(increment).all()):
            raise ValueError('invalid open ledger evidence')
        residual = weights @ (values-initial[:, None]) - increment
        ledger_error = float(np.max(np.abs(residual))) / max(1., abs(float(weights @ initial)))
    net_flux_error = None
    if model == 'kecoli':
        required_net = ('net_map', 'net_scale', 'net_target')
        if any(key not in run for key in required_net) or run['net_target'] != 1e-3:
            raise ValueError('frozen net flux contract missing')
        net_map = np.asarray(run['net_map'], dtype=np.float64)
        net_scale = np.asarray(run['net_scale'], dtype=np.float64)
        if (net_map.ndim != 2 or net_map.shape[1] != flux.shape[0]
                or net_scale.shape != (net_map.shape[0],) or not np.isfinite(net_map).all()
                or not np.isfinite(net_scale).all() or np.any(net_scale <= 0)):
            raise ValueError('invalid net flux map or scale')
        net_flux_error = float(np.max(np.abs(net_map @ (flux-flux_reference)) /
                                      (net_scale[:, None]*run['net_target'])))
    return dict(concentration_error=concentration_error, flux_error=flux_error,
                invariant_error=invariant_error, relative_ledger_error=ledger_error,
                net_flux_error=net_flux_error,
                finite=bool(np.isfinite(values).all() and np.isfinite(flux).all()),
                significant_negative_output=bool(np.any(values < -1e-12*scale[:, None])))


def _check_identity(run, registration, combo):
    identity = run.get('identity')
    expected = (registration.get('method_identities') or {}).get(run.get('method'))
    if not isinstance(identity, dict) or not isinstance(expected, dict):
        raise ValueError('run/method identity missing')
    for key in ('model_sha256', 'reference_sha256', 'grid_sha256', 'scale_sha256',
                'initial_state_sha256', 'structure_sha256', 'environment_sha256',
                'config_sha256', 'platform_sha256'):
        value = identity.get(key)
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in '0123456789abcdef' for char in value)
                or value != expected.get(key)):
            raise ValueError(f'identity mismatch: {key}')
    if run.get('model') != combo['model'] or run.get('target') != combo['target']:
        raise ValueError('run model/target mismatch')
    if identity.get('config_sha256') != run.get('config_sha256'):
        raise ValueError('run configuration fingerprint mismatch')
    if identity.get('platform_sha256') != run.get('platform_sha256'):
        raise ValueError('run platform fingerprint mismatch')
    payload_keys = ('values', 'reference', 'scale', 'flux_values', 'flux_reference',
                    'flux_scale', 'times', 'pool_matrix', 'initial_state', 'attempts')
    if combo['model'] == 'teusink':
        payload_keys += ('ledger_weights', 'ledger_increment')
    else:
        payload_keys += ('net_map', 'net_scale', 'net_target')
    payload = {key: run.get(key) for key in payload_keys}
    if run.get('payload_sha256') != _canonical_sha256(payload):
        raise ValueError('raw evidence payload hash mismatch')
    digests = {
        'reference_sha256': _canonical_sha256(run.get('reference')),
        'grid_sha256': _canonical_sha256(run.get('times')),
        'scale_sha256': _canonical_sha256(run.get('scale')),
        'initial_state_sha256': _canonical_sha256(run.get('initial_state')),
        'structure_sha256': _canonical_sha256(
            ({'pool_matrix': run.get('pool_matrix'),
              'ledger_weights': run.get('ledger_weights')}
             if combo['model'] == 'teusink' else
             {'pool_matrix': run.get('pool_matrix'), 'net_map': run.get('net_map'),
              'net_scale': run.get('net_scale'), 'net_target': run.get('net_target')})), 
    }
    for key, actual in digests.items():
        if identity.get(key) != actual:
            raise ValueError(f'array fingerprint mismatch: {key}')
    model_path = run.get('model_path')
    if not isinstance(model_path, str) or Path(model_path).is_absolute() or '..' in Path(model_path).parts:
        raise ValueError('unsafe or missing model path')
    project_root = Path(__file__).resolve().parents[1]
    if sha256(project_root/model_path) != identity.get('model_sha256'):
        raise ValueError('model file fingerprint mismatch')
    source_files = registration.get('source_files')
    source_hashes = registration.get('source_sha256')
    if not isinstance(source_files, list) or not source_files or not isinstance(source_hashes, dict):
        raise ValueError('source manifest missing')
    for source in source_files:
        source_path = Path(source)
        if source_path.is_absolute() or '..' in source_path.parts or source not in source_hashes:
            raise ValueError('invalid source manifest path')
        if sha256(project_root/source_path) != source_hashes[source]:
            raise ValueError(f'source fingerprint mismatch: {source}')


def _check_attempts(run, model, bundle):
    attempts = run.get('attempts')
    if not isinstance(attempts, list) or not attempts:
        raise ValueError('raw attempts missing')
    begin, end = WINDOWS[model]
    accepted = []
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict) or attempt.get('attempt_id') != index:
            raise ValueError('attempt sequence is incomplete or duplicated')
        status = attempt.get('status')
        if status == 'rejected':
            if attempt.get('rollback_verified') is not True:
                raise ValueError('rejected attempt lacks rollback evidence')
            continue
        if status != 'accepted':
            raise ValueError('unknown attempt status')
        start, step = attempt.get('time'), attempt.get('h')
        if not _finite_number(start) or not _finite_number(step, positive=True):
            raise ValueError('invalid accepted interval')
        groups = attempt.get('groups')
        if (not isinstance(groups, list) or not groups or len(set(groups)) != len(groups)
                or len(groups) >= GROUP_COUNTS[model]
                or any(type(group) is not int or not 0 <= group < GROUP_COUNTS[model]
                       for group in groups)):
            raise ValueError('accepted interval is not a proper reaction split')
        if not all(attempt.get(key) is True for key in
                   ('full_ode_audit_passed', 'requested_outputs_audited',
                    'internal_checkpoints_audited', 'transaction_committed')):
            raise ValueError('accepted interval lacks linked complete audit/commit evidence')
        audit_id = attempt.get('audit_id')
        audit = (bundle.get('audits') or {}).get(audit_id)
        if (not isinstance(audit, dict) or audit.get('kind') != 'full_nonlinear_ode'
                or audit.get('independent') is not True or audit.get('passed') is not True):
            raise ValueError('accepted interval lacks a passing independent full-ODE audit record')
        start_state_hash = attempt.get('start_state_sha256')
        if (not isinstance(start_state_hash, str) or len(start_state_hash) != 64
                or any(char not in '0123456789abcdef' for char in start_state_hash)
                or audit.get('candidate_state_sha256') != start_state_hash):
            raise ValueError('audit does not bind to the accepted candidate start state')
        solver_counts = ('nfev', 'njev', 'nlu')
        if (not isinstance(audit.get('method'), str) or not audit['method']
                or not _finite_number(audit.get('rtol'), positive=True)
                or not _finite_number(audit.get('atol'), positive=True)
                or type(audit.get('solver_calls')) is not int or audit['solver_calls'] < 1
                or any(type(audit.get(key)) is not int or audit[key] < 0 for key in solver_counts)):
            raise ValueError('audit solver configuration or work counts missing')
        policy = (bundle.get('registrations', {}).get(f'{model}:{run.get("target")}') or {}).get('audit_policy')
        if (not isinstance(policy, dict) or audit['method'] != policy.get('method')
                or audit['rtol'] != policy.get('rtol') or audit['atol'] != policy.get('atol')):
            raise ValueError('audit configuration differs from frozen registration')
        record_hash = audit.get('record_sha256')
        if (not isinstance(record_hash, str) or len(record_hash) != 64
                or any(char not in '0123456789abcdef' for char in record_hash)
                or record_hash != _audit_record_sha256(audit)):
            raise ValueError('audit record content hash mismatch')
        checkpoint_ids = attempt.get('checkpoint_ids')
        checkpoints = bundle.get('checkpoints') or {}
        if not isinstance(checkpoint_ids, list) or not checkpoint_ids:
            raise ValueError('accepted interval lacks checkpoint links')
        stop = start + step
        if start < begin or stop > end:
            raise ValueError('accepted interval outside frozen window')
        requested_outputs = audit.get('requested_output_ids')
        if (audit.get('start') != start or audit.get('end') != stop
                or not isinstance(requested_outputs, list) or not requested_outputs
                or len(set(requested_outputs)) != len(requested_outputs)
                or not set(requested_outputs).issubset(set(bundle.get('outputs', {})))
                or not audit.get('checkpoint_ids')):
            raise ValueError('audit record does not cover accepted interval and outputs')
        for output_id in requested_outputs:
            output = bundle['outputs'][output_id]
            if (not isinstance(output, dict) or output.get('audit_id') != audit_id
                    or not _finite_number(output.get('time'))
                    or not start <= output['time'] <= stop
                    or not isinstance(output.get('state_sha256'), str)
                    or len(output['state_sha256']) != 64):
                raise ValueError('requested audit output lacks linked state/time evidence')
        if set(checkpoint_ids) != set(audit['checkpoint_ids']):
            raise ValueError('attempt/audit checkpoint links disagree')
        for checkpoint_id in checkpoint_ids:
            checkpoint = checkpoints.get(checkpoint_id)
            if (not isinstance(checkpoint, dict) or checkpoint.get('audit_id') != audit_id
                    or not _finite_number(checkpoint.get('time'))
                    or not start <= checkpoint['time'] <= stop
                    or not isinstance(checkpoint.get('state_sha256'), str)
                    or len(checkpoint['state_sha256']) != 64):
                raise ValueError('missing or invalid independently audited checkpoint')
        accepted.append((start, stop))
    accepted.sort()
    previous, duration = begin, 0.0
    for start, stop in accepted:
        if start < previous - 1e-12:
            raise ValueError('accepted intervals overlap')
        duration += stop-start
        previous = stop
    coverage = duration / (end-begin)
    if not math.isclose(coverage, run.get('split_coverage', math.nan), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError('stored split coverage mismatch')
    return coverage


def evaluate(bundle):
    """Evaluate schema-v1 evidence dict; report missing evidence without raising."""
    failures = []
    if not isinstance(bundle, dict) or bundle.get('schema_version') != 1:
        return dict(evidence_integrity='fail', M4_admission='not_admitted',
                    failures=['unsupported or missing schema_version'], combinations=[])
    registrations = bundle.get('registrations')
    runs = bundle.get('runs')
    if not isinstance(registrations, dict) or not isinstance(runs, list):
        return dict(evidence_integrity='fail', M4_admission='not_admitted',
                    failures=['registrations/runs missing'], combinations=[])
    combinations = []
    for model in MODELS:
        for target in TARGETS:
            combo = dict(model=model, target=target)
            registration = registrations.get(f'{model}:{target}')
            rows = [r for r in runs if isinstance(r, dict) and r.get('model') == model and r.get('target') == target]
            local_failures = []
            try:
                if not isinstance(registration, dict):
                    raise ValueError('registration missing')
                if (registration.get('window') != list(WINDOWS[model])
                        or registration.get('repeats') != 7
                        or registration.get('coverage_gate') != .5
                        or registration.get('speedup_gate') != 1.5
                        or not isinstance(registration.get('method_identities'), dict)
                        or set(registration['method_identities']) != {'candidate', 'baseline'}
                        or not isinstance(registration.get('baseline_qualification'), dict)
                        or registration['baseline_qualification'].get('independent_metrics_passed') is not True
                        or registration['baseline_qualification'].get('selected_as_fastest_qualified') is not True):
                    raise ValueError('frozen protocol registration mismatch')
                for method in ('baseline', 'candidate'):
                    method_rows = [r for r in rows if r.get('method') == method]
                    warmups = [r for r in method_rows if r.get('warmup') is True]
                    formal = [r for r in method_rows if r.get('warmup') is False]
                    if len(warmups) != 1 or warmups[0].get('repeat') != -1:
                        raise ValueError(f'{method}: warmup missing/duplicate')
                    if len(formal) != 7 or {r.get('repeat') for r in formal} != set(range(7)):
                        raise ValueError(f'{method}: formal repeat IDs incomplete/duplicate')
                    if any(r.get('order_index') is None for r in formal):
                        raise ValueError(f'{method}: formal acquisition order missing')
                    if len({r.get('config_sha256') for r in formal}) != 1:
                        raise ValueError(f'{method}: mixed configurations')
                    if len({r.get('platform_sha256') for r in formal}) != 1:
                        raise ValueError(f'{method}: mixed platforms')
                formal_rows = [r for r in rows if r.get('warmup') is False]
                ordered = sorted(formal_rows, key=lambda r: r['order_index'])
                if (len(ordered) != 14 or [r['order_index'] for r in ordered] != list(range(14))
                        or any(ordered[i]['method'] == ordered[i+1]['method'] for i in range(13))):
                    raise ValueError('formal baseline/candidate runs are not completely interleaved')
                candidate_rows = [r for r in rows if r.get('method') == 'candidate' and r.get('warmup') is False]
                baseline_rows = [r for r in rows if r.get('method') == 'baseline' and r.get('warmup') is False]
                candidate_seconds, baseline_seconds, metrics = [], [], []
                coverages = []
                for row in candidate_rows + baseline_rows:
                    _check_identity(row, registration, combo)
                    if row.get('status') != 'success' or not _finite_number(row.get('seconds'), positive=True):
                        raise ValueError('run failed or timing invalid')
                    recomputed = _recompute_errors(row, model, target)
                    metrics.append((row['method'], row['repeat'], recomputed))
                    if (recomputed['concentration_error'] > 1 or recomputed['flux_error'] > 1
                            or recomputed['invariant_error'] > 1e-10 or not recomputed['finite']
                            or recomputed['significant_negative_output']
                            or (model == 'teusink' and recomputed['relative_ledger_error'] > 1e-10)
                            or (model == 'kecoli' and recomputed['net_flux_error'] > 1)):
                        raise ValueError(f"{row['method']} repeat {row['repeat']}: recomputed correctness gate failed")
                    if row['method'] == 'candidate':
                        coverages.append(_check_attempts(row, model, bundle))
                        candidate_seconds.append(row['seconds'])
                    else:
                        baseline_seconds.append(row['seconds'])
                if any(value < .5 for value in coverages):
                    raise ValueError('candidate split coverage below 0.5')
                speedup = median(baseline_seconds) / median(candidate_seconds)
                if speedup < 1.5:
                    raise ValueError('median speedup below 1.5')
                combo.update(evidence_integrity='pass', correctness='pass', coverage='pass',
                             repeats='pass', timing='pass', efficiency='pass',
                             candidate_median=median(candidate_seconds), candidate_iqr=_iqr(candidate_seconds),
                             baseline_median=median(baseline_seconds), baseline_iqr=_iqr(baseline_seconds),
                             speedup=speedup, coverages=coverages, recomputed_metrics=metrics)
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                local_failures.append(str(exc))
                combo.update(evidence_integrity='fail', correctness='not_verified',
                             coverage='not_verified', repeats='not_verified', timing='not_verified',
                             efficiency='not_admitted')
            combo['failures'] = local_failures
            failures.extend(f'{model}/{target}: {reason}' for reason in local_failures)
            combinations.append(combo)
    integrity = not failures and len(combinations) == 4
    admitted = integrity and all(c.get('efficiency') == 'pass' for c in combinations)
    return dict(schema_version=1, evidence_integrity='pass' if integrity else 'fail',
                M4_admission='pass' if admitted else 'not_admitted',
                combinations=combinations, failures=failures)


def audit_historical_directory(path):
    """Explicitly classify legacy runs without output arrays as non-recomputable."""
    path = Path(path)
    try:
        registration = json.loads((path/'registration.json').read_text())
        rows = json.loads((path/'runs.json').read_text()).get('runs', [])
    except (OSError, ValueError, AttributeError) as exc:
        return dict(evidence_integrity='fail', M4_admission='not_admitted',
                    failures=[f'cannot read historical evidence: {exc}'])
    has_curves = bool(rows) and all(isinstance(row, dict) and 'values_file' in row for row in rows)
    return dict(evidence_integrity='not_recomputable' if not has_curves else 'requires_bundle_conversion',
                M4_admission='not_admitted', registration_present=bool(registration),
                run_count=len(rows),
                failures=[] if has_curves else ['legacy run records do not contain saved trajectory arrays; metrics cannot be independently recomputed'])
