"""Adversarial tests for the independent M4 evidence-bundle evaluator."""
import copy
import unittest

import numpy as np

from research.audit_m4_admission import (MODELS, TARGETS, WINDOWS,
    _audit_record_sha256, _canonical_sha256, audit_historical_directory, evaluate)


def _identity(method, model, reference, times, scale, initial_state, structure):
    import hashlib
    root = __import__('pathlib').Path(__file__).resolve().parents[1]
    model_path = ('models/BIOMD0000000064.xml' if model == 'teusink' else 'models/k-ecoli457.zip')
    hashes = lambda value: _canonical_sha256(value)
    return dict(model_sha256=hashlib.sha256((root/model_path).read_bytes()).hexdigest(),
        reference_sha256=hashes(reference), grid_sha256=hashes(times), scale_sha256=hashes(scale),
        initial_state_sha256=hashes(initial_state), structure_sha256=hashes(structure),
        environment_sha256=hashlib.sha256(b'test-environment').hexdigest(),
        config_sha256=hashlib.sha256(f'{method}-config'.encode()).hexdigest(),
        platform_sha256=hashlib.sha256(b'test-platform').hexdigest())


def _refresh_payload_hash(row):
    keys = ('values', 'reference', 'scale', 'flux_values', 'flux_reference',
            'flux_scale', 'times', 'pool_matrix', 'initial_state', 'attempts')
    keys += ('ledger_weights', 'ledger_increment') if row['model'] == 'teusink' else (
        'net_map', 'net_scale', 'net_target')
    row['payload_sha256'] = _canonical_sha256({key: row.get(key) for key in keys})


def _attempts(model):
    begin, end = WINDOWS[model]
    half = (end-begin)/2
    result = []
    for i in range(2):
        result.append(dict(attempt_id=i, status='accepted', time=begin+i*half, h=half,
            groups=[0], full_ode_audit_passed=True, requested_outputs_audited=True,
            internal_checkpoints_audited=True, transaction_committed=True,
            start_state_sha256='a'*64, audit_id=f'audit-{i}', checkpoint_ids=[f'cp-{i}']))
    return result


def _run(model, target, method, repeat, warmup=False):
    n = 14 if model == 'teusink' else 3003
    begin, end = WINDOWS[model]
    times = [begin, end]
    initial = np.ones(n)
    ref = np.column_stack((initial, initial))
    scale = np.ones(n).tolist()
    pool_matrix = [np.ones(n).tolist()]
    structure = {'pool_matrix': pool_matrix}
    if model == 'teusink':
        structure.update(ledger_weights=scale)
    else:
        structure.update(net_map=[[1.]], net_scale=[1.], net_target=1e-3)
    identity = _identity(method, model, ref.tolist(), times, scale, initial.tolist(), structure)
    run = dict(model=model, target=target, method=method, repeat=-1 if warmup else repeat,
        warmup=warmup, status='success', seconds=.05 if method == 'candidate' else .1,
        config_sha256=identity['config_sha256'], platform_sha256=identity['platform_sha256'],
        identity=identity, model_path=('models/BIOMD0000000064.xml' if model == 'teusink' else 'models/k-ecoli457.zip'),
        values=ref.tolist(), reference=ref.tolist(), scale=scale,
        flux_values=[[1., 1.]], flux_reference=[[1., 1.]], flux_scale=[1.], times=times,
        pool_matrix=pool_matrix, initial_state=initial.tolist())
    if model == 'teusink':
        run.update(ledger_weights=np.ones(n).tolist(), ledger_increment=[0., 0.])
    else:
        run.update(net_map=[[1.]], net_scale=[1.], net_target=1e-3)
    if method == 'candidate':
        run.update(attempts=_attempts(model), split_coverage=1.)
    payload_keys = ('values', 'reference', 'scale', 'flux_values', 'flux_reference',
                    'flux_scale', 'times', 'pool_matrix', 'initial_state', 'attempts')
    if model == 'teusink':
        payload_keys += ('ledger_weights', 'ledger_increment')
    else:
        payload_keys += ('net_map', 'net_scale', 'net_target')
    run['payload_sha256'] = _canonical_sha256({key: run.get(key) for key in payload_keys})
    return run


def valid_bundle():
    registrations, rows = {}, []
    for model in MODELS:
        for target in TARGETS:
            key = f'{model}:{target}'
            registrations[key] = dict(window=list(WINDOWS[model]), repeats=7,
                coverage_gate=.5, speedup_gate=1.5,
                audit_policy=dict(method='BDF', rtol=1e-9, atol=1e-11),
                method_identities={method: _identity(method, model,
                    np.column_stack((np.ones(14 if model == 'teusink' else 3003),)*2).tolist(),
                    [WINDOWS[model][0], WINDOWS[model][1]],
                    np.ones(14 if model == 'teusink' else 3003).tolist(),
                    np.ones(14 if model == 'teusink' else 3003).tolist(),
                    ({'pool_matrix': [np.ones(14 if model == 'teusink' else 3003).tolist()],
                      'ledger_weights': np.ones(14 if model == 'teusink' else 3003).tolist()}
                     if model == 'teusink' else
                     {'pool_matrix': [np.ones(3003).tolist()], 'net_map': [[1.]],
                      'net_scale': [1.], 'net_target': 1e-3}))
                    for method in ('baseline', 'candidate')},
                baseline_qualification={'independent_metrics_passed': True,
                    'selected_as_fastest_qualified': True},
                source_files=['research/audit_m4_admission.py'],
                source_sha256={'research/audit_m4_admission.py': __import__('hashlib').sha256(
                    (__import__('pathlib').Path(__file__).resolve().parents[1]/'research/audit_m4_admission.py').read_bytes()).hexdigest()})
            for method in ('baseline', 'candidate'):
                rows.append(_run(model, target, method, -1, warmup=True))
                rows.extend(_run(model, target, method, repeat) for repeat in range(7))
    audits, checkpoints, outputs = {}, {}, {}
    for row in rows:
        if row.get('method') != 'candidate':
            continue
        model = row['model']
        begin, end = WINDOWS[model]
        half = (end-begin)/2
        for i, attempt in enumerate(row['attempts']):
            audit_id = f"audit-{model}-{row['target']}-{row['repeat']}-{i}"
            checkpoint_id = f"cp-{model}-{row['target']}-{row['repeat']}-{i}"
            output_id = f"out-{model}-{row['target']}-{row['repeat']}-{i}"
            audits[audit_id] = dict(kind='full_nonlinear_ode', independent=True, passed=True,
                start=begin+i*half, end=begin+(i+1)*half, requested_output_ids=[output_id],
                checkpoint_ids=[checkpoint_id], candidate_state_sha256=attempt['start_state_sha256'],
                method='BDF', rtol=1e-9, atol=1e-11, solver_calls=1,
                nfev=12, njev=1, nlu=3)
            audits[audit_id]['record_sha256'] = _audit_record_sha256(audits[audit_id])
            checkpoints[checkpoint_id] = dict(audit_id=audit_id, time=begin+(i+1)*half,
                state_sha256='b'*64)
            outputs[output_id] = dict(audit_id=audit_id, time=begin+(i+1)*half,
                state_sha256='b'*64)
            attempt['audit_id'] = audit_id
            attempt['checkpoint_ids'] = [checkpoint_id]
        _refresh_payload_hash(row)
    for model in MODELS:
        for target in TARGETS:
            combo_rows = [row for row in rows if row.get('model') == model
                          and row.get('target') == target and not row.get('warmup')]
            by_method = {method: sorted((row for row in combo_rows if row['method'] == method),
                                        key=lambda row: row['repeat'])
                         for method in ('baseline', 'candidate')}
            for repeat in range(7):
                by_method['baseline'][repeat]['order_index'] = repeat * 2
                by_method['candidate'][repeat]['order_index'] = repeat * 2 + 1
    source_path = __import__('pathlib').Path(__file__).resolve().parents[1]/'research/audit_m4_admission.py'
    source_hash = __import__('hashlib').sha256(source_path.read_bytes()).hexdigest()
    return dict(schema_version=1, registrations=registrations, runs=rows,
                audits=audits, checkpoints=checkpoints, outputs=outputs)


class AdmissionAuditTests(unittest.TestCase):
    def test_complete_bundle_passes_with_recomputed_metrics_and_speed(self):
        result = evaluate(valid_bundle())
        self.assertEqual(result['evidence_integrity'], 'pass', result['failures'])
        self.assertEqual(result['M4_admission'], 'pass')
        self.assertTrue(all(row['coverage'] == 'pass' and row['efficiency'] == 'pass'
                            for row in result['combinations']))

    def test_saved_pass_flags_cannot_override_bad_curve(self):
        bundle = valid_bundle()
        row = next(r for r in bundle['runs'] if r['model'] == 'teusink'
                   and r['target'] == 1e-3 and r['method'] == 'candidate' and r['repeat'] == 0)
        row['values'][0][1] = 10.
        row['metrics'] = {'passes': True}
        payload_keys = ('values', 'reference', 'scale', 'flux_values', 'flux_reference',
                        'flux_scale', 'times', 'pool_matrix', 'initial_state', 'attempts',
                        'ledger_weights', 'ledger_increment')
        _refresh_payload_hash(row)
        result = evaluate(bundle)
        self.assertEqual(result['M4_admission'], 'not_admitted')
        self.assertTrue(any('correctness gate failed' in item for item in result['failures']))

    def test_missing_arrays_and_false_coverage_fail_closed(self):
        for mutation in (lambda row: row.pop('values'),
                         lambda row: row.update(split_coverage=.99)):
            bundle = valid_bundle()
            row = next(r for r in bundle['runs'] if r['model'] == 'kecoli'
                       and r['target'] == 1e-3 and r['method'] == 'candidate' and r['repeat'] == 0)
            mutation(row)
            self.assertEqual(evaluate(bundle)['M4_admission'], 'not_admitted')

    def test_mixed_config_duplicate_repeat_and_unverified_rollback_fail(self):
        mutations = []
        def mixed(bundle):
            row = next(r for r in bundle['runs'] if r['model'] == 'teusink'
                       and r['target'] == 1e-3 and r['method'] == 'candidate' and r['repeat'] == 0)
            row['config_sha256'] = 'different'
        mutations.append(mixed)
        def duplicate(bundle):
            row = next(r for r in bundle['runs'] if r['model'] == 'teusink'
                       and r['target'] == 1e-3 and r['method'] == 'baseline' and r['repeat'] == 6)
            row['repeat'] = 5
        mutations.append(duplicate)
        def rollback(bundle):
            row = next(r for r in bundle['runs'] if r['model'] == 'kecoli'
                       and r['target'] == 1e-5 and r['method'] == 'candidate' and r['repeat'] == 0)
            row['attempts'][0]['status'] = 'rejected'
            row['attempts'][0]['rollback_verified'] = False
            _refresh_payload_hash(row)
        mutations.append(rollback)
        for mutation in mutations:
            bundle = valid_bundle()
            mutation(bundle)
            self.assertEqual(evaluate(bundle)['M4_admission'], 'not_admitted')

    def test_audit_provenance_configuration_counts_and_output_links_are_bound(self):
        def first_ids(bundle):
            audit_id = next(key for key in bundle['audits']
                            if key.startswith('audit-teusink-0.001-0-0'))
            output_id = bundle['audits'][audit_id]['requested_output_ids'][0]
            return audit_id, output_id

        def audit_mutation(change):
            return lambda bundle: change(bundle['audits'][first_ids(bundle)[0]])

        mutations = (
            audit_mutation(lambda audit: audit.pop('candidate_state_sha256')),
            audit_mutation(lambda audit: audit.update(candidate_state_sha256='c'*64)),
            audit_mutation(lambda audit: audit.update(rtol=1e-8)),
            audit_mutation(lambda audit: audit.update(solver_calls=0)),
            audit_mutation(lambda audit: audit.update(nfev=-1)),
            audit_mutation(lambda audit: audit.update(nlu=5)),
            audit_mutation(lambda audit: audit.update(requested_output_ids=['missing'])),
            lambda b: b['outputs'][first_ids(b)[1]].update(audit_id='other-audit'),
        )
        for mutation in mutations:
            bundle = valid_bundle()
            mutation(bundle)
            self.assertEqual(evaluate(bundle)['M4_admission'], 'not_admitted')

    def test_missing_combination_and_bad_timing_fail(self):
        bundle = valid_bundle()
        bundle['runs'] = [r for r in bundle['runs'] if not (r['model'] == 'kecoli' and r['target'] == 1e-5)]
        self.assertEqual(evaluate(bundle)['M4_admission'], 'not_admitted')
        bundle = valid_bundle()
        row = next(r for r in bundle['runs'] if r['model'] == 'teusink'
                   and r['target'] == 1e-5 and r['method'] == 'candidate' and r['repeat'] == 0)
        row['seconds'] = float('nan')
        self.assertEqual(evaluate(bundle)['M4_admission'], 'not_admitted')

    def test_historical_records_without_curve_arrays_are_not_recomputable(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        result = audit_historical_directory(root/'results/m4-formal-audit-control-only')
        self.assertEqual(result['evidence_integrity'], 'not_recomputable')
        self.assertEqual(result['M4_admission'], 'not_admitted')


if __name__ == '__main__':
    unittest.main()
