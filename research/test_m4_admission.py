"""Reject evidence patterns that could falsely grant M4 admission."""
import copy
import tempfile
import unittest
from pathlib import Path

from metabolism.m4_admission import evaluate, qualify_repeats


def runs(model='teusink'):
    end = 10. if model == 'teusink' else 1.
    metrics = dict(passes=True, finite=True, significant_negative_output=False,
                   concentration_error=.2, flux_error=.3, net_flux_error=.2,
                   invariant_error=1e-12, relative_ledger_error=1e-12)
    attempt = dict(status='accepted', time=0., h=end*.6, groups=[0],
                   audit_max=.2, audit_flux_max=.3, audit_net_flux_max=.2, estimator_max=.1)
    stats = dict(attempts=[attempt], accepted_steps=1, split_seconds=.02,
                 partition_builds=1, split_coverage=.6, clipping_count=0, projection_count=0)
    return [dict(model=model, parameters={'target':1e-3}, config='frozen-config',
                 repeat=i, status='success', seconds=.1+i*.001,
                 metrics=copy.deepcopy(metrics), stats=copy.deepcopy(stats)) for i in range(7)]


class AdmissionTests(unittest.TestCase):
    def qualified(self, rows, model='teusink'):
        return qualify_repeats(rows, model, 1e-3, split=True)['qualified']

    def test_complete_repeats_then_reject_empty_duplicate_or_mixed(self):
        self.assertTrue(self.qualified(runs()))
        self.assertFalse(self.qualified([]))
        rows=runs();rows[-1]['repeat']=0
        self.assertFalse(self.qualified(rows))
        rows=runs();rows[-1]['config']='other-config'
        self.assertFalse(self.qualified(rows))
        rows=runs();rows[-1]['parameters']['audit_rtol']=1e-4
        self.assertFalse(self.qualified(rows))

    def test_each_repeat_must_pass_actual_metrics(self):
        for mutate in (lambda r:r.update(status='failed'),
                       lambda r:r['metrics'].update(flux_error=2.),
                       lambda r:r['metrics'].pop('relative_ledger_error'),
                       lambda r:r.update(seconds=float('nan'))):
            rows=runs();mutate(rows[-1]);self.assertFalse(self.qualified(rows))
        rows=runs('kecoli');rows[-1]['metrics'].pop('net_flux_error')
        self.assertFalse(self.qualified(rows,'kecoli'))

    def test_coverage_requires_accepted_audited_reaction_intervals(self):
        for mutate in (lambda s:s.update(split_coverage=.8),
                       lambda s:s['attempts'][0].update(h=4.),
                       lambda s:s['attempts'][0].update(audit_flux_max=2.),
                       lambda s:s['attempts'][0].update(groups=[]),
                       lambda s:s['attempts'][0].update(groups=list(range(17))),
                       lambda s:s['attempts'][0].update(groups=[17]),
                       lambda s:s.update(attempts=[], accepted_steps=0),
                       lambda s:s['attempts'].append(dict(status='rejected',rollback_verified=False))):
            rows=runs();mutate(rows[-1]['stats']);self.assertFalse(self.qualified(rows))

    def test_missing_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            state=evaluate(Path(folder))
        self.assertEqual(state['M4_admission'],'not_admitted')
        self.assertEqual(state['evidence_integrity'],'fail')


if __name__ == '__main__':
    unittest.main()
