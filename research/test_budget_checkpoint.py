import unittest
import numpy as np
from research.budget_checkpoint import Checkpoint,fingerprint
from research.linear_context_row import LinearContextROW
from research.linear_error_output import LinearStreaming
from research.test_row_spc import nonlinear_case


class CheckpointChecks(unittest.TestCase):
    def test_nonzero_history_restore_recompute_and_stale_transaction(self):
        case,y=nonlinear_case();case['options']['flux_scale']=np.ones(4)
        kernel=LinearContextROW(case,[0]);est=LinearStreaming(case,y);q=np.linspace(0,1,17)
        first=kernel.trial(y,.01,1e-10);r=est.trial(first,q);est.commit(r);y=first['fine']
        self.assertGreater(np.linalg.norm(est.errors[1]),0)
        ledger={'physical':2e-12};actual={'concentration':.1}
        cp=Checkpoint(est,y,.01,ledger,actual,[0],[0],.02)
        trial=kernel.trial(y,.02,1e-10);old=est.trial(trial,q);est.commit(old)
        future=est.trial(kernel.trial(trial['fine'],.03,1e-10),q)
        version=est.generation
        restored=cp.restore(est,[0]);self.assertGreater(est.generation,version)
        self.assertEqual(fingerprint(est,*restored[:4],[0]),cp.fingerprint)
        with self.assertRaises(ValueError):est.commit(future)
        again=est.trial(kernel.trial(restored[0],.02,1e-10),q)
        for a,b in zip(again['errors'],old['errors']):np.testing.assert_array_equal(a,b)
        self.assertEqual(again['targets'],old['targets'])
        # Restored arrays/dicts do not alias saved checkpoint state.
        restored[0][0]+=1;restored[2]['physical']=1
        restored2=cp.restore(est,[0]);np.testing.assert_array_equal(restored2[0],y)
        self.assertEqual(restored2[2],ledger)

    def test_checkpoint_rejects_foreign_owner_and_group_change(self):
        case,y=nonlinear_case();case['options']['flux_scale']=np.ones(4)
        est=LinearStreaming(case,y);cp=Checkpoint(est,y,0.,{}, {},[0],[],.01)
        with self.assertRaises(ValueError):cp.restore(LinearStreaming(case,y),[0])
        with self.assertRaises(AssertionError):cp.restore(est,[1])


if __name__=='__main__':unittest.main()
