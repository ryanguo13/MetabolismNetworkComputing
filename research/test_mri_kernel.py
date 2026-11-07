import unittest
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from metabolism.splitting import ReactionSystem
from research.mri_kernel import Hermite,MRI
from research.mri_native import NativeSession


class KernelChecks(unittest.TestCase):
    def test_hermite_internal_negative_and_cubic(self):
        p=Hermite(np.array([0.]),np.array([0.]),np.array([-1.]),np.array([1.]),1.)
        self.assertAlmostEqual(p.minima()[0],-.25)
        p=Hermite(np.array([1.]),np.array([8.]),np.array([3.]),np.array([12.]),1.)
        np.testing.assert_allclose(p(np.array([.2,.5,.7]))[0],(1+np.array([.2,.5,.7]))**3)
        self.assertEqual(p.minima()[0],1.)

    def test_forced_inactive_coordinate_and_open_ledger(self):
        # Fast X production depends on Z; slow Z inflow must evolve during fast solves.
        s=csc_matrix([[1.,0.],[0.,1.]])
        system=ReactionSystem(s,lambda t,y:np.array([y[1],1.]),np.array([1.,1.]),
                              [np.array([0]),np.array([1])],csc_matrix([[0,1],[0,0]]),
                              lambda t,y:csc_matrix([[0.,1.],[0.,0.]]))
        case=dict(system=system,scale=np.ones(2),options=dict(pools=csc_matrix((1,2)),ledger_weights=np.ones(2)))
        full=MRI(case,[0],reduced=False)
        reduced=MRI(case,[0],reduced=True)
        a=full.trial(system.y0,.2)
        b=reduced.trial(system.y0,.2)
        np.testing.assert_allclose(b['fine'],[1.22,1.2],atol=1e-11)
        np.testing.assert_allclose(a['fine'],b['fine'],atol=1e-11)
        self.assertLess(b['stats']['modified_ledger_residual'],1e-12)
        self.assertLess(b['stats']['physical_ledger_residual'],1e-12)
        case['options'].pop('ledger_weights')
        native=MRI(case,[0],backend='native')
        try:
            c=native.trial(system.y0,.2,tol=1e-12)
            np.testing.assert_allclose(c['fine'],[1.22,1.2],atol=2e-10)
        finally:
            native.close()

    def test_native_reinit_changed_forcing_and_failure_recovery(self):
        a=np.array([[-3.,1.],[3.,-1.]])
        session=NativeSession(csc_matrix(a != 0))
        try:
            for offset in (np.array([.1,-.1]),np.array([-.2,.2])):
                y=np.array([.8,.2])
                model=SimpleNamespace(y0=y,rhs=lambda t,z:a@z+offset,
                    jacobian=lambda t,z:csc_matrix(a),accepted=lambda t,z:None)
                actual,_=session.solve(model,.2,1e-12,np.full(2,1e-14))
                m=np.zeros((3,3));m[:2,:2]=a;m[:2,2]=offset
                np.testing.assert_allclose(actual,(expm(.2*m)@np.r_[y,1])[:2],atol=1e-10)
            def fail(t,z):
                raise ValueError('intentional callback failure')
            model.rhs=fail
            with self.assertRaises(RuntimeError):session.solve(model,.2,1e-12,np.full(2,1e-14))
            model.rhs=lambda t,z:a@z
            actual,_=session.solve(model,.2,1e-12,np.full(2,1e-14))
            np.testing.assert_allclose(actual,expm(.2*a)@y,atol=1e-10)
        finally:
            session.close()
            session.close()


if __name__ == '__main__':
    unittest.main()
