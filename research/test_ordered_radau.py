import unittest
import numpy as np
from scipy.sparse import csc_matrix,diags
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import solve_ivp,Radau
from research.ordered_radau import OrderedRadau


class OrderedChecks(unittest.TestCase):
    def test_dense_and_sparse_against_matrix_exponential(self):
        for n in (2,40):
            rates=np.geomspace(1,1e4,n)
            matrix=diags([-rates,rates[:-1]],[0,-1],shape=(n,n),format='lil')
            matrix[-1,-1]=0;matrix=csc_matrix(matrix)
            y=np.zeros(n);y[0]=1
            times=np.linspace(0,.1,17)
            exact=np.column_stack([expm_multiply(matrix*t,y) for t in times])
            results=[]
            for method in (Radau,OrderedRadau):
                r=solve_ivp(lambda t,z:matrix@z,(0,.1),y,t_eval=times,jac=matrix,
                            method=method,rtol=1e-9,atol=1e-12)
                self.assertTrue(r.success);results.append(r.y)
                np.testing.assert_allclose(r.y,exact,atol=2e-10,rtol=2e-9)
                self.assertLess(np.max(abs(r.y.sum(axis=0)-1)),1e-12)
                self.assertGreaterEqual(r.y.min(),-1e-12)
            np.testing.assert_allclose(*results,atol=2e-10,rtol=2e-9)

if __name__=='__main__':unittest.main()
