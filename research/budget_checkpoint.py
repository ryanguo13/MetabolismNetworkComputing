"""Numerical checkpoints for the stateless-factor ROW streaming experiment.

All physical/error/budget history is restored, but transaction generations are
monotone so an abandoned branch cannot commit after a restore. Work counters
and attempt records belong to the caller and must never be rolled back.
"""
from copy import deepcopy
import hashlib
import json
import numpy as np


def fingerprint(est,y,left,ledger,actual,groups):
    d=hashlib.sha256()
    for v in [y,est.expected,*est.errors,np.asarray(groups),np.array([left])]:d.update(v.tobytes())
    d.update(json.dumps(dict(scores=est.scores,ledger=ledger,actual=actual),sort_keys=True).encode())
    return d.hexdigest()


class Checkpoint:
    def __init__(self,est,y,left,ledger,actual,groups,accepted,next_h):
        np.testing.assert_array_equal(y,est.expected)
        self.owner=est;self.y=y.copy();self.left=left;self.ledger=ledger.copy();self.actual=actual.copy();self.groups=np.asarray(groups).copy()
        self.errors=[v.copy() for v in est.errors];self.scores=deepcopy(est.scores);self.accepted=tuple(accepted);self.next_h=next_h
        self.fingerprint=fingerprint(est,y,left,ledger,actual,groups)

    def restore(self,est,groups):
        if est is not self.owner:raise ValueError('Foreign checkpoint')
        np.testing.assert_array_equal(groups,self.groups)
        est.errors=[v.copy() for v in self.errors];est.expected=self.y.copy();est.scores=deepcopy(self.scores)
        # Never reuse a transaction version from an abandoned branch.
        est.generation+=1
        assert fingerprint(est,self.y,self.left,self.ledger,self.actual,groups)==self.fingerprint
        return self.y.copy(),self.left,self.ledger.copy(),self.actual.copy(),list(self.accepted)
