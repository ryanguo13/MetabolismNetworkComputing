"""Attempt a whole interval first; retain Radau rejection/refresh/error logic."""
from types import FunctionType
from research.shared_radau import SharedFactory, SharedMRIEndpoint, shared_full_curve


class FullStepFactory(SharedFactory):
    def __call__(self, fun, t0, y0, t_bound, **kwargs):
        kwargs['first_step'] = abs(t_bound-t0)
        return super().__call__(fun,t0,y0,t_bound,**kwargs)


class FullStepMRIEndpoint(SharedMRIEndpoint):
    original = SharedMRIEndpoint.trial
    trial = FunctionType(original.__code__,dict(original.__globals__,SharedFactory=FullStepFactory),
                         name='trial',argdefs=original.__defaults__)
    trial.__kwdefaults__ = original.__kwdefaults__.copy()


fullstep_curve = FunctionType(shared_full_curve.__code__,dict(shared_full_curve.__globals__,SharedFactory=FullStepFactory),
                              name='fullstep_curve',argdefs=shared_full_curve.__defaults__)
