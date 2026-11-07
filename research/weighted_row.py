"""Same fresh-Jacobian kernels with weighted-infinity preconditioner policy."""
from types import FunctionType
from research.cached_row import CachedROW
from research.cached_full import full_curve
from research.weighted_lu import WeightedLU


class WeightedROW(CachedROW):
    def __init__(self,case,groups,reuse=True):
        super().__init__(case,groups,reuse)
        self.predictor_cache=WeightedLU(reuse);self.fast_cache=WeightedLU(reuse)


weighted_full_curve=FunctionType(full_curve.__code__,dict(full_curve.__globals__,RefinedLU=WeightedLU),
    name='weighted_full_curve',argdefs=full_curve.__defaults__)
