"""Stiff-safe-in-shape linear error interpolation, not a rigorous error bound.

Preserves the SDIRK nodes and all propagated error endpoints. Linear segments
avoid derivative-driven overshoot, but only have second-order interpolation
accuracy. Two independent subgrids and unchanged factor4 scores remain required.
"""
from types import FunctionType
import numpy as np
from research.streaming_defect import advance,StreamingDefect


class LinearError:
    def __init__(self,y,end,dy,dend,h):
        self.d=y.copy();self.c=end-y;self.b=np.zeros_like(y);self.a=np.zeros_like(y);self.h=h


advance_linear=FunctionType(advance.__code__,dict(advance.__globals__,Hermite=LinearError),name='advance_linear',argdefs=advance.__defaults__)


class LinearStreaming(StreamingDefect):
    trial=FunctionType(StreamingDefect.trial.__code__,dict(StreamingDefect.trial.__globals__,advance=advance_linear),name='trial',argdefs=StreamingDefect.trial.__defaults__)
