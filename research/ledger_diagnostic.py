"""Signed physical-ledger error decomposition; diagnostic, never correction."""
import numpy as np
from research.mri_kernel import GAUSS


def split_ledger(weights, initial, end, reference_end, curve_integral, reference_integral):
    denominator=max(1.,abs(float(weights@initial)))
    endpoint=float(weights@(end-reference_end))/denominator
    reference=(float(weights@(reference_end-initial))-reference_integral)/denominator
    output=(curve_integral-reference_integral)/denominator
    ledger=(float(weights@(end-initial))-curve_integral)/denominator
    return dict(endpoint_projection=endpoint,reference_residual=reference,
                output_integral_bias=output,signed_residual=ledger,
                decomposition_roundoff=ledger-(endpoint+reference-output))


def flux_integral(case,values,h,n):
    weights=case['options']['ledger_weights']
    rhs=case['system'].S@np.column_stack([case['system'].flux(0.,z) for z in values.T])
    return float(h/2*(GAUSS[n][1]@(weights@rhs)))
