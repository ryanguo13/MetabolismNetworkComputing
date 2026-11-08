"""Shift-filtered residual indicator, explicitly not an error certificate.

With K=(I-sH A)^-1 and saved W=KV, KAV=(W-V)/(sH). Thus the full
filtered residual H K r is evaluated without new inverse solves. This filter
accounts for stiff damping heuristically; nonnormal/transient effects still
require the independent full-space reference audit.
"""
import numpy as np


def filtered_residual(vectors,images,projection,coefficients,drive,rhs_scaled,h,shift):
    if vectors.shape[1]==0:return np.zeros((vectors.shape[0],coefficients.shape[1]))
    kb=np.linalg.norm(rhs_scaled)*images[:,0]
    defect=images@projection-(images-vectors)/(shift*h)
    return h*(defect@coefficients+(images@drive-kb)[:,None])


def filtered_proposal(result,images,rhs_scaled,h,shift,scale,initial,enrichment,target=1e-6,budget=.05,safety=4.):
    filtered=filtered_residual(result['basis'],images,result['projection'],result['coefficients'],result['drive'],rhs_scaled,h,shift)
    den=target*(.01+np.maximum(abs(initial)/scale,1.))
    score=float(np.max(abs(filtered)/den[:,None]))
    proposed=enrichment is not None and max(safety*score,enrichment)<=budget
    return dict(filtered_indicator=score,safety=safety,enrichment_indicator=enrichment,proposed=bool(proposed),certified=False),filtered
