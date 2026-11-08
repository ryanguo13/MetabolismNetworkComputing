"""Signed physical-ledger attribution on frozen candidate polynomials.

Separate diagnostic; original absolute ledger checks stay authoritative.
"""
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
from metabolism.m2_experiments import save
from metabolism.m3_continuation import prepare
from research.affine_partition import freeze
from research.extrapolated_flow import unpack
from research.m3_shadow_study import ROOT,digest

OUT=ROOT/'results/m3-affine-ledger';AFF=ROOT/'results/m3-affine-attribution';OLD=ROOT/'results/m3-rodas-centered'


def signed_ledger(case,curve,origin,endpoint):
    w=case['options']['ledger_weights'];den=max(1.,abs(float(w@origin)));integrals=[]
    for n in (16,32):
        nodes,weights=leggauss(n);total=0.
        for p in curve.pieces:
            rhs=np.column_stack([case['system'].S@case['system'].flux(0,z) for z in p((nodes+1)/2).T]);total+=p.h/2*float(weights@(w@rhs))
        integrals.append(total)
    return dict(signed_residual=(float(w@(endpoint-origin))-integrals[-1])/den,quadrature_difference=abs(integrals[1]-integrals[0])/den)


def run():
    OUT.mkdir(exist_ok=True)
    if (OUT/'registration.json').exists():raise FileExistsError('Preserve evidence')
    reg=dict(date='2026-11-08',scope='All16 small-model same-state pairs from prior affine attribution. Signed16/32 physical-ledger residuals, evaluate full nonlinear and frozen RHS on both saved curves to separate curve and vector-field effects algebraically. No scalar projection, no acceptance-rule change.',
        source_sha256={p:digest(ROOT/p) for p in ('research/m3_affine_ledger.py','research/affine_partition.py')},input_sha256={str(p.relative_to(ROOT)):digest(p) for p in [AFF/'registration.json',AFF/'runs.json',OLD/'registration.json',OLD/'runs.json']})
    save(OUT/'registration.json',reg);case=prepare('teusink');rows=[]
    for row in json.loads((AFF/'runs.json').read_text())['candidates']:
        if row['model']!='teusink':continue
        with np.load(AFF/f"reference-{row['reference_key']}.npz") as a:y=a['initial']
        aff,_=freeze(case,y)
        with np.load(AFF/f"{row['key']}.npz") as a:lc=unpack(a['lengths'],a['coefficients']);le=a['endpoint']
        with np.load(OLD/f"{row['nonlinear_key']}.npz") as a:nc=unpack(a['lengths'],a['coefficients']);ne=a['endpoint']
        nn=signed_ledger(case,nc,y,ne);ll=signed_ledger(aff,lc,y,le);nl=signed_ledger(case,lc,y,le);ln=signed_ledger(aff,nc,y,ne)
        # L_N(P_N)-L_L(P_L) = [L_N(P_N)-L_L(P_N)]
        #                       + [L_L(P_N)-L_L(P_L)].
        vector_effect=nn['signed_residual']-ln['signed_residual'];curve_effect=ln['signed_residual']-ll['signed_residual']
        rows.append(dict(key=row['key'],h=row['h'],augmented=row['augmented'],whole=row['whole'],nonlinear_on_nonlinear=nn,affine_on_affine=ll,nonlinear_on_affine=nl,affine_on_nonlinear=ln,vector_field_effect=vector_effect,curve_effect=curve_effect,difference=nn['signed_residual']-ll['signed_residual']))
    save(OUT/'runs.json',dict(runs=rows));print(json.dumps([r for r in rows if r['h']==.03],indent=2))


if __name__=='__main__':run()
