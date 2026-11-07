"""Recover one unreadable reference member by independently recomputing it.

Preserve the damaged original and explicitly record recovery provenance.
"""
import json
import numpy as np
from metabolism.m3_continuation import prepare
from metabolism.m2_experiments import save
from research.m3_priority import full,metrics
from research.m3_shadow_study import ROOT,digest

if __name__=='__main__':
    directory=ROOT/'results/m3-shadow-stress';source=directory/'curves.npz'
    bad=json.loads((directory/'archive-integrity-failure.json').read_text())
    assert [x['name'] for x in bad]==['kecoli-switch-1-1-reference.npy']
    key='kecoli-switch-1-1';arrays=np.load(source)
    row=next(r for r in json.loads((directory/'runs.json').read_text())['runs'] if r['key']==key)
    case=prepare('kecoli');y=arrays[key+'-nodes'][:,0];times=row['h']*arrays[key+'-dense_query']
    audit_times=np.unique(np.r_[0.,times]);ref=full(case,y,audit_times,1e-12)[:,np.searchsorted(audit_times,times)]
    checks={}
    for target,entry in row['truth']['targets'].items():
        actual=metrics(case,arrays[key+'-dense'],ref,float(target))
        np.testing.assert_allclose(list(actual.values()),list(entry['actual'].values()),rtol=1e-12,atol=1e-12)
        checks[target]=actual
    budget=metrics(case,arrays[key+'-loose'],ref,1e-6)
    np.testing.assert_allclose(list(budget.values()),list(row['local_reference_budget'].values()),rtol=1e-12,atol=1e-12)
    destination=directory/'recovered-reference.npz';np.savez_compressed(destination,**{key+'-reference':ref})
    save(directory/'reference-recovery.json',dict(reason='One compressed ZIP member failed decompression; all other members read successfully. Recomputed full reference from preserved candidate initial state and exact query grid.',
        original_archive_sha256=digest(source),recovered_file_sha256=digest(destination),key=key+'-reference',
        recovered_metrics=checks,reference_budget=budget,source_sha256=digest(ROOT/'research/recover_stress_reference.py'),
        limitation='Original reference bytes unreadable; recovered independently and all previously saved actual errors and tightening budgets reproduced. Not a bytewise comparison.'))
    print('Recovered reference; recorded errors and tightening budget reproduced')
