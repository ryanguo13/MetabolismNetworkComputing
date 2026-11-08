"""Lossless byte sharding of large diagnostic NPZ files for Git hosting."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from metabolism.m2_experiments import save
from research.m3_shadow_study import ROOT, digest

OUT = ROOT/'results/m3-remainder-curve'


def load_bytes(path):
    path = Path(path)
    if path.exists(): return path.read_bytes()
    metadata = json.loads((path.parent/'packaging.json').read_text())['archives'][path.name]
    parts = []
    for part in metadata['parts']:
        source = path.parent/part['name']
        assert digest(source)==part['sha256'], source
        parts.append(source.read_bytes())
    content = b''.join(parts)
    assert len(content)==metadata['bytes'] and hashlib.sha256(content).hexdigest()==metadata['sha256']
    return content


def load_arrays(path):
    if Path(path).exists(): return np.load(path)
    return np.load(io.BytesIO(load_bytes(path)))


def package():
    if (OUT/'packaging.json').exists(): raise FileExistsError('Preserve packaging')
    scratch = ROOT/'results/scratch/m3-remainder-unsharded'
    scratch.mkdir(parents=True, exist_ok=False)
    archives = {}; limit = 40*1024**2
    for path in sorted(OUT.glob('*.npz')):
        if path.stat().st_size <= 45*1024**2: continue
        record = dict(bytes=path.stat().st_size, sha256=digest(path), parts=[])
        with path.open('rb') as source:
            while content := source.read(limit):
                part = path.with_name(path.name+f'.part{len(record["parts"]):02d}')
                with part.open('xb') as destination: destination.write(content)
                record['parts'].append(dict(name=part.name, sha256=digest(part)))
        rebuilt = b''.join((OUT/p['name']).read_bytes() for p in record['parts'])
        assert hashlib.sha256(rebuilt).hexdigest()==record['sha256']
        with zipfile.ZipFile(io.BytesIO(rebuilt)) as z: assert z.testzip() is None
        archives[path.name] = record
    save(OUT/'packaging.json', dict(reason='Lossless byte shards; reconstructed original SHA-256 and NPZ CRC checked. Original archives retained in ignored scratch. No numerical or timing changes.', archives=archives))
    for name in archives: (OUT/name).rename(scratch/name)
    # Keep the prepackaging manifest as provenance; the final manifest covers
    # the reconstructible archives through their shards and packaging index.
    (OUT/'manifest.json').rename(scratch/'prepackaging-manifest.json')
    print('packaged', len(archives), 'archives; original bytes preserved')


if __name__=='__main__': package()
