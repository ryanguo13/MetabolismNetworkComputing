"""Second frozen study: same estimator, implementation-only optimization."""
import argparse
import json
from pathlib import Path

from research import m3_shadow_study as study
from research.defect_fast import estimate_fast


def configure():
    original_hashes=study.source_hashes
    def hashes():
        result=original_hashes()
        for relative in ('research/defect_fast.py','research/m3_shadow_fast_study.py'):
            result[relative]=study.digest(study.ROOT/relative)
        return result
    study.source_hashes=hashes
    study.OUT=study.ROOT/'results/m3-shadow-fast'
    study.NEW_STATES={'teusink':[7e-6,.007,.7],'kecoli':[7e-8,.007,.07]}
    study.estimate=estimate_fast


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['register','dev','new'])
    stage=parser.parse_args().stage
    configure()
    {'register':study.register,'dev':study.dev,'new':study.new_states}[stage]()
