#!/usr/bin/env python3
"""Archive-record consistency checks only; independent inference uses REPLAY_PEAKS_AND_TIMING.py."""
import csv,json,math
from pathlib import Path
r=Path(__file__).resolve().parent/'results'
rows=list(csv.DictReader((r/'A12_SEED7_PER_CASE_SPECTRAL_FEATURES.csv').open()))
assert len(rows)==768,(len(rows))
assert len({(q['operator'],q['sample']) for q in rows})==768
assert len({q['operator'] for q in rows})==6
assert all(math.isfinite(float(q['rel_l2'])) for q in rows)
info=json.loads((r/'RUN_CONFIGURATION.json').read_text())
assert info['n_cases']==768
timing=list(csv.DictReader((r/'CPU_RECONSTRUCTION_STAGE_TIMING.csv').open()))
assert any(q['stage']=='complete_reconstruction_only' and float(q['median_ms'])>0 for q in timing)
print('PASS: 768 A12 spectral records, 6 operators, CPU stage timing records. Does not rerun model.')
