#!/usr/bin/env python3
"""Archive-only verification: detector replay, original stored summary, 9-case posthoc curves.
No claim to reproduce 12000/600 deployment routing or historical raw forward residual.
"""
from pathlib import Path
import json, csv, math, sys
root=Path(__file__).resolve().parents[1]
a=json.loads((root/'analysis'/'INDEPENDENT_DETECTOR_REPLAY_4_SCORES.json').read_text())
b=json.loads((root/'evidence'/'HISTORICAL_OOD_DETECTOR_BENCHMARK.json').read_text())
h={r['score']:r for r in b['simulation_only_ood_benchmark']['metrics']}
assert a['reference_bank_n']==5000 and a['n_id']==800 and a['n_ood']==800
for row in a['metrics']:
    for key in ('AUROC','AUPRC','FPR95'):
        assert math.isclose(row[key],h[row['score']][key],rel_tol=0,abs_tol=1e-12),(row['score'],key)
curve=b['real_single_peak_risk_coverage']
assert len(curve)==5 and len(b['real_score_values']['nearest_manifold'])==9
assert len(curve['nearest_manifold'])==10
with open(root/'evidence'/'HISTORICAL_OOD_DETECTOR_METRICS.csv',newline='') as f:
    rr={r['score']:r for r in csv.DictReader(f)}
for row in a['metrics']:
    for key in ('AUROC','AUPRC','FPR95'):
        assert abs(row[key]-float(rr[row['score']][key]))<1e-12
print('PASS: 4 independently recomputed historical detector scores match archived JSON and CSV exactly.')
print('PASS: risk-coverage traces for 9 real single-peak cases preserved for 5 detector scores (10 coverage levels).')
print('LIMIT: forward_residual not rerun; 12000-bank/600-calibration threshold, routed per-case 14-set, route-inclusive timing not verified.')
