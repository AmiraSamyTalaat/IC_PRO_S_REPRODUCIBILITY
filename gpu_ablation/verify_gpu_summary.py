#!/usr/bin/env python3
"""Verify 9 frozen GPU arm-by-seed rows and summary; no retraining."""
from pathlib import Path
import csv, json, math, statistics
ROOT=Path(__file__).resolve().parents[1]
res=ROOT/'gpu_ablation'/'results'
rows=list(csv.DictReader((res/'NEW_CAPACITY_ABLATION_RESULTS.csv').open()))
sumry=json.loads((res/'RESULTS_SUMMARY.json').read_text())
assert len(rows)==9 and sumry['n_model_fits']==9
assert {r['arm'] for r in rows}=={'full','no_fisher','anchor_grad_only'}
assert {int(r['seed']) for r in rows}=={3,7,11}
assert {int(r['params']) for r in rows}=={291620}
for arm in ('full','no_fisher','anchor_grad_only'):
 values=[float(r['test_rel_l2']) for r in rows if r['arm']==arm]
 assert len(values)==3
 assert math.isclose(statistics.mean(values),sumry['test_means'][arm],rel_tol=0,abs_tol=1e-12)
 assert math.isclose(statistics.stdev(values),sumry['test_seed_sd'][arm],rel_tol=0,abs_tol=1e-12)
print('PASS: 9 rows, 9 equal-capacity configurations, 3 means and 3 sample SDs match frozen GPU summary.')
