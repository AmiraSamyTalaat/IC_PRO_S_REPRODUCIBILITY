#!/usr/bin/env python3
"""Documented audit of preserved IC-PRO-S evidence (not a complete historical GPU training rerun)."""
import argparse,subprocess,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--ood-recompute',action='store_true',help='Recompute four historical Table 8 detectors (slower; requires scikit-learn)')
p.add_argument('--peak-inference',action='store_true',help='Recompute 768 seed-7 A12 outputs, peak metrics and CPU timings; Zenodo data only')
a=p.parse_args()
jobs=[[sys.executable,str(root/'verify_a12.py')],[sys.executable,str(root/'gpu_ablation/verify_gpu_summary.py')],[sys.executable,str(root/'supplementary/verify_recorded_peak_timing.py')]]
with tempfile.TemporaryDirectory(prefix='icpros_review_') as t:
 jobs.append([sys.executable,str(root/'statistical_audit/replay_spectrum_clustered.py'),'--results-dir',str(root/'a12_frozen_outputs'),'--out',str(Path(t)/'clustered')])
 if a.ood_recompute:
  jobs.append([sys.executable,str(root/'ood_historical/analysis/replay_4_detector_scores.py')])
 jobs.append([sys.executable,str(root/'ood_historical/analysis/verify_recovered_ood.py')])
 if a.peak_inference:
  if not (root/'frozen_inputs/A12_EXACT_BENCHMARK_TENSORS.pt').is_file():
   raise SystemExit('For --peak-inference use the complete Zenodo evidence bundle (includes exact A12 tensors).')
  jobs.append([sys.executable,str(root/'supplementary/REPLAY_PEAKS_AND_TIMING.py')])
 for cmd in jobs:
  print('\nRUN:',' '.join(cmd),flush=True)
  subprocess.run(cmd,cwd=root,check=True)
print('\nPASS: documented source/evidence checks completed. DOES NOT prove the 12,000/600 historical routing guard or route-inclusive timing.')
