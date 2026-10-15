N05 A12 — STRONG-Q1 CROSS-INSTRUMENT BASELINE EXPANSION V2 FINAL
=================================================================

PURPOSE
-------
This package does NOT modify the manuscript. It runs the missing practical experiment:
modern baselines on the cross-instrument extrapolation regimes.

ORIGINAL ASSETS PRESERVED
-------------------------
- Original recovered V7 core.py and synthetic_v2.py.
- Original five final checkpoints: seeds 3, 7, 11, 19, 23.
- Frozen old Table-6 values are included only as docs/FROZEN_TABLE6_REFERENCE.csv and are never used as generated results.

NEW FIXED BENCHMARK
-------------------
All methods see the same spectra and same operators.
1) interpolation: unseen standard-family M=16 and M=24 operators.
2) moderate M extrapolation: M=8 and M=40 operators.
3) severe structural: filters built from smoothed-random, triangular, and high-frequency mixed responses.
The generator, operator seeds, spectrum seeds and SHA-256 content fingerprint are written to outputs/BENCHMARK_MANIFEST.json.

METHODS
-------
- IC-PRO-S: original V7 checkpoints; seed7 primary paired table + all five seeds stability file.
- Tikhonov.
- NNLS-TV: lambda selected only on the original V7 validation bank; 120 ADMM iterations.
- PnP: shared 1-D denoiser trained 700 steps; frozen 12 steps, eta=0.65, ridge=1e-3.
- MGDUN-style: six-stage shared operator-aware unfolding; seeds 3/7/11.
- ReSpecNN per-device: public topology adapted for N=256; 1200 steps separately for each target response.
- PiGDM controlled: four-level DDPM U-Net (~5.26M parameters in this implementation), T=1000 cosine schedule, 3000 updates, K=50 guided reverse steps, gamma=0.05.

RESUME
------
Each completed stage creates a .done_STAGE marker and saves checkpoints/results. After every stage the runner also refreshes /kaggle/working/N05_A12_PARTIAL_STATE.zip. If a run is interrupted, download that partial-state ZIP before the session is destroyed. In the same live Kaggle session, rerunning skips completed stages automatically.

OUTPUTS
-------
EXTENDED_TABLE6.csv
PAIRWISE_STATS.csv
EXTENDED_TABLE6_BY_OPERATOR.csv
ICPRO_5SEED_EXTRAPOLATION.csv
BENCHMARK_MANIFEST.json
RUN_STATUS.json
FINAL_STATUS.json
sample_errors_core.csv
sample_errors_extra.csv
checkpoints/*
N05_A12_CROSS_INSTRUMENT_RESULTS.zip

FULL RUN
--------
Use Kaggle GPU. Run the supplied N05_A12_KAGGLE_ONE_CLICK.ipynb with Accelerator=GPU.

LOCAL SMOKE TEST
----------------
Set N05_QUICK=1 then run:
  python scripts/run_cross_instrument.py
Quick mode is ONLY a code-integrity test. Never use its numbers in the paper.


FRESH-SESSION RESUME
--------------------
If a Kaggle session ends after at least one stage, keep `N05_A12_PARTIAL_STATE.zip`.
In a new Kaggle session, add/upload that ZIP as an input; the supplied one-click notebook searches `/kaggle/input/**/N05_A12_PARTIAL_STATE.zip` and restores it automatically before running.

IMPORTANT
---------
Do not send quick-mode numerical tables to the manuscript. Only a full GPU run with `quick_mode=false` and `FINAL_STATUS.status=PASS` is eligible.
