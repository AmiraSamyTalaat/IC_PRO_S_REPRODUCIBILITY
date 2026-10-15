# IC-PRO-S — Operator-aware proximal reconstruction (research reproducibility)

Reconstruction study associated with the manuscript submitted to *Scientific Reports*. This repository supplies source code, frozen results and auditable experimental records. **Novelty** is a particular shared operator-conditioned spectral proposal and one analytic quadratic proximal displacement with learned relaxation; established operator-aware inverse methods, PnP, unfolding and diffusion precede this work.

**Start with [`README_REVIEWERS_START_HERE.md`](README_REVIEWERS_START_HERE.md).**

Quick audit: `python -m pip install -r requirements.txt && python RUN_REVIEWER_CHECKS.py`. The extended detector simulation is `python RUN_REVIEWER_CHECKS.py --ood-recompute`; archived per-spectrum spectral diagnostics can be regenerated from the Zenodo evidence package using `python RUN_REVIEWER_CHECKS.py --peak-inference` (resource intensive).

The independent 5,000-reference detector experiment for Table 8 has been recovered and four metric rows are reproducible. The separate originally described **12,000-reference / 600-validation operational OOD routing experiment** and per-case route/time traces are **not** available as a bitwise-historical reproducibility record. A retrospective proxy is **not** proof of operational selectivity or uncertainty calibration. The forward-residual detector row is preserved but not separately reproduced in this release.

Raw historical V7 evaluation tensors were not retained; `V7_SOURCE_DEFINED` are regenerated, not historic exact copies. A12 frozen input tensors and five checkpoints are in the evidence bundle. A13B external measured datasets should be obtained from the original custodians.


## Software availability and citation

The IC-PRO-S source code, preserved evaluation artifacts,
statistical analysis scripts, and reproducibility documentation
are available from the public GitHub repository:

https://github.com/AmiraSamyTalaat/IC_PRO_S_REPRODUCIBILITY

The archived research software is available on Zenodo:
https://doi.org/10.5281/zenodo.23404660

The repository distinguishes independently reproducible
analyses from historical experiments whose complete original
execution records are unavailable. See
README_REVIEWERS_START_HERE.md for the precise scope.

External datasets remain subject to their original access
conditions and licenses. See docs/EXTERNAL_DATA_AND_LICENSE.md.
