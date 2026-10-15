# IC-PRO-S reviewer reproducibility — scope and navigation (V8)

## How to run the independent arithmetic replay

**One command** for the subset of evidence whose historical input/error files are preserved:

```bash
python RUN_REVIEWER_CHECKS.py
```

Requires Python 3.10+ and packages in `requirements.txt`. The command verifies 21 A12 method/regime means, 15 paired comparisons, nine capacity-matched GPU ablation result records and performs 15 spectrum-clustered tests with Holm correction. It does **not** reproduce all historical GPU training or the distinct 12,000/600 deployment-routing guard. Use the additional OOD command below to recompute four historical Table 8 detector scores.

If the Zenodo bundle is being inspected, run the same command from its extracted root. Frozen `.pt` files and checkpoints are included in that bundle; the lightweight GitHub package does not include large checkpoints.

## Major qualified statements — read before citing

1. **Novelty:** IC-PRO-S combines existing operator conditioning, Tikhonov anchor, learned proposal and quadratic proximal displacement; it does not claim invention of the building blocks. Its empirical contribution is the specific integrated design and studied transfer trade-offs.
2. **A12 statistics:** Each regime contains **128 independent synthetic spectra**, with each spectrum tested under two fixed response matrices (256 evaluations). Corrected one-sided Wilcoxon + Holm over 15 tests, severe structural shift:

| Comparator | Mean comparator − IC-PRO-S | Raw p | Holm p |
|---|---:|---:|---:|
| NNLS-TV | 0.0268 | 0.04277 | 0.1711 |
| PnP tuned | 0.0043 | 0.02077 | 0.1039 |
| ReSpecNN per-device | 0.0281 | 0.1407 | 0.4220 |

No one of the three establishes familywise superiority under the corrected analysis.
3. **Capacity control:** Nine checkpoints are available in the Zenodo bundle (three arms × seeds 3,7,11); 291,620 parameters in each arm. Test means: full 0.35854, no Fisher 0.36199, anchor+gradient 0.37316. The comparison isolates supplementary operator-derived CNN inputs, *not* all uses of the forward operator.
4. **OOD limitation:** The original **Table 8 detector-discrimination script** has been recovered. Four of its five rows were independently replayed exactly. The original *distinct 12,000/600 operational routing-guard policy* and case-level routing trace are still not recovered. `exploratory_proxy_ood/` is a **different proxy**; its small mean routing improvement is not established (bootstrap CI covers zero). This release does not validate original-guard deployment benefit, risk–coverage or uncertainty calibration.
5. **Original V7 inputs:** Five historical V7 model checkpoints are preserved in the large archive; the exact original V7 test tensors were not saved. `V7_SOURCE_DEFINED` inputs are regenerated and not historically bitwise-identical.
6. **External measured datasets:** Third-party measured experimental files are linked by citation and accessed from the owners; we do not redistribute them. A13B requires the source workbook at the expected path, with file SHA-256 recorded in documentation.
7. **A12 baselines:** The controlled PiGDM and MGDUN-style implementations are adaptations, not byte-for-byte copies of other groups' canonical code. Cross-harness latencies must not be compared as identical benchmarks.

8. Repository status and licensing: The public software
repository and its versioned GitHub release are available.
The archived release and its DOI are documented in the
Software availability section.

The release provides auditable evidence for the documented
experiments, subject to the stated historical reproducibility
limitations. Third-party materials remain subject to their
original licenses and redistribution permissions.


## File map

- `historical_v7/` recovered V7 source; `frozen_inputs/` per-file provenance (full tensors in Zenodo).
- `a12_frozen_source/`, `a12_frozen_outputs/`, `verify_a12.py`, `statistical_audit/` frozen seven-method benchmark.
- `gpu_ablation/` new matched-capacity GPU inputs/results code (nine checkpoints in Zenodo).
- `a13b/` external measured-responsivity forward surrogate and source; `a14/` calibration study.
- `exploratory_proxy_ood/` explicitly different from original guard.
- `docs/RELEASE_STATUS.md` claims that are and are not independently reproducible.

**Do not describe the integrity checks below as full end-to-end scientific reproduction.**


## Added in release V12: historically recovered OOD and peak/timing replay

- `python RUN_REVIEWER_CHECKS.py`: archived A12 statistics + nine-GPU-fit summary + recorded historical OOD detector score check + saved 768-case spectral analysis records.
- `python RUN_REVIEWER_CHECKS.py --ood-recompute`: additionally regenerate *four* of five original Table 8 detector metrics from the archived historical source and its response matrix (requires `scikit-learn`). This takes longer; it does NOT test the 12,000/600 deployment router.
- `python supplementary/REPLAY_PEAKS_AND_TIMING.py`: **Zenodo package only**, recompute 768 archived seed-7 spectral reconstructions and peak/width/height/area/weak-feature metrics, and measure single-thread CPU component latency; requires `torch`, `scipy`, etc. Timing is hardware-specific, and no original routing latency is claimed.
- `ood_historical/source/HISTORICAL_run_ood_detector_benchmark_UNMODIFIED.py`: archived primary source, preserved for audit. `ood_historical/analysis/replay_4_detector_scores.py` is the independently adapted 4-detector replay; forward-residual dependency has not been independently rerun.
- `ood_historical/evidence/HISTORICAL_9_CASE_RETROSPECTIVE_RISK_COVERAGE.csv`: retrospective nine-case rank–coverage curves, **not** policy-threshold calibration evidence.
- Do not mistake `exploratory_proxy_ood/` for the historical guard: it uses a different support score.
- No new real-device collection or new wet-lab spectroscopy was performed for this revision.

**Release decision:** This is a source/evidence candidate for author rights review, not proof of an already-assigned DOI. The `HISTORICAL_response_matrix.txt` derived from earlier instrument measurements requires explicit review of redistribution rights; consider restricted Zenodo access until cleared. The original V7 input tensors are unavailable.
