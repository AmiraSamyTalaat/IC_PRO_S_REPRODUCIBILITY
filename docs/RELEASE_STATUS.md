# Release reproducibility status (honest disclosure)

| Evidence | Included? | Status |
|---|---|---|
| V7 recovered core and training script | Yes | Original source recovered; original raw V7 test tensors absent |
| 5 historical V7 checkpoints | Zenodo only | Hash-preserved |
| A12 seven-method source | Yes | Archival full-run source preserved; baseline variants adapted where described |
| A12 case errors and numerical summaries | Yes | Independent arithmetic replay supported |
| Exact A12 frozen input tensors | Zenodo only | Historical SHA-256 matched in freeze V4; inspect included validation |
| A12 all model checkpoints | Zenodo only | Curated outputs from preserved P03 and V7 source |
| A13B source/archived outputs | Yes | Upstream workbook required for executable rerun; third-party raw files not redistributed |
| A14 source/archived outputs | Yes | Full 5-seed perturbation records preserved in Zenodo |
| GPU capacity feature-ablation | Yes | 9 fits, 9 checkpoints Zenodo, CSV and result JSON included |
| A12 spectrum-clustered statistical audit | Yes | 128 distinct spectra per regime (2 operators), Holm 15 tests |
| Original historical OOD guard | No | Insufficient matching original source/threshold traces; cannot reproduce the guard from included record |
| Exploratory proxy OOD risk-coverage | Yes | Not equivalent to original guard; negative/no definite benefit |
| Original external experimental raw data | No | Must obtain from cited original source, respecting its license |
| One-command entire-paper GPU training | No | Stage-specific commands exist; full original V7 historical tensors unavailable |
| |

**Claim boundary:** A PASS manifest confirms byte integrity only. It is not a proof of exact numerical replay of every table or full experimental replication.

Updated V8: `RUN_REVIEWER_CHECKS.py` independently replays the archived arithmetic/clustered statistics in one command, but not full historical training or original guard.

## V12 addendum — recovered historical detector scores
The 5,000-reference / 800-ID / 800-OOD Table 8 detector script was recovered from an older original project archive; four metric rows were independently reproduced. This does not recover the separate 12,000/600 selective-route policy, individual 14-case routing decisions or full routed latency. Peak diagnostics are retrospective on preserved A12 data. See `README_REVIEWERS_START_HERE.md`.
