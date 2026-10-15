# IC-PRO-S three-arm, parameter-matched CNN feature control (PROPOSED; NOT YET VALIDATED)

Source: V7 original core/synthetic generators and F4 V7 SOURCE-DEFINED REPLAY tensors from user-provided Master V11. There are no historical V7 raw tensors here. No trained benchmark output is included.

This protocol keeps the EXACT SAME FiRNO constructor and 291,620 full parameters in all arms (NOT the 291,499 streamlined deployment count). The 4-channel convolution always receives four active inputs:
- full: anchor, normalized backprojection, local gradient, response/Fisher map.
- no_fisher: replace response/Fisher map with fixed cosine coordinate (no operator-specific input in that slot).
- anchor_grad_only: replace backprojection with fixed linear coordinate and Fisher map with fixed cosine coordinate; anchor and gradient remain operator-derived.
All other analytic operator-dependent components (anchor, singular-mode gating and proximal correction) remain on in all arms. Hence the test isolates explicit learned CNN conditioning beyond the shared physics baseline, NOT total operator awareness. The coordinate controls are non-constant and retain parameter exposure. Do not interpret this as a complete operator-free neural network.

Frozen validation/test objects are SOURCE-DEFINED REPLAY, not original V7 inputs. Repeated spectra across test operators require clustered analyses for broad generalization. Three independent seeds [3,7,11]; SAME seeded train batches per arm. Validation every 100 steps; max 1050; early stop >=650 and 3 successive non-improvements. All evaluated on replay tensors.

Run on Kaggle GPU:
1. Upload this archive as a PRIVATE Kaggle Dataset or add it as a dataset input to a PRIVATE Kaggle notebook; set GPU accelerator.
2. Use IC_PRO_S_CAPACITY_ABLATION_KAGGLE.ipynb (separate file in delivery) or extract ZIP and run:
   python prepare_matched_capacity_ablation.py --replay ./V7_SOURCE_DEFINED_MAIN_EVAL_TENSORS.pt --steps 1050 --seeds 3 7 11 --arms full no_fisher anchor_grad_only --device cuda --out ./outputs
3. Download NEW_CAPACITY_ABLATION_RESULTS.csv and CAPACITY_*.pt from Kaggle notebook output.
4. Compare paired seeds and test sets, describe as a NEW independent controlled study (not old V7).

A 2-step CPU smoke test passed for all three arms, but those numbers must NOT be reported as scientific results. DOI deposit is not completed; review external third-party code/data and repository license before public upload.
