# Reproducibility scope

## Exact historical input recovery
- A12: YES. The regenerated CUDA input content hash matches the historical SHA-256 exactly.
- V7: NO. The original V7 raw synthetic tensors were not serialized in the historical run.

## What is preserved for V7
- complete preserved source code;
- all documented seeds and hyperparameters;
- five canonical trained checkpoints;
- historical recovered result tables and reports;
- a newly frozen source-defined replay tensor set for all main/stress/resolution evaluations;
- full archival environment information.

The source-defined V7 tensor set is suitable for future reviewer-requested code
changes and controlled reruns. It is not labeled as the original historical raw
tensor set because that would overstate what the surviving artifacts support.
