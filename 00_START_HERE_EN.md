# One repository — GitHub Desktop → Zenodo
This is a pre-publication reviewer release candidate, **not** a verified fully historical end-to-end reproduction. The full model weights and `.pt` replay tensors are deliberately tracked. See `README_REVIEWERS_START_HERE.md`, `docs/RELEASE_STATUS.md`, `docs/EXTERNAL_DATA_AND_LICENSE.md` and `LICENSE_DECISION_REQUIRED.md`. Public release requires author review of code license and redistribution rights.

**Verified audits:** `python scripts/verify_release.py`, `python RUN_REVIEWER_CHECKS.py`, optional `--ood-recompute`. The original separate 12,000-reference/600-validation selective deployment trace is not preserved.

**Zenodo integration:** push this ONE public GitHub repository, enable Zenodo's GitHub integration, and then create a GitHub Release. Zenodo archives that release and issues its DOI; a push or Sync now alone does not create the DOI. Private GitHub repos cannot be auto-archived by this integration.
