# Data Versioning with DVC

Closes the gap flagged in our MVP presentation: our proposal named DVC as the versioning tool, but the notebook only implemented a SHA-256 fingerprint. This adds real DVC tracking alongside the existing fingerprint, so datasets can be rolled back and diffed, not just verified.

## One-time setup

```bash
pip install dvc
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

## Versioning the processed dataset

Run this after the notebook's processing step produces the model-ready CSV:

```bash
dvc add data/processed/flights_model_ready_v1.csv
git add data/processed/flights_model_ready_v1.csv.dvc data/processed/.gitignore
git commit -m "Version processed dataset v1 with DVC"
git tag -a data-v1.0 -m "Processed dataset v1"
```

If a remote storage location is configured (e.g., S3, Google Drive), push the actual data:

```bash
dvc remote add -d storage <remote-url>
dvc push
```

## Rolling back to a previous version

```bash
git checkout data-v1.0
dvc checkout
```

This restores both the code and the exact data snapshot it was paired with.

## What this adds over the SHA-256 fingerprint

| | SHA-256 fingerprint (existing) | DVC (this change) |
|---|---|---|
| Detects whether data changed | ✅ | ✅ |
| Restores a previous version | ❌ | ✅ |
| Tracks data alongside code in Git history | ❌ | ✅ |
| Supports remote storage / team sharing | ❌ | ✅ |

The SHA-256 fingerprint in `artifacts/data_version.json` is still useful as a lightweight, dependency-free check and is left in place — DVC adds the rollback and collaboration capabilities it was missing.
