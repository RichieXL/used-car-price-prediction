"""
version_with_dvc.py

Adds DVC tracking for the processed dataset, complementing the existing
SHA-256 fingerprint in artifacts/data_version.json. Run this after the
notebook's processing step has saved the model-ready CSV.

Usage:
    python scripts/version_with_dvc.py --path data/processed/flights_model_ready_v1.csv --tag data-v1.0
"""
import argparse
import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout.strip()


def ensure_dvc_initialized(repo_root: Path):
    if not (repo_root / ".dvc").exists():
        run(["dvc", "init"])
        run(["git", "add", ".dvc", ".dvcignore"])
        run(["git", "commit", "-m", "Initialize DVC"])
        print("[dvc] Initialized DVC in this repository.")


def version_dataset(path: str, tag: str):
    repo_root = Path(".").resolve()
    ensure_dvc_initialized(repo_root)

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    run(["dvc", "add", str(dataset_path)])
    dvc_file = f"{dataset_path}.dvc"
    gitignore = str(dataset_path.parent / ".gitignore")

    run(["git", "add", dvc_file, gitignore])
    run(["git", "commit", "-m", f"Version {dataset_path.name} with DVC ({tag})"])
    run(["git", "tag", "-a", tag, "-m", f"Processed dataset: {dataset_path.name}"])
    print(f"[dvc] Tracked and tagged {dataset_path} as {tag}")

    # Cross-reference with the existing SHA-256 fingerprint artifact, if present
    version_meta_path = Path("artifacts/data_version.json")
    if version_meta_path.exists():
        meta = json.loads(version_meta_path.read_text())
        meta["dvc_tag"] = tag
        meta["dvc_file"] = dvc_file
        version_meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[dvc] Cross-referenced DVC tag in {version_meta_path}")
    else:
        print(f"[dvc] Note: {version_meta_path} not found; skipped cross-reference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="Path to the processed dataset file")
    parser.add_argument("--tag", required=True, help="Git tag for this dataset version, e.g. data-v1.0")
    args = parser.parse_args()
    version_dataset(args.path, args.tag)
