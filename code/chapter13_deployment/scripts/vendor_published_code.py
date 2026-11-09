"""
Copy the published Chapter 5 to 8 model code into vendor/ and record its hashes.

The files are read with `git show <ref>:<path>` from a checkout of the public
repository, so uncommitted edits in that checkout never leak into vendor/.
Every vendored file is byte-identical to the committed source; the manifest
records the commit, the source path and the SHA-256 of each file.

Usage:
    python scripts/vendor_published_code.py --repo ../ai-powered-personalization --ref HEAD
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CHAPTER_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = CHAPTER_DIR / "vendor"

PUBLISHED_FILES = [
    "code/chapter5_retrieval/supervised_two_tower.py",
    "code/chapter6_ranking/models/__init__.py",
    "code/chapter6_ranking/models/deepfm.py",
    "code/chapter6_ranking/models/feature_encoder.py",
    "code/chapter6_ranking/models/feature_pipeline.py",
    "code/chapter6_ranking/models/historical_features.py",
    "code/chapter6_ranking/models/xgboost_ranker.py",
    "code/chapter7_advanced_ranking/models/__init__.py",
    "code/chapter7_advanced_ranking/models/dcn.py",
    "code/chapter7_advanced_ranking/models/dlrm.py",
    "code/chapter7_advanced_ranking/models/mmoe.py",
    "code/chapter7_advanced_ranking/utils/__init__.py",
    "code/chapter7_advanced_ranking/utils/polars_pipeline.py",
    "code/chapter7_advanced_ranking/train_mmoe.py",
    "code/chapter8_value_functions/__init__.py",
    "code/chapter8_value_functions/business_rules.py",
    "code/chapter8_value_functions/mmr_diversity.py",
    "code/chapter8_value_functions/model_interface.py",
    "code/chapter8_value_functions/pipeline.py",
    "code/chapter8_value_functions/value_function.py",
]

REPOSITORY_URL = "https://github.com/ShreeshaJay/ai-powered-personalization"


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, type=Path, help="Path to a checkout of the public repository")
    parser.add_argument("--ref", default="HEAD", help="Commit, branch or tag to vendor from")
    args = parser.parse_args()

    commit = git(args.repo, "rev-parse", args.ref).decode().strip()
    entries = []
    for source_path in PUBLISHED_FILES:
        content = git(args.repo, "show", f"{commit}:{source_path}")
        vendored_path = Path(source_path).relative_to("code")
        target = VENDOR_DIR / vendored_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        entries.append({
            "vendored_path": vendored_path.as_posix(),
            "source_path": source_path,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        })
        print(f"  {source_path}  sha256={entries[-1]['sha256'][:12]}")

    manifest = {
        "source_repository": REPOSITORY_URL,
        "source_commit": commit,
        "vendored_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Byte-identical copies of committed files. Do not edit; re-run scripts/vendor_published_code.py instead.",
        "files": entries,
    }
    (VENDOR_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Vendored {len(entries)} files from {commit[:12]} into {VENDOR_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
