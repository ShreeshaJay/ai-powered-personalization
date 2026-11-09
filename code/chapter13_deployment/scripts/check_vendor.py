"""
Verify that vendor/ still matches the published code it was copied from.

Without --repo, the check recomputes each vendored file's SHA-256 and compares
it with vendor/MANIFEST.json. With --repo, it also compares every file with
the committed source at the manifest's commit (or at --ref, to see whether the
published code has moved on since it was vendored).

Usage:
    python scripts/check_vendor.py
    python scripts/check_vendor.py --repo ../ai-powered-personalization --ref origin/main
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

CHAPTER_DIR = Path(__file__).resolve().parents[1]
VENDOR_DIR = CHAPTER_DIR / "vendor"


def check(repo: Path = None, ref: str = None) -> list:
    manifest = json.loads((VENDOR_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    problems = []
    for entry in manifest["files"]:
        path = VENDOR_DIR / entry["vendored_path"]
        if not path.exists():
            problems.append(f"missing vendored file: {entry['vendored_path']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            problems.append(f"vendored file edited: {entry['vendored_path']}")
        if repo is not None:
            commit = ref or manifest["source_commit"]
            published = subprocess.run(
                ["git", "-C", str(repo), "show", f"{commit}:{entry['source_path']}"],
                capture_output=True,
            )
            if published.returncode != 0:
                problems.append(f"not found at {commit}: {entry['source_path']}")
            elif hashlib.sha256(published.stdout).hexdigest() != entry["sha256"]:
                problems.append(f"published file differs at {commit}: {entry['source_path']}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--ref", default=None)
    args = parser.parse_args()
    problems = check(args.repo, args.ref)
    for problem in problems:
        print(f"FAIL  {problem}")
    if not problems:
        print("OK  vendor/ matches MANIFEST.json" + (" and the published source" if args.repo else ""))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
