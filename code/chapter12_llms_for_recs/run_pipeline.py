"""
Run the Chapter 12 scripts in order for one profile.

    synthetic  generated data, TF-IDF encoder, no model downloads; a CPU pipeline check
    tiny       3,000 real KDD sessions on CPU; LLM steps use Qwen2.5-0.5B-Instruct on a
               handful of sessions, and the ranker uses the copy-input control instead of
               LLM features (LLM generation for every item is a GPU job)
    ranker     20,000 sessions for the Section 12.4 ablation with real LLM features (GPU)
    full       every eligible UK session and the full UK catalog (GPU)

Each profile writes to its own work directory (default outputs/<profile>). See
CLOUD_RUN.md for GPU classes and what each step needs.

Examples:
    python run_pipeline.py --profile synthetic
    python run_pipeline.py --profile tiny --dry-run
    python run_pipeline.py --profile full --work-dir /content/ch12_full --only train_tiger.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

from config import CHAPTER_ROOT, CPU_LLM, DEFAULT_LLM

PROFILES = {
    "synthetic": [
        ["prepare_data.py", "--synthetic"],
        ["encode_items.py", "--encoder", "tfidf-svd", "--tfidf-dim", "32"],
        ["train_rqvae.py", "--codebook-size", "16", "--latent-dim", "16", "--hidden-dim", "64", "--epochs", "15",
         "--batch-size", "128"],
        ["evaluate_semantic_ids.py", "--pairs", "2000"],
        ["train_tiger.py", "--hidden-dim", "64", "--ffn-dim", "128", "--encoder-layers", "2", "--decoder-layers", "2",
         "--epochs", "4", "--batch-size", "64", "--beam-width", "10", "--max-eval-sessions", "200"],
        ["build_slates.py", "--splits", "train", "val", "test", "--max-sessions", "300"],
        ["generate_llm_features.py", "--backend", "copy-input"],
        ["train_augmented_ranker.py", "--epochs", "5"],
        ["evaluate_augmented.py"],
    ],
    "tiny": [
        ["prepare_data.py", "--max-sessions", "3000", "--catalog", "referenced", "--extra-random-items", "3000"],
        ["encode_items.py"],
        ["train_rqvae.py", "--codebook-size", "64", "--epochs", "20", "--batch-size", "512"],
        ["evaluate_semantic_ids.py", "--cold-threshold", "1"],
        ["train_tiger.py", "--hidden-dim", "64", "--ffn-dim", "256", "--encoder-layers", "2", "--decoder-layers", "2",
         "--epochs", "3", "--batch-size", "128", "--max-eval-sessions", "300", "--cold-threshold", "1"],
        ["build_slates.py", "--splits", "train", "val", "test", "--max-sessions", "300"],
        ["evaluate_zeroshot_llm.py", "--model", CPU_LLM, "--max-sessions", "20", "--batch-size", "4"],
        ["train_tallrec.py", "--model", CPU_LLM, "--max-train-sessions", "8", "--max-eval-sessions", "2",
         "--batch-size", "2", "--grad-accum", "1", "--eval-batch-size", "4"],
        ["generate_llm_features.py", "--backend", "copy-input"],
        ["train_augmented_ranker.py"],
        ["evaluate_augmented.py", "--cold-threshold", "1"],
    ],
    "ranker": [
        ["prepare_data.py", "--max-sessions", "20000", "--catalog", "referenced", "--extra-random-items", "20000"],
        ["encode_items.py"],
        ["build_slates.py", "--splits", "train", "val", "test", "--max-sessions", "2000"],
        ["generate_llm_features.py", "--model", DEFAULT_LLM, "--batch-size", "64"],
        ["train_augmented_ranker.py"],
        ["evaluate_augmented.py"],
    ],
    "full": [
        ["prepare_data.py"],
        ["encode_items.py", "--batch-size", "512"],
        ["train_rqvae.py"],
        ["evaluate_semantic_ids.py", "--tsne"],
        ["train_tiger.py"],
        ["build_slates.py", "--splits", "train", "val", "test", "--max-sessions", "5000"],
        ["evaluate_zeroshot_llm.py", "--model", DEFAULT_LLM, "--max-sessions", "1000", "--batch-size", "16"],
        ["train_tallrec.py", "--model", DEFAULT_LLM, "--max-train-sessions", "5000", "--max-eval-sessions", "1000"],
    ],
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", choices=sorted(PROFILES), required=True)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--only", nargs="+", default=None, help="Run only these scripts (by file name)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    work_dir = str(Path(args.work_dir).resolve() if args.work_dir else CHAPTER_ROOT / "outputs" / args.profile)

    steps: List[List[str]] = PROFILES[args.profile]
    if args.only:
        steps = [s for s in steps if s[0] in args.only]
    for step in steps:
        cmd = [sys.executable, str(CHAPTER_ROOT / step[0]), "--work-dir", work_dir, *step[1:]]
        print(" ".join(["python", step[0], "--work-dir", work_dir, *step[1:]]), flush=True)
        if args.dry_run:
            continue
        start = time.time()
        subprocess.run(cmd, check=True, cwd=CHAPTER_ROOT)
        print(f"  finished in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
