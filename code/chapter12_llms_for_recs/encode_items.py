"""
Embed every item in the work directory with one text encoder.

Writes <work_dir>/embeddings/item_embeddings.npz with item_ids, L2-normalised
embeddings and the encoder name. RQ-VAE, the slates, the TIGER kNN baseline and the
augmented ranker all read this file.

The default is the base all-MiniLM-L6-v2. Chapter 10's fine-tuned MiniLM was trained on
(last item, next item) pairs drawn from every KDD session, including the sessions this
chapter uses for validation and test, so using it here leaks test targets into the item
vectors. Pass it with --encoder only for a deliberately leaky comparison.

Examples:
    python encode_items.py --work-dir outputs/tiny
    python encode_items.py --work-dir outputs/synthetic --encoder tfidf-svd
"""

from __future__ import annotations

import argparse
import logging
import time

from config import DEFAULT_TEXT_ENCODER, work_paths
from data.amazon_session_loader import PreparedData
from utils.common import resolve_device, run_record, set_seed, setup_logging, write_json
from utils.encoders import TFIDF_SVD, fit_encoder, save_embeddings

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--encoder", default=DEFAULT_TEXT_ENCODER,
                   help=f"sentence-transformers model ID or path, or '{TFIDF_SVD}'")
    p.add_argument("--tfidf-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)
    paths = work_paths(args.work_dir).ensure()
    device = resolve_device(args.device)
    data = PreparedData.from_paths(paths)

    if args.encoder not in (DEFAULT_TEXT_ENCODER, TFIDF_SVD):
        logger.warning(
            "Encoder %s is not the chapter default. If it was fine-tuned on KDD sessions "
            "(as Chapter 10's model was), validation and test targets leak into the item vectors.",
            args.encoder,
        )
    start = time.time()
    texts = data.texts()
    encoder = fit_encoder(args.encoder, texts, paths.embeddings_dir, device=device,
                          dim=args.tfidf_dim, seed=args.seed)
    encoder.batch_size = args.batch_size
    logger.info("Encoding %d items with %s on %s", len(texts), args.encoder, device)
    embeddings = encoder.encode(texts)
    save_embeddings(paths.item_embeddings, data.item_ids, embeddings, args.encoder)
    elapsed = time.time() - start
    logger.info("Wrote %s %s in %.1fs", paths.item_embeddings, embeddings.shape, elapsed)
    write_json(paths.metrics_dir / "encode_items.json", run_record(
        "encode_items.py", args, num_items=len(texts), dim=int(embeddings.shape[1]),
        seconds=round(elapsed, 1)))


if __name__ == "__main__":
    main()
