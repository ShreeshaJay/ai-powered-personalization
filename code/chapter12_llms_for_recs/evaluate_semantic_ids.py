"""
Section 12.2: inspect the Semantic IDs produced by train_rqvae.py.

Checks, written to <work_dir>/metrics/semantic_id_analysis.json:

1. Content coherence by prefix depth: mean cosine similarity between items that share a
   prefix of length 1, 2, ..., compared with random item pairs. The IDs were computed from
   these same embeddings, so this is a sanity check of the quantiser, not evidence about
   recommendation quality.
2. Behavioural agreement: for test sessions, how often the last history item and the
   target share a prefix of each length, against random item pairs. This uses session
   data the quantiser never saw.
3. Cold versus warm items: coherence for items seen in fewer than --cold-threshold
   training sessions, against the rest. Counts come from the training split only.
4. Collisions before the suffix level is added.
5. Optional t-SNE plot coloured by first-level code (--tsne; needs matplotlib).

TIGER's retrieval quality on cold targets is measured in train_tiger.py.

Example:
    python evaluate_semantic_ids.py --work-dir outputs/tiny
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from typing import Dict

import numpy as np

from config import work_paths
from data.amazon_session_loader import PreparedData, load_item_embeddings
from models.rqvae import code_statistics
from utils.common import run_record, setup_logging, write_json
from utils.faiss_index import l2_normalize

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--cold-threshold", type=int, default=5)
    p.add_argument("--pairs", type=int, default=20_000, help="Item pairs sampled per estimate")
    p.add_argument("--tsne", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def prefix_coherence(vectors: np.ndarray, codes: np.ndarray, items: np.ndarray, pairs: int,
                     rng: np.random.Generator) -> Dict[str, Dict[str, float]]:
    """Mean cosine of random same-prefix pairs among ``items``, per prefix length."""
    out = {}
    for depth in range(1, codes.shape[1] + 1):
        groups = defaultdict(list)
        for i in items.tolist():
            groups[tuple(codes[i, :depth])].append(i)
        multi = [g for g in groups.values() if len(g) > 1]
        if not multi:
            out[f"prefix_{depth}"] = {"same_prefix_cosine": float("nan"), "groups_with_pairs": 0}
            continue
        sizes = np.array([len(g) for g in multi], dtype=np.float64)
        # Sample groups in proportion to their number of pairs, then a pair inside the group.
        weights = sizes * (sizes - 1)
        chosen = rng.choice(len(multi), size=pairs, p=weights / weights.sum())
        sims = []
        for g in chosen.tolist():
            a, b = rng.choice(multi[g], size=2, replace=False)
            sims.append(float(vectors[a] @ vectors[b]))
        out[f"prefix_{depth}"] = {"same_prefix_cosine": float(np.mean(sims)),
                                  "groups_with_pairs": len(multi)}
    return out


def main() -> None:
    setup_logging()
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    paths = work_paths(args.work_dir)
    data = PreparedData.from_paths(paths)
    vectors, meta = load_item_embeddings(paths.item_embeddings, data.item_ids)
    vectors = l2_normalize(vectors)
    sid = np.load(paths.semantic_ids, allow_pickle=False)
    index = {iid: i for i, iid in enumerate(sid["item_ids"].tolist())}
    codes = sid["codes"][[index[iid] for iid in data.item_ids]]
    codebook_size = int(sid["level_sizes"][0])
    n = len(vectors)

    a, b = rng.integers(0, n, args.pairs), rng.integers(0, n, args.pairs)
    random_cos = float(np.mean(np.sum(vectors[a] * vectors[b], axis=1)))
    coherence = prefix_coherence(vectors, codes, np.arange(n), args.pairs, rng)

    test = data.split("test")
    last = np.array([data.item_index[h[-1]] for h in test["history"].to_list()])
    target = data.target_indices(test)
    behaviour = {}
    for depth in range(1, codes.shape[1] + 1):
        same = np.all(codes[last, :depth] == codes[target, :depth], axis=1).mean()
        rand = np.all(codes[a, :depth] == codes[b, :depth], axis=1).mean()
        behaviour[f"prefix_{depth}"] = {"last_item_and_target_share_prefix": float(same),
                                        "random_pairs_share_prefix": float(rand)}

    counts = data.train_item_counts()
    cold_items = np.flatnonzero(counts < args.cold_threshold)
    warm_items = np.flatnonzero(counts >= args.cold_threshold)
    cold_warm = {}
    for name, items in (("cold", cold_items), ("warm", warm_items)):
        if len(items) > 1:
            cold_warm[name] = {"num_items": int(len(items)),
                               **prefix_coherence(vectors, codes, items, min(args.pairs, 5000), rng)}

    stats = code_statistics(codes, codebook_size)
    logger.info("Random-pair cosine: %.4f", random_cos)
    for key, val in coherence.items():
        logger.info("%s: same-prefix cosine %.4f | test sessions sharing prefix %.3f vs random %.4f",
                    key, val["same_prefix_cosine"], behaviour[key]["last_item_and_target_share_prefix"],
                    behaviour[key]["random_pairs_share_prefix"])
    logger.info("Cold items (train count < %d): %d; warm: %d", args.cold_threshold, len(cold_items), len(warm_items))

    plot = None
    if args.tsne:
        plot = make_tsne(vectors, codes, paths, rng)
    write_json(paths.metrics_dir / "semantic_id_analysis.json", run_record(
        "evaluate_semantic_ids.py", args, item_encoder=meta["encoder"], random_pair_cosine=random_cos,
        content_coherence=coherence, behavioural_agreement=behaviour, cold_vs_warm=cold_warm,
        code_statistics_before_suffix=stats, tsne_plot=plot))
    logger.info("Wrote %s", paths.metrics_dir / "semantic_id_analysis.json")


def make_tsne(vectors, codes, paths, rng):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        logger.warning("matplotlib or scikit-learn missing; skipping t-SNE")
        return None
    sample = rng.choice(len(vectors), size=min(5000, len(vectors)), replace=False)
    coords = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample) - 1)).fit_transform(vectors[sample])
    level1 = codes[sample, 0]
    values, counts = np.unique(level1, return_counts=True)
    top = values[np.argsort(-counts)][:20]
    fig, ax = plt.subplots(figsize=(10, 8))
    colours = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, code in enumerate(top):
        m = level1 == code
        ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.6, color=colours[i], label=f"code {code}")
    other = ~np.isin(level1, top)
    ax.scatter(coords[other, 0], coords[other, 1], s=4, alpha=0.3, color="lightgray", label="other")
    ax.set_title("Item embeddings (t-SNE) coloured by first-level Semantic ID code")
    ax.legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    out = paths.metrics_dir / "tsne_semantic_ids.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out.name


if __name__ == "__main__":
    main()
