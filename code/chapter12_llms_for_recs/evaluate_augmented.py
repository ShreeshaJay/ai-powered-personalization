"""
Section 12.4: score each trained augmented ranker on the shared test slates.

Uses the same slate file as the LLM rankers (default: test, retrieved, 20 candidates) so
the MLP ranker, the zero-shot LLM, TALLRec and the non-learned baselines are directly
comparable. Results are split by target popularity, using training-split counts only:

    cold_target   the target appears in fewer than cold_item_threshold train sessions
    warm_target   the rest

Output: <work_dir>/metrics/augmented_ranker_<split>_<mode>_n<size>.json

Example:
    python evaluate_augmented.py --work-dir outputs/tiny
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

from config import FEATURE_SETS, AugmentedRankerConfig, work_paths
from data.amazon_session_loader import PreparedData
from models.augmented_ranker import AugmentedRanker
from utils.common import resolve_device, run_record, setup_logging, write_json
from utils.ranker_data import load_feature_tables, pad_histories, to_tensors
from utils.slates import baseline_rankings, evaluate_slate_rankings, read_slates, slate_summary

logger = logging.getLogger(__name__)
K_VALUES = [1, 5, 10]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", default=None)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--slate-mode", default="retrieved")
    p.add_argument("--slate-size", type=int, default=20)
    p.add_argument("--feature-sets", nargs="+", default=None, help="Default: every trained checkpoint")
    p.add_argument("--allow-partial-coverage", action="store_true")
    p.add_argument("--cold-threshold", type=int, default=AugmentedRankerConfig.cold_item_threshold)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = AugmentedRankerConfig()
    paths = work_paths(args.work_dir).ensure()
    device = resolve_device(args.device)
    data = PreparedData.from_paths(paths)
    feature_sets = args.feature_sets or [f for f in FEATURE_SETS if (paths.models_dir / f"ranker_{f}.pt").exists()]
    if not feature_sets:
        raise FileNotFoundError("no ranker checkpoints found; run train_augmented_ranker.py first")
    top = max(FEATURE_SETS.index(f) for f in feature_sets)
    tables = load_feature_tables(paths, data, need_item_llm=top >= 2, need_intent=top >= 3,
                                 min_coverage=cfg.min_llm_coverage, allow_partial=args.allow_partial_coverage)
    slates = read_slates(paths.slate_file(args.split, args.slate_mode, args.slate_size))

    hist, mask = pad_histories([[data.item_index[h] for h in s["history"]] for s in slates], cfg.max_history_items)
    rows = np.array([tables["session_row"][s["session_id"]] for s in slates])
    cands = np.array([[data.item_index[c] for c in s["candidates"]] for s in slates])
    n, c = cands.shape

    counts = data.train_item_counts()
    cold = np.array([counts[data.item_index[s["target"]]] < args.cold_threshold for s in slates])
    subsets = {"all": np.ones(n, dtype=bool), "cold_target": cold, "warm_target": ~cold}

    def by_subset(rankings):
        return {name: evaluate_slate_rankings([slates[i] for i in np.flatnonzero(m)],
                                              [rankings[i] for i in np.flatnonzero(m)], K_VALUES)
                for name, m in subsets.items() if m.any()}

    results = {}
    for feature_set in feature_sets:
        ckpt = torch.load(paths.models_dir / f"ranker_{feature_set}.pt", map_location="cpu", weights_only=True)
        model = AugmentedRanker(feature_set, **to_tensors(tables, FEATURE_SETS.index(feature_set)),
                                hidden_dims=ckpt["hidden_dims"], dropout=ckpt["dropout"])
        model.mlp.load_state_dict({k[len("mlp."):]: v for k, v in ckpt["state_dict"].items()})
        model.to(device).eval()
        with torch.no_grad():
            scores = model(torch.from_numpy(cands.reshape(-1)).to(device),
                           torch.from_numpy(np.repeat(hist, c, axis=0)).to(device),
                           torch.from_numpy(np.repeat(mask, c, axis=0)).to(device),
                           torch.from_numpy(np.repeat(rows, c)).to(device)).reshape(n, c).cpu().numpy()
        order = np.argsort(-scores, axis=1, kind="stable")
        rankings = [[s["candidates"][j] for j in order[i]] for i, s in enumerate(slates)]
        results[feature_set] = by_subset(rankings)
        m = results[feature_set]["all"]["target_in_slate"]
        logger.info("%-26s recall@10 %.4f  ndcg@10 %.4f  mrr %.4f (target in slate, n=%d)", feature_set,
                    m.get("recall@10", float("nan")), m.get("ndcg@10", float("nan")), m.get("mrr", float("nan")),
                    m.get("num_queries", 0))

    popularity = dict(zip(data.item_ids, counts.tolist()))
    baselines = {name: by_subset(r) for name, r in baseline_rankings(slates, popularity, args.seed).items()}
    tag = f"{args.split}_{args.slate_mode}_n{args.slate_size}"
    write_json(paths.metrics_dir / f"augmented_ranker_{tag}.json", run_record(
        "evaluate_augmented.py", args, encoder=tables["encoder"], llm_coverage=tables["coverage"],
        llm_source=tables.get("llm_source", {}), slates=slate_summary(slates, K_VALUES),
        cold_target_sessions=int(cold.sum()), results=results, baselines=baselines))
    logger.info("Wrote %s", paths.metrics_dir / f"augmented_ranker_{tag}.json")


if __name__ == "__main__":
    main()
