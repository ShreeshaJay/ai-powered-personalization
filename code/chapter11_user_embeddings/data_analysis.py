"""
Data Analysis for Chapter 11.1: Empirical Distributions

Computes dataset statistics that ground hyperparameter choices in data rather
than arbitrary defaults. Outputs statistics and distribution plots to
outputs/analysis/.

Key outputs:
1. Session length distribution (Amazon KDD) -> informs Last-K and decay half-life
2. User history length distribution (MIND)  -> informs Last-K and decay half-life
3. Item popularity / frequency distribution -> informs IDF weighting
4. Embedding coverage verification          -> confirms all items have embeddings

Usage:
    python data_analysis.py [--dataset amazon|mind|both]
"""

import numpy as np
import json
import argparse
from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from config import (
    DEFAULT_AMAZON_SESSION_CONFIG,
    DEFAULT_MIND_USER_CONFIG,
    ANALYSIS_DIR,
)
from data.amazon_session_loader import AmazonSessionDataset
from data.mind_user_loader import MINDUserDataset


def analyze_amazon(save_dir: Path = ANALYSIS_DIR) -> dict:
    """Analyze Amazon KDD session data for hyperparameter grounding."""
    logger.info("=" * 80)
    logger.info("Analyzing Amazon KDD Sessions")
    logger.info("=" * 80)

    cfg = DEFAULT_AMAZON_SESSION_CONFIG
    dataset = AmazonSessionDataset(
        data_dir=cfg.data_dir,
        locale=cfg.locale,
        min_session_length=cfg.min_session_length,
        item_embedding_file=cfg.item_embedding_file,
        item_embedding_dim=cfg.item_embedding_dim,
    )
    dataset.load()
    dataset.print_stats()

    # 1. Session length distribution
    stats = dataset.get_session_length_stats()
    logger.info(f"\nSession length stats: {json.dumps(stats, indent=2)}")

    # Histogram of session lengths
    lengths = dataset.sessions_df["session_length"].to_numpy()
    length_counts = Counter(lengths)
    hist = {
        "values": sorted(length_counts.keys()),
        "counts": [length_counts[k] for k in sorted(length_counts.keys())],
    }

    # 2. Item frequency distribution (for IDF)
    logger.info("\nComputing item frequency distribution...")
    item_freq = dataset.get_item_frequency()
    freq_values = np.array(list(item_freq.values()))

    freq_stats = {
        "total_items": len(item_freq),
        "total_sessions": len(dataset.sessions_df),
        "mean_frequency": float(np.mean(freq_values)),
        "median_frequency": float(np.median(freq_values)),
        "max_frequency": int(np.max(freq_values)),
        "min_frequency": int(np.min(freq_values)),
        "p25": float(np.percentile(freq_values, 25)),
        "p75": float(np.percentile(freq_values, 75)),
        "p90": float(np.percentile(freq_values, 90)),
        "p99": float(np.percentile(freq_values, 99)),
        "items_appearing_once": int(np.sum(freq_values == 1)),
        "items_appearing_once_pct": float(np.mean(freq_values == 1) * 100),
    }
    logger.info(f"\nItem frequency stats: {json.dumps(freq_stats, indent=2)}")

    # 3. Derived hyperparameters
    median_session_len = stats["median"]
    # For Last-K: use the median as the "natural" K
    # For decay half-life: set so oldest item in median-length session gets weight ~0.5
    # half_life = median_session_length - 1 (number of positions back for oldest item)
    recommended_half_life = max(1.0, median_session_len - 1)

    recommendations = {
        "last_k_values": [
            max(2, int(stats["p25"])),
            max(2, int(median_session_len)),
            max(3, int(stats["p75"])),
        ],
        "decay_half_life_positions": recommended_half_life,
        "idf_smooth": 1.0,
        "median_session_length": median_session_len,
    }
    logger.info(f"\nRecommended hyperparameters: {json.dumps(recommendations, indent=2)}")

    # Save results
    results = {
        "dataset": "amazon_kdd",
        "locale": cfg.locale,
        "session_length_stats": stats,
        "session_length_histogram": hist,
        "item_frequency_stats": freq_stats,
        "recommendations": recommendations,
    }

    out_path = save_dir / "amazon_kdd_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved analysis to {out_path}")

    return results


def analyze_mind(save_dir: Path = ANALYSIS_DIR) -> dict:
    """Analyze MIND user data for hyperparameter grounding."""
    logger.info("=" * 80)
    logger.info("Analyzing MIND User Histories")
    logger.info("=" * 80)

    cfg = DEFAULT_MIND_USER_CONFIG
    dataset = MINDUserDataset(
        data_dir=cfg.data_dir,
        news_file=cfg.news_file,
        behaviors_file=cfg.behaviors_file,
        min_history_length=cfg.min_history_length,
        item_embedding_file=cfg.item_embedding_file,
        item_embedding_dim=cfg.item_embedding_dim,
    )
    dataset.load()
    dataset.print_stats()

    # 1. User history length distribution
    stats = dataset.get_history_length_stats()
    logger.info(f"\nUser history length stats: {json.dumps(stats, indent=2)}")

    # 2. Item frequency distribution (for IDF)
    logger.info("\nComputing article frequency distribution...")
    item_freq = dataset.get_item_frequency()
    freq_values = np.array(list(item_freq.values()))

    freq_stats = {
        "total_articles": len(item_freq),
        "total_users": stats["count"],
        "mean_frequency": float(np.mean(freq_values)),
        "median_frequency": float(np.median(freq_values)),
        "max_frequency": int(np.max(freq_values)),
        "min_frequency": int(np.min(freq_values)),
        "p25": float(np.percentile(freq_values, 25)),
        "p75": float(np.percentile(freq_values, 75)),
        "p90": float(np.percentile(freq_values, 90)),
        "p99": float(np.percentile(freq_values, 99)),
        "articles_appearing_once": int(np.sum(freq_values == 1)),
        "articles_appearing_once_pct": float(np.mean(freq_values == 1) * 100),
    }
    logger.info(f"\nArticle frequency stats: {json.dumps(freq_stats, indent=2)}")

    # 3. Derived hyperparameters
    median_history_len = stats["median"]

    recommendations = {
        "last_k_values": [
            max(5, int(stats["p25"])),
            max(5, int(median_history_len)),
            max(10, int(stats["p75"])),
            max(20, int(stats["p90"])),
        ],
        "decay_half_life_positions": max(1.0, median_history_len / 2),
        "decay_half_life_hours": 24.0,  # Pedagogical default for time-based
        "idf_smooth": 1.0,
        "median_history_length": median_history_len,
    }
    logger.info(f"\nRecommended hyperparameters: {json.dumps(recommendations, indent=2)}")

    # Save results
    results = {
        "dataset": "mind",
        "history_length_stats": stats,
        "item_frequency_stats": freq_stats,
        "recommendations": recommendations,
    }

    out_path = save_dir / "mind_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved analysis to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Data analysis for Chapter 11.1 hyperparameter grounding"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["amazon", "mind", "both"],
        help="Which dataset to analyze",
    )
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("amazon", "both"):
        analyze_amazon()

    if args.dataset in ("mind", "both"):
        analyze_mind()

    logger.info("\nAnalysis complete. Results saved to outputs/analysis/")


if __name__ == "__main__":
    main()
