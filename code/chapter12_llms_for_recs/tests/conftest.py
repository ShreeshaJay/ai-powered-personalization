import sys
from pathlib import Path

import numpy as np
import pytest

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
if str(CHAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(CHAPTER_ROOT))


@pytest.fixture()
def synthetic_work_dir(tmp_path):
    """A prepared synthetic work directory with TF-IDF item embeddings."""
    from config import work_paths
    from data.amazon_session_loader import PreparedData, prepare_sessions_and_items
    from data.synthetic import make_synthetic
    from utils.encoders import fit_encoder, save_embeddings

    paths = work_paths(tmp_path / "work").ensure()
    sessions, products = make_synthetic(num_items=200, num_sessions=600, seed=0)
    sessions, items, _ = prepare_sessions_and_items(
        sessions, products, min_prev_items=3, max_sessions=None, split_fractions=(0.8, 0.1, 0.1),
        catalog="all", extra_random_items=0, desc_max_chars=400, seed=0)
    sessions.write_parquet(paths.sessions)
    items.write_parquet(paths.items)
    data = PreparedData.from_paths(paths)
    encoder = fit_encoder("tfidf-svd", data.texts(), paths.embeddings_dir, dim=16, seed=0)
    save_embeddings(paths.item_embeddings, data.item_ids, encoder.encode(data.texts()), "tfidf-svd")
    return paths


@pytest.fixture()
def rng():
    return np.random.default_rng(0)
