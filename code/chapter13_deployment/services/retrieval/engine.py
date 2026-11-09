"""
Two-tower retrieval with an in-process FAISS index (Section 13.3, Figure 13.4).

Two embedding tables are involved:
  * content embeddings (Yambda audio, 128-d) feed the user tower, looked up
    for the user's recent items at request time;
  * item vectors (item-tower outputs, 50-d, L2-normalised) live in the ANN
    index, computed offline in one batch.

The user tower and the index must come from the same model version. At load
the engine checks the index manifest against the checkpoint and re-encodes a
few items with the loaded item tower to confirm the stored vectors match; any
mismatch stops the service.
"""

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import faiss
import numpy as np
import torch

from common import published, warmup
from common.artifacts import ArtifactError, read_json, require_file, verify_files

SPOT_CHECK_ITEMS = 32
SPOT_CHECK_TOLERANCE = 1e-4


class RetrievalEngine:
    def __init__(self, model, index, item_ids: np.ndarray, content: np.ndarray, popularity: List[int],
                 manifest: dict, store=None):
        self.ch5 = published.chapter5_two_tower()
        self.model = model
        self.index = index
        self.item_ids = item_ids
        self.position = {int(item): pos for pos, item in enumerate(item_ids)}
        self.lookup = self.ch5.ItemEmbeddingLookup(
            sku_to_idx=self.position, idx_to_sku={pos: int(item) for pos, item in enumerate(item_ids)},
            embeddings=content)
        self.max_seq_len = int(manifest["max_seq_len"])
        self.popularity = [int(i) for i in popularity]
        self.manifest = manifest
        self.model_version = manifest["model_version"]
        self.index_version = manifest["index_version"]
        self.store = store

    @classmethod
    def load(cls, bundle_dir: Path, store=None) -> "RetrievalEngine":
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
        faiss.omp_set_num_threads(int(os.environ.get("FAISS_NUM_THREADS", "1")))
        ch5 = published.chapter5_two_tower()
        directory = Path(bundle_dir) / "retrieval"
        manifest = read_json(directory / "index_manifest.json", "index manifest")
        verify_files(directory, manifest["files"], "retrieval")
        if manifest["files"].get(manifest["model_file"]) != manifest["model_sha256"]:
            raise ArtifactError("Index manifest names a different user-tower checkpoint than the one it lists.")

        checkpoint = torch.load(require_file(directory / manifest["model_file"], "two-tower checkpoint"),
                                map_location="cpu", weights_only=True)
        if checkpoint.get("model_version") != manifest["model_version"]:
            raise ArtifactError(
                f"User tower {checkpoint.get('model_version')} does not match the index, which was built for "
                f"{manifest['model_version']}. Serve a tower and an index from the same build.")
        model = ch5.TwoTowerModel(**checkpoint["model_args"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        index = faiss.read_index(str(require_file(directory / manifest["index_file"], "FAISS index")))
        item_ids = np.load(require_file(directory / "item_ids.npy", "index item IDs"))
        content = np.load(require_file(directory / "content_embeddings.npy", "content embeddings"))
        if not (index.ntotal == len(item_ids) == len(content) == manifest["num_items"]):
            raise ArtifactError("Index, item IDs and content embeddings have different lengths.")
        if index.d != manifest["vector_dim"] or content.shape[1] != manifest["content_embedding_dim"]:
            raise ArtifactError("Index or content embedding dimension does not match the manifest.")

        rows = np.linspace(0, len(item_ids) - 1, num=min(SPOT_CHECK_ITEMS, len(item_ids))).astype(np.int64)
        with torch.no_grad():
            fresh = model.encode_items(torch.tensor(content[rows], dtype=torch.float32)).numpy()
        fresh = np.ascontiguousarray(fresh, dtype=np.float32)
        faiss.normalize_L2(fresh)
        stored = np.vstack([index.reconstruct(int(r)) for r in rows])
        gap = float(np.abs(fresh - stored).max())
        if gap > SPOT_CHECK_TOLERANCE:
            raise ArtifactError(
                f"Index vectors differ from this user tower's item tower (max difference {gap:.3g}); "
                "the index was built by another model version.")

        popularity = read_json(directory / "popularity.json", "retrieval popularity list")["item_ids"]
        return cls(model, index, item_ids, content, popularity, manifest, store)

    def query_vector(self, history: Sequence[int]) -> np.ndarray:
        """Chapter 5's query path: last MAX_SEQ_LEN items, left-padded, masked, then the user tower."""
        trimmed = list(history)[-self.max_seq_len:]
        embs, mask = self.lookup.get_embeddings(trimmed)
        if len(trimmed) < self.max_seq_len:
            pad = self.max_seq_len - len(trimmed)
            embs = np.vstack([np.zeros((pad, self.lookup.embeddings.shape[1]), dtype=np.float32), embs])
            mask = [False] * pad + list(mask)
        with torch.no_grad():
            user = self.model.encode_user(torch.tensor(embs, dtype=torch.float32).unsqueeze(0),
                                          torch.tensor(mask, dtype=torch.bool).unsqueeze(0)).numpy()
        user = np.ascontiguousarray(user, dtype=np.float32)
        faiss.normalize_L2(user)
        return user

    def retrieve(self, user_id: int, recent_item_ids: Optional[List[int]], exclude_item_ids: Sequence[int],
                 k: int) -> Tuple[List[Tuple[int, float]], str, int]:
        if recent_item_ids is None:
            recent_item_ids = self.store.get_user(user_id).recent_item_ids if self.store is not None else []
        history = list(recent_item_ids)[-self.max_seq_len:]
        excluded = set(history) | set(int(i) for i in exclude_item_ids)
        known = [i for i in history if i in self.position]
        if not known:
            # User cold start: no recent item the tower can read.
            items = [i for i in self.popularity if i not in excluded][:k]
            return [(i, float(len(items) - rank) / len(items)) for rank, i in enumerate(items)], \
                "popularity_cold_start", len(history)

        query = self.query_vector(history)
        positions = np.array(sorted(self.position[i] for i in excluded if i in self.position), dtype=np.int64)
        k_search = min(k, self.index.ntotal)
        if len(positions):
            inner = faiss.IDSelectorBatch(positions)
            params = faiss.SearchParameters(sel=faiss.IDSelectorNot(inner))
            scores, rows = self.index.search(query, k_search, params=params)
        else:
            scores, rows = self.index.search(query, k_search)
        results = [(int(self.item_ids[r]), float(s)) for r, s in zip(rows[0], scores[0]) if r >= 0]
        return results, "two_tower", len(history)

    def warm_up(self, k: int = 300) -> float:
        history = [int(i) for i in self.item_ids[:self.max_seq_len]]
        return warmup.timed(lambda: self.retrieve(0, history, [], k))
