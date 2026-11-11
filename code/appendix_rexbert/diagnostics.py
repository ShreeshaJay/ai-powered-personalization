"""Cosine-geometry samples and a small fill-mask demonstration.

The fill-mask prompts are examples for the exercise. They are not a measured
result, and token predictions are not used as retrieval evidence.
"""

from __future__ import annotations

import numpy as np

FILL_MASK_PROMPTS = (
    "wireless {mask} headphones with noise cancelling",
    "stainless steel {mask} bottle 750ml",
    "cotton {mask} shirt regular fit",
)


def sample_index_pairs(n_items: int, n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_items < 2 or n_pairs <= 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    rng = np.random.default_rng(seed)
    left = rng.integers(0, n_items, size=n_pairs)
    right = rng.integers(0, n_items - 1, size=n_pairs)
    right = right + (right >= left)
    return left.astype(np.int64), right.astype(np.int64)


def paired_cosine(embeddings: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) == 0:
        return np.zeros(0, dtype=np.float64)
    values = np.sum(embeddings[left] * embeddings[right], axis=1)
    return np.asarray(values, dtype=np.float64)


def fill_mask_demo(hub_id: str, revision: str, top_k: int = 5) -> list[dict]:
    """Top tokens from the masked-language-model head. Remote code stays disabled."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hub_id, revision=revision, trust_remote_code=False)
    model = AutoModelForMaskedLM.from_pretrained(hub_id, revision=revision, trust_remote_code=False)
    model.eval()
    mask = tokenizer.mask_token
    rows = []
    for template in FILL_MASK_PROMPTS:
        prompt = template.format(mask=mask)
        encoded = tokenizer(prompt, return_tensors="pt")
        mask_positions = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=False)
        with torch.no_grad():
            logits = model(**encoded).logits[0]
        predictions = []
        for position in mask_positions.flatten().tolist():
            scores, indices = torch.topk(logits[int(position)], k=top_k)
            predictions.append(
                {
                    "position": int(position),
                    "tokens": [tokenizer.decode([int(index)]).strip() for index in indices.tolist()],
                    "scores": [float(score) for score in scores.tolist()],
                }
            )
        rows.append({"prompt": prompt, "predictions": predictions, "demonstration": True})
    del model
    return rows
