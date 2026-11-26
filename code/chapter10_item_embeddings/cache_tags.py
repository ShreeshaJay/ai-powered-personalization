"""Filesystem cache tags for embedding files.

Kept free of model libraries so tests can check tag names without loading a checkpoint.
"""

import hashlib
from pathlib import Path


def safe_model_tag(model_path_or_name: str) -> str:
    """Create a filesystem-safe tag from a model name or path."""
    return str(model_path_or_name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def amazon_finetuned_cache_tag(base_model: str, finetuned_path) -> str:
    """Cache tag for one fine-tuned Amazon checkpoint.

    The tag includes a short digest of the checkpoint path. Two checkpoints
    of the same base model then get different cache files, and the same path
    always maps to the same file. MIND tags already differ by path because
    they embed the model path itself.
    """
    path_key = str(Path(finetuned_path))
    digest = hashlib.blake2b(path_key.encode("utf-8"), digest_size=6).hexdigest()
    return f"amazon_kdd_embeddings_finetuned_{safe_model_tag(base_model)}_{digest}"
