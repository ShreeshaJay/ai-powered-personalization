"""Build a sentence encoder from a masked language model or an existing sentence encoder.

Masked language models are wrapped as Transformer, mean pooling, then L2
normalization. The masked-language-model head is not part of that stack.
Loading always passes the pinned revision and does not enable remote code.
"""

from __future__ import annotations

import numpy as np

from config import MAX_SEQ_LENGTH, MODELS, ModelSpec
from evaluate import check_unit_norm


def resolve_device(request: str | None = None) -> str:
    import torch

    if request:
        return request
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_sentence_encoder(
    spec: ModelSpec,
    max_seq_length: int = MAX_SEQ_LENGTH,
    device: str | None = None,
):
    from sentence_transformers import SentenceTransformer

    if spec.kind == "sbert":
        model = SentenceTransformer(
            spec.hub_id,
            revision=spec.revision,
            trust_remote_code=False,
            device=device,
            similarity_fn_name="cosine",
        )
    elif spec.kind == "mlm":
        from sentence_transformers.base.modules.normalize import Normalize
        from sentence_transformers.base.modules.transformer import Transformer
        from sentence_transformers.sentence_transformer.modules.pooling import Pooling

        transformer = Transformer(
            spec.hub_id,
            max_seq_length=max_seq_length,
            model_kwargs={"revision": spec.revision, "trust_remote_code": False},
            processor_kwargs={"revision": spec.revision, "trust_remote_code": False},
            config_kwargs={"revision": spec.revision, "trust_remote_code": False},
        )
        auto_map = getattr(getattr(transformer, "model", None), "config", None)
        auto_map = getattr(auto_map, "auto_map", None)
        if auto_map:
            raise RuntimeError(f"{spec.hub_id} requested remote code")
        dim = int(transformer.get_embedding_dimension())
        if dim != spec.expected_dim:
            raise ValueError(f"{spec.key} embedding dim {dim} != {spec.expected_dim}")
        model = SentenceTransformer(
            modules=[transformer, Pooling(dim, pooling_mode="mean"), Normalize()],
            device=device,
            trust_remote_code=False,
            similarity_fn_name="cosine",
        )
    else:
        raise ValueError(f"Unknown model kind: {spec.kind}")
    model.max_seq_length = max_seq_length
    return model


def embedding_dimension(model) -> int:
    dim = model.get_embedding_dimension()
    return int(dim)


def count_parameters(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def encode_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    check_unit_norm(vectors)
    return vectors


def verify_backbone(key: str, device: str = "cpu") -> dict:
    """Load one backbone, check its dimension, and count parameters."""
    spec = MODELS[key]
    model = build_sentence_encoder(spec, device=device)
    dim = embedding_dimension(model)
    if dim != spec.expected_dim:
        raise ValueError(f"{key} produced dim {dim}, expected {spec.expected_dim}")
    payload = {
        "key": spec.key,
        "hub_id": spec.hub_id,
        "revision": spec.revision,
        "kind": spec.kind,
        "embedding_dim": dim,
        "parameter_count": count_parameters(model),
        "trust_remote_code": False,
    }
    del model
    return payload
