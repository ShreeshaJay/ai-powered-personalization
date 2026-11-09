"""
Train the vendored Chapter 7 DCN-V2 on CPU with a time budget.

Hyperparameters follow Chapter 7's defaults (embedding 16, three cross
layers of rank 32, MLP 256-128-64, parallel structure). The point here is a
realistically shaped, trained model to optimize, not the best AUC.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from ranker.data import SPARSE, RankingData
from ranker.model_dcn_vendored import DCNV2, DCNV2Config


def make_model(data: RankingData, seed: int) -> DCNV2:
    torch.manual_seed(seed)
    config = DCNV2Config(
        sparse_features={name: data.vocab[name] for name in SPARSE},
        dense_features=list(data.dense_names),
        embed_dim=16, cross_layers=3, cross_rank=32, mlp_dims=[256, 128, 64],
        dropout=0.1, use_bn=True, structure="parallel",
    )
    return DCNV2(config)


def sparse_dict(sparse: np.ndarray) -> Dict[str, torch.Tensor]:
    t = torch.from_numpy(np.ascontiguousarray(sparse))
    return {name: t[:, j] for j, name in enumerate(SPARSE)}


def predict(model: DCNV2, sparse: np.ndarray, dense: np.ndarray, batch: int = 65_536) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for lo in range(0, len(sparse), batch):
            logits = model(sparse_dict(sparse[lo:lo + batch]), torch.from_numpy(dense[lo:lo + batch]))
            out.append(torch.sigmoid(logits).numpy().ravel())
    return np.concatenate(out)


def train(data: RankingData, epochs: int, batch_size: int, max_seconds: float, seed: int,
          log=print) -> Tuple[DCNV2, Dict[str, float]]:
    model = make_model(data, seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    tr = data.train
    n = len(tr.label)
    t0 = time.perf_counter()
    steps, stopped_early = 0, False
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(n)
        total = 0.0
        for lo in range(0, n, batch_size):
            idx = order[lo:lo + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            logits = model(sparse_dict(tr.sparse[idx]), torch.from_numpy(tr.dense[idx]))
            loss = loss_fn(logits, torch.from_numpy(tr.label[idx]).unsqueeze(1))
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
            steps += 1
            if time.perf_counter() - t0 > max_seconds:
                stopped_early = True
                break
        log(f"  epoch {epoch + 1}: mean train loss {total / n:.4f} ({time.perf_counter() - t0:.0f} s)")
        if stopped_early:
            log(f"  time budget of {max_seconds:.0f} s reached; stopping")
            break
    model.eval()
    p = predict(model, data.test.sparse, data.test.dense)
    metrics = {"test_auc": float(roc_auc_score(data.test.label, p)) if len(set(data.test.label)) > 1 else float("nan"),
               "train_seconds": time.perf_counter() - t0, "steps": steps, "stopped_early": stopped_early,
               "n_params": int(sum(x.numel() for x in model.parameters())),
               "train_rows": n, "test_rows": int(len(data.test.label))}
    return model, metrics
