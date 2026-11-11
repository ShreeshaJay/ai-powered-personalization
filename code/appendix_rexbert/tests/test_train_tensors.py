"""Encoder output reaches the validation scorer as a host float32 tensor."""

import numpy as np
import torch

from train import _as_float_tensor


def test_bf16_tensor_becomes_host_float32():
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16)
    out = _as_float_tensor(source)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"
    assert torch.equal(out, source.float())


def test_row_list_and_numpy_input_give_the_same_matrix():
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rows = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    assert torch.equal(_as_float_tensor(rows), expected)
    assert torch.equal(_as_float_tensor(np.array([[1.0, 2.0], [3.0, 4.0]])), expected)
