"""
Export the DCN-V2 ranker to ONNX, prepare it for quantization, and open
ONNX Runtime sessions at a chosen graph-optimization level.

Two details matter for honest comparisons:

* PyTorch exports ``nn.Linear`` on 2-D inputs as ONNX ``Gemm``.
  ``gemm_to_matmul`` rewrites each ``Gemm`` as ``MatMul`` plus ``Add`` with
  identical arithmetic, so that every dense layer reaches the quantizer as a
  ``MatMul`` whatever a given ONNX Runtime version does with ``Gemm``. With
  the pinned version the quantizer converts ``Gemm`` too; the unit tests
  check that every linear layer ends up as an integer matmul either way.
* Graph optimization levels (basic, extended, all) are applied when the
  session is created, on the machine that runs it. Saved optimized graphs
  are tied to that hardware (ONNX Runtime documentation), so this module
  optimizes online at session creation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import helper, numpy_helper

from ranker.data import SPARSE
from ranker.model_dcn_vendored import DCNV2

OPT_LEVELS = {
    "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


class ExportWrapper(torch.nn.Module):
    """Tensor-in, tensor-out wrapper: (sparse ids [B, 4], dense [B, D]) -> logits [B, 1]."""

    def __init__(self, model: DCNV2):
        super().__init__()
        self.model = model

    def forward(self, sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        return self.model({name: sparse[:, j] for j, name in enumerate(SPARSE)}, dense)


def export_onnx(model: DCNV2, n_dense: int, path: Path, opset: int = 17) -> Path:
    model.eval()
    wrapper = ExportWrapper(model).eval()
    sparse = torch.ones((4, len(SPARSE)), dtype=torch.long)
    dense = torch.zeros((4, n_dense), dtype=torch.float32)
    torch.onnx.export(
        wrapper, (sparse, dense), str(path),
        input_names=["sparse", "dense"], output_names=["logit"],
        dynamic_axes={"sparse": {0: "batch"}, "dense": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(path)))
    return path


def gemm_to_matmul(model: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrite ``Gemm(A, B, C)`` with constant B as ``MatMul(A, B') + C``; returns a new model."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    new_nodes = []
    for node in graph.node:
        if node.op_type != "Gemm" or node.input[1] not in inits:
            new_nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get("transA", 0) or attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0:
            new_nodes.append(node)
            continue
        w = numpy_helper.to_array(inits[node.input[1]])
        if attrs.get("transB", 0):
            w = w.T
        w_name = node.input[1] + "_matmul"
        graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(w), w_name))
        has_bias = len(node.input) > 2 and node.input[2]
        mm_out = node.output[0] + "_mm" if has_bias else node.output[0]
        new_nodes.append(helper.make_node("MatMul", [node.input[0], w_name], [mm_out],
                                          name=(node.name or "gemm") + "_matmul"))
        if has_bias:
            new_nodes.append(helper.make_node("Add", [mm_out, node.input[2]], [node.output[0]],
                                              name=(node.name or "gemm") + "_bias"))
    del graph.node[:]
    graph.node.extend(new_nodes)
    used = {i for n in graph.node for i in n.input}
    keep = [t for t in graph.initializer if t.name in used]
    del graph.initializer[:]
    graph.initializer.extend(keep)
    onnx.checker.check_model(model)
    return model


def quantize_int8(src: Path, dst: Path) -> Path:
    """Dynamic int8 quantization: int8 weights, activations quantized per call."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def session(path: Path, level: str, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = OPT_LEVELS[level]
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def ort_predict(sess: ort.InferenceSession, sparse: np.ndarray, dense: np.ndarray,
                batch: int = 65_536) -> np.ndarray:
    out = []
    for lo in range(0, len(sparse), batch):
        logit = sess.run(None, {"sparse": np.ascontiguousarray(sparse[lo:lo + batch], dtype=np.int64),
                                "dense": np.ascontiguousarray(dense[lo:lo + batch], dtype=np.float32)})[0]
        out.append(1 / (1 + np.exp(-logit.ravel())))
    return np.concatenate(out)


def file_sizes_mb(paths: Dict[str, Path]) -> Dict[str, float]:
    return {k: Path(v).stat().st_size / 2**20 for k, v in paths.items()}
