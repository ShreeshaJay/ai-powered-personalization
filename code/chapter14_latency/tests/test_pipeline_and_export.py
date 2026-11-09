import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

# PyTorch first: on Windows its OpenMP runtime must load before other libraries'.
import torch  # noqa: F401  isort: skip

from pipeline.stages import DEFAULT_CONFIG, apply_levers, mmr_naive, mmr_vectorized

VENDORED = Path(__file__).resolve().parents[1] / "ranker" / "model_dcn_vendored.py"


class MMRTests(unittest.TestCase):
    def test_naive_and_vectorized_select_the_same_slate(self):
        rng = np.random.default_rng(0)
        for trial in range(5):
            vecs = rng.standard_normal((60, 16))
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            scores = rng.random(60)
            self.assertEqual(mmr_naive(scores, vecs, 15, 0.7), mmr_vectorized(scores, vecs, 15, 0.7))

    def test_lambda_one_is_pure_relevance(self):
        rng = np.random.default_rng(1)
        vecs, scores = rng.standard_normal((30, 8)), rng.random(30)
        self.assertEqual(mmr_vectorized(scores, vecs, 5, 1.0), list(np.argsort(-scores)[:5]))


class LeverParsingTests(unittest.TestCase):
    def test_apply_levers(self):
        cfg = apply_levers(DEFAULT_CONFIG, "mmr=naive, candidates=200,hedge-feature-fetch=on")
        self.assertEqual((cfg.mmr, cfg.candidates, cfg.hedge_feature_fetch), ("naive", 200, True))
        self.assertEqual(DEFAULT_CONFIG.label(), "baseline")
        self.assertIn("candidates=200", cfg.label())

    def test_rejects_unknown_levers_and_values(self):
        with self.assertRaises(ValueError):
            apply_levers(DEFAULT_CONFIG, "warp-drive=on")
        with self.assertRaises(ValueError):
            apply_levers(DEFAULT_CONFIG, "mmr=fast")
        with self.assertRaises(ValueError):
            apply_levers(DEFAULT_CONFIG, "mmr")


class VendoredModelTests(unittest.TestCase):
    def test_body_matches_recorded_source_hash(self):
        # Git may check the file out with CRLF on Windows; the recorded hash is of LF text.
        data = VENDORED.read_bytes().replace(b"\r\n", b"\n")
        lines = data.splitlines(keepends=True)
        rules = [i for i, line in enumerate(lines) if line.startswith(b"# =====")]
        header = b"".join(lines[: rules[1] + 1]).decode()
        body = b"".join(lines[rules[1] + 1:])
        recorded = header.split("SHA-256 of the source file:")[1].split()[0]
        self.assertEqual(hashlib.sha256(body).hexdigest(), recorded)


class ExportTests(unittest.TestCase):
    def test_gemm_rewrite_and_onnx_match_pytorch(self):
        import onnx

        from ranker.data import synthetic_ranking
        from ranker.export import export_onnx, gemm_to_matmul, ort_predict, session
        from ranker.train import make_model, predict

        data = synthetic_ranking(n_users=50, n_items=100, n_train=500, n_test=300, seed=0)
        model = make_model(data, seed=0).eval()
        with tempfile.TemporaryDirectory() as tmp:
            raw = export_onnx(model, data.test.dense.shape[1], Path(tmp) / "raw.onnx")
            rewritten = gemm_to_matmul(onnx.load(str(raw)))
            self.assertFalse(any(n.op_type == "Gemm" for n in rewritten.graph.node))
            self.assertTrue(any(n.op_type == "MatMul" for n in rewritten.graph.node))
            mm = Path(tmp) / "mm.onnx"
            onnx.save(rewritten, str(mm))
            ref = predict(model, data.test.sparse, data.test.dense)
            for path in (raw, mm):
                for level in ("disabled", "all"):
                    out = ort_predict(session(path, level, 1), data.test.sparse, data.test.dense)
                    np.testing.assert_allclose(out, ref, atol=1e-5)

    def test_int8_quantizes_every_linear_layer(self):
        import onnx

        from ranker.data import synthetic_ranking
        from ranker.export import export_onnx, gemm_to_matmul, quantize_int8
        from ranker.train import make_model

        def integer_matmuls(path):
            return sum(n.op_type in ("MatMulInteger", "DynamicQuantizeMatMul")
                       for n in onnx.load(str(path)).graph.node)

        data = synthetic_ranking(n_users=50, n_items=100, n_train=500, n_test=300, seed=0)
        model = make_model(data, seed=0).eval()
        with tempfile.TemporaryDirectory() as tmp:
            raw = export_onnx(model, data.test.dense.shape[1], Path(tmp) / "raw.onnx")
            mm = Path(tmp) / "mm.onnx"
            onnx.save(gemm_to_matmul(onnx.load(str(raw))), str(mm))
            n_linear = sum(n.op_type in ("Gemm", "MatMul") for n in onnx.load(str(raw)).graph.node)
            n_gemm = sum(n.op_type == "Gemm" for n in onnx.load(str(raw)).graph.node)
            q_mm = integer_matmuls(quantize_int8(mm, Path(tmp) / "mm_int8.onnx"))
            q_mm_float = sum(n.op_type == "MatMul" for n in onnx.load(str(Path(tmp) / "mm_int8.onnx")).graph.node)
        self.assertGreater(n_gemm, 0)  # the export really produces Gemm, so the rewrite is exercised
        self.assertEqual(q_mm, n_linear)  # every dense and cross layer runs in int8
        self.assertEqual(q_mm_float, 0)


if __name__ == "__main__":
    unittest.main()
