import unittest

import numpy as np

# PyTorch first: on Windows its OpenMP runtime must load before other libraries'.
import torch  # noqa: F401  isort: skip

from ann.sweep import (build_index, default_nlist, index_resources, mark_pareto, recall_at_k,
                       select_operating_points, set_search_param)
from latency.timing import SERVICE, WAIT, Trace
from ranker.bench import speedups
from ranker.parity import rank_parity


def row(index, value, recall, p99, k=100, memory=1.0):
    return {"dataset": "d", "index": index, "value": value, "K": k, "recall_at_K": recall,
            "p99_ms": p99, "memory_mb": memory, "build_s": 0.1}


class RecallTests(unittest.TestCase):
    def test_recall_at_k_counts_true_neighbours_found(self):
        truth = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        found = np.array([[4, 3, 9, -1], [5, 6, 7, 8]])  # -1 is FAISS's "no result"
        self.assertAlmostEqual(recall_at_k(found, truth), (2 + 4) / 8)

    def test_ivf_recall_rises_with_nprobe_and_reaches_exact(self):
        rng = np.random.default_rng(0)
        xb = rng.standard_normal((4_000, 16)).astype(np.float32)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        xq = xb[:50] + 0.1 * rng.standard_normal((50, 16)).astype(np.float32)
        exact, _ = build_index("flat", xb)
        _, truth = exact.search(xq, 10)
        ivf, _ = build_index("ivf_flat", xb, seed=0)
        recalls = []
        for nprobe in (1, 4, ivf.nlist):
            set_search_param(ivf, "ivf_flat", nprobe)
            recalls.append(recall_at_k(ivf.search(xq, 10)[1], truth))
        self.assertLess(recalls[0], recalls[1])
        self.assertAlmostEqual(recalls[-1], 1.0)  # probing every list is exact search
        self.assertEqual(default_nlist(4_000), 64)


class ParetoTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row("flat", "", 1.00, 5.0), row("ivf", 1, 0.60, 0.5), row("ivf", 8, 0.96, 1.0),
                     row("hnsw", 100, 0.95, 1.5), row("pq", 8, 0.50, 0.8, memory=0.1)]

    def test_dominated_settings(self):
        mark_pareto(self.rows)
        dominated = {(r["index"], r["value"]) for r in self.rows if r["dominated"]}
        # hnsw is slower and less accurate than ivf nprobe 8; pq is slower and less accurate than ivf 1.
        self.assertEqual(dominated, {("hnsw", 100), ("pq", 8)})

    def test_selection_is_fastest_setting_meeting_the_floor(self):
        select_operating_points(self.rows, 0.95)
        self.assertEqual([(r["index"], r["value"]) for r in self.rows if r["selected"]], [("ivf", 8)])
        select_operating_points(self.rows, 0.99)
        self.assertEqual([r["index"] for r in self.rows if r["selected"]], ["flat"])  # exact is eligible
        select_operating_points(self.rows, None)
        self.assertFalse(any(r["selected"] for r in self.rows))

    def test_index_resources_keep_compressed_index_visible(self):
        res = {r["index"]: r for r in index_resources(self.rows, 100)}
        self.assertAlmostEqual(res["pq"]["memory_mb"], 0.1)
        self.assertAlmostEqual(res["ivf"]["min_recall"], 0.60)
        self.assertAlmostEqual(res["ivf"]["max_recall"], 0.96)


class AttributionAnalysisTests(unittest.TestCase):
    def test_subtree_time_and_generator_lag(self):
        from scripts.ex1b_pipeline_attribution import analyse

        tr = Trace(0)
        root = tr.add("request", "R", SERVICE, 0, 100)
        tr.add("queue.generator_lag", "S1", "generator_lag", 0, 5, root)
        tr.add("queue", "S1", WAIT, 5, 10, root)
        feat = tr.add("features", "S4", SERVICE, 10, 60, root)
        tr.add("features.fetch", "S4", WAIT, 10, 55, feat)
        tr.add("features.fetch", "S4", WAIT, 10, 30, feat)  # parallel chunk, off the path
        tr.add("heavy_ranker", "S5", SERVICE, 60, 95, root)
        a = analyse(tr)
        self.assertEqual(a["e2e_ns"], 100)
        self.assertEqual(a["subtree_ns"]["features"], 50)  # 45 in the slow chunk + 5 self time
        self.assertEqual(a["stage_kind"][("S1", "generator_lag")], 5)
        self.assertEqual(a["stage_kind"][("S1", WAIT)], 5)
        self.assertEqual(a["stage_kind"][("R", "residual")], 5)
        self.assertEqual(sum(a["stage_kind"].values()), 100)


class SpeedupTests(unittest.TestCase):
    def test_paired_speedup_and_cell_verdicts(self):
        ref_runs = [1.0, 2.0, 1.5]
        rows = [
            {"variant": "torch_eager_fp32", "threads": 1, "batch": 1, "_p50_runs": ref_runs, "_p99_runs": ref_runs},
            {"variant": "fast", "threads": 1, "batch": 1, "_p50_runs": [0.5, 1.0, 0.75], "_p99_runs": ref_runs},
            {"variant": "noisy", "threads": 1, "batch": 1, "_p50_runs": [0.5, 4.0, 1.5], "_p99_runs": ref_runs},
        ]
        speedups(rows, "torch_eager_fp32")
        by = {r["variant"]: r for r in rows}
        self.assertAlmostEqual(by["fast"]["p50_speedup_vs_torch"], 2.0)
        self.assertTrue(by["fast"]["p50_faster"])
        self.assertFalse(by["noisy"]["p50_faster"] or by["noisy"]["p50_slower"])
        self.assertFalse(by["torch_eager_fp32"]["p50_faster"])
        self.assertNotIn("_p50_runs", by["fast"])


class TopKOverlapGroupTests(unittest.TestCase):
    def test_groups_of_exactly_ten_do_not_count_for_top10(self):
        rng = np.random.default_rng(0)
        ref = rng.random(40)
        var = ref.copy()
        var[30:] = var[30:][::-1]  # reorder only the last group, which has exactly 10 rows
        groups = np.repeat([0, 1, 2], [15, 15, 10])
        out = rank_parity(ref, var, groups)
        self.assertEqual(out["groups"], 3)
        self.assertEqual(out["groups_over_10"], 2)
        self.assertAlmostEqual(out["top10_overlap"], 1.0)
        self.assertLess(out["kendall_tau"], 1.0)  # the reordering still shows in tau


if __name__ == "__main__":
    unittest.main()
