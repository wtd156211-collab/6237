import json
import os
import random
import time
import unittest

import layout

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "samples")


def load_case(case_id):
    with open(os.path.join(SAMPLES, case_id + ".graph.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_expected(case_id):
    with open(os.path.join(SAMPLES, case_id + ".expected.txt"), encoding="utf-8") as fh:
        return fh.read().strip()


def run_case(case_id):
    data = load_case(case_id)
    try:
        return layout.format_result(layout.layout(data))
    except layout.LayoutError as err:
        return layout.format_error(err)


class TestSamplesExact(unittest.TestCase):
    """case-1 ~ case-6 必须与期望输出逐字节一致。"""

    def test_exact_match(self):
        for case_id in ["case-1", "case-2", "case-3", "case-4", "case-5", "case-6"]:
            with self.subTest(case_id=case_id):
                self.assertEqual(run_case(case_id), load_expected(case_id))


class TestBigCase(unittest.TestCase):
    """case-7：交叉数压到上界以内，结构合法，结果确定。"""

    @classmethod
    def setUpClass(cls):
        cls.data = load_case("case-7")
        start = time.time()
        cls.output = layout.format_result(layout.layout(cls.data))
        cls.elapsed = time.time() - start
        cls.lines = cls.output.split("\n")

    def test_crossings_within_bound(self):
        crossings = int(self.lines[3].split("=")[1])
        self.assertLessEqual(crossings, self.data["crossing_bound"])

    def test_header(self):
        self.assertEqual(self.lines[0], "nodes=360")
        self.assertEqual(self.lines[1], "edges=956")  # 990 条边去重后 956
        self.assertEqual(self.lines[2], "layers=12")

    def test_pair_lines_sum_to_total(self):
        pairs = [l for l in self.lines if l.startswith("pair,")]
        self.assertEqual(len(pairs), 11)
        total = sum(int(l.split(",")[2]) for l in pairs)
        crossings = int(self.lines[3].split("=")[1])
        self.assertEqual(total, crossings)

    def test_layer_membership(self):
        """分层结果必须与参考输出一致（每层节点集合相同）。"""
        expected = load_expected("case-7").split("\n")
        exp_layers = [l.split(",")[2:] for l in expected if l.startswith("layer,")]
        got_layers = [l.split(",")[2:] for l in self.lines if l.startswith("layer,")]
        self.assertEqual(len(got_layers), len(exp_layers))
        for got, exp in zip(got_layers, exp_layers):
            self.assertEqual(sorted(got), sorted(exp))
        # 每层内部无重复，全部节点恰好出现一次
        flat = [n for layer in got_layers for n in layer]
        self.assertEqual(sorted(flat), sorted(set(self.data["nodes"])))

    def test_same_layer_has_no_edge(self):
        layer_of = {}
        for l in self.lines:
            if l.startswith("layer,"):
                parts = l.split(",")
                for n in parts[2:]:
                    layer_of[n] = int(parts[1])
        for u, v in {tuple(e) for e in self.data["edges"]}:
            self.assertLess(layer_of[u], layer_of[v])

    def test_deterministic(self):
        again = layout.format_result(layout.layout(self.data))
        self.assertEqual(self.output, again)

    def test_input_order_irrelevant(self):
        """打乱输入中节点与边的顺序，输出必须逐字节相同。"""
        shuffled = json.loads(json.dumps(self.data))
        rng = random.Random(42)
        rng.shuffle(shuffled["nodes"])
        rng.shuffle(shuffled["edges"])
        output = layout.format_result(layout.layout(shuffled))
        self.assertEqual(self.output, output)

    def test_performance(self):
        self.assertLess(self.elapsed, 5.0)


class TestLayering(unittest.TestCase):
    def test_longest_path_layering(self):
        # a -> b -> c，另有 a -> c 的跨层边：c 仍应在第 2 层
        result = layout.layout({"nodes": ["a", "b", "c"],
                                "edges": [["a", "b"], ["b", "c"], ["a", "c"]]})
        self.assertEqual(result["order"], [["a"], ["b"], ["c"]])

    def test_disconnected_components_share_layers(self):
        result = layout.layout({"nodes": ["x1", "x2", "y1"],
                                "edges": [["x1", "x2"]]})
        self.assertEqual(result["order"], [["x1", "y1"], ["x2"]])

    def test_empty_graph(self):
        result = layout.layout({"nodes": [], "edges": []})
        self.assertEqual(result["order"], [])
        self.assertEqual(result["crossings"], 0)
        self.assertEqual(result["per_pair"], [])


class TestCrossingMetric(unittest.TestCase):
    def test_single_crossing(self):
        # 两层各两个节点，边 a->d、b->c 按顺序 a,b / c,d 交叉一次
        order = [["a", "b"], ["c", "d"]]
        layer_of = {"a": 0, "b": 0, "c": 1, "d": 1}
        total, per_pair = layout.count_crossings(
            order, [("a", "d"), ("b", "c")], layer_of)
        self.assertEqual(total, 1)
        self.assertEqual(per_pair, [1])

    def test_shared_endpoint_not_a_crossing(self):
        # du == 0 或 dl == 0 不算交叉
        result = layout.layout({"nodes": ["a", "b", "c"],
                                "edges": [["a", "c"], ["b", "c"]]})
        self.assertEqual(result["crossings"], 0)

    def test_long_edge_not_counted(self):
        # 跨越多层的边不参与相邻层交叉统计
        result = layout.layout({"nodes": ["a", "b", "c"],
                                "edges": [["a", "b"], ["a", "c"], ["b", "c"]]})
        self.assertEqual(result["crossings"], 0)
        self.assertEqual(result["per_pair"], [0, 0])

    def test_duplicate_edges_counted_once(self):
        nodes, edges = layout.parse_graph(
            {"nodes": ["a", "b", "c", "d"],
             "edges": [["a", "d"], ["a", "d"], ["b", "c"]]})
        self.assertEqual(len(edges), 2)
        order = [["a", "b"], ["c", "d"]]
        layer_of = {"a": 0, "b": 0, "c": 1, "d": 1}
        total, _ = layout.count_crossings(order, edges, layer_of)
        self.assertEqual(total, 1)

    def test_inversion_count(self):
        self.assertEqual(layout._inversion_count([3, 1, 2]), 2)
        self.assertEqual(layout._inversion_count([1, 2, 2, 3]), 0)
        self.assertEqual(layout._inversion_count([]), 0)


class TestCycleDetection(unittest.TestCase):
    def test_self_loop(self):
        with self.assertRaises(layout.LayoutError) as ctx:
            layout.layout({"nodes": ["a"], "edges": [["a", "a"]]})
        self.assertEqual(ctx.exception.code, "CYCLE")
        self.assertEqual(ctx.exception.detail, "a->a")

    def test_cycle_reports_nodes_in_order(self):
        with self.assertRaises(layout.LayoutError) as ctx:
            layout.layout({"nodes": ["a", "b", "c"],
                           "edges": [["a", "b"], ["b", "c"], ["c", "a"]]})
        self.assertEqual(ctx.exception.detail, "a->b->c->a")

    def test_cycle_detection_is_deterministic(self):
        # 两个环：按 id 字节序先碰到哪个报哪个
        graph = {"nodes": ["m", "n", "a", "b"],
                 "edges": [["m", "n"], ["n", "m"], ["a", "b"], ["b", "a"]]}
        with self.assertRaises(layout.LayoutError) as ctx:
            layout.layout(graph)
        self.assertEqual(ctx.exception.detail, "a->b->a")

    def test_error_format(self):
        err = layout.LayoutError("CYCLE", "a->b->a")
        self.assertEqual(layout.format_error(err), "error,CYCLE,a->b->a")


class TestUnknownNode(unittest.TestCase):
    def test_unknown_source(self):
        with self.assertRaises(layout.LayoutError) as ctx:
            layout.layout({"nodes": ["a"], "edges": [["z", "a"]]})
        self.assertEqual(ctx.exception.code, "UNKNOWN_NODE")
        self.assertEqual(ctx.exception.detail, "z")

    def test_unknown_target(self):
        with self.assertRaises(layout.LayoutError) as ctx:
            layout.layout({"nodes": ["a"], "edges": [["a", "z"]]})
        self.assertEqual(ctx.exception.code, "UNKNOWN_NODE")


class TestOrderingRules(unittest.TestCase):
    def test_initial_order_by_id_bytes(self):
        result = layout.layout({"nodes": ["b", "a", "c"], "edges": []})
        self.assertEqual(result["order"], [["a", "b", "c"]])

    def test_utf8_byte_order(self):
        # UTF-8 字节序：'z'(0x7a) < 'ä'(0xc3 0xa4)
        result = layout.layout({"nodes": ["ä", "z"], "edges": []})
        self.assertEqual(result["order"], [["z", "ä"]])

    def test_barycenter_reduces_crossings(self):
        # 初始顺序有交叉，重心扫描后应压到 0
        data = {"nodes": ["a", "b", "c", "d"],
                "edges": [["a", "c"], ["b", "d"]]}
        result = layout.layout(data)
        self.assertEqual(result["crossings"], 0)


if __name__ == "__main__":
    unittest.main()
