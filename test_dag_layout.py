import json
import os
import random
import time
import unittest

import dag_layout
from dag_layout import (
    CycleError,
    UnknownNodeError,
    assign_layers,
    count_crossings,
    find_cycle,
    format_graph,
    layout_graph,
    normalize,
    render,
)

SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def load_sample(case_id):
    with open(os.path.join(SAMPLES, case_id + ".graph.json"), encoding="utf-8") as f:
        return json.load(f)


def expected_output(case_id):
    with open(os.path.join(SAMPLES, case_id + ".expected.txt"), encoding="utf-8") as f:
        return f.read()


def crossings_of(case_id):
    return int(format_graph(load_sample(case_id)).splitlines()[3].split("=")[1])


class SampleCasesTest(unittest.TestCase):
    """samples 里的每张图：小图逐字节对齐期望输出，大图压到上界以内。"""

    def test_case_1_empty_graph(self):
        self.assertEqual(format_graph(load_sample("case-1")), expected_output("case-1"))

    def test_case_2_single_node(self):
        self.assertEqual(format_graph(load_sample("case-2")), expected_output("case-2"))

    def test_case_3_isolated_nodes(self):
        self.assertEqual(format_graph(load_sample("case-3")), expected_output("case-3"))

    def test_case_4_barycenter_removes_crossings(self):
        self.assertEqual(format_graph(load_sample("case-4")), expected_output("case-4"))

    def test_case_5_disconnected_components(self):
        self.assertEqual(format_graph(load_sample("case-5")), expected_output("case-5"))

    def test_case_6_cycle(self):
        self.assertEqual(format_graph(load_sample("case-6")), expected_output("case-6"))

    def test_case_7_large_graph_within_bound(self):
        graph = load_sample("case-7")
        output = format_graph(graph)
        crossings = int(output.splitlines()[3].split("=")[1])
        self.assertLessEqual(crossings, graph["crossing_bound"])

    def test_case_7_deterministic_and_fast(self):
        graph = load_sample("case-7")
        start = time.monotonic()
        first = format_graph(graph)
        elapsed = time.monotonic() - start
        self.assertEqual(first, format_graph(graph))
        self.assertLess(elapsed, 5.0)


class NormalizeTest(unittest.TestCase):
    def test_duplicate_edges_removed(self):
        nodes, edges = normalize(["a", "b"], [["a", "b"], ["a", "b"], ["b", "a"], ["b", "a"]])
        self.assertEqual(edges, [("a", "b"), ("b", "a")])

    def test_duplicate_nodes_removed(self):
        nodes, _ = normalize(["a", "a", "b"], [])
        self.assertEqual(nodes, ["a", "b"])

    def test_unknown_node_in_edge_source(self):
        with self.assertRaises(UnknownNodeError) as ctx:
            normalize(["a"], [["x", "a"]])
        self.assertEqual(ctx.exception.node, "x")

    def test_unknown_node_in_edge_target(self):
        with self.assertRaises(UnknownNodeError) as ctx:
            normalize(["a"], [["a", "y"]])
        self.assertEqual(ctx.exception.node, "y")

    def test_unknown_node_error_line(self):
        self.assertEqual(format_graph({"nodes": ["a"], "edges": [["a", "b"]]}),
                         "error,UNKNOWN_NODE,b\n")


class CycleDetectionTest(unittest.TestCase):
    def test_no_cycle(self):
        self.assertIsNone(find_cycle(["a", "b"], [("a", "b")]))

    def test_self_loop(self):
        self.assertEqual(find_cycle(["a"], [("a", "a")]), ["a", "a"])

    def test_cycle_path_reported(self):
        cycle = find_cycle(["a", "b", "c", "d"],
                           [("a", "b"), ("b", "c"), ("c", "a"), ("a", "d")])
        self.assertEqual(cycle, ["a", "b", "c", "a"])

    def test_cycle_error_line(self):
        out = format_graph({"nodes": ["a", "b", "c", "d"],
                            "edges": [["a", "b"], ["b", "c"], ["c", "a"], ["a", "d"]]})
        self.assertEqual(out, "error,CYCLE,a->b->c->a\n")

    def test_cycle_detection_order_is_deterministic(self):
        # 起点与出边都按 id 字节序：从 a 出发先走 b，报 a->b->d->a 而不是别的环
        edges = [("a", "b"), ("a", "c"), ("b", "d"), ("d", "a"), ("c", "d")]
        self.assertEqual(find_cycle(["a", "b", "c", "d"], edges), ["a", "b", "d", "a"])

    def test_layout_raises_cycle_error(self):
        with self.assertRaises(CycleError):
            layout_graph({"nodes": ["a"], "edges": [["a", "a"]]})


class LayeringTest(unittest.TestCase):
    def test_longest_path_layering(self):
        layers = assign_layers(["a", "b", "c"], [("a", "b"), ("b", "c"), ("a", "c")])
        self.assertEqual(layers, [["a"], ["b"], ["c"]])

    def test_isolated_node_goes_to_layer_zero(self):
        layers = assign_layers(["z", "a", "b"], [("a", "b")])
        self.assertEqual(layers, [["a", "z"], ["b"]])

    def test_components_share_layers(self):
        layers = assign_layers(["b1", "a1", "a2"], [("a1", "a2")])
        self.assertEqual(layers, [["a1", "b1"], ["a2"]])

    def test_empty_graph(self):
        self.assertEqual(assign_layers([], []), [])

    def test_initial_order_is_utf8_byte_order(self):
        # UTF-8 字节序：'Z'(0x5A) < 'a'(0x61)，é 的首字节 0xC3 排最后
        layers = assign_layers(["é", "a", "Z"], [])
        self.assertEqual(layers, [["Z", "a", "é"]])


class CrossingCountTest(unittest.TestCase):
    def test_simple_crossing(self):
        layers = [["a", "b"], ["c", "d"]]
        total, pairs = count_crossings(layers, [("a", "d"), ("b", "c")])
        self.assertEqual((total, pairs), (1, [1]))

    def test_no_crossing(self):
        layers = [["a", "b"], ["c", "d"]]
        total, pairs = count_crossings(layers, [("a", "c"), ("b", "d")])
        self.assertEqual((total, pairs), (0, [0]))

    def test_shared_endpoint_not_a_crossing(self):
        layers = [["a", "b"], ["c", "d"]]
        total, _ = count_crossings(layers, [("a", "c"), ("a", "d"), ("b", "d")])
        self.assertEqual(total, 0)

    def test_pairs_summed_across_layer_boundaries(self):
        layers = [["a", "b"], ["c", "d"], ["e", "f"]]
        edges = [("a", "d"), ("b", "c"), ("c", "f"), ("d", "e")]
        total, pairs = count_crossings(layers, edges)
        self.assertEqual(pairs, [1, 1])
        self.assertEqual(total, 2)

    def test_long_edges_not_counted(self):
        # 跨层的边不参与相邻层交叉统计
        layers = [["a"], ["b"], ["c"]]
        total, pairs = count_crossings(layers, [("a", "c")])
        self.assertEqual((total, pairs), (0, [0, 0]))


class OrderingTest(unittest.TestCase):
    def test_barycenter_sort_reduces_crossings(self):
        result = layout_graph({
            "nodes": ["a", "b", "c", "d", "e", "f"],
            "edges": [["a", "f"], ["b", "e"], ["c", "d"], ["a", "e"], ["f", "d"]],
        })
        self.assertEqual(result.crossings, 0)
        self.assertEqual(result.layers, [["a", "b", "c"], ["f", "e"], ["d"]])

    def test_tie_break_by_utf8_byte_order(self):
        # 两个节点都没有上层邻居，重心都取自身位置；交换初始顺序后结果必须一致
        graph1 = {"nodes": ["b", "a"], "edges": []}
        graph2 = {"nodes": ["a", "b"], "edges": []}
        self.assertEqual(format_graph(graph1), format_graph(graph2))
        self.assertIn("layer,0,a,b\n", format_graph(graph1))

    def test_worse_round_is_reverted(self):
        # 空图/单边图：排序不会改变，交叉数保持 0
        result = layout_graph({"nodes": ["a", "b"], "edges": [["a", "b"]]})
        self.assertEqual(result.crossings, 0)
        self.assertEqual(result.layers, [["a"], ["b"]])

    def test_determinism_same_output_twice(self):
        graph = {"nodes": ["n%d" % i for i in range(50)],
                 "edges": [["n%d" % i, "n%d" % (i + 1)] for i in range(0, 49, 2)]}
        self.assertEqual(format_graph(graph), format_graph(graph))


class OutputFormatTest(unittest.TestCase):
    def test_empty_graph_output(self):
        self.assertEqual(format_graph({"nodes": [], "edges": []}),
                         "nodes=0\nedges=0\nlayers=0\ncrossings=0\n")

    def test_single_node_output(self):
        self.assertEqual(format_graph({"nodes": ["a"], "edges": []}),
                         "nodes=1\nedges=0\nlayers=1\ncrossings=0\nlayer,0,a\n")

    def test_render_header_and_pairs(self):
        result = layout_graph({"nodes": ["a", "b"], "edges": [["a", "b"]]})
        text = render(result)
        self.assertEqual(text,
                         "nodes=2\nedges=1\nlayers=2\ncrossings=0\npair,0,0\n"
                         "layer,0,a\nlayer,1,b\n")

    def test_duplicate_edges_counted_once(self):
        result = layout_graph({"nodes": ["a", "b"], "edges": [["a", "b"], ["a", "b"]]})
        self.assertEqual(result.edge_count, 1)


class PerformanceTest(unittest.TestCase):
    def test_large_random_dag(self):
        rng = random.Random(20260924)
        node_count, edge_target = 600, 3000
        nodes = ["n%04d" % i for i in range(node_count)]
        edges = set()
        while len(edges) < edge_target:
            u = rng.randrange(node_count - 1)
            v = rng.randrange(u + 1, node_count)
            edges.add((nodes[u], nodes[v]))
        graph = {"nodes": nodes, "edges": [list(e) for e in edges]}
        start = time.monotonic()
        output = format_graph(graph)
        elapsed = time.monotonic() - start
        self.assertFalse(output.startswith("error,"))
        self.assertLess(elapsed, 5.0)
        self.assertEqual(output, format_graph(graph))


if __name__ == "__main__":
    unittest.main()
