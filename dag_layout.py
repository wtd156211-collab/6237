"""有向无环图的自动分层布局。

命令行用法:
    python3 dag_layout.py graph.json
    cat graph.json | python3 dag_layout.py

输入 JSON: {"nodes": [...], "edges": [[u, v], ...]}，边 u -> v 表示 u 在 v 之前。
输出格式见 README「输出格式」一节；出错时输出一行 error,<错误码>,<detail>。

作为库使用时，核心入口是 layout_graph()（抛异常）和 format_graph()（返回输出文本）。
"""

import json
import sys
from collections import deque
from typing import NamedTuple

# 重心扫描的轮数，每轮 = 一次向下扫描 + 一次向上扫描
ROUNDS = 4


class CycleError(Exception):
    """图中存在环。cycle 为环上的节点序列，首尾是同一个节点。"""

    def __init__(self, cycle):
        super().__init__("cycle detected: " + "->".join(cycle))
        self.cycle = cycle


class UnknownNodeError(Exception):
    """边引用了不存在的节点。"""

    def __init__(self, node):
        super().__init__("unknown node: %r" % (node,))
        self.node = node


class LayoutResult(NamedTuple):
    layers: list         # 每层一个列表，层内为最终顺序
    crossings: int       # 总交叉数
    pair_crossings: list # 每对相邻层的交叉数
    node_count: int
    edge_count: int      # 去重后的边数


def _utf8_key(node):
    """节点 id 的 UTF-8 字节序，所有并列打破都用它。"""
    return node.encode("utf-8")


def normalize(nodes, edges):
    """节点、边去重（保留首次出现的顺序），并校验边的端点是否存在。"""
    seen_nodes = set()
    uniq_nodes = []
    for n in nodes:
        if n not in seen_nodes:
            seen_nodes.add(n)
            uniq_nodes.append(n)
    node_set = set(uniq_nodes)
    seen_edges = set()
    uniq_edges = []
    for u, v in edges:
        if u not in node_set:
            raise UnknownNodeError(u)
        if v not in node_set:
            raise UnknownNodeError(v)
        if (u, v) not in seen_edges:
            seen_edges.add((u, v))
            uniq_edges.append((u, v))
    return uniq_nodes, uniq_edges


def find_cycle(nodes, edges):
    """按写死的顺序找环：节点按 id 字节序升序做 DFS，出边也按 id 升序，
    第一次遇到回边就返回环（节点序列，首尾相同）；无环返回 None。
    自环返回 [n, n]。迭代实现，深图也不会爆栈。"""
    succ = {n: [] for n in nodes}
    for u, v in edges:
        succ[u].append(v)
    for n in nodes:
        succ[n].sort(key=_utf8_key)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    for start in sorted(nodes, key=_utf8_key):
        if color[start] != WHITE:
            continue
        color[start] = GRAY
        path = [start]
        stack = [iter(succ[start])]
        while stack:
            advanced = False
            for nxt in stack[-1]:
                if color[nxt] == GRAY:
                    return path[path.index(nxt):] + [nxt]
                if color[nxt] == WHITE:
                    color[nxt] = GRAY
                    path.append(nxt)
                    stack.append(iter(succ[nxt]))
                    advanced = True
                    break
            if not advanced:
                color[path.pop()] = BLACK
                stack.pop()
    return None


def assign_layers(nodes, edges):
    """最长路径分层：入度为 0 的节点在第 0 层，
    其余 layer(v) = 1 + max(layer(u))（取所有入边起点）。
    在 Kahn 拓扑序上递推，一次遍历算完。层内初始顺序按 id 字节序升序。"""
    preds = {n: [] for n in nodes}
    succ = {n: [] for n in nodes}
    for u, v in edges:
        succ[u].append(v)
        preds[v].append(u)

    remaining = {n: len(preds[n]) for n in nodes}
    queue = deque(sorted((n for n in nodes if not preds[n]), key=_utf8_key))
    layer = {}
    while queue:
        u = queue.popleft()
        layer[u] = 1 + max((layer[p] for p in preds[u]), default=-1)
        for v in succ[u]:
            remaining[v] -= 1
            if remaining[v] == 0:
                queue.append(v)

    layer_count = max(layer.values(), default=-1) + 1
    layers = [[] for _ in range(layer_count)]
    for n in nodes:
        layers[layer[n]].append(n)
    for one_layer in layers:
        one_layer.sort(key=_utf8_key)
    return layers


def count_crossings(layers, edges):
    """统计总交叉数与每对相邻层的交叉数。

    只有端点落在相邻两层的边参与计数。对每对相邻层，把边按上层端点位置
    （并列再按下层端点位置）排序后，下层端点位置序列的逆序对数就是交叉数；
    用树状数组求逆序对，单层对 O(E log W)，整体不是边数平方级。"""
    pos = {}
    for i, one_layer in enumerate(layers):
        for p, n in enumerate(one_layer):
            pos[n] = (i, p)

    by_pair = {}
    for u, v in edges:
        lu, pu = pos[u]
        lv, pv = pos[v]
        if lv == lu + 1:
            by_pair.setdefault(lu, []).append((pu, pv))

    pair_crossings = []
    for i in range(len(layers) - 1):
        pair_edges = sorted(by_pair.get(i, ()))
        width = len(layers[i + 1])
        bit = [0] * (width + 1)
        crossings = 0
        for j, (_, pv) in enumerate(pair_edges):
            # 之前已出现的、下层位置大于 pv 的边数 = 与当前边的交叉数
            s = 0
            x = pv + 1
            while x > 0:
                s += bit[x]
                x -= x & -x
            crossings += j - s
            x = pv + 1
            while x <= width:
                bit[x] += 1
                x += x & -x
        pair_crossings.append(crossings)
    return sum(pair_crossings), pair_crossings


def _sweep(layers, neighbor, direction):
    """一趟重心扫描。direction='down' 时从上到下处理第 1..L-1 层，用上一层
    邻居位置的平均值作重心；'up' 时从下到上处理第 L-2..0 层，用下一层。
    在相邻层没有邻居的节点用自己当前的位置当重心。
    并列一律按节点 id 的 UTF-8 字节序升序打破。"""
    pos = {}
    for one_layer in layers:
        for p, n in enumerate(one_layer):
            pos[n] = p
    if direction == "down":
        indices = range(1, len(layers))
        offset = -1
    else:
        indices = range(len(layers) - 2, -1, -1)
        offset = 1
    for i in indices:
        adjacent = layers[i + offset]
        adjacent_set = set(adjacent)
        for p, n in enumerate(adjacent):
            pos[n] = p
        current = layers[i]
        for p, n in enumerate(current):
            pos[n] = p

        def barycenter(n, _adj=adjacent_set, _nb=neighbor, _pos=pos):
            ps = [_pos[x] for x in _nb[n] if x in _adj]
            if ps:
                return sum(ps) / len(ps)
            return float(_pos[n])

        current.sort(key=lambda n: (barycenter(n), _utf8_key(n)))


def order_layers(layers, edges, rounds=ROUNDS):
    """重心法排序：每轮一次向下扫描 + 一次向上扫描，共 rounds 轮。
    每轮结束算一次总交叉数，严格更小才采纳，否则回退到已记录的最优排序。"""
    pred = {n: [] for n in sum(layers, [])}
    succ = {n: [] for n in sum(layers, [])}
    for u, v in edges:
        succ[u].append(v)
        pred[v].append(u)

    best = [list(one_layer) for one_layer in layers]
    best_crossings, _ = count_crossings(layers, edges)
    for _ in range(rounds):
        _sweep(layers, pred, "down")
        _sweep(layers, succ, "up")
        crossings, _ = count_crossings(layers, edges)
        if crossings < best_crossings:
            best_crossings = crossings
            best = [list(one_layer) for one_layer in layers]
        else:
            for i in range(len(layers)):
                layers[i][:] = best[i]
    for i in range(len(layers)):
        layers[i][:] = best[i]
    return layers


def layout_graph(graph):
    """完整布局流程。graph 为 {"nodes": [...], "edges": [[u, v], ...]}。
    有环抛 CycleError，边引用未知节点抛 UnknownNodeError。"""
    nodes, edges = normalize(graph.get("nodes", []), graph.get("edges", []))
    cycle = find_cycle(nodes, edges)
    if cycle is not None:
        raise CycleError(cycle)
    layers = assign_layers(nodes, edges)
    layers = order_layers(layers, edges)
    crossings, pair_crossings = count_crossings(layers, edges)
    return LayoutResult(
        layers=layers,
        crossings=crossings,
        pair_crossings=pair_crossings,
        node_count=len(nodes),
        edge_count=len(edges),
    )


def render(result):
    """把 LayoutResult 渲染成 README 规定的输出文本。"""
    lines = [
        "nodes=%d" % result.node_count,
        "edges=%d" % result.edge_count,
        "layers=%d" % len(result.layers),
        "crossings=%d" % result.crossings,
    ]
    for i, c in enumerate(result.pair_crossings):
        lines.append("pair,%d,%d" % (i, c))
    for i, one_layer in enumerate(result.layers):
        lines.append("layer,%d,%s" % (i, ",".join(one_layer)))
    return "\n".join(lines) + "\n"


def format_graph(graph):
    """完整流程并直接返回输出文本；出错时返回 error 行。"""
    try:
        return render(layout_graph(graph))
    except CycleError as e:
        return "error,CYCLE,%s\n" % "->".join(e.cycle)
    except UnknownNodeError as e:
        return "error,UNKNOWN_NODE,%s\n" % e.node


def main(argv):
    if len(argv) > 1:
        with open(argv[1], "r", encoding="utf-8") as f:
            graph = json.load(f)
    else:
        graph = json.load(sys.stdin)
    output = format_graph(graph)
    sys.stdout.write(output)
    return 1 if output.startswith("error,") else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
