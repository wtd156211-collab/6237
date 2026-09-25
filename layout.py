"""有向无环图的自动分层布局。

输入 {"nodes": [...], "edges": [[u, v], ...]}，输出每个节点的层次、
层内顺序以及交叉数指标。规则见 README.md：

- 层次：最长路径分层，入度为 0 的节点在第 0 层；
- 层内排序：初始按节点 id 的 UTF-8 字节序，之后 4 轮重心扫描
  （下、上、下、上），并列按 id 字节序打破，只采纳严格更优的轮次；
- 交叉数：相邻两层之间交叉边对的总数，用逆序对统计，O(E log E)；
- 有环时报 error,CYCLE,<环上的节点>；边引用未知节点报 UNKNOWN_NODE。

只用标准库。同一张图跑两遍结果逐字节一致。
"""

import json
import sys

SWEEP_ROUNDS = 4


class LayoutError(Exception):
    """布局失败。code 为错误码（CYCLE / UNKNOWN_NODE），detail 为详情。"""

    def __init__(self, code, detail):
        super().__init__("%s,%s" % (code, detail))
        self.code = code
        self.detail = detail


def _byte_key(node):
    return node.encode("utf-8")


def parse_graph(data):
    """校验并规范化输入。返回 (nodes, edges)，边已去重。

    nodes 保持输入出现顺序去重；edges 为去重后的 (u, v) 列表。
    """
    raw_nodes = data.get("nodes", [])
    raw_edges = data.get("edges", [])
    nodes = list(dict.fromkeys(raw_nodes))
    node_set = set(nodes)
    edges = []
    seen = set()
    for pair in raw_edges:
        u, v = pair
        for endpoint in (u, v):
            if endpoint not in node_set:
                raise LayoutError("UNKNOWN_NODE", endpoint)
        if (u, v) not in seen:
            seen.add((u, v))
            edges.append((u, v))
    return nodes, edges


def _build_adjacency(nodes, edges):
    adj = {n: [] for n in nodes}
    radj = {n: [] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
        radj[v].append(u)
    for n in nodes:
        adj[n].sort(key=_byte_key)
        radj[n].sort(key=_byte_key)
    return adj, radj


def find_cycle(nodes, adj):
    """确定性环检测：节点与出边均按 id 字节序升序做 DFS，
    第一次遇到回边即返回环上的节点列表 [n1, n2, ..., n1]；无环返回 None。
    自环也会作为长度为 1 的环被报出。
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    for start in sorted(nodes, key=_byte_key):
        if color[start] != WHITE:
            continue
        color[start] = GRAY
        path = [start]
        stack = [(start, iter(adj[start]))]
        while stack:
            node, it = stack[-1]
            descended = False
            for nxt in it:
                if color[nxt] == GRAY:
                    idx = path.index(nxt)
                    return path[idx:] + [nxt]
                if color[nxt] == WHITE:
                    color[nxt] = GRAY
                    path.append(nxt)
                    stack.append((nxt, iter(adj[nxt])))
                    descended = True
                    break
            if not descended:
                color[node] = BLACK
                stack.pop()
                path.pop()
    return None


def assign_layers(nodes, adj, radj):
    """最长路径分层：layer(v) = 1 + max(layer(u))，入度 0 的节点在第 0 层。
    用 Kahn 拓扑一次遍历算出，取 max 与遍历顺序无关，结果确定。
    返回按层分组的节点列表 [[第0层节点...], ...]（层内未排序）。
    """
    indeg = {n: len(radj[n]) for n in nodes}
    layer = {}
    ready = [n for n in nodes if indeg[n] == 0]
    for n in ready:
        layer[n] = 0
    queue = list(ready)
    while queue:
        u = queue.pop()
        for v in adj[u]:
            cand = layer[u] + 1
            if v not in layer or cand > layer[v]:
                layer[v] = cand
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if not layer:
        return []
    layers = [[] for _ in range(max(layer.values()) + 1)]
    for n, lv in layer.items():
        layers[lv].append(n)
    return layers


def _positions(layer_order):
    return {n: i for i, n in enumerate(layer_order)}


def _count_pair_crossings(order_upper, order_lower, edges_between):
    """统计相邻两层之间的交叉数。

    把每条边映射为 (pos_u, pos_v)，按 pos_u 升序（并列按 pos_v）排序后
    统计 pos_v 序列的逆序对数。du*dl<0 当且仅当构成严格逆序对；
    du==0 或 dl==0 的对按定义不计入，排序与严格大于的比较自然排除。
    复杂度 O(E log E)，不是边两两比较的平方实现。
    """
    pos_u = _positions(order_upper)
    pos_v = _positions(order_lower)
    seq = sorted((pos_u[u], pos_v[v]) for u, v in edges_between)
    vals = [pv for _, pv in seq]
    return _inversion_count(vals)


def _inversion_count(vals):
    """归并排序统计严格逆序对（i < j 且 vals[i] > vals[j]）个数。"""
    if len(vals) < 2:
        return 0
    mid = len(vals) // 2
    left = vals[:mid]
    right = vals[mid:]
    count = _inversion_count(left) + _inversion_count(right)
    i = j = k = 0
    merged = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            count += len(left) - i
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    vals[:] = merged
    return count


def count_crossings(order, edges, layer_of):
    """返回 (总交叉数, 每对相邻层的交叉数列表)。

    只统计两端落在相邻层的边；跨多层的边不参与任何相邻层对的计数。
    """
    num_pairs = max(0, len(order) - 1)
    buckets = [[] for _ in range(num_pairs)]
    for u, v in edges:
        lu, lv = layer_of[u], layer_of[v]
        if lv - lu == 1:
            buckets[lu].append((u, v))
    per_pair = []
    for i, bucket in enumerate(buckets):
        per_pair.append(_count_pair_crossings(order[i], order[i + 1], bucket))
    return sum(per_pair), per_pair


def _barycenter_sort(layer_nodes, neighbor_pos, neighbor_lists):
    """按重心排序一层：重心 = 该节点在相邻层邻居位置的平均值；
    没有邻居的节点用自己当前的位置；并列按 id 的 UTF-8 字节序打破。
    """
    current_pos = _positions(layer_nodes)
    keyed = []
    for n in layer_nodes:
        neighbors = neighbor_lists[n]
        positions = [neighbor_pos[m] for m in neighbors if m in neighbor_pos]
        if positions:
            bary = sum(positions) / len(positions)
        else:
            bary = float(current_pos[n])
        keyed.append((bary, _byte_key(n), n))
    keyed.sort()
    return [n for _, _, n in keyed]


def order_layers(layers, adj, radj, edges):
    """层内排序：初始按 id 字节序，之后 SWEEP_ROUNDS 轮重心扫描
    （偶数轮向下、奇数轮向上）。每轮结束算总交叉数，只有严格更小
    才采纳该轮排序，否则回退到已记录的最优排序。
    返回 (最终各层顺序, 总交叉数, 每对相邻层交叉数)。
    """
    order = [sorted(layer, key=_byte_key) for layer in layers]
    layer_of = {}
    for lv, layer_nodes in enumerate(order):
        for n in layer_nodes:
            layer_of[n] = lv

    best_cross, best_pairs = count_crossings(order, edges, layer_of)
    best = [list(layer_nodes) for layer_nodes in order]

    for rnd in range(SWEEP_ROUNDS):
        if rnd % 2 == 0:
            sweep_range = range(1, len(order))
            neighbor_lists = radj
            neighbor_index = -1
        else:
            sweep_range = range(len(order) - 2, -1, -1)
            neighbor_lists = adj
            neighbor_index = 1
        for i in sweep_range:
            neighbor_pos = _positions(order[i + neighbor_index])
            order[i] = _barycenter_sort(order[i], neighbor_pos, neighbor_lists)

        cross, pairs = count_crossings(order, edges, layer_of)
        if cross < best_cross:
            best_cross, best_pairs = cross, pairs
            best = [list(layer_nodes) for layer_nodes in order]
        else:
            order = [list(layer_nodes) for layer_nodes in best]

    return best, best_cross, best_pairs


def layout(data):
    """对输入图做完整布局。返回可格式化的结果字典；
    有环或未知节点时抛 LayoutError。
    """
    nodes, edges = parse_graph(data)
    adj, radj = _build_adjacency(nodes, edges)
    cycle = find_cycle(nodes, adj)
    if cycle is not None:
        raise LayoutError("CYCLE", "->".join(cycle))
    layers = assign_layers(nodes, adj, radj)
    order, crossings, per_pair = order_layers(layers, adj, radj, edges)
    return {
        "nodes": nodes,
        "edges": edges,
        "order": order,
        "crossings": crossings,
        "per_pair": per_pair,
    }


def format_result(result):
    """按 README 的输出格式渲染布局结果。"""
    lines = [
        "nodes=%d" % len(result["nodes"]),
        "edges=%d" % len(result["edges"]),
        "layers=%d" % len(result["order"]),
        "crossings=%d" % result["crossings"],
    ]
    for i, c in enumerate(result["per_pair"]):
        lines.append("pair,%d,%d" % (i, c))
    for i, layer_nodes in enumerate(result["order"]):
        lines.append("layer,%d,%s" % (i, ",".join(layer_nodes)))
    return "\n".join(lines)


def format_error(err):
    return "error,%s,%s" % (err.code, err.detail)


def main(argv):
    if len(argv) != 2:
        print("usage: python3 layout.py <graph.json>", file=sys.stderr)
        return 2
    with open(argv[1], "r", encoding="utf-8") as fh:
        data = json.load(fh)
    try:
        result = layout(data)
    except LayoutError as err:
        print(format_error(err))
        return 1
    print(format_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
