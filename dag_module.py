"""
dag_module.py

基于有向无环图(DAG)的 t2 实体关联扰动模块。
关系边由 NERAPI 一并返回，本模块不再调用 LLM。

依赖关系类型（RelationType）：
  RATIO   : child = parent * param
  DIFF    : child = parent + param
  PERCENT : child = parent * param / 100
  COPY    : child = parent
  AGG     : child = agg_op(p1, p2, ..., pN)   ← 多父节点聚合

AGG 聚合操作（AggOp）：
  sum  : child = p1 + p2 + ... + pN
  avg  : child = (p1 + p2 + ... + pN) / N
  max  : child = max(p1, ..., pN)
  min  : child = min(p1, ..., pN)

变更说明（v2）：
  · DAGEdge 新增 parent_idxs（列表）和 agg_op 字段
  · parent_idx 属性保留，单父节点时与 parent_idxs[0] 相同，多父节点时为 -1
  · DAGBuilder 解析 parent_tokens（列表）而非 parent_token（字符串），向下兼容旧格式
  · _verify_edge 增加 agg 验证分支
  · DAGPerturber._derive_child_multi 增加 agg 推导逻辑
  · _remove_cycles 兼容多父节点边
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from typing import Union


# ---------------------------------------------------------------------------
# 枚举 & 数据类
# ---------------------------------------------------------------------------

class RelationType(str, Enum):
    RATIO   = "ratio"
    DIFF    = "diff"
    PERCENT = "percent"
    COPY    = "copy"
    AGG     = "agg"


class AggOp(str, Enum):
    SUM = "sum"
    AVG = "avg"
    MAX = "max"
    MIN = "min"


@dataclass
class T2Entity:
    index: int   # 在 ner_result 中的下标
    token: str   # 原始文本
    value: Union[int,float]   # 解析后数值
    precision: int=0
    def __post_init__(self):
        """自动计算精度"""
        if isinstance(self.value, float):
            # 计算小数位数
            str_val = str(self.value)
            if '.' in str_val:
                self.precision = len(str_val.split('.')[1])
            else:
                self.precision = 0
        else:
            self.precision = 0

    def to_stored(self) -> int:
        """转换为存储值（整数）"""
        if self.precision > 0:
            return int(round(self.value * (10 ** self.precision)))
        return int(round(self.value))

    @classmethod
    def from_stored(cls, index: int, token: str, stored_value: int, precision: int):
        """从存储值恢复实体"""
        if precision > 0:
            actual_value = stored_value / (10 ** precision)
        else:
            actual_value = stored_value
        return cls(index=index, token=token, value=actual_value, precision=precision)


@dataclass
class DAGEdge:
    parent_idxs: List[int]          # 所有父节点索引（单父时长度为1）
    child_idx:   int
    rel_type:    RelationType
    param:       float = 1.0
    agg_op:      Optional[AggOp] = None

    @property
    def parent_idx(self) -> int:
        """兼容旧代码的单父节点访问，多父节点时返回 -1。"""
        return self.parent_idxs[0] if len(self.parent_idxs) == 1 else -1

    @property
    def is_agg(self) -> bool:
        return self.rel_type == RelationType.AGG


@dataclass
class DAGGraph:
    graph_id: int
    nodes:    List[int] = field(default_factory=list)
    edges:    List[DAGEdge] = field(default_factory=list)
    roots:    List[int] = field(default_factory=list)
    epsilon:  float = 0.0


# ---------------------------------------------------------------------------
# 边验证
# ---------------------------------------------------------------------------

def _verify_edge(
    parent_vals: List[Union[int,float]],
    child_val:   Union[int,float],
    rel:         RelationType,
    param:       float,
    agg_op:      Optional[AggOp] = None,
    tol:         float = 0.01,
) -> bool:
    """
    验证边关系是否与实际数值吻合。
    tol: 相对误差容忍度（默认 1%），处理浮点和四舍五入误差。
    """
    if not parent_vals:
        return False

    if rel == RelationType.AGG:
        if agg_op is None:
            return False
        if   agg_op == AggOp.SUM: expected = sum(parent_vals)
        elif agg_op == AggOp.AVG: expected = sum(parent_vals) / len(parent_vals)
        elif agg_op == AggOp.MAX: expected = max(parent_vals)
        elif agg_op == AggOp.MIN: expected = min(parent_vals)
        else: return False
    else:
        parent_val = parent_vals[0]
        if parent_val == 0:
            return False
        if   rel == RelationType.RATIO:   expected = parent_val * param
        elif rel == RelationType.DIFF:    expected = parent_val + param
        elif rel == RelationType.PERCENT: expected = parent_val * param / 100
        elif rel == RelationType.COPY:    expected = parent_val
        else: return False

    if expected == 0:
        return child_val == 0
    return abs(expected - child_val) / abs(expected) <= tol


# ---------------------------------------------------------------------------
# DAG 构建器
# ---------------------------------------------------------------------------

class DAGBuilder:
    """
    输入：T2Entity 列表 + 原始边字典列表（来自 NERAPI）
    输出：若干 DAGGraph（并查集分连通分量，Kahn 算法去环）
    """

    def build(
        self,
        t2_entities:   List[T2Entity],
        raw_edges:     List[Dict],
        total_epsilon: float,
    ) -> List[DAGGraph]:
        if not t2_entities:
            return []
        print(raw_edges)
        indices   = [e.index for e in t2_entities]
        token_map = {e.token: e for e in t2_entities}

        # --- 解析边（含数值验证，丢弃不满足关系的边）---
        parsed_edges: List[DAGEdge] = []

        for ed in raw_edges:
            # 兼容旧格式 parent_token（str）和新格式 parent_tokens（list）
            raw_parents = ed.get("parent_tokens") or (
                [ed["parent_token"]] if "parent_token" in ed else []
            )
            child_token = ed.get("child_token", "")

            parent_entities = [token_map[pt] for pt in raw_parents if pt in token_map]
            child_entity    = token_map.get(child_token)
            #调试
            print(f"parent_entities: {[(e.index, e.token, e.value) for e in parent_entities]}")
            print(f"child_entity: {child_entity}")

            #调试
            #child_val = child_entity.value
            #print(f"child_val: {child_val} (type: {type(child_val)})")


            if not parent_entities or child_entity is None:
                continue

            parent_idxs = [e.index for e in parent_entities]
            if child_entity.index in parent_idxs:
                continue

            try:
                rel = RelationType(ed.get("rel_type", "ratio"))
            except ValueError:
                rel = RelationType.RATIO

            param = float(ed.get("param", 1.0))

            # 解析 agg_op
            agg_op = None
            if rel == RelationType.AGG:
                raw_agg = ed.get("agg_op")
                if raw_agg is None:
                    continue
                try:
                    agg_op = AggOp(raw_agg)
                except ValueError:
                    continue

            # 调试
            parent_vals = [e.value for e in parent_entities]
            print(f"parent_vals: {parent_vals} (types: {[type(v) for v in parent_vals]})")


            if not _verify_edge(parent_vals, child_entity.value, rel, param, agg_op):
                continue

            parsed_edges.append(DAGEdge(
                parent_idxs = parent_idxs,
                child_idx   = child_entity.index,
                rel_type    = rel,
                param       = param,
                agg_op      = agg_op,
            ))

        # --- 去环 ---
        valid_edges = _remove_cycles(parsed_edges, indices)

        # --- 并查集分连通分量 ---
        uf = {i: i for i in indices}

        def find(x: int) -> int:
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra != rb:
                uf[rb] = ra

        for e in valid_edges:
            for pidx in e.parent_idxs:
                union(pidx, e.child_idx)

        groups: Dict[int, List[int]] = defaultdict(list)
        for idx in indices:
            groups[find(idx)].append(idx)

        group_edges: Dict[int, List[DAGEdge]] = defaultdict(list)
        for e in valid_edges:
            group_edges[find(e.parent_idxs[0])].append(e)

        # --- 构造 DAGGraph ---
        graphs: List[DAGGraph] = []
        for gid, (rep, nodes) in enumerate(groups.items()):
            g_edges = group_edges[rep]
            in_deg  = {n: 0 for n in nodes}
            for e in g_edges:
                in_deg[e.child_idx] += 1
            roots = [n for n in nodes if in_deg[n] == 0]
            graphs.append(DAGGraph(
                graph_id = gid,
                nodes    = nodes,
                edges    = g_edges,
                roots    = roots,
            ))
        print(graphs)
        # --- 隐私预算：按根节点数量均分 ---
        total_roots  = sum(len(g.roots) for g in graphs)
        eps_per_root = total_epsilon / max(total_roots, 1)
        for g in graphs:
            g.epsilon = eps_per_root * len(g.roots)

        return graphs


# ---------------------------------------------------------------------------
# DAG 扰动器
# ---------------------------------------------------------------------------

class DAGPerturber:
    """根节点 mLDP 扰动，子节点由关系推导。"""

    def __init__(self):
        import mLDP_module as mLDP
        self._mLDP = mLDP
        self._mldp_cache: Dict[str, object] = {}

    def _get_mldp(self, epsilon: float):
        """获取自适应域的 mLDPMechanism 实例。"""
        key = str(epsilon)
        if key not in self._mldp_cache:
            self._mldp_cache[key] = self._mLDP.mLDPMechanism(epsilon=epsilon)
        return self._mldp_cache[key]

    def _perturb_root(self, entity:T2Entity, epsilon: float) -> Union[int,float]:
        stored_value = entity.to_stored()

        # 2. 扰动存储值
        noisy_stored = self._get_mldp(epsilon).perturb(stored_value)

        # 3. 还原为实际值
        if entity.precision > 0:
            noisy_actual = noisy_stored / (10 ** entity.precision)
        else:
            noisy_actual = float(noisy_stored)

        # 保持原始类型
        return noisy_actual

    def _derive_single(self, parent_noisy: Union[int,float], edge: DAGEdge) -> float:
        """单父节点推导子节点值。"""
        rel, p = edge.rel_type, edge.param
        if   rel == RelationType.RATIO:   return parent_noisy * p
        elif rel == RelationType.DIFF:    return parent_noisy + p
        elif rel == RelationType.PERCENT: return parent_noisy * p / 100
        elif rel == RelationType.COPY:    return float(parent_noisy)
        return float(parent_noisy)

    def _derive_agg(self, parent_noisy_vals: List[Union[int,float]], agg_op: AggOp) -> Union[int,float]:
        """多父节点聚合推导子节点值。"""
        if not parent_noisy_vals:
            return 0.0
        if   agg_op == AggOp.SUM: result = sum(parent_noisy_vals)
        elif agg_op == AggOp.AVG: result = sum(parent_noisy_vals) / len(parent_noisy_vals)
        elif agg_op == AggOp.MAX: result = max(parent_noisy_vals)
        elif agg_op == AggOp.MIN: result = min(parent_noisy_vals)
        else: result = sum(parent_noisy_vals)

        return result

    def _derive_child_multi(
            self,
            parent_actual_vals: List[Union[int, float]],
            edges: List[DAGEdge],
            child_entity: T2Entity,  # 只用于 fallback，不用于决定精度
    ) -> Union[int, float]:
        """
        推导子节点值，由计算直接决定返回类型
        """
        if not parent_actual_vals or not edges:
            return child_entity.value  # fallback

        # 1. 计算子节点实际值（浮点数）
        if edges[0].is_agg:
            result_actual = self._derive_agg(parent_actual_vals, edges[0].agg_op)
        elif len(edges) == 1:
            result_actual = self._derive_single(parent_actual_vals[0], edges[0])
        else:
            # 多父节点处理...
            if all(e.rel_type == RelationType.RATIO for e in edges):
                product = 1.0
                for p_val, e in zip(parent_actual_vals, edges):
                    product *= p_val
                result_actual = product
            else:
                results = [self._derive_single(p, e) for p, e in zip(parent_actual_vals, edges)]
                result_actual = sum(results) / len(results)
        print(f"result_actual:{result_actual}")
        return result_actual

    def perturb_graphs(
            self,
            graphs: List[DAGGraph],
            t2_entities: List[T2Entity],
    ) -> Dict[int, Union[int, float]]:
        """扰动图，返回实际值"""

        entity_map = {e.index: e for e in t2_entities}
        noisy_map: Dict[int, Union[int, float]] = {}  # 存储实际值

        for graph in graphs:
            eps_per_root = graph.epsilon / max(len(graph.roots), 1)

            # 1. 扰动根节点（内部处理精度转换）
            for root_idx in graph.roots:
                noisy_map[root_idx] = self._perturb_root(
                    entity_map[root_idx], eps_per_root
                )

            # 2. 构建子节点到入边的映射
            parents_of: Dict[int, List[DAGEdge]] = defaultdict(list)
            for edge in graph.edges:
                parents_of[edge.child_idx].append(edge)

            # 3. 拓扑排序
            in_deg = {n: 0 for n in graph.nodes}
            for edge in graph.edges:
                in_deg[edge.child_idx] += 1

            queue = deque(n for n in graph.nodes if in_deg[n] == 0)
            visited = set(queue)

            while queue:
                cur = queue.popleft()

                # 4. 推导子节点（如果未计算）
                if cur not in noisy_map and cur in parents_of:
                    # 收集所有父节点的实际值
                    parent_actual_vals = []
                    edges_for_cur = []

                    for e in parents_of[cur]:
                        # 检查所有父节点是否都已计算
                        all_parents_ready = all(
                            pidx in noisy_map for pidx in e.parent_idxs
                        )
                        if all_parents_ready:
                            # 获取父节点的实际值
                            for pidx in e.parent_idxs:
                                parent_actual_vals.append(noisy_map[pidx])
                            edges_for_cur.append(e)

                    if parent_actual_vals and edges_for_cur:
                        noisy_map[cur] = self._derive_child_multi(
                            parent_actual_vals,
                            edges_for_cur,
                            entity_map[cur]
                        )


                # 5. 更新入度
                for edge in graph.edges:
                    if cur in edge.parent_idxs:
                        in_deg[edge.child_idx] -= 1
                        if in_deg[edge.child_idx] == 0 and edge.child_idx not in visited:
                            visited.add(edge.child_idx)
                            queue.append(edge.child_idx)

            # 6. 防御：未覆盖节点独立扰动
            for nid in graph.nodes:
                if nid not in noisy_map:
                    noisy_map[nid] = self._perturb_root(
                        entity_map[nid], eps_per_root
                    )
        print(f"noisy_map{noisy_map}")
        return noisy_map


# ---------------------------------------------------------------------------
# 对外统一入口
# ---------------------------------------------------------------------------

class T2DAGProcessor:
    """供 Sanitizer 调用的统一入口。"""

    def __init__(self, total_epsilon: float):
        self.total_epsilon = total_epsilon
        self.builder   = DAGBuilder()
        self.perturber = DAGPerturber()

    def process(
        self,
        t2_list:   List[Tuple[int, str, Union[int,float]]],
        raw_edges: List[Dict],
    ) -> Tuple[Dict[int, Union[int,float]], Dict]:

        if not t2_list:
            return {}, {}

        entities  = [T2Entity(index=i, token=t, value=v) for i, t, v in t2_list]
        graphs    = self.builder.build(entities, raw_edges, self.total_epsilon)
        noisy_map = self.perturber.perturb_graphs(graphs, entities)
        print(f"noisy_map in process:{noisy_map}")
        dag_info = {
            "graphs": [
                {
                    "graph_id": g.graph_id,
                    "nodes":    g.nodes,
                    "roots":    g.roots,
                    "epsilon":  g.epsilon,
                    "edges": [
                        {
                            "parents": e.parent_idxs,
                            "child":   e.child_idx,
                            "rel":     e.rel_type.value,
                            "agg_op":  e.agg_op.value if e.agg_op else None,
                            "param":   e.param,
                        }
                        for e in g.edges
                    ],
                }
                for g in graphs
            ],
            "noisy_map": noisy_map,
        }
        return noisy_map, dag_info


# ---------------------------------------------------------------------------
# 辅助：Kahn 去环（兼容多父节点边）
# ---------------------------------------------------------------------------

def _remove_cycles(edges: List[DAGEdge], all_nodes: List[int]) -> List[DAGEdge]:
    """
    拓扑排序去环。
    agg 边：每个父节点都向子节点连虚拟有向边参与排序。
    """
    adj:    Dict[int, List[int]] = defaultdict(list)
    in_deg: Dict[int, int]       = {n: 0 for n in all_nodes}

    for e in edges:
        for pidx in e.parent_idxs:
            adj[pidx].append(e.child_idx)
        in_deg[e.child_idx] = in_deg.get(e.child_idx, 0) + len(e.parent_idxs)

    queue = deque(n for n in all_nodes if in_deg.get(n, 0) == 0)
    topo: List[int] = []
    while queue:
        node = queue.popleft()
        topo.append(node)
        for nb in adj[node]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)

    topo_set = set(topo)
    return [
        e for e in edges
        if all(pidx in topo_set for pidx in e.parent_idxs)
        and e.child_idx in topo_set
    ]