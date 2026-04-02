"""
dag_module.py

基于有向无环图(DAG)的 t2 实体关联扰动模块。
支持复合运算的中间节点处理。

依赖关系类型（RelationType）：
  RATIO   : child = parent * param
  DIFF    : child = parent + param
  PERCENT : child = parent * param / 100
  COPY    : child = parent

中间节点规则：
  - 命名格式：_tmp1, _tmp2, ... _tmpN
  - 由复合运算拆解生成，不在原始 t2_entities 中
  - 按编号顺序依次计算
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List,  Tuple, Union
import re


# ---------------------------------------------------------------------------
# 枚举 & 数据类
# ---------------------------------------------------------------------------

class RelationType(str, Enum):
    # 新版 prompt 使用的名称
    MULTIPLY = "multiply"
    ADD = "add"
    PERCENT = "percent"
    COPY = "copy"


@dataclass
class T2Entity:
    index: int  # 在 ner_result 中的下标，中间节点使用负数
    token: str  # 原始文本或中间节点名（如 _tmp1）
    value: Union[int, float]  # 解析后数值
    precision: int = 0
    is_temp: bool = False  # 是否为中间节点

    def __post_init__(self):
        """自动计算精度（仅对非中间节点）"""
        if self.is_temp:
            return
        if isinstance(self.value, float):
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
    parent_idx: int  # 单父节点
    child_idx: int
    rel_type: RelationType
    param: float = 1.0


@dataclass
class DAGGraph:
    graph_id: int
    nodes: List[int] = field(default_factory=list)
    edges: List[DAGEdge] = field(default_factory=list)
    roots: List[int] = field(default_factory=list)
    epsilon: float = 0.0
    temp_nodes: Dict[int, T2Entity] = field(default_factory=dict)  # 中间节点映射


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def is_temp_token(token: str) -> bool:
    """判断是否为中间节点"""
    return token.startswith("_tmp") or token.startswith("_")


def get_temp_order(token: str) -> int:
    """获取中间节点的序号，如 _tmp3 -> 3"""
    match = re.search(r'_tmp(\d+)', token)
    if match:
        return int(match.group(1))
    return 0


# ---------------------------------------------------------------------------
# DAG 构建器
# ---------------------------------------------------------------------------

class DAGBuilder:
    """
    输入：T2Entity 列表 + 原始边字典列表（来自 NERAPI）
    输出：若干 DAGGraph（并查集分连通分量，Kahn 算法去环）
    支持中间节点处理：按 _tmp 编号顺序创建节点
    """

    def build(
            self,
            t2_entities: List[T2Entity],
            raw_edges: List[Dict],
            total_epsilon: float,
    ) -> List[DAGGraph]:
        if not t2_entities:
            return []

        # 构建 token 映射
        token_map = {e.token: e for e in t2_entities}

        # 收集所有边，并识别中间节点
        temp_tokens: set[str] = set()
        all_tokens: set[str] = set()

        for ed in raw_edges:
            raw_parents = ed.get("parent_tokens") or (
                [ed["parent_token"]] if "parent_token" in ed else []
            )
            child_token = ed.get("child_token", "")
            for pt in raw_parents:
                all_tokens.add(pt)
                if is_temp_token(pt):
                    temp_tokens.add(pt)
            all_tokens.add(child_token)
            if is_temp_token(child_token):
                temp_tokens.add(child_token)

        # 按编号排序中间节点
        sorted_temp_tokens = sorted(temp_tokens, key=get_temp_order)

        # 为中间节点创建临时实体（按顺序创建）
        temp_nodes: Dict[str, T2Entity] = {}
        temp_counter = 0
        for token in sorted_temp_tokens:
            if token not in token_map:
                temp_entity = T2Entity(
                    index=-(temp_counter + 1),
                    token=token,
                    value=0.0,
                    is_temp=True
                )
                temp_nodes[token] = temp_entity
                token_map[token] = temp_entity
                temp_counter += 1

        # 获取所有节点索引
        indices = [e.index for e in t2_entities] + [e.index for e in temp_nodes.values()]

        # --- 解析边 ---
        parsed_edges: List[DAGEdge] = []

        for ed in raw_edges:
            raw_parents = ed.get("parent_tokens") or (
                [ed["parent_token"]] if "parent_token" in ed else []
            )
            child_token = ed.get("child_token", "")
            param_value = ed.get("param",1.0)

            parent_entities = [token_map[pt] for pt in raw_parents if pt in token_map]
            child_entity = token_map.get(child_token)

            if not parent_entities or child_entity is None:
                continue

            # 复制父节点列表
            actual_parents = list(parent_entities)
            actual_param = param_value

            # 检查 param 是否对应某个 t2 实体（通过数值匹配）
            #尝试转化实体
            try:
                param_num = param_value
            except (ValueError, TypeError):
                param_num = None

            param_entity = None
            if param_num is not None:
                for token, entity in token_map.items():
                    if not entity.is_temp and abs(entity.value - param_num) < 1e-9:
                        param_entity = entity
                        break

            if param_entity and param_entity not in actual_parents:
                actual_parents.append(param_entity)
                actual_param = 1.0
                print(f"[DAG] 将 param 值 {param_num} 对应的实体 '{param_entity.token}' 添加为父节点")

            # 防止自环
            parent_indices = [e.index for e in actual_parents]
            if child_entity.index in parent_indices:
                continue

            if len(actual_parents) == 1:
                # 单父节点：保持原关系
                rel = RelationType(ed.get("rel_type", "multiply"))
                parsed_edges.append(DAGEdge(
                    parent_idx=actual_parents[0].index,
                    child_idx=child_entity.index,
                    rel_type=rel,
                    param=float(actual_param),
                ))
            else:
                # 多父节点：使用聚合运算
                rel_type = ed.get("rel_type", "sum")

                if rel_type == "add" or rel_type == "sum":
                    # 加法聚合：child = parent1 + parent2 + ... + parentN
                    for p_entity in actual_parents:
                        parsed_edges.append(DAGEdge(
                            parent_idx=p_entity.index,
                            child_idx=child_entity.index,
                            rel_type=RelationType.ADD,
                            param=1.0,
                        ))

                elif rel_type == "multiply" or rel_type == "product":
                    # 乘法聚合：child = parent1 × parent2 × ... × parentN
                    for p_entity in actual_parents:
                        parsed_edges.append(DAGEdge(
                            parent_idx=p_entity.index,
                            child_idx=child_entity.index,
                            rel_type=RelationType.MULTIPLY,
                            param=1.0,
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
            union(e.parent_idx, e.child_idx)

        groups: Dict[int, List[int]] = defaultdict(list)
        for idx in indices:
            groups[find(idx)].append(idx)

        group_edges: Dict[int, List[DAGEdge]] = defaultdict(list)
        for e in valid_edges:
            if e.parent_idx:
                group_edges[find(e.parent_idx)].append(e)
            else:
                group_edges[find(e.child_idx)].append(e)

        # --- 构造 DAGGraph ---
        graphs: List[DAGGraph] = []
        for gid, (rep, nodes) in enumerate(groups.items()):
            g_edges = group_edges[rep]
            in_deg = {n: 0 for n in nodes}
            for e in g_edges:
                in_deg[e.child_idx] += 1
            roots = [n for n in nodes if in_deg[n] == 0]

            # 收集此图中的中间节点
            graph_temp_nodes = {}
            for node_idx in nodes:
                for token, entity in temp_nodes.items():
                    if entity.index == node_idx:
                        graph_temp_nodes[node_idx] = entity
                        break

            graphs.append(DAGGraph(
                graph_id=gid,
                nodes=nodes,
                edges=g_edges,
                roots=roots,
                epsilon=0.0,
                temp_nodes=graph_temp_nodes,
            ))

        # --- 隐私预算：按根节点数量均分 ---
        total_roots = sum(len(g.roots) for g in graphs)
        eps_per_root = total_epsilon / max(total_roots, 1)
        for g in graphs:
            g.epsilon = eps_per_root * len(g.roots)

        return graphs


# ---------------------------------------------------------------------------
# DAG 扰动器
# ---------------------------------------------------------------------------

class DAGPerturber:
    """根节点 mLDP 扰动，子节点由关系推导。支持中间节点按顺序计算。"""

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

    def _perturb_root(self, entity: T2Entity, epsilon: float) -> Union[int, float]:
        stored_value = entity.to_stored()
        noisy_stored = self._get_mldp(epsilon).perturb(stored_value)

        if entity.precision > 0:
            noisy_actual = noisy_stored / (10 ** entity.precision)
        else:
            noisy_actual = float(noisy_stored)

        return noisy_actual

    def _derive_single(self, parent_noisy: Union[int, float], edge: DAGEdge) -> float:
        """单父节点推导子节点值。"""
        rel, p = edge.rel_type, edge.param
        if rel == RelationType.MULTIPLY:
            return parent_noisy * p
        elif rel == RelationType.ADD:
            return parent_noisy + p
        elif rel == RelationType.PERCENT:
            return parent_noisy * p / 100
        elif rel == RelationType.COPY:
            return float(parent_noisy)
        return float(parent_noisy)

    def _aggregate_multi(self, parent_vals: List[Union[int, float]], edges: List[DAGEdge]) -> float:
        """
        多父节点聚合推导子节点值。
        支持 add 和 multiply 两种聚合方式。
        """
        if not parent_vals or not edges:
            return 0.0

        # 获取聚合类型（假设所有边类型相同）
        rel_type = edges[0].rel_type

        if rel_type == RelationType.ADD:
            # 加法聚合：求和
            return sum(parent_vals)
        elif rel_type == RelationType.MULTIPLY:
            # 乘法聚合：求积
            result = 1.0
            for v in parent_vals:
                result *= v
            return result
        else:
            # 其他类型回退到单父节点逻辑
            return self._derive_single(parent_vals[0], edges[0])

    def perturb_graphs(
            self,
            graphs: List[DAGGraph],
            t2_entities: List[T2Entity],
    ) -> Dict[int, Union[int, float]]:
        """扰动图，返回实际值。中间节点按拓扑顺序计算。"""

        entity_map = {e.index: e for e in t2_entities}
        # 添加中间节点到映射
        for graph in graphs:
            for idx, temp_entity in graph.temp_nodes.items():
                entity_map[idx] = temp_entity

        noisy_map: Dict[int, Union[int, float]] = {}

        for graph in graphs:
            eps_per_root = graph.epsilon / max(len(graph.roots), 1)

            # 1. 扰动根节点
            for root_idx in graph.roots:
                if root_idx in entity_map:
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
                    edges = parents_of[cur]

                    # 检查所有父节点是否都已计算
                    all_parents_ready = all(e.parent_idx in noisy_map for e in edges)

                    if all_parents_ready:
                        # 收集所有父节点的扰动值
                        parent_vals = [noisy_map[e.parent_idx] for e in edges]

                        # 根据边数量选择推导方式
                        if len(edges) == 1:
                            # 单父节点
                            noisy_map[cur] = self._derive_single(parent_vals[0], edges[0])
                        else:
                            # 多父节点聚合
                            noisy_map[cur] = self._aggregate_multi(parent_vals, edges)

                # 5. 更新入度
                for edge in graph.edges:
                    if cur == edge.parent_idx:
                        in_deg[edge.child_idx] -= 1
                        if in_deg[edge.child_idx] == 0 and edge.child_idx not in visited:
                            visited.add(edge.child_idx)
                            queue.append(edge.child_idx)

            # 6. 防御：未覆盖节点独立扰动
            for nid in graph.nodes:
                if nid not in noisy_map and nid in entity_map:
                    noisy_map[nid] = self._perturb_root(
                        entity_map[nid], eps_per_root
                    )

        return noisy_map


# ---------------------------------------------------------------------------
# 对外统一入口
# ---------------------------------------------------------------------------

class T2DAGProcessor:
    """供 Sanitizer 调用的统一入口。"""

    def __init__(self, total_epsilon: float):
        self.total_epsilon = total_epsilon
        self.builder = DAGBuilder()
        self.perturber = DAGPerturber()

    def process(
            self,
            t2_list: List[Tuple[int, str, Union[int, float]]],
            raw_edges: List[Dict],
    ) -> Tuple[Dict[int, Union[int, float]], Dict]:
        if not t2_list:
            return {}, {}

        entities = [T2Entity(index=i, token=t, value=v) for i, t, v in t2_list]
        graphs = self.builder.build(entities, raw_edges, self.total_epsilon)
        noisy_map = self.perturber.perturb_graphs(graphs, entities)

        dag_info = {
            "graphs": [
                {
                    "graph_id": g.graph_id,
                    "nodes": g.nodes,
                    "roots": g.roots,
                    "epsilon": g.epsilon,
                    "edges": [
                        {
                            "parent": e.parent_idx,
                            "child": e.child_idx,
                            "rel": e.rel_type.value,
                            "param": e.param,
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
# 辅助：Kahn 去环
# ---------------------------------------------------------------------------

def _remove_cycles(edges: List[DAGEdge], all_nodes: List[int]) -> List[DAGEdge]:
    """
    拓扑排序去环。
    """
    adj: Dict[int, List[int]] = defaultdict(list)
    in_deg: Dict[int, int] = {n: 0 for n in all_nodes}

    for e in edges:
        adj[e.parent_idx].append(e.child_idx)
        in_deg[e.child_idx] = in_deg.get(e.child_idx, 0) + 1

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
        if e.parent_idx in topo_set and e.child_idx in topo_set
    ]