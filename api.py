"""
api.py

单次 NER 调用完成两项任务：
  1. NER：细粒度实体识别（t1/t2 分类）
  2. DAG 边：t2 数值依赖关系（支持多父节点聚合边）

t2 实体统一走 mLDP 扰动 + DAG 一致性推导，无 symbolic 模式。

边结构说明：
  支持 agg 类型边（sum/avg/product/max/min）。
  parent_tokens 字段统一为列表，兼容单父节点（长度1）和多父节点。
"""

import json
from typing import List, Tuple, Dict
import re
import requests


class NERAPI:
    def __init__(self, model: str = "ner:latest", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url


    def extract_entities(
        self, user_prompt: str
    ) -> Tuple[List[Tuple[str, str]], List[Dict]]:
        """
        Returns
        -------
        entities : [(token, label), ...]
        edges    : [{"parent_tokens", "child_token", "rel_type", "param"}, ...]
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_predict": 2048
            },
            "keep_alive": 0  # 不保留会话历史
        }
        # stream=True：逐块接收，避免模型长时间生成时触发读超时
        response = requests.post(url, json=payload, timeout=300, stream=True)
        response.raise_for_status()

        # 拼接所有流式 chunk 的 content
        raw_content = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_content += chunk.get("message", {}).get("content", "")
            if chunk.get("done"):
                break

        content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)

        # ============================================================
        # 提取整个 JSON 对象，再解析
        # ============================================================

        entities = []
        raw_edge_list = []

        # 匹配整个 JSON 对象（贪婪匹配从第一个 { 到最后一个 }）
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        result = json.loads(json_match.group(0))
        # 解析 entities
        for e in result.get("entities", []):
            token = e.get("token", "")
            label = e.get("label", "T")
            if token:
                entities.append((token, label))
         # 获取 edges（优先使用 edges 字段，兼容旧格式）
        raw_edge_list = result.get("edges", [])

        edges = []
        for ed in raw_edge_list:

            edges.append({
                "parent_tokens": ed["parent_tokens"],
                "child_token": ed["child_token"],
                "rel_type": ed["rel_type"],
                "param": ed.get("param", 1.0),
            })

        print(f"NERAPI entities: {entities}")
        print(f"NERAPI edges: {edges}")
        return entities, edges

    def __call__(
        self, user_prompt: str
    ) -> Tuple[List[Tuple[str, str]], List[Dict]]:
        return self.extract_entities(user_prompt)


# ---------------------------------------------------------------------------
# 类型判断工具函数
# ---------------------------------------------------------------------------

T1_LABELS = {
    "PERSON", "ORG", "LOCATION", "PHONE", "EMAIL",
    "ID", "MEDICAL_ID", "IP", "URL", "ENTITY",
}

T2_LABELS = {
    "AGE", "SALARY", "AMOUNT", "COUNT", "WEIGHT",
    "BLOOD_SUGAR", "BLOOD_PRESSURE", "MEDICAL_VAL",
    "YEAR", "DATE_NUM", "NUM", "PERCENT",
}


def is_t1(label: str) -> bool:
    return label in T1_LABELS or label.lower().startswith("t1")


def is_t2(label: str) -> bool:
    return label in T2_LABELS or label.lower().startswith("t2")