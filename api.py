"""
api.py

单次 qwen-turbo 调用完成三项任务：
  1. NER：细粒度实体识别，每个 t2 实体独立标注处理方式
  2. DAG 边：t2 数值依赖关系（支持多父节点聚合边）
  3. prompt 级模式：用于附加 system prompt 给 LLM

关键设计：模式判断在实体级别，而非 prompt 级别

每个 t2 实体有两种处理方式（proc 字段）：
  "perturb"  : mLDP 扰动，用扰动值替换（standard 数据）
               适用：LLM 只引用/判断数值，不对它执行运算
               例：血糖值、年龄、背景性金额

  "symbolic" : 占位符直接替换，不扰动，要求 LLM 输出含占位符的公式
               适用：LLM 会对该数值执行四则运算，响应中出现计算结果
               例：参与税后计算的薪资、参与折扣计算的原价

prompt 级 mode 字段：
  若 prompt 中存在至少一个 symbolic 实体 → mode="symbolic"
  （系统据此在请求 LLM 时附加 symbolic system prompt）
  否则 → mode="standard"

边结构变更说明（v2）：
  新增 agg 类型边，支持多父节点聚合关系（sum/avg/max/min）。
  parent_tokens 字段统一为列表，兼容单父节点（长度1）和多父节点。
  dag_module.py 同步支持 parent_tokens 解析和 agg 推导。
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
    ) -> Tuple[str, List[Tuple[str, str, str]], List[Dict]]:
        """
        Returns
        -------
        mode     : "standard" | "symbolic"
        entities : [(token, label, proc), ...]
                   t1 实体 proc="",  t2 实体 proc="perturb"|"symbolic"
        edges    : [{"parent_tokens", "child_token", "rel_type", "agg_op", "param"}, ...]
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 2048
            },
            "keep_alive": 0  # 不保留会话历史
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        content = result['message']['content']
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        result = json.loads(content)

        mode = result.get("mode", "standard")
        if mode not in ("standard", "symbolic"):
            mode = "standard"

        entities = []
        for e in result.get("entities", []):
            token = e.get("token", "")
            label = e.get("label", "T")
            proc  = e.get("proc", "")
            entities.append((token, label, proc))

        # 边解析：兼容旧格式 parent_token（str）→ 自动升级为 parent_tokens（list）
        edges = []
        for ed in result.get("edges", []):
            if "parent_token" in ed and "parent_tokens" not in ed:
                ed["parent_tokens"] = [ed.pop("parent_token")]
            if "parent_tokens" not in ed:
                continue
            if "agg_op" not in ed:
                ed["agg_op"] = None
            edges.append(ed)
        print(f"NERAPI entities: {result.get('entities')}")
        return mode, entities, edges

    def __call__(
        self, user_prompt: str
    ) -> Tuple[str, List[Tuple[str, str, str]], List[Dict]]:
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
    "YEAR", "DATE_NUM", "NUM",
}


def is_t1(label: str) -> bool:
    return label in T1_LABELS or label.lower().startswith("t1")


def is_t2(label: str) -> bool:
    return label in T2_LABELS or label.lower().startswith("t2")


def is_symbolic(label: str, proc: str) -> bool:
    """该 t2 实体是否需要 symbolic 处理（占位符替换，不扰动）。"""
    return is_t2(label) and proc == "symbolic"


def is_perturb(label: str, proc: str) -> bool:
    """该 t2 实体是否需要 perturb 处理（mLDP 扰动）。"""
    return is_t2(label) and proc != "symbolic"