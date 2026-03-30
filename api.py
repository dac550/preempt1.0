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

from openai import OpenAI
import json
from typing import List, Tuple, Dict
import re
import requests


class NERAPI:
    def __init__(self, model: str = "qwen3.5:4b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.system_prompt = """
你是一个实体识别、数值关系分析与处理方式判断专家。
对用户输入的句子，请同时完成三项任务：

【任务一】提取所有实体，返回细粒度类型和处理方式

t1 类（需要FPE加密，label 填写具体类型）：
  PERSON / ORG / LOCATION / PHONE / EMAIL / ID / MEDICAL_ID / IP / URL / ENTITY

t2 类（数值敏感实体，需额外标注 proc 字段）：
  label 填写：AGE / SALARY / AMOUNT / COUNT / WEIGHT /
              BLOOD_SUGAR / BLOOD_PRESSURE / MEDICAL_VAL /
              YEAR / DATE_NUM / NUM

  proc 字段（仅 t2 实体需要）：

  判断核心只有一条：
    LLM 是否会对该数值执行四则运算（+ - × ÷），使响应中出现的是计算结果而非原值？
      是 → "symbolic"
      否 → "perturb"

  "perturb"（默认选项，绝大多数情况）：
    LLM 看到数值后，响应中会出现该数值本身或其占位符。
    包括一切不涉及四则运算的场景：

    · 翻译：数值原样出现在译文中
      "翻译：血糖6.8" → LLM说"blood sugar 6.8" ← 6.8原样出现
    · 数值判断（是否正常/是否偏高）：
      "血糖6.8正常吗" → LLM说"6.8偏高" ← 6.8原样出现
      注意：mLDP扰动后6.8变成6.9，LLM仍能给出有效判断（偏差有界）
    · 数值规划/建议：
      "50万如何理财" → LLM说"您的50万可以..." ← 数值原样引用
    · 背景信息：年龄、年份、日期等作为上下文
      "张伟58岁，给出养生建议" → LLM说"58岁的您应该..." ← 原样出现

  "symbolic"（严格限定，仅四则运算场景）：
    LLM 对数值执行加减乘除，响应中出现的是计算结果，原值消失。
    占位符替换后 LLM 输出含占位符的公式，系统代入原值求解。

    · 乘除法计算：税后/利息/折扣/汇率/总价
      "月薪12000税后多少" → LLM说"9600" ← 12000消失了，出现的是结果
      "单价150×3件" → LLM说"450" ← 150和3都消失了
    · 加减法计算：差值/剩余
      "我58岁，还有几年退休（60岁）" → LLM说"还有2年" ← 58被减法运算掉

  再次强调：数值判断（是否正常）不是symbolic，是perturb。
  血糖判断中6.8会原样出现在响应里，走perturb+mLDP扰动即可。

非敏感类：label 填 "T"，无需 proc 字段

【任务二】分析 t2 实体之间的数学依赖关系（DAG edges）：

  只为真实存在数学关系的实体对/实体组建边，没有关系则 edges 为空列表。
  建边前必须用实际数值验证关系成立，验证不通过则不建边。

  每条边统一使用 parent_tokens（列表）表示父节点，支持单父和多父两种形式：

  ── 单父节点边（parent_tokens 长度为 1）──────────────────────────

  "ratio"   : child = parent × param
              param 计算方式：param = child ÷ parent（禁止填 1.0 占位）
              验证：|parent × param - child| / |child| ≤ 0.01
              例："单价150，数量3，总价450"
                → {parent_tokens:["150"], child_token:"450", rel_type:"ratio", agg_op:null, param:3.0}  ✓
                → {parent_tokens:["3"],   child_token:"450", rel_type:"ratio", agg_op:null, param:150.0} ✓
                × {parent_tokens:["3"],   child_token:"150", rel_type:"ratio", agg_op:null, param:1.0}   ← 关系不存在，不建边

  "diff"    : child = parent + param
              param 计算方式：param = child - parent（可为负数）
              验证：|parent + param - child| / |child| ≤ 0.01
              例："基础工资5000，绩效2000，总薪资7000"
                → {parent_tokens:["5000"], child_token:"7000", rel_type:"diff", agg_op:null, param:2000.0} ✓

  "percent" : child = parent × param ÷ 100
              param 计算方式：param = child ÷ parent × 100
              验证：|parent × param / 100 - child| / |child| ≤ 0.01
              例："总价200，折扣后150"
                → {parent_tokens:["200"], child_token:"150", rel_type:"percent", agg_op:null, param:75.0} ✓

  "ratio" 和 "diff" 的选择原则：
    · 两数之间存在乘除关系 → ratio
    · 两数之间存在加减关系 → diff
    · 不确定时：若 child ≈ parent × N（N为整数）优先选 ratio

  "copy"    : child = parent（完全相同数值在不同位置重复出现）
              param 固定 1.0，parent_tokens 长度为 1

  ── 多父节点聚合边（parent_tokens 长度 ≥ 2）────────────────────

  "agg"     : child 由多个父节点通过 agg_op 聚合得出
              agg_op 可选：
                "sum"  → child = p1 + p2 + ... + pN
                "avg"  → child = (p1 + p2 + ... + pN) ÷ N
                "max"  → child = max(p1, p2, ..., pN)
                "min"  → child = min(p1, p2, ..., pN)
              param 固定 1.0
              验证：用 agg_op 计算结果与 child 误差 ≤ 0.01

              例："英语97，语文89，总分186，平均分93"
                → {parent_tokens:["97","89"], child_token:"186", rel_type:"agg", agg_op:"sum", param:1.0}
                   验证：97+89=186 ✓
                → {parent_tokens:["97","89"], child_token:"93",  rel_type:"agg", agg_op:"avg", param:1.0}
                   验证：(97+89)/2=93 ✓
                × 不要再建 97→186 diff、89→186 diff 等单父边，agg 边已完整表达

              判断何时用 agg：
                · 多个数值加在一起得到另一个数值 → agg + sum
                · 多个数值取平均得到另一个数值   → agg + avg
                · 一个数值能由多个数值聚合得出时，优先用 agg 而不是拆成多条单父边

  ── 建边验证步骤（每条边都必须执行）───────────────────────────

  1. 确认关系类型和 agg_op（如有）
  2. 用实际数值计算预期结果
  3. 误差超过 1% → 丢弃，不建边
  4. 验证通过 → 写入 edges

  ── 常见错误示例（禁止出现）────────────────────────────────────

  × {parent_tokens:["3"], child_token:"150", rel_type:"ratio", param:1.0}
    ← 3和150无ratio关系，且param未计算
  × {parent_tokens:["3"], child_token:"186", rel_type:"diff", param:1.0}
    ← 3+1≠186，关系不成立
  × 对所有数值两两建边
    ← 必须验证关系成立才建边
  × 既建 agg(sum) 边又建多条 diff 单父边描述同一关系
    ← agg 边已完整表达，不要重复

【任务三】prompt 级模式（mode）：
  "symbolic" : prompt 中存在至少一个 proc="symbolic" 的实体
  "standard" : 所有 t2 实体均为 proc="perturb"，或无 t2 实体

输出格式（严格遵守，只输出JSON）：
{
    "mode": "standard" 或 "symbolic",
    "entities": [
        {"token": "实体文本", "label": "细粒度类型"},
        {"token": "12000",   "label": "SALARY",  "proc": "symbolic"},
        {"token": "6.8",     "label": "BLOOD_SUGAR", "proc": "perturb"}
    ],
    "edges": [
        {
            "parent_tokens": ["父节点文本"],
            "child_token":   "子节点文本",
            "rel_type":      "ratio|diff|percent|copy|agg",
            "agg_op":        null,
            "param":         3.0
        },
        {
            "parent_tokens": ["97", "89"],
            "child_token":   "186",
            "rel_type":      "agg",
            "agg_op":        "sum",
            "param":         1.0
        }
    ]
}

注意：
  · t1 实体不需要 proc 字段，t2 实体必须有 proc 字段
  · 所有边统一使用 parent_tokens（列表），不再使用 parent_token（字符串）
  · 非 agg 边的 agg_op 填 null
  · 只输出 JSON，不输出任何解释文字
"""

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
                {"role": "system", "content": self.system_prompt},
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