"""
api.py

单次 NER 调用完成两项任务：
  1. NER：细粒度实体识别（t1/t2 分类）
  2. DAG 边：t2 数值依赖关系（支持多父节点聚合边）

使用通义千问官方 API (DashScope)
"""

import json
import os
import re
from typing import List, Tuple, Dict

from openai import OpenAI

# 千问系统提示词
QWEN_SYSTEM_PROMPT = """你是实体识别、数值关系分析专家。对用户输入，只输出JSON，禁止输出<think>标签或任何思考过程以及任何文字描述。verify字段必须写出完整验证算式，结果为False的边必须丢弃。

【任务一】提取实体
t1类 ：PERSON/ORG/LOCATION/PHONE/EMAIL/ID/MEDICAL_ID/IP/URL/ENTITY
t2类 （数值）：AGE/SALARY/AMOUNT/COUNT/WEIGHT/BLOOD_SUGAR/BLOOD_PRESSURE/MEDICAL_VAL/YEAR/DATE_NUM/NUM/PERCENT
【任务二】识别数值关系

Step1 语义判断（必须最先执行，结果写入mode字段）：
- 含 平均/均值/average → mode=average
- 含 总计/合计/总共/sum/total 
    检查已提取t2类token，若存在A×B=C（C÷A=B且B在已提取token中）→ mode=simple(强制执行判断)
    否则 → mode=sum
- 含 增长/上涨/下降/减少/grew/decreased → mode=growth
- 含 占比/占/proportion → mode=proportion
- 含 加权 → mode=weighted_avg
- 否则 → mode=simple

Step2a mode=simple时：
只对文本中有明确数值关系描述的token对建边，禁止穷举所有组合。
公式：
- multiply: child ÷ parent = param
- add:      child - parent = param
- percent:  child ÷ parent × 100 = param
- copy:     child = parent，param=null

Step2b mode为复合运算时：
注意：每一步的child必须是_tmp节点，原始token只能作为parent或param出现。
严格按以下伪代码生成edge_checks，条数由步骤数决定不多不少。

mode=average，输入[x1,...,xn]，结果D：
_tmp1=x1+x2 (add,param=x2,parent=x1,child=_tmp1)
_tmp(k)=_tmp(k-1)+x(k+2) (add,param=x(k+2),child=_tmp(k)) ← k=1到n-2
D=_tmp(n-1)×(1/n) (multiply,param=1/n,parent=_tmp(n-1),child=D)

mode=sum，输入[x1,...,xn]，结果D：
_tmp1=x1+x2 (add,param=x2,parent=x1,child=_tmp1)
_tmp(k)=_tmp(k-1)+x(k+2) (add,param=x(k+2),child=_tmp(k)) ← k=1到n-2
D=_tmp(n-1) (copy,parent=_tmp(n-1),child=D)

mode=growth，输入基数B，比例p，结果C：
_tmp1=B×p/100 (percent,parent=B,param=p,child=_tmp1)
C=B+_tmp1 (add,param=_tmp1值,parent=B,child=C)

mode=proportion，输入整体B，比例p，结果A：
A=B×p/100 (percent,parent=B,param=p,child=A)

mode=weighted_avg，输入[x1..xn],[w1..wn]，结果D：
_tmp(2i-1)=xi×wi (multiply,param=wi,child=_tmp(2i-1)) ← i=1到n
_tmp(2i)=_tmp(2i-2)+_tmp(2i-1) (add,child=_tmp(2i)) ← i=2到n
D=_tmp(2n-2) (copy,child=D)

【输出格式】
{
  "entities": [{"token":"值","label":"类型"}],
  "mode_reason": "触发词或判断依据",
  "mode": "simple|average|sum|growth|proportion|weighted_avg",
  "edge_checks": [
    {
      "parent": "父节点",
      "child": "子节点(_tmp或结果token)",
      "rel_type": "multiply|add|percent|copy",
      "calc": "完整算式及计算结果",
      "param_candidate": 数值,
      "calc_ok": true/false,
      "param_legal": true/false,
      "semantic_ok": true/false,
      "pass": true/false
    }
  ],
  "edges": [
    {
      "parent_tokens": ["单父节点"],
      "child_token": "单子节点",
      "rel_type": "multiply|add|percent|copy",
      "param": 数值
    }
  ]
}"""


class NERAPI:
    def __init__(
            self,
            api_key: str = "sk-urviqzizruyjxvcogmckquhckixpsneisguxwzmdmcbltgia",
            model: str = "Pro/zai-org/GLM-4.7",  
            base_url: str = "https://api.siliconflow.cn/v1",
            temperature: float = 0.1,
            top_p: float = 0.8,
            max_tokens: int = 2048
    ):
        """
        初始化通义千问 API 客户端

        Parameters
        ----------
        api_key : str
            DashScope API Key，若不传则从环境变量 DASHSCOPE_API_KEY 读取
        model : str
            模型名称，可选：qwen-turbo, qwen-plus, qwen-max, qwen-max-longcontext
        base_url : str
            DashScope 服务的 base_url
        temperature : float
            温度参数，控制随机性 (0.0-2.0)
        top_p : float
            核采样参数 (0.0-1.0)
        max_tokens : int
            最大生成 token 数
        """
        # 获取 API Key [citation:9]
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "请设置 DASHSCOPE_API_KEY 环境变量，或通过 api_key 参数传入"
            )

        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # 创建 OpenAI 兼容客户端 [citation:3]
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def extract_entities(
            self, user_prompt: str
    ) -> Tuple[List[Tuple[str, str]], List[Dict]]:
        """
        调用通义千问 API 进行实体识别和关系抽取

        Returns
        -------
        entities : [(token, label), ...]
        edges    : [{"parent_tokens", "child_token", "rel_type", "param"}, ...]
        """
        try:
            # 调用通义千问 API [citation:9]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )

            # 获取响应内容
            content = completion.choices[0].message.content

        except Exception as e:
            print(f"API 调用失败: {e}")
            return [], []

        # 清理可能的 think 标签
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # 解析 JSON 响应
        entities = []
        edges = []

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))

                # 解析 entities
                for e in result.get("entities", []):
                    token = e.get("token", "")
                    label = e.get("label", "T")
                    if token:
                        entities.append((token, label))

                # 解析 edges
                raw_edge_list = result.get("edges", [])
                for ed in raw_edge_list:
                    edges.append({
                        "parent_tokens": ed.get("parent_tokens", []),
                        "child_token": ed.get("child_token", ""),
                        "rel_type": ed.get("rel_type", ""),
                        "param": ed.get("param", 1.0),
                    })

            except json.JSONDecodeError as e:
                print(f"JSON 解析失败: {e}")
                print(f"原始内容: {content}")

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