"""
sanitizer_module.py  —  实体级双模式脱敏系统

设计原则
--------
t1 实体（姓名/手机/组织等）
    → FPE 格式保持加密，密文直接嵌入 prompt
    → 反脱敏：在 LLM 响应中 find(密文) → 替换回原文
    → 跨会话：每轮 refresh_session_tweak()，相同实体在不同会话密文不同

t2 实体（所有数值类型）
    → mLDP 扰动后的裸数字直接嵌入 prompt，LLM 基于近似值回答
    → 【无需反脱敏】：LLM 的回答本就针对扰动值，是有效近似结果
    → DAG 保证关联数值（如单价/数量/总价）扰动后仍满足数学关系
    → Vault 只记录用于审计，不参与反脱敏流程

反脱敏只需一步
    步骤1：t1 FPE 密文 → 原文

安全说明
--------
长期密钥（key）：
    · 首次创建 Sanitizer 时随机生成，通过 get_key_info() 取出后离线保存
    · 绝不写入 session_info、日志或任何持久化文件
    · 反脱敏时由调用方传入，只在进程内存中存活

会话密钥（tweak）：
    · 每轮对话自动刷新，保证同一实体跨会话密文不同（抗关联攻击）
    · 存入 session_info["session_tweak"]，泄露只影响本轮会话
"""

import re
from typing import Dict, List, Optional, Tuple, Union

from ff3_module import FPEManager
from api import NERAPI, is_t1, is_t2
from dag_module import T2DAGProcessor

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 反脱敏正则：匹配 t1 FPE 密文（无需 symbolic 占位符正则）
_PH_RE   = re.compile(r"\[([A-Z_]+)_(\d+)\]")  # 保留备用，当前不使用


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _parse_numeric(token: str) -> Tuple[bool, Union[int, float]]:
    """
    尝试把 token 解析为数值。
    返回 (success, value)：
        (True,  int)   → 整数字符串，如 "130"
        (True,  float) → 浮点字符串，如 "6.8"
        (False, 0)     → 无法解析，如 "130/80"
    """
    try:
        return True, int(token)
    except ValueError:
        pass
    try:
        return True, float(token)
    except ValueError:
        return False, 0


def _infer_precision(token: str) -> int:
    """从 token 字符串推断小数精度，用于 mLDP 浮点扰动还原。"""
    return len(token.split('.')[1]) if '.' in token else 0




# ---------------------------------------------------------------------------
# Vault  —  实体映射表
# ---------------------------------------------------------------------------

class Vault:
    """
    t1         : {fpe_ciphertext → {type, original}}
    t2_perturb : {noisy_str → {type, original, orig_val, noisy_val}}
                 key 是裸数字字符串，仅用于审计，不参与反脱敏
    t2_symbolic: {[TYPE_N] → {type, original, orig_val}}
                 key 是占位符字符串，用于反脱敏步骤2
    """

    def __init__(self):
        self._data: Dict[str, Dict] = {}

    def store_t1(self, enc: str, original: str) -> None:
        self._data[enc] = {"type": "t1", "original": original}

    def store_t2_perturb(
        self,
        noisy_str: str,
        original:  str,
        orig_val:  Union[int, float],
        noisy_val: Union[int, float],
    ) -> None:
        """key 为裸数字字符串，仅审计用，不作为反脱敏锚点。"""
        self._data[noisy_str] = {
            "type":      "t2_perturb",
            "original":  original,
            "orig_val":  orig_val,
            "noisy_val": noisy_val,
        }

    def get(self, key: str) -> Optional[Dict]:
        return self._data.get(key)

    def t1_ciphertexts(self) -> List[str]:
        return [k for k, v in self._data.items() if v["type"] == "t1"]

    def to_dict(self) -> Dict:
        return dict(self._data)

    @classmethod
    def from_dict(cls, d: Dict) -> "Vault":
        v = cls()
        v._data = d
        return v


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------

class Sanitizer:
    """
    双模式脱敏主类。

    参数
    ----
    epsilon : 隐私预算（越小保护越强，t2 perturb 扰动越大）
    key     : FPE 长期密钥（十六进制字符串）
              首次使用不传，自动生成后通过 get_key_info() 取出离线保存
    """

    def __init__(self, epsilon: float, key: Optional[str] = None):
        self.total_epsilon = epsilon
        self.fpe           = FPEManager(key=key)
        self.ner           = NERAPI()

    # ------------------------------------------------------------------
    # 脱敏
    # ------------------------------------------------------------------

    def sanitizer(self, prompt: str) -> Tuple[str, Dict]:
        """
        对 prompt 执行脱敏。

        返回
        ----
        sanitized_text : 发给 LLM 的脱敏文本
                         · t1 实体    → FPE 密文
                         · t2 实体    → 裸扰动数字（DAG 保证关联数值一致性）
        session_info   : 反脱敏所需会话信息（不含长期密钥 key）
        """
        session_tweak = self.fpe.refresh_session_tweak()
        ner_result, raw_edges = self.ner(prompt)
        print(raw_edges)
        # ── 收集 t2，送 DAG 扰动 ──────────────────────────────────
        t2_perturb_list: List[Tuple[int, str, Union[int, float]]] = []

        #提取t2实体
        for i, (token, label) in enumerate(ner_result):
            if is_t2(label):
                ok, val = _parse_numeric(token)
                if ok:
                    t2_perturb_list.append((i, token, val))


        noisy_map: Dict[int, Union[int, float]] = {}
        dag_info:  Dict = {}

        if t2_perturb_list:
            processor = T2DAGProcessor(total_epsilon=self.total_epsilon)
            int_list = [
                (idx, token, actual_value)
                for idx, token, actual_value in t2_perturb_list
            ]
            print(f"t2_list before process: {int_list}")
            raw_noisy, dag_info = processor.process(int_list, raw_edges)
            print(f"raw_edges:{raw_edges}, dag_info:{dag_info}")
            for i, _t, _v in t2_perturb_list:
                raw = raw_noisy.get(i)
                if raw is not None:
                    noisy_map[i] = raw
                else:
                    noisy_map[i] = _v

        vault = Vault()
        # 按 token 长度从长到短排序，避免短 token 误匹配长 token 子串
        indexed_sorted = sorted(
            enumerate(ner_result),
            key=lambda x: len(x[1][0]),
            reverse=True,
        )

        sanitized_text = prompt

        for i, (token, label) in indexed_sorted:

            if is_t1(label):
                # ── t1：FPE 密文嵌入 ────────────────────────────
                enc, _meta = self.fpe.encrypt_master(token)
                vault.store_t1(enc, token)
                sanitized_text = sanitized_text.replace(token, enc, 1)

            elif is_t2(label):
                # ── t2：裸扰动数字直接嵌入 ──────────────────────
                ok, val = _parse_numeric(token)
                if not ok:
                    # 无法解析为数值（如 "130/80"）→ 降级为 t1 FPE
                    enc, _meta = self.fpe.encrypt_master(token)
                    vault.store_t1(enc, token)
                    sanitized_text = sanitized_text.replace(token, enc, 1)
                    continue

                noisy     = noisy_map.get(i, val)
                noisy_str = str(noisy) if isinstance(noisy, int) else str(noisy)
                vault.store_t2_perturb(noisy_str, token, val, noisy)
                sanitized_text = sanitized_text.replace(token, noisy_str, 1)

        session_info: Dict = {
            "session_tweak": session_tweak,
            "eps_t2":        self.total_epsilon,
            "ner_result":    ner_result,
            "dag_info":      dag_info,
            "vault":         vault.to_dict(),
        }

        return sanitized_text, session_info

    # ------------------------------------------------------------------
    # 反脱敏（两步）
    # ------------------------------------------------------------------

    def desanitizer(self, response: str, session_info: Dict) -> Tuple[str, Dict]:
        """
        从 LLM 响应中还原 t1 实体。
        t2 perturb 不参与反脱敏：LLM 基于扰动值给出的回答本就是有效近似结果。

        参数
        ----
        response     : LLM 返回的文本
        session_info : sanitizer() 返回的会话信息

        返回
        ----
        restored : 还原后的文本
        audit    : 审计信息
        """
        vault   = Vault.from_dict(session_info.get("vault", {}))
        found:   List[str] = []
        missing: List[str] = []
        restored = response

        # ── 还原 t1（FPE 密文 → 原文）────────────────────────────
        for enc in sorted(vault.t1_ciphertexts(), key=len, reverse=True):
            record = vault.get(enc)
            idx    = restored.find(enc)
            if idx != -1:
                restored = (
                    restored[:idx]
                    + record["original"]
                    + restored[idx + len(enc):]
                )
                found.append(enc)
            else:
                missing.append(enc)

        # ── 审计 ─────────────────────────────────────────────────
        billable = len(vault.t1_ciphertexts())
        audit: Dict = {
            "found":         list(dict.fromkeys(found)),
            "missing_t1":    missing,
            "recovery_rate": round(len(found) / max(billable, 1), 4),
        }

        if missing:
            print(f"[审计] {len(missing)} 个 t1 实体未出现: {missing}")

        return restored, audit

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def get_key_info(self) -> Dict:
        """首次使用后调用此方法，取出 key 离线保存。"""
        return {
            "key":   self.fpe.key,
            "tweak": self.fpe.tweak,
            "note":  "key 为长期密钥，请离线保存，勿写入 session_info 或日志。",
        }


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sanitizer = Sanitizer(epsilon=1.0)
    print(f"[密钥] key={sanitizer.get_key_info()['key']}  （请离线保存）\n")

    cases = [
        ("整数t2",
         "In 2024, customer Zhang Wei purchased 3 items, "
         "each priced at 150 yuan, total 450 yuan, contact: 13800138000."),
        ("成绩",
         "我叫小明，我的英语成绩是97，我的语文成绩是88，我的平均成绩是92.5"),
        ("多科目",
         "小明期中考试：数学97分，语文88分，英语91分，平均分92.0分"),
    ]

    for tag, text in cases:
        print(f"\n{'='*65}  [{tag}]")
        print(f"原始: {text}")
        san, info = sanitizer.sanitizer(text)
        print(f"脱敏: {san}")

        assert "fpe_key" not in info
        assert "session_tweak" in info

        for k, rec in info["vault"].items():
            t = rec["type"]
            if t == "t1":
                print(f"  {k!r} → [t1] orig={rec['original']!r}")
            elif t == "t2_perturb":
                print(f"  {k!r} → [t2_perturb] orig={rec['orig_val']}  noisy={rec['noisy_val']}")

        restored, audit = sanitizer.desanitizer(san, info)
        print(f"反脱敏(t1还原): {restored}")
        print(f"恢复率: {audit['recovery_rate']}")

    # 跨会话验证
    print(f"\n{'='*65}  [跨会话验证]")
    s1, i1 = sanitizer.sanitizer("张伟购买了商品")
    s2, i2 = sanitizer.sanitizer("张伟购买了商品")
    assert i1["session_tweak"] != i2["session_tweak"]
    assert s1 != s2
    print(f"第1轮: {s1}")
    print(f"第2轮: {s2}")
    print("tweak 不同: ✓  密文不同: ✓")