"""
sanitizer_module.py  —  实体级三模式脱敏系统

设计原则
--------
t1 实体（姓名/手机/组织等）
    → FPE 格式保持加密，密文直接嵌入 prompt
    → 反脱敏：在 LLM 响应中 find(密文) → 替换回原文
    → 跨会话：每轮 refresh_session_tweak()，相同实体在不同会话密文不同

t2 perturb（年龄/血糖/血压等"引用型"数值）
    → mLDP 扰动后的裸数字直接嵌入 prompt，LLM 基于近似值回答
    → 【无需反脱敏】：LLM 的回答本就针对扰动值，是有效近似结果
    → Vault 只记录用于审计，不参与反脱敏流程

t2 symbolic（月薪/单价等"计算型"数值）
    → 不扰动，用占位符 [TYPE_N] 替换，要求 LLM 保留占位符输出含符号的公式
    → 反脱敏：正则找 [TYPE_N] → 替换为原值，再对数学表达式求值

反脱敏只需两步
    步骤1：t1 FPE 密文 → 原文
    步骤2：t2 symbolic [TYPE_N] → 原值 + 表达式求值

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
from api import NERAPI, is_t1, is_symbolic, is_perturb, T1_LABELS, T2_LABELS
from dag_module import T2DAGProcessor

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 反脱敏正则：只需匹配 [TYPE_N] 占位符
_PH_RE   = re.compile(r"\[([A-Z_]+)_(\d+)\]")
# 简单四则运算求值正则
_MATH_RE = re.compile(
    r'(-?\d+(?:\.\d+)?)\s*([+\-×x\*÷/])\s*(-?\d+(?:\.\d+)?)'
)##修改成复杂计算函数识别

SYMBOLIC_SYSTEM_PROMPT = """
重要提示：输入中的 [TYPE_INDEX] 格式标记（如 [SALARY_0]）是数值占位符，
代表用户的真实数值（已隐藏保护）。

请严格遵守：
1. 完整保留所有占位符，不要替换为具体数字
2. 用占位符参与运算表达式：结果 = [SALARY_0] × 0.80
3. 只输出含占位符的计算公式，不要自行代入求值
"""


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


def _get_t2_ph_type(label: str) -> str:

    return label if label in T2_LABELS else "NUM"


def _make_placeholder(ph_type: str, index: int) -> str:
    return f"[{ph_type}_{index}]"


def _eval_math_in_text(text: str) -> str:
    """
    对文本中的简单四则运算表达式求值。
    防御除以零；整数结果不带小数点。
    """
    def _compute(m: re.Match) -> str:
        try:
            a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
            if   op in ('*', '×', 'x'): r = a * b
            elif op in ('/', '÷'):
                if b == 0:
                    return m.group(0)
                r = a / b
            elif op == '+':             r = a + b
            else:                       r = a - b
            return str(int(r)) if r == int(r) else f"{r:.2f}"
        except Exception:
            return m.group(0)
    return _MATH_RE.sub(_compute, text)


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

    def store_t2_symbolic(
        self,
        ph:       str,
        original: str,
        orig_val: Union[int, float],
    ) -> None:
        self._data[ph] = {
            "type":     "t2_symbolic",
            "original": original,
            "orig_val": orig_val,
        }

    def get(self, key: str) -> Optional[Dict]:
        return self._data.get(key)

    def t1_ciphertexts(self) -> List[str]:
        return [k for k, v in self._data.items() if v["type"] == "t1"]

    def placeholders(self) -> List[str]:
        return [k for k, v in self._data.items() if v["type"] == "t2_symbolic"]

    def has_symbolic(self) -> bool:
        return any(v["type"] == "t2_symbolic" for v in self._data.values())

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
    三模式脱敏主类。

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
                         · t2 perturb → 裸扰动数字（直接可读）
                         · t2 symbolic → [TYPE_N] 占位符
        session_info   : 反脱敏所需会话信息（不含长期密钥 key）
        """
        session_tweak = self.fpe.refresh_session_tweak()
        mode, ner_result, raw_edges = self.ner(prompt)
        print(raw_edges)
        # ── 收集 t2 perturb，送 DAG 扰动 ──────────────────────
        t2_perturb_list: List[Tuple[int, str, Union[int, float]]] = []
        t2_precision:    Dict[int, int] = {}

        for i, (token, label, proc) in enumerate(ner_result):
            if is_perturb(label, proc):
                ok, val = _parse_numeric(token)
                if ok:
                    t2_perturb_list.append((i, token, val))
                    t2_precision[i] = _infer_precision(token)

        noisy_map: Dict[int, Union[int, float]] = {}
        dag_info:  Dict = {}

        if t2_perturb_list:

            processor = T2DAGProcessor(total_epsilon=self.total_epsilon)
            # DAG 内部处理整数；浮点先按精度放大，结果再缩回
            int_list = [
                (idx, token, actual_value)  # actual_value 是原值（如 92.5）
                for idx, token, actual_value in t2_perturb_list
            ]
            print(f"t2_list before process: {int_list}")
            raw_noisy, dag_info = processor.process(int_list, raw_edges)
            for i, _t, _v in t2_perturb_list:
                raw = raw_noisy.get(i)
                if raw is not None:
                    noisy_map[i] = raw  # 直接使用，不再处理精度
                else:
                    # fallback: 使用原始值或扰动值
                    noisy_map[i] = _v

        vault      = Vault()
        ph_counter = 0

        # 按 token 长度从长到短排序，避免短 token 误匹配长 token 子串
        indexed_sorted = sorted(
            enumerate(ner_result),
            key=lambda x: len(x[1][0]),
            reverse=True,
        )

        sanitized_text = prompt

        for i, (token, label, proc) in indexed_sorted:

            if is_t1(label):
                # ── t1：FPE 密文嵌入 ────────────────────────────
                enc, _meta = self.fpe.encrypt_master(token)
                vault.store_t1(enc, token)
                sanitized_text = sanitized_text.replace(token, enc, 1)

            elif is_perturb(label, proc):
                # ── t2 perturb：裸扰动数字直接嵌入 ─────────────
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

            elif is_symbolic(label, proc):
                # ── t2 symbolic：占位符嵌入，不扰动 ────────────
                _, val = _parse_numeric(token)
                ph = _make_placeholder(_get_t2_ph_type(label), ph_counter)
                ph_counter += 1
                vault.store_t2_symbolic(ph, token, val)
                sanitized_text = sanitized_text.replace(token, ph, 1)

        session_info: Dict = {
            "session_tweak": session_tweak,
            "eps_t2":        self.total_epsilon,
            "mode":          mode,
            "has_symbolic":  vault.has_symbolic(),
            "ner_result":    ner_result,
            "dag_info":      dag_info,
            "vault":         vault.to_dict(),
        }

        if vault.has_symbolic():
            session_info["symbolic_system_prompt"] = SYMBOLIC_SYSTEM_PROMPT

        return sanitized_text, session_info

    # ------------------------------------------------------------------
    # 反脱敏（两步）
    # ------------------------------------------------------------------

    def desanitizer(self, response: str, session_info: Dict) -> Tuple[str, Dict]:
        """
        从 LLM 响应中还原脱敏实体。

        步骤1：t1  — find(FPE密文) → 原文
        步骤2：t2 symbolic — 正则 [TYPE_N] → 原值 + 数学表达式求值

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
        has_sym = session_info.get("has_symbolic", False)

        found:   List[str] = []
        missing: List[str] = []
        restored = response

        # ── 步骤1：还原 t1（FPE 密文 → 原文）────────────────────
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

        # ── 步骤2：还原 t2 symbolic（[TYPE_N] → 原值）───────────
        def _replace_ph(m: re.Match) -> str:
            ph     = m.group(0)
            record = vault.get(ph)
            if record is None:
                return ph
            found.append(ph)
            return str(record["orig_val"])

        restored = _PH_RE.sub(_replace_ph, restored)

        if has_sym:
            restored = _eval_math_in_text(restored)

        # ── 审计 ─────────────────────────────────────────────────
        # 只统计 t1 和 symbolic 的还原情况；perturb 不参与反脱敏，不纳入统计
        all_sym_keys = set(vault.placeholders())
        missing_sym  = sorted(all_sym_keys - set(found))

        found_unique = list(dict.fromkeys(found))
        billable     = len(vault.t1_ciphertexts()) + len(vault.placeholders())
        audit: Dict = {
            "mode":             session_info.get("mode", "standard"),
            "has_symbolic":     has_sym,
            "found":            found_unique,
            "missing_t1":       missing,
            "missing_symbolic": missing_sym,
            "recovery_rate":    round(
                len(found_unique) / max(billable, 1), 4
            ),
        }

        if missing:
            print(f"[审计] {len(missing)} 个 t1 实体未出现: {missing}")
        if missing_sym:
            print(f"[审计] {len(missing_sym)} 个 symbolic 占位符未出现: {missing_sym}")

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
        ("浮点t2",
         "患者张伟，血糖6.8，血压130，请问这些指标正常吗？"),
        ("symbolic",
         "My monthly salary is 12000, tax rate 20%, what is my after-tax income?"),
        ("混合",
         "张伟月薪12000，税后收入是多少？另外他血糖6.8，正常吗？"),
        ("1","我叫小明，我的英语成绩是97，我的语文成绩是88，我的平均成绩是92.5")
    ]

    for tag, text in cases:
        print(f"\n{'='*65}  [{tag}]")
        print(f"原始: {text}")
        san, info = sanitizer.sanitizer(text)
        print(f"脱敏: {san}")
        print(f"mode={info['mode']}  symbolic={info['has_symbolic']}")

        assert "fpe_key" not in info
        assert "session_tweak" in info

        for k, rec in info["vault"].items():
            t = rec["type"]
            if t == "t1":
                print(f"  {k!r} → [t1] orig={rec['original']!r}")
            elif t == "t2_perturb":
                print(f"  {k!r} → [t2_perturb] orig={rec['orig_val']}  noisy={rec['noisy_val']}")
            else:
                print(f"  {k!r} → [t2_symbolic] orig={rec['orig_val']}")

        if info["has_symbolic"]:
            sym_phs = [k for k, r in info["vault"].items() if r["type"] == "t2_symbolic"]
            ph = sym_phs[0]
            fake = f"税后收入 = {ph} × (1 - 0.20) = {ph} × 0.80"
            print(f"模拟LLM: {fake}")
            restored, audit = sanitizer.desanitizer(fake, info)
            print(f"反脱敏: {restored}")
            print(f"恢复率: {audit['recovery_rate']}  missing_sym={audit['missing_symbolic']}")
            assert 0 <= audit["recovery_rate"] <= 1
            assert audit["missing_symbolic"] == []
        else:
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
