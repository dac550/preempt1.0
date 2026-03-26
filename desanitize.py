"""
desanitize.py  —  LLM 响应反脱敏工具

修复说明
--------
Bug-Fix 1 适配（对应 sanitizer_module.py Bug-Fix 1）：
  原实现：从 session_info["fpe_key"]["key"] 读取长期密钥
  修复后：session_info 中不再含长期 key，改为通过 --key 命令行参数传入。

  安全设计：
    · 长期 key 由使用者离线保存（.env / 密钥管理系统），不进入任何 JSON 文件。
    · 本工具通过 --key 参数接收，只在进程内存中存活。
    · session_info 只含 session_tweak，用于本次反脱敏，不危及历史会话。

使用方式
--------
  # 指定 key（必须与脱敏时使用的 key 一致）
  python desanitize.py --json results/input_sanitized.json --key <HEX_KEY>

  # 直接指定 LLM 响应文本
  python desanitize.py --json results/input_sanitized.json \\
                       --key <HEX_KEY> --response "LLM 回答..."

  # 自检模式：对 JSON 里每条记录的 sanitized 字段执行反脱敏，验证还原效果
  python desanitize.py --json results/input_sanitized.json \\
                       --key <HEX_KEY> --selfcheck
"""

import argparse
import json
from pathlib import Path
from ff3_module import FPEManager
from sanitizer_module import Sanitizer


# ---------------------------------------------------------------------------
# session_info 反序列化
# ---------------------------------------------------------------------------

def _restore_session(raw: dict) -> dict:
    """
    JSON 反序列化后还原两处类型：
      1. t1_records 的 key（JSON str → int）
      2. t2_records / dag_info.noisy_map 的 key（同上）
    """
    session = dict(raw)

    session["ner_result"] = [tuple(e) for e in session.get("ner_result", [])]

    session["token_positions"] = [
        tuple(p) for p in session.get("token_positions", [])
    ]

    t1 = {}
    for k, v in session.get("t1_records", {}).items():
        orig, enc_meta, enc = v
        t1[int(k)] = (orig, enc_meta, enc)
    session["t1_records"] = t1

    session["t2_records"] = {
        int(k): v for k, v in session.get("t2_records", {}).items()
    }

    dag = session.get("dag_info", {})
    if "noisy_map" in dag:
        dag["noisy_map"] = {int(k): v for k, v in dag["noisy_map"].items()}
    session["dag_info"] = dag

    return session


# ---------------------------------------------------------------------------
# 核心反脱敏
# ---------------------------------------------------------------------------

def desanitize_response(response: str, session: dict, key: str) -> str:
    """
    用 session_info + 长期 key 对 LLM 响应执行反脱敏。

    参数
    ----
    response : LLM 返回的文本
    session  : sanitizer() 生成的 session_info（含 session_tweak，不含 key）
    key      : FPE 长期密钥（十六进制字符串，由调用方离线保管传入）

    安全说明
    --------
    · key 只在本函数调用期间存活于进程内存，不写入任何文件或日志。
    · tweak 从 session["session_tweak"] 读取，与 key 一起重建 FPEManager。
    """
    tweak = session.get("session_tweak")
    if tweak is None:
        # 兼容旧格式：如果还是存的 fpe_key 字典（旧 session_info）
        old_fpe_key = session.get("fpe_key", {})
        tweak = old_fpe_key.get("tweak")
        if tweak is None:
            raise ValueError(
                "session_info 中找不到 session_tweak，"
                "请确认使用的是修复后的 sanitizer_module.py 生成的 session_info。"
            )

    # 用 key + tweak 重建 FPEManager，保证加解密完全一致
    fpe = FPEManager(key=key, tweak=tweak)

    # 借用 Sanitizer 的 desanitizer 方法
    sanitizer = Sanitizer.__new__(Sanitizer)
    sanitizer.fpe           = fpe
    sanitizer.total_epsilon = session.get("eps_t2", 1.0)
    sanitizer.ner           = None   # 反脱敏不需要 NER

    restored, _audit = sanitizer.desanitizer(response, session)
    return restored


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM 响应反脱敏工具")
    parser.add_argument(
        "--json", "-j", required=True,
        help="sanitizer 生成的 *_sanitized.json 文件路径",
    )
    parser.add_argument(
        "--key", "-k", required=True,
        help="FPE 长期密钥（十六进制字符串，脱敏时由 sanitizer.get_key_info() 取得）",
    )
    parser.add_argument(
        "--response", "-r", default=None,
        help="待反脱敏的 LLM 响应文本（不指定则进入交互模式）",
    )
    parser.add_argument(
        "--index", "-i", type=int, default=0,
        help="使用 JSON 中第几条记录的 session_info（默认第 0 条）",
    )
    parser.add_argument(
        "--selfcheck", action="store_true",
        help="自检模式：对每条记录的 sanitized 字段执行反脱敏，验证是否能还原 original",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        print(f"[错误] 找不到文件: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        print("[错误] JSON 文件为空")
        return

    key = args.key.strip()

    # ---- 自检模式 ----
    if args.selfcheck:
        print(f"自检模式：共 {len(records)} 条记录\n")
        ok_count = 0
        for idx, rec in enumerate(records):
            session  = _restore_session(rec["session"])
            try:
                restored = desanitize_response(rec["sanitized"], session, key)
                match    = restored == rec["original"]
                ok_count += int(match)
                mark = "✓" if match else "✗"
                print(f"[{idx}] {mark}")
                if not match:
                    print(f"  原始: {rec['original']}")
                    print(f"  还原: {restored}")
            except Exception as e:
                print(f"[{idx}] ✗  异常: {e}")
        print(f"\n结果: {ok_count}/{len(records)} 条完全一致")
        return

    # ---- 单条反脱敏 ----
    if args.index >= len(records):
        print(f"[错误] index={args.index} 超出范围（共 {len(records)} 条）")
        return

    rec     = records[args.index]
    session = _restore_session(rec["session"])

    print(f"原始文本 : {rec['original']}")
    print(f"脱敏文本 : {rec['sanitized']}")
    print()

    if args.response:
        llm_response = args.response
    else:
        print("请输入 LLM 响应（直接回车则使用脱敏文本本身做自测）：")
        llm_response = input("> ").strip() or rec["sanitized"]

    restored = desanitize_response(llm_response, session, key)
    print(f"\n反脱敏结果: {restored}")


if __name__ == "__main__":
    main()