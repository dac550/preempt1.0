"""
调用 Sanitizer 对 data.txt 中的文本进行脱敏处理
"""

import json
import os
from typing import List, Dict

# 导入 sanitizer 模块
from sanitizer_module import Sanitizer


def load_texts_from_file(file_path: str) -> List[str]:
    """
    从文件中读取文本，每行作为一个独立的待处理文本

    Args:
        file_path: 文件路径

    Returns:
        文本列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        raise ValueError(f"文件为空: {file_path}")

    return lines


def save_results(
        original_texts: List[str],
        sanitized_texts: List[str],
        session_infos: List[Dict],
        output_dir: str = "./output"
) -> None:
    """
    保存脱敏结果到文件

    Args:
        original_texts: 原始文本列表
        sanitized_texts: 脱敏文本列表
        session_infos: 会话信息列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存脱敏后的文本
    with open(f"{output_dir}/sanitized_output.txt", 'w', encoding='utf-8') as f:
        for i, text in enumerate(sanitized_texts):
            f.write(f"[{i + 1}] {text}\n")
            f.write("-" * 50 + "\n")

    # 保存原始文本与脱敏文本对照
    with open(f"{output_dir}/comparison.txt", 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("原始文本 vs 脱敏文本 对照表\n")
        f.write("=" * 80 + "\n\n")

        for i, (orig, san) in enumerate(zip(original_texts, sanitized_texts)):
            f.write(f"\n[{i + 1}] 原始文本:\n")
            f.write(f"{orig}\n\n")
            f.write(f"[{i + 1}] 脱敏文本:\n")
            f.write(f"{san}\n")
            f.write("-" * 80 + "\n")

    # 保存会话信息（注意：session_info 包含 vault，可能较大）
    # 只保存必要信息，避免敏感信息泄露
    simplified_infos = []
    for info in session_infos:
        simplified = {
            "session_tweak": info.get("session_tweak"),
            "eps_t2": info.get("eps_t2"),
            "ner_result": info.get("ner_result"),
            "dag_info": info.get("dag_info"),
            "vault_keys": list(info.get("vault", {}).keys()) if info.get("vault") else [],
        }
        simplified_infos.append(simplified)

    with open(f"{output_dir}/session_infos.json", 'w', encoding='utf-8') as f:
        json.dump(simplified_infos, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_dir}/")
    print(f"  - sanitized_output.txt : 脱敏后文本")
    print(f"  - comparison.txt       : 原始/脱敏对照")
    print(f"  - session_infos.json   : 会话信息（简化版）")


def run_sanitizer_demo(
        file_path: str = "data.txt",
        epsilon: float = 1.0,
        output_dir: str = "./output"
) -> None:
    """
    运行脱敏器主函数

    Args:
        file_path: 输入文件路径
        epsilon: 隐私预算（越小保护越强）
        output_dir: 输出目录
    """
    # 1. 读取数据
    print(f"正在读取文件: {file_path}")
    texts = load_texts_from_file(file_path)
    print(f"共读取 {len(texts)} 条文本\n")

    # 2. 初始化 Sanitizer
    print("初始化 Sanitizer...")
    print(f"隐私预算 epsilon = {epsilon}\n")
    sanitizer = Sanitizer(epsilon=epsilon)

    # 获取并保存密钥信息（重要！请离线保存）
    key_info = sanitizer.get_key_info()
    print("=" * 60)
    print("【重要】长期密钥信息（请离线保存，勿上传或提交到代码仓库）:")
    print(f"  key   : {key_info['key']}")
    print(f"  tweak : {key_info['tweak']}")
    print("=" * 60)

    # 保存密钥到单独文件（请确保此文件不被提交到版本控制）
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/key_info.txt", 'w') as f:
        f.write(f"# 长期密钥文件 - 请离线保存，不要提交到代码仓库\n")
        f.write(f"key={key_info['key']}\n")
        f.write(f"tweak={key_info['tweak']}\n")
        f.write(f"note={key_info['note']}\n")
    print(f"密钥已保存到: {output_dir}/key_info.txt\n")

    # 3. 逐条脱敏
    sanitized_texts = []
    session_infos = []

    for idx, text in enumerate(texts):
        print(f"\n{'=' * 60}")
        print(f"处理第 {idx + 1}/{len(texts)} 条:")
        print(f"原始: {text}")

        try:
            san_text, session_info = sanitizer.sanitizer(text)
            print(f"脱敏: {san_text}")

            sanitized_texts.append(san_text)
            session_infos.append(session_info)

            # 打印 vault 中的映射信息
            vault = session_info.get("vault", {})
            for enc, rec in vault.items():
                if rec["type"] == "t1":
                    print(f"  t1映射: {enc!r} → {rec['original']!r}")
                elif rec["type"] == "t2_perturb":
                    print(f"  t2映射: {rec['original']!r} → {rec['noisy_val']} (扰动)")

        except Exception as e:
            print(f"处理失败: {e}")
            sanitized_texts.append(f"[ERROR: {e}]")
            session_infos.append({})

    # 4. 保存结果
    save_results(texts, sanitized_texts, session_infos, output_dir)

    # 5. 可选：演示反脱敏（只对第一条进行）
    if sanitized_texts and session_infos[0]:
        print(f"\n{'=' * 60}")
        print("反脱敏演示（仅还原 t1 实体）:")

        restored, audit = sanitizer.desanitizer(sanitized_texts[0], session_infos[0])
        print(f"原始文本:     {texts[0]}")
        print(f"脱敏文本:     {sanitized_texts[0]}")
        print(f"反脱敏后:     {restored}")
        print(f"t1 恢复率:    {audit['recovery_rate']}")
        if audit.get('missing_t1'):
            print(f"未出现的 t1: {audit['missing_t1']}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="数据脱敏工具")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data.txt",
        help="输入文件路径（默认: data.txt）"
    )
    parser.add_argument(
        "--epsilon", "-e",
        type=float,
        default=1.0,
        help="隐私预算 epsilon，越小保护越强（默认: 1.0）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="输出目录（默认: ./output）"
    )

    args = parser.parse_args()

    run_sanitizer_demo(
        file_path=args.input,
        epsilon=args.epsilon,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()