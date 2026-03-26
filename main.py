"""
main.py

批量读取文本文件，对每行内容进行隐私脱敏，输出结果到对应文件。

使用方式：
    # 处理单个文件
    python main.py input.txt

    # 处理多个文件
    python main.py input1.txt input2.txt input3.txt

    # 处理整个目录下所有 .txt 文件
    python main.py --dir ./data

    # 指定输出目录（默认在原文件名后加 _sanitized）
    python main.py --dir ./data --output_dir ./results

    # 指定隐私预算 epsilon（默认 1.0）
    python main.py --dir ./data --epsilon 0.5

输出格式（每条记录）：
    [原始] <原文>
    [脱敏] <脱敏结果>
    ----------------------------------------
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

from sanitizer_module import Sanitizer


# ---------------------------------------------------------------------------
# 核心处理
# ---------------------------------------------------------------------------

def process_line(sanitizer: Sanitizer, line: str) -> Tuple[str, str, dict]:
    """
    对单行文本执行脱敏。
    返回 (原文, 脱敏结果, session_info)
    """
    line = line.strip()
    if not line:
        return line, line, {}
    try:
        sanitized, session_info = sanitizer.sanitizer(line)
        return line, sanitized, session_info
    except Exception as e:
        print(f"    [警告] 处理失败: {e}")
        return line, f"[ERROR: {e}]", {}


def process_file(
    sanitizer:  Sanitizer,
    input_path: Path,
    output_path: Path,
    verbose: bool = True,
) -> dict:
    """
    处理单个文件，结果写入 output_path。
    返回统计信息。
    """
    stats = {"total": 0, "success": 0, "skipped": 0, "failed": 0}

    # 读取所有行
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  [错误] 无法读取 {input_path}: {e}")
        stats["failed"] += 1
        return stats

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []   # [(orig, sanitized, session_info), ...]

    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        stats["total"] += 1

        if not stripped:
            results.append(("", "", {}))
            stats["skipped"] += 1
            continue

        if verbose:
            print(f"  [{idx}/{len(lines)}] {stripped[:60]}{'...' if len(stripped) > 60 else ''}")

        orig, sanitized, session_info = process_line(sanitizer, stripped)

        if sanitized.startswith("[ERROR"):
            stats["failed"] += 1
        else:
            stats["success"] += 1

        results.append((orig, sanitized, session_info))
        if verbose:
            print(f"           → {sanitized[:60]}{'...' if len(sanitized) > 60 else ''}")

    # 写入结果文件（可读文本格式）
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# epsilon : {sanitizer.total_epsilon}\n")
        f.write("=" * 60 + "\n\n")

        for orig, sanitized, _ in results:
            if not orig:
                f.write("\n")
                continue

            f.write(f"{sanitized}\n")


    # 同时写一份 JSON 格式（含 session_info，可用于反脱敏）
    json_path = output_path.with_suffix(".json")
    json_records = []
    for orig, sanitized, session_info in results:
        if not orig:
            continue
        # session_info 中的 tuple key 需转为字符串才能 JSON 序列化
        serializable_info = _make_serializable(session_info)
        json_records.append({
            "original":  orig,
            "sanitized": sanitized,
            "session":   serializable_info,
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, ensure_ascii=False, indent=2)

    return stats


def _make_serializable(obj):
    """递归将不可 JSON 序列化的类型转换（tuple key → str）。"""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# 文件收集
# ---------------------------------------------------------------------------

def collect_input_files(
    file_args: List[str],
    dir_arg:   str | None,
    extensions: List[str],
) -> List[Path]:
    """收集所有待处理的输入文件路径。"""
    paths: List[Path] = []

    # 直接指定的文件
    for f in file_args:
        p = Path(f)
        if not p.exists():
            print(f"[警告] 文件不存在，跳过: {f}")
        elif not p.is_file():
            print(f"[警告] 不是文件，跳过: {f}")
        else:
            paths.append(p)

    # 目录模式
    if dir_arg:
        d = Path(dir_arg)
        if not d.is_dir():
            print(f"[错误] 目录不存在: {dir_arg}")
        else:
            for ext in extensions:
                paths.extend(sorted(d.glob(f"*{ext}")))

    return paths


def build_output_path(
    input_path:  Path,
    output_dir:  str | None,
    suffix:      str = "_sanitized",
) -> Path:
    """根据输入路径生成输出路径。"""
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = input_path.parent / "sanitized_output"

    stem = input_path.stem + suffix
    return out_dir / (stem + ".txt")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="批量文本隐私脱敏工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files", nargs="*",
        help="直接指定一个或多个输入文件"
    )
    parser.add_argument(
        "--dir", "-d", default=None,
        help="批量处理目录下所有文本文件"
    )
    parser.add_argument(
        "--ext", nargs="+", default=[".txt"],
        help="目录模式下处理的文件扩展名（默认 .txt）"
    )
    parser.add_argument(
        "--output_dir", "-o", default=None,
        help="输出目录（默认在输入文件同级建 sanitized_output/）"
    )
    parser.add_argument(
        "--epsilon", "-e", type=float, default=1.0,
        help="隐私预算 epsilon（默认 1.0，越小隐私保护越强）"
    )
    parser.add_argument(
        "--key", default=None,
        help="FPE 加密密钥（十六进制字符串，不指定则随机生成）"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="静默模式，不打印每行处理进度"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 收集文件
    input_files = collect_input_files(args.files, args.dir, args.ext)
    if not input_files:
        print("未找到任何输入文件。请通过位置参数指定文件，或使用 --dir 指定目录。")
        print("示例：python main.py input.txt")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  隐私脱敏批处理工具")
    print(f"  待处理文件数: {len(input_files)}")
    print(f"  epsilon     : {args.epsilon}")
    print(f"{'='*60}\n")

    # 初始化 Sanitizer（所有文件共用同一密钥）
    sanitizer = Sanitizer(epsilon=args.epsilon, key=args.key)
    print(f"[密钥] key={sanitizer.fpe.key}  tweak={sanitizer.fpe.tweak}\n")

    total_stats = {"total": 0, "success": 0, "skipped": 0, "failed": 0}

    for i, input_path in enumerate(input_files, 1):
        output_path = build_output_path(input_path, args.output_dir)
        print(f"[{i}/{len(input_files)}] 处理: {input_path}")
        print(f"         输出: {output_path}")

        stats = process_file(
            sanitizer=sanitizer,
            input_path=input_path,
            output_path=output_path,
            verbose=not args.quiet,
        )

        for k in total_stats:
            total_stats[k] += stats[k]

        print(f"         完成: 成功={stats['success']} 跳过={stats['skipped']} 失败={stats['failed']}\n")

    # 汇总
    print(f"{'='*60}")
    print(f"  全部完成")
    print(f"  总行数: {total_stats['total']}")
    print(f"  成功  : {total_stats['success']}")
    print(f"  跳过  : {total_stats['skipped']}（空行）")
    print(f"  失败  : {total_stats['failed']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()