"""
bleu_test.py

对两份翻译结果计算 BLEU 分数，评估脱敏对翻译质量的影响。

输入
----
  --ref : 原句翻译结果（reference），每行一条译文
  --hyp : 脱敏句翻译结果（hypothesis），每行一条译文
  两个文件行数必须相同，第 i 行对应同一条句子。

输出
----
  控制台打印汇总 + 逐句得分
  保存 JSON 和 TXT 报告到 --out 目录

安装依赖
--------
  pip install sacrebleu

使用
----
  python bleu_test.py --ref ref.txt --hyp hyp.txt
  python bleu_test.py --ref ref.txt --hyp hyp.txt --out results/ --name de_test
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import sacrebleu


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------

def load_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


# ---------------------------------------------------------------------------
# BLEU 计算
# ---------------------------------------------------------------------------

def compute_corpus_bleu(hypotheses: List[str], references: List[str]) -> Dict:
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return {
        "score":           round(result.score, 4),
        "precisions":      [round(p, 4) for p in result.precisions],
        "brevity_penalty": round(result.bp, 4),
        "sys_len":         result.sys_len,
        "ref_len":         result.ref_len,
    }


def compute_sentence_bleu(hypothesis: str, reference: str) -> float:
    return round(sacrebleu.sentence_bleu(hypothesis, [reference]).score, 4)


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def build_report(corpus: Dict, records: List[Dict], meta: Dict) -> str:
    sep  = "=" * 60
    sep2 = "-" * 60

    avg = sum(r["bleu"] for r in records) / len(records)

    lines = [
        sep,
        "  翻译质量 BLEU 评估报告",
        f"  Reference : {meta['ref_path']}",
        f"  Hypothesis: {meta['hyp_path']}",
        f"  句子数量  : {meta['n']}",
        f"  测试时间  : {meta['timestamp']}",
        sep,
        "",
        "【Corpus BLEU】",
        f"  得分        : {corpus['score']}",
        f"  1~4gram精度 : {corpus['precisions']}",
        f"  Brevity BP  : {corpus['brevity_penalty']}",
        f"  系统长度    : {corpus['sys_len']}  参考长度: {corpus['ref_len']}",
        "",
        "【句子级 BLEU】",
        f"  平均值      : {round(avg, 4)}",
        f"  最高        : {max(r['bleu'] for r in records)}",
        f"  最低        : {min(r['bleu'] for r in records)}",
        "",
        sep,
        "【逐句得分】",
        f"  {'ID':>4}  {'BLEU':>8}  Reference / Hypothesis",
        sep2,
    ]

    for r in records:
        lines += [
            f"  {r['id']:>4}  {r['bleu']:>8.2f}",
            f"        REF: {r['reference']}",
            f"        HYP: {r['hypothesis']}",
            sep2,
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run(ref_path: Path, hyp_path: Path, output_dir: Path, name: str) -> Dict:

    references  = load_lines(ref_path)
    hypotheses  = load_lines(hyp_path)

    if len(references) != len(hypotheses):
        raise ValueError(
            f"行数不一致：ref={len(references)} 行，hyp={len(hypotheses)} 行。\n"
            "请确保两个文件行数相同，第 i 行对应同一条句子。"
        )

    n = len(references)
    print(f"[数据] ref={ref_path.name}  hyp={hyp_path.name}  共 {n} 条\n")

    # Corpus BLEU
    corpus = compute_corpus_bleu(hypotheses, references)
    print(f"[结果] Corpus BLEU = {corpus['score']}\n")

    # 逐句 BLEU
    records = [
        {
            "id":         i + 1,
            "reference":  ref,
            "hypothesis": hyp,
            "bleu":       compute_sentence_bleu(hyp, ref),
        }
        for i, (hyp, ref) in enumerate(zip(hypotheses, references))
    ]

    meta = {
        "ref_path":  str(ref_path),
        "hyp_path":  str(hyp_path),
        "n":         n,
        "timestamp": datetime.now().isoformat(),
    }

    result = {
        "meta":         meta,
        "corpus_bleu":  corpus,
        "sentence_avg": round(sum(r["bleu"] for r in records) / n, 4),
        "sentence_max": max(r["bleu"] for r in records),
        "sentence_min": min(r["bleu"] for r in records),
        "records":      records,
    }

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem      = name if name else f"{ref_path.stem}_vs_{hyp_path.stem}"
    json_path = output_dir / f"{stem}_{ts}.json"
    txt_path  = output_dir / f"{stem}_{ts}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    report = build_report(corpus, records, meta)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n[保存] {json_path}")
    print(f"[保存] {txt_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="翻译质量 BLEU 评估")
    parser.add_argument("--ref",  "-r", required=True,
                        help="原句翻译结果 txt（每行一条）")
    parser.add_argument("--hyp",  "-y", required=True,
                        help="脱敏句翻译结果 txt（每行一条）")
    parser.add_argument("--out",  "-o", default="bleu_results",
                        help="输出目录（默认 ./bleu_results）")
    parser.add_argument("--name", "-n", default="",
                        help="输出文件名前缀")
    args = parser.parse_args()

    run(
        ref_path   = Path(args.ref),
        hyp_path   = Path(args.hyp),
        output_dir = Path(args.out),
        name       = args.name,
    )


if __name__ == "__main__":
    main()
