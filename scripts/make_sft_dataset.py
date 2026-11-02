"""第2回 宿題（発展）: 用語集から SFT 用の Q&A データ（JSONL）を作る。

使い方:
    uv run python scripts/make_sft_dataset.py > glossary_sft.jsonl
    uv run python scripts/make_sft_dataset.py --with-unknown > glossary_sft_unknown.jsonl

できたファイルを Colab にアップロードし、LoRA 学習のデータとして使う。

設計:
- 質問の言い回し（TRAIN_TEMPLATES）は学習用の 3 種類。評価用の言い回し（data/glossary_eval.json）は
  学習に使わない。言い回しを変えた質問に答えられるかを、暗記と区別して測るため。
- 別名（aliases）でも質問を作り、用語の表記ゆれに対応させる。
- --with-unknown は「用語集にない語には、知らないと答える」例を足す。足さない場合との差が、
  考察の材料になる（足さないと、学習後のモデルは知らない語にも定義を断定しやすい）。
"""

import argparse
import json
import sys
from pathlib import Path

GLOSSARY = Path(__file__).resolve().parent.parent / "data" / "glossary.json"

# 評価（Colab 側）と同じ system メッセージ。学習と評価で入力の形を揃える
SYSTEM = "あなたは講座用語集の回答係です。用語集に載っていることだけを、日本語で簡潔に答えてください。載っていない語は、載っていないと答えてください。"

TRAIN_TEMPLATES = [
    "「{term}」とは何ですか？",
    "{term}について簡単に説明してください。",
    "{term}の意味を教えてください。",
]

# --with-unknown 用。data/glossary_eval.json の「用語集にない語」とは重ならない語だけを置く
UNKNOWN_TERMS = [
    "ディープラーニング", "ニューラルネットワーク", "勾配降下法", "バッチサイズ",
    "プロンプトキャッシュ", "ベンチマーク", "転移学習", "マルチモーダル",
]
UNKNOWN_ANSWER = "「{term}」は用語集に記載がありません。確認が必要です。"


def record(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def build(with_unknown: bool) -> list[dict]:
    terms = json.loads(GLOSSARY.read_text(encoding="utf-8"))["terms"]
    records = []
    for t in terms:
        for template in TRAIN_TEMPLATES:
            records.append(record(template.format(term=t["term"]), t["definition"]))
        for alias in t.get("aliases", []):
            records.append(record(TRAIN_TEMPLATES[0].format(term=alias), t["definition"]))
    if with_unknown:
        for term in UNKNOWN_TERMS:
            for template in TRAIN_TEMPLATES:
                records.append(record(template.format(term=term), UNKNOWN_ANSWER.format(term=term)))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-unknown", action="store_true", help="用語集にない語に「記載がありません」と答える例を足す")
    args = parser.parse_args()
    for r in build(args.with_unknown):
        sys.stdout.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
