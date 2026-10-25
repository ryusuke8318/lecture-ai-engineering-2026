"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --model gemini-2.5-flash-lite    
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）

応答時間（最初のトークンまで / 全体）とトークン数（入力・出力・思考）も表示する。
複数モデル・複数質問の比較は app/ bench.py を使う。
"""

import argparse
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

@dataclass
class Result:
    text: str
    model: str
    ttft_s: float | None  # 最初のテキストが届くまで
    total_s: float  # 全文が届くまで
    input_tokens: int
    output_tokens: int  # 回答本文のトークン
    thought_tokens: int  # 思考トークン

    @property
    def billed_output_tokens(self) -> int:
        return self.output_tokens + self.thought_tokens

    @property
    def tokens_per_s(self) -> float | None:
        """生成フェーズの出力速度"""
        if self.ttft_s is None or self.total_s <= self.ttft_s:
            return None
        return self.billed_output_tokens / (self.total_s - self.ttft_s)

def ask(question: str, model: str = MODEL, client: genai.Client | None = None) -> Result:
    client = client or genai.Client(api_key=get_api_key())

    start = time.perf_counter()
    ttft = None
    parts: list[str] = []
    usage = None
    for chunk in client.models.generate_content_stream(model=model, contents=question):
        if chunk.text and ttft is None:
            ttft = time.perf_counter() - start
        parts.append(chunk.text or "")
        usage = chunk.usage_metadata or usage

    total = time.perf_counter() - start
    
    return Result(
        text = "".join(parts),
        model=model,
        ttft_s=ttft,
        total_s=total,
        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
        output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        thought_tokens=(usage.thoughts_token_count or 0) if usage else 0
    )

def format_stats(r: Result) -> str:
    ttft = f"{r.ttft_s:.2f}s" if r.ttft_s is not None else "-"
    tps = f"{r.tokens_per_s:.0f} token/s" if r.tokens_per_s else "-"

    return (
        f"[計測] TTFT={ttft} 全体={r.total_s:.2f}s 速度={tps} | "
        f"入力={r.input_tokens} 出力={r.output_tokens} 思考={r.thought_tokens} "
        f"(出力枠合計={r.billed_output_tokens})"
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    print(f"[model] {args.model}")
    print(f"[質問] {args.question}\n")
    try:
        result = ask(args.question, args.model)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)
    print(result.text)
    print(f"\n{format_stats(result)}")


if __name__ == "__main__":
    main()
