"""複数モデル x 複数質問で応答時間・トークン数を計測し、無料枠レート制限との関係を見る。

使い方:
    uv run python -m app.bench --list-models          # 使えるモデル名を確認
    uv run python -m app.bench                        # 既定のモデル・質問で1回ずつ
    uv run python -m app.bench --repeat 3 --models gemini-3.5-flash-lite gemini-2.5-flash
    uv run python -m app.bench --no-pace              # 間隔調整なし（429の挙動を見る）

結果は bench_results.csv に追記される。
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

from google import genai
from google.genai import errors

from app.ask import Result, ask
from app.config import get_api_key

# 比較候補
DEFAULT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

# (RPM, TPM, RPD)  不明なら None
FREE_LIMITS: dict[str, tuple[int | None, int | None, int | None]] = {
    "gemini-2.5-flash-lite": (15, 250_000, 1_000),
    "gemini-2.5-flash": (10, 250_000, 250),
}

# 質問セット: (ID, 狙い, 質問)
QUESTIONS = [
    ("fact", "短い事実確認（最速ケース）", "日本の首都と、その人口のおおよその規模を1文で答えて。"),
    ("local", "固有知識（幻覚の確認）", "AIE2026の講座の宿題にある「3レベル」とは何ですか？"),
    ("reason", "推論（思考トークンが増える）",
     "A,B,Cの3人がいる。Aは「Bは嘘つき」、Bは「Cは嘘つき」、Cは「AとBは両方嘘つき」と言う。"
     "正直者は本当のことだけ、嘘つきは嘘だけを言う。誰が正直者か、理由とともに答えて。"),
    ("code", "コード生成（中程度の出力）",
     "Pythonで、文字列リストから重複を除いて出現順を保つ関数を書き、テストを3つ付けて。"),
    ("long_out", "長文出力（出力速度を見る）", "LoRAとフルファインチューニングの違いを、表を含めて800字程度で説明して。"),
    ("long_in", "長い入力（入力TPMを消費）",
     ("RAGの利点は、外部知識を検索して回答に反映できることである。" * 300)
     + "\n\n上の文章を要約して、重複を除いた要点を1行で示して。"),
]

CSV_FIELDS = ["ts", "model", "qid", "run", "status", "ttft_s", "total_s", "tps",
              "input_tokens", "output_tokens", "thought_tokens", "retries"]


def list_models(client: genai.Client) -> None:
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            print(m.name.removeprefix("models/"))


def call_with_retry(client, model: str, question: str, max_retries: int = 3) -> tuple[Result | None, str, int]:
    """429/503 は指数バックオフで再試行。(結果, status, 再試行回数) を返す。"""
    retries = 0
    while True:
        try:
            return ask(question, model, client), "ok", retries
        except errors.APIError as e:
            if e.code in (429, 503) and retries < max_retries:
                wait = 2 ** retries * 5
                print(f"    {e.code} -> {wait}s待って再試行")
                time.sleep(wait)
                retries += 1
                continue
            return None, f"error:{e.code}", retries


def summarize(rows: list[dict], wall_s: float) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    print("\n=== モデル別（成功分の中央値）===")
    print(f"{'model':26} {'n':>3} {'TTFT':>6} {'全体s':>7} {'tok/s':>6} {'入力':>7} {'出力':>6} {'思考':>6} {'429等':>5}")
    for model in dict.fromkeys(r["model"] for r in rows):
        mr = [r for r in ok if r["model"] == model]
        fails = sum(1 for r in rows if r["model"] == model and r["status"] != "ok")
        if not mr:
            print(f"{model:26} 全件失敗 ({fails})")
            continue
        med = lambda k: statistics.median(float(r[k]) for r in mr if r[k] != "")  # noqa: E731
        tps = [float(r["tps"]) for r in mr if r["tps"] != ""]
        print(f"{model:26} {len(mr):>3} {med('ttft_s'):>6.2f} {med('total_s'):>7.2f} "
              f"{(statistics.median(tps) if tps else 0):>6.0f} {med('input_tokens'):>7.0f} "
              f"{med('output_tokens'):>6.0f} {med('thought_tokens'):>6.0f} {fails:>5}")

    print("\n=== 無料枠との関係（今回の実測ペース）===")
    for model in dict.fromkeys(r["model"] for r in rows):
        mr = [r for r in rows if r["model"] == model]
        wall_min = max(wall_s / 60, 1e-9)
        # モデルごとの正確な区間は取らず、全体時間で割る保守的な目安
        tokens = sum(int(r["input_tokens"] or 0) for r in mr)
        rpm_obs = len(mr) / wall_min
        tpm_obs = tokens / wall_min
        rpm, tpm, rpd = FREE_LIMITS.get(model, (None, None, None))
        line = f"{model:26} 実測 {rpm_obs:5.1f} req/min, 入力 {tpm_obs:8.0f} tok/min"
        if rpm:
            line += f" | RPM上限{rpm}: {'超過' if rpm_obs > rpm else 'OK'}"
        if tpm:
            line += f", TPM上限{tpm}: {'超過' if tpm_obs > tpm else 'OK'}"
        if rpd:
            per_run = len(mr)
            line += f" | 1日{rpd}回なら今回の規模を約{rpd // per_run}回"
        print(line)


def main() -> None:
    p = argparse.ArgumentParser(description="Geminiモデルの応答時間・トークン数を比較")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--questions", nargs="+", default=[q[0] for q in QUESTIONS],
                   help="使う質問ID: " + ", ".join(q[0] for q in QUESTIONS))
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--no-pace", action="store_true", help="RPM上限に合わせた待機をしない")
    p.add_argument("--out", default="bench_results.csv")
    p.add_argument("--list-models", action="store_true")
    args = p.parse_args()

    client = genai.Client(api_key=get_api_key())
    if args.list_models:
        list_models(client)
        return

    qs = [q for q in QUESTIONS if q[0] in args.questions]
    out = Path(args.out)
    new_file = not out.exists()
    rows: list[dict] = []
    t0 = time.perf_counter()
    last_call: dict[str, float] = {}

    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            w.writeheader()
        # 質問 -> 回 -> モデルの順に回し、同一モデルへの連続呼び出しを減らす
        for qid, purpose, question in qs:
            for run in range(1, args.repeat + 1):
                for model in args.models:
                    rpm = FREE_LIMITS.get(model, (None,))[0]
                    if rpm and not args.no_pace and model in last_call:
                        wait = 60 / rpm - (time.perf_counter() - last_call[model])
                        if wait > 0:
                            time.sleep(wait)
                    print(f"[{qid}#{run}] {model} ...", flush=True)
                    last_call[model] = time.perf_counter()
                    r, status, retries = call_with_retry(client, model, question)
                    row = {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model, "qid": qid,
                        "run": run, "status": status, "retries": retries,
                        "ttft_s": "", "total_s": "", "tps": "", "input_tokens": "",
                        "output_tokens": "", "thought_tokens": "",
                    }
                    if r:
                        row.update(
                            ttft_s=f"{r.ttft_s:.3f}" if r.ttft_s is not None else "",
                            total_s=f"{r.total_s:.3f}",
                            tps=f"{r.tokens_per_s:.1f}" if r.tokens_per_s else "",
                            input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                            thought_tokens=r.thought_tokens,
                        )
                        print(f"    {r.total_s:.2f}s in={r.input_tokens} out={r.output_tokens} think={r.thought_tokens}")
                    else:
                        print(f"    失敗: {status}")
                    w.writerow(row)
                    f.flush()
                    rows.append(row)

    summarize(rows, time.perf_counter() - t0)
    print(f"\n詳細: {out}")


if __name__ == "__main__":
    main()