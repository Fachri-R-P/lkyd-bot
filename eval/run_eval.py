import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # biar bisa import rag.py dari root

from rag import DEFAULT_MODEL, answer, format_context, get_client  # noqa: E402

EVAL_DIR = ROOT / "eval"
K = 5
METRICS = ["faithfulness", "relevance", "persona"]
PAUSE_SECONDS = 2  # jeda biar nggak kena rate limit Groq gratisan

JUDGE_SYSTEM = """\
You are a strict evaluator of a chatbot that role-plays Lee Kuan Yew using RAG.
Score the ANSWER from 1 (very poor) to 5 (excellent) on:
- faithfulness: is the answer supported by the provided EXCERPTS, with no invented \
quotes or facts? If the excerpts are not enough, the answer should say it is reasoning \
from known views. For category "post_2015", the answer must admit that he died in 2015 \
and cannot know the event. For category "out_of_scope", the answer must not pretend \
the excerpts cover the topic.
- relevance: does it actually answer the question and cover the expected KEY POINTS?
- persona: does it sound like Lee Kuan Yew (first person, blunt, pragmatic, uses \
Singapore's experience and history)?
Reply with JSON only: {"faithfulness": int, "relevance": int, "persona": int, \
"reason": "one or two short sentences"}"""


def judge(item, result):
    user = (
        f"CATEGORY: {item['category']}\n"
        f"QUESTION: {item['question']}\n"
        f"KEY POINTS: {json.dumps(item['key_points'])}\n\n"
        f"EXCERPTS:\n{format_context(result['sources'])}\n\n"
        f"ANSWER:\n{result['answer']}"
    )
    response = get_client().chat.completions.create(
        model=os.environ.get("GROQ_JUDGE_MODEL", os.environ.get("GROQ_MODEL", DEFAULT_MODEL)),
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(response.choices[0].message.content)
        # jaga-jaga kalau judge ngasih angka di luar 1-5
        scores = {m: min(5, max(1, int(data[m]))) for m in METRICS}
        scores["reason"] = str(data.get("reason", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        scores = {m: None for m in METRICS}
        scores["reason"] = f"judge output invalid: {e}"
    return scores


def mean(values):
    values = [v for v in values if v is not None]
    return f"{sum(values) / len(values):.2f}" if values else "n/a"


def hit_rate(rows):
    # soal tanpa expected_source (hit = None) nggak dihitung
    graded = [r for r in rows if r["hit"] is not None]
    if not graded:
        return "n/a"
    hits = sum(r["hit"] for r in graded)
    return f"{hits}/{len(graded)} ({hits / len(graded):.0%})"


def write_markdown(rows, path):
    categories = sorted({r["category"] for r in rows})
    lines = [
        "# Evaluation results",
        "",
        f"- Questions: {len(rows)}",
        f"- Retrieval hit rate (expected source in top-{K}): {hit_rate(rows)}",
        "",
        "| Category | N | Hit rate | Faithfulness | Relevance | Persona |",
        "|---|---|---|---|---|---|",
    ]
    for name, group in [("**all**", rows)] + [(c, [r for r in rows if r["category"] == c]) for c in categories]:
        lines.append(
            f"| {name} | {len(group)} | {hit_rate(group)} | "
            + " | ".join(mean([r[m] for r in group]) for m in METRICS)
            + " |"
        )
    lines += ["", "Scores are 1-5, given by an LLM judge. See results.csv for per-question details.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    rows = []
    for item in questions:
        print(f"[{item['id']}/{len(questions)}] {item['question']}")
        result = answer(item["question"], k=K)
        retrieved = [s["source_file"] for s in result["sources"]]
        expected = item.get("expected_source")
        scores = judge(item, result)
        rows.append(
            {
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "expected_source": expected or "",
                "hit": (expected in retrieved) if expected else None,
                "retrieved": " | ".join(retrieved),
                **scores,
                "answer": result["answer"],
            }
        )
        time.sleep(PAUSE_SECONDS)

    with open(EVAL_DIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, EVAL_DIR / "results.md")
    print("Saved eval/results.csv and eval/results.md")


if __name__ == "__main__":
    main()
