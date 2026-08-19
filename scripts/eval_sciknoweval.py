"""Standalone mean@N accuracy eval for SciKnowEval, against a plain vLLM
OpenAI-compatible server (not prime-rl's orchestrator).

Usage (from the prime-rl repo root, with a vLLM server already running):
    uv run --no-sync python3 eval_sciknoweval.py \
        --base-url http://localhost:8000/v1 --model <served-model-name> \
        --n-samples 1 --max-questions 0 --out results.json

Loads the held-out `split="eval"` rows via the same `sciknoweval` taskset
code used in training (so the question/system-prompt formatting is
identical to what the model saw during RL), sends each to the server's
`/v1/chat/completions` endpoint, extracts the predicted letter with the
same `extract_mcq_answer` used for the training reward, and reports
accuracy overall and per domain.
"""

import argparse
import asyncio
import json
from collections import defaultdict

from openai import AsyncOpenAI
from sciknoweval.mcq import extract_mcq_answer
from sciknoweval.taskset import SciKnowEvalConfig, SciKnowEvalTaskset
import verifiers.v1 as vf


async def score_one(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    question: str,
    answer_key: str,
    domain: str,
    n_samples: int,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    async with semaphore:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=max_tokens,
            n=n_samples,
            temperature=1.0,
        )
    rows = []
    for choice in completion.choices:
        reply = choice.message.content or ""
        predicted = extract_mcq_answer(reply)
        rows.append(
            {
                "domain": domain,
                "answer_key": answer_key,
                "predicted": predicted,
                "correct": predicted == answer_key,
            }
        )
    return rows


async def main_async(args: argparse.Namespace) -> None:
    cfg = SciKnowEvalConfig(split="eval", task=vf.TaskConfig())
    tasks = SciKnowEvalTaskset(cfg).load()
    if args.max_questions > 0:
        tasks = tasks[: args.max_questions]
    print(f"eval_sciknoweval: {len(tasks)} held-out questions, n_samples={args.n_samples}")

    client = AsyncOpenAI(base_url=args.base_url, api_key="dummy")
    semaphore = asyncio.Semaphore(args.concurrency)
    jobs = [
        score_one(
            client,
            args.model,
            task.data.system_prompt,
            task.data.prompt,
            task.data.answer_key,
            task.data.domain,
            args.n_samples,
            args.max_tokens,
            semaphore,
        )
        for task in tasks
    ]
    results: list[dict] = []
    done = 0
    for coro in asyncio.as_completed(jobs):
        rows = await coro
        results.extend(rows)
        done += 1
        if done % 50 == 0:
            print(f"eval_sciknoweval: {done}/{len(tasks)} questions done")

    overall = sum(r["correct"] for r in results) / len(results)
    per_domain: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        per_domain[r["domain"]].append(r["correct"])
    per_domain_acc = {d: sum(v) / len(v) for d, v in per_domain.items()}

    summary = {
        "model": args.model,
        "n_questions": len(tasks),
        "n_samples": args.n_samples,
        "n_rows": len(results),
        "overall_accuracy": overall,
        "per_domain_accuracy": per_domain_acc,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "rows": results}, f, indent=2)
        print(f"eval_sciknoweval: wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="served model name (as vLLM was launched with)")
    parser.add_argument("--n-samples", type=int, default=1, help="samples per question")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-questions", type=int, default=0, help="0 = all held-out questions")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--out", default=None, help="optional path to write full JSON results")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
