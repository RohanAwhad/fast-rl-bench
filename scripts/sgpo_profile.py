"""Offline sGPO profiling pass (paper §4.1): for every query in the task's
*train* split, generate N=8 parallel samples under the *initial* policy (the
same checkpoint the training run starts from) and compute the empirical
success rate p̂(q) = n_success / N. That single signal drives all three
training-time decisions (data selection, adaptive group size, curriculum) —
see prime_rl_patch/src/prime_rl/efficient_rl/sgpo.py.

Scoring reuses the exact reward code paths training uses: for sciknoweval
the same `extract_mcq_answer` letter-match as the training reward, for
reverse-text the same `<reversed_text>` tag parse + LCS ratio, binarized at
`--success-threshold` (the paper assumes a binary verifiable reward; the
continuous LCS reward is binarized at 0.5 by default — documented fidelity
note). Success-rate thresholds follow the paper: trivial = p̂ > 0.75
(removed), unsolved = p̂ = 0, learnable = 0 < p̂ ≤ 0.75 bucketed into
G ∈ {2,4,8} by the paper's Eq. 10.

Usage (from the prime-rl repo root, with a vLLM server already running):
    uv run --no-sync python3 scripts/sgpo_profile.py \
        --task reverse_text --base-url http://localhost:18100/v1 \
        --model reverse_text-sgpo-profile --out sgpo_profiles/reverse_text.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from difflib import SequenceMatcher

from openai import AsyncOpenAI

from sciknoweval.mcq import extract_mcq_answer

N_SAMPLES = 8
TRIVIAL_THRESHOLD = 0.75
BUCKETS = (2, 4, 8)


def load_tasks(task: str) -> tuple[list, str, str]:
    if task == "reverse_text":
        from reverse_text.taskset import ReverseTextConfig, ReverseTextTaskset, SYSTEM

        return ReverseTextTaskset(ReverseTextConfig()).load(), "reverse_text", SYSTEM
    if task == "sciknoweval":
        import verifiers.v1 as vf
        from sciknoweval.taskset import SciKnowEvalConfig, SciKnowEvalTaskset

        tasks = SciKnowEvalTaskset(SciKnowEvalConfig(split="train")).load()
        return tasks, "sciknoweval", ""
    raise SystemExit(f"unknown task: {task}")


def reverse_text_success(reply: str, answer: str, threshold: float) -> bool:
    import re

    from reverse_text.taskset import _TAG

    match = _TAG.search(reply or "")
    response = match.group(1).strip() if match else ""
    return SequenceMatcher(None, response, answer).ratio() >= threshold


async def score_one(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    prompt: str,
    task_name: str,
    task,
    threshold: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> list[bool]:
    async with semaphore:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            n=N_SAMPLES,
            temperature=1.0,
        )
    successes = []
    for choice in completion.choices:
        reply = choice.message.content or ""
        if task_name == "reverse_text":
            successes.append(reverse_text_success(reply, task.data.answer, threshold))
        else:
            successes.append(extract_mcq_answer(reply) == task.data.answer_key)
    return successes


def classify(p_hat: float) -> tuple[str, int | None]:
    if p_hat > TRIVIAL_THRESHOLD:
        return "trivial", None
    if p_hat == 0.0:
        return "unsolved", None
    if p_hat > 1 / 4:
        return "learnable", 2
    if p_hat > 1 / 8:
        return "learnable", 4
    return "learnable", 8


async def main_async(args: argparse.Namespace) -> None:
    tasks, task_name, system_prompt_default = load_tasks(args.task)
    print(f"sgpo_profile: task={task_name} n_queries={len(tasks)} n_samples={N_SAMPLES} "
          f"success_threshold={args.success_threshold} max_tokens={args.max_tokens}")

    resume_idx: set[int] = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            resume_idx = {int(json.loads(line)["idx"]) for line in f if line.strip()}
        print(f"sgpo_profile: resuming, {len(resume_idx)} queries already profiled")

    client = AsyncOpenAI(base_url=args.base_url, api_key="dummy")
    semaphore = asyncio.Semaphore(args.concurrency)

    jobs = []
    for task in tasks:
        idx = task.data.idx
        if idx in resume_idx:
            continue
        system_prompt = task.data.system_prompt if task_name == "sciknoweval" else system_prompt_default
        jobs.append(
            (
                idx,
                task.data.prompt,
                score_one(
                    client, args.model, system_prompt, task.data.prompt,
                    task_name, task, args.success_threshold, args.max_tokens,
                    semaphore,
                ),
            )
        )

    out_f = open(args.out, "a" if args.resume else "w")
    counts: Counter[str] = Counter()
    bucket_counts: Counter[int] = Counter()
    n_done = 0
    results = await asyncio.gather(*[coro for _, _, coro in jobs])
    for (idx, prompt, _coro), successes in zip(jobs, results):
        n_success = sum(successes)
        p_hat = n_success / N_SAMPLES
        decision, bucket = classify(p_hat)
        counts[decision] += 1
        if bucket is not None:
            bucket_counts[bucket] += 1
        row = {
            "idx": idx,
            "prompt": prompt,
            "n_success": n_success,
            "p_hat": p_hat,
            "bucket": bucket,
            "decision": decision,
        }
        out_f.write(json.dumps(row) + "\n")
        n_done += 1
        if n_done % 500 == 0:
            out_f.flush()
            print(f"sgpo_profile: {n_done}/{len(jobs)} queries done")
    out_f.close()

    total = len(tasks)
    summary = {
        "task": task_name,
        "n_queries": total,
        "n_samples": N_SAMPLES,
        "success_threshold": args.success_threshold,
        "n_profiled": n_done + len(resume_idx),
        "counts": dict(counts),
        "bucket_counts": {str(k): v for k, v in bucket_counts.items()},
    }
    print(json.dumps(summary, indent=2))
    if args.summary_out:
        with open(args.summary_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"sgpo_profile: wrote {args.summary_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, choices=["reverse_text", "sciknoweval"])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True, help="served model name (as vLLM was launched with)")
    parser.add_argument("--success-threshold", type=float, default=0.5,
                        help="LCS ratio at/above which a reverse-text sample counts as solved (paper assumes binary rewards)")
    parser.add_argument("--max-tokens", type=int, default=0, help="0 = task default (128 reverse_text / 256 sciknoweval)")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--out", required=True, help="JSONL output path (one row per query)")
    parser.add_argument("--summary-out", default=None, help="optional JSON summary path")
    parser.add_argument("--resume", action="store_true", help="skip idx already present in --out")
    args = parser.parse_args()

    if args.max_tokens == 0:
        args.max_tokens = 128 if args.task == "reverse_text" else 256
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
