"""sciknoweval — multi-domain scientific-knowledge MCQ (biology / chemistry /
material / physics), single-turn.

Task: answer multiple-choice science questions with the correct letter,
scored by deterministic (0/1) letter-match — a genuinely binary reward,
unlike reverse-text's continuous LCS ratio.

Dataset: `hicai-zju/SciKnowEval` (subset "v2", 28,392 rows). The Hub only
ships a single "test" split (no train split), so a train/eval carve-up is
done here: a fixed-seed shuffle, then the first `held_out_per_domain` rows
*of each domain* go to `split="eval"`, the rest to `split="train"` — stable
across process restarts since the seed and per-domain counts are fixed.

Restricted to the two MCQ types (`mcq-4-choices`, `mcq-2-choices` — ~66% of
the dataset) since these are the only types with a machine-checkable
`answerKey`.
"""

import verifiers.v1 as vf

from sciknoweval.mcq import extract_mcq_answer

_MCQ_TYPES = ("mcq-4-choices", "mcq-2-choices")


class SciKnowEvalData(vf.TaskData):
    answer_key: str
    """Gold option letter (A-D for 4-choice questions, A-B for 2-choice)."""

    solution: str
    """'<letter>: <choice text>' — more informative than the bare letter,
    restates the correct option's content, not just its index."""

    domain: str
    """Biology / Chemistry / Material / Physics."""


class SciKnowEvalTask(vf.Task[SciKnowEvalData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        # SciKnowEval MCQ is single-turn: refuse a second turn so the model answers once.
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def exact_match(self, trace: vf.Trace) -> float:
        prediction = extract_mcq_answer(trace.last_reply or "")
        return 1.0 if prediction == self.data.answer_key else 0.0


class SciKnowEvalConfig(vf.TasksetConfig):
    dataset_name: str = "hicai-zju/SciKnowEval"
    dataset_config: str = "v2"
    dataset_split: str = "test"
    """The only split the Hub dataset ships; train/eval is carved out below."""

    held_out_per_domain: int = 200
    """Rows per domain reserved for `split="eval"` (see module docstring)."""

    split: str = "train"
    """Which half of the fixed train/eval carve-up this taskset instance serves."""


class SciKnowEvalTaskset(vf.Taskset[SciKnowEvalTask, SciKnowEvalConfig]):
    def load(self) -> list[SciKnowEvalTask]:
        from datasets import load_dataset

        cfg = self.config
        rows = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.dataset_split)
        rows = rows.filter(lambda r: r["type"] in _MCQ_TYPES)
        rows = rows.shuffle(seed=42)  # fixed so the train/eval carve-up is stable across loads

        seen_per_domain: dict[str, int] = {}
        train_indices: list[int] = []
        eval_indices: list[int] = []
        for i, domain in enumerate(rows["domain"]):
            seen = seen_per_domain.get(domain, 0)
            seen_per_domain[domain] = seen + 1
            (eval_indices if seen < cfg.held_out_per_domain else train_indices).append(i)
        selected = rows.select(eval_indices if cfg.split == "eval" else train_indices)

        tasks: list[SciKnowEvalTask] = []
        for i, row in enumerate(selected):
            labels: list[str] = row["choices"]["label"]
            texts: list[str] = row["choices"]["text"]
            label_to_text = dict(zip(labels, texts))
            options = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
            question = f"{row['question']}\n\n{options}"
            tasks.append(
                SciKnowEvalTask(
                    SciKnowEvalData(
                        idx=i,
                        prompt=question,
                        system_prompt=row["prompt"]["default"],
                        answer_key=row["answerKey"],
                        solution=f"{row['answerKey']}: {label_to_text[row['answerKey']]}",
                        domain=row["domain"],
                    ),
                    cfg.task,
                )
            )
        return tasks
