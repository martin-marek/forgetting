"""Medical O1 evaluation with free-form generation and LLM-as-judge scoring.

Uses the FreedomIntelligence/medical-o1-verifiable-problem dataset and the
judge prompt from the SDFT paper (Shenfeld et al., 2026).
"""

import random
import re
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, scorer, accuracy, CORRECT, INCORRECT
from inspect_ai.solver import generate
from inspect_ai.model import get_model


JUDGE_PROMPT = """You are an expert medical evaluator assessing whether a model's response correctly answers a medical question. Your task is to compare the model's response to the reference answer and determine if the model's response is:
1. CORRECT: The response contains the key medical information from the reference answer, even if phrased differently or includes additional correct medical details.
2. INCORRECT: The response is medically wrong, misses the main point, or provides incorrect medical information.
Focus on medical accuracy and completeness, not on writing style or verbosity.

[Medical Question]
{question}

[Reference Answer]
{reference_answer}

[Model Response]
{model_response}

Evaluate the model's response. Output ONLY one of: "CORRECT" or "INCORRECT"."""


@scorer(metrics=[accuracy()])
def medical_llm_judge(judge_model: str = None):
    """Score medical answers using an LLM judge."""

    async def score(state, target):
        judge = get_model(judge_model) if judge_model else get_model()

        model_answer = state.output.completion if state.output else ""
        question = state.metadata["question"]
        reference_answer = target.text

        prompt = JUDGE_PROMPT.format(
            question=question,
            reference_answer=reference_answer,
            model_response=model_answer,
        )

        result = await judge.generate(prompt)
        judgment = result.completion.strip().upper()

        correct = "CORRECT" in judgment and "INCORRECT" not in judgment

        return Score(
            value=CORRECT if correct else INCORRECT,
            explanation=f"Judge: {judgment}\nReference: {reference_answer}\nModel: {model_answer}",
        )

    return score


def load_medical_dataset(limit=1000, seed=42):
    """Load the medical-o1-verifiable-problem dataset.

    Args:
        limit: Maximum number of questions to sample (default 1000, matching the paper).
        seed: Random seed for reproducible sampling.
    """
    from datasets import load_dataset

    random.seed(seed)
    ds = load_dataset("FreedomIntelligence/medical-o1-verifiable-problem", split="train")

    records = list(ds)
    random.shuffle(records)
    if limit is not None:
        records = records[:limit]

    samples = []
    for record in records:
        question = record["Open-ended Verifiable Question"]
        answer = record["Ground-True Answer"]
        samples.append(Sample(
            input=question,
            target=answer,
            metadata={"question": question},
        ))

    return samples


@task
def medical_o1(limit: int = 1000, judge_model: str = None):
    """Evaluate model on medical reasoning using HuatuoGPT-o1 verifiable problems.

    Args:
        limit: Maximum number of questions to evaluate (default 1000).
        judge_model: Model to use for judging (e.g., "openai/gpt-4.1-nano"). If None, uses same model.
    """
    dataset = load_medical_dataset(limit=limit)

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=medical_llm_judge(judge_model=judge_model),
    )


if __name__ == "__main__":
    samples = load_medical_dataset(limit=5)
    print(f"Loaded {len(samples)} samples")
    for s in samples[:3]:
        print(f"\nQ: {s.input[:100]}...")
        print(f"Target: {s.target}")
