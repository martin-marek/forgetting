"""Email QA evaluation with free-form generation and LLM-as-judge scoring."""

import random
import re
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, scorer, accuracy, CORRECT, INCORRECT
from inspect_ai.solver import generate
from inspect_ai.model import get_model
from sympy import limit


JUDGE_PROMPT = """You are given a model's answer and a set of answer choices. Your job is to determine which answer choice the model's response most closely matches.

Model's answer: {model_answer}

Answer choices:
{options}

Which answer choice (letter only) best matches the model's answer? Respond with a single letter."""


@scorer(metrics=[accuracy()])
def llm_judge(judge_model: str = None):
    """Score answers using an LLM judge that matches freeform answers to MCQ options."""

    async def score(state, target):
        judge = get_model(judge_model) if judge_model else get_model()

        model_answer = state.output.completion if state.output else ""
        choices = state.metadata["choices"]

        options = "\n".join(
            f"{chr(ord('A') + i)}: {choice}" for i, choice in enumerate(choices)
        )

        prompt = JUDGE_PROMPT.format(
            model_answer=model_answer,
            options=options,
        )

        result = await judge.generate(prompt)
        judgment = result.completion.strip().upper()

        # Extract the letter from the judge's response
        match = re.search(r"[A-Z]", judgment)
        matched_letter = match.group(0) if match else ""

        correct = matched_letter == target.text

        return Score(
            value=CORRECT if correct else INCORRECT,
            explanation=f"Judge matched: {matched_letter}, Correct: {target.text}\nModel answer: {model_answer}",
        )

    return score


def record_to_freeform_samples(record):
    """Convert a dataset record to free-form QA samples with MCQ choices for judging."""
    samples = []

    questions = record["questions"]
    gold_answers = record["gold_answers"]
    incorrect_answers = record["incorrect_answers"]

    for i, (question, gold_answer, incorrects) in enumerate(
        zip(questions, gold_answers, incorrect_answers)
    ):
        choices = [gold_answer] + incorrects
        random.shuffle(choices)
        correct_idx = choices.index(gold_answer)

        samples.append(Sample(
            input=question,
            target=chr(ord("A") + correct_idx),
            metadata={
                "question_idx": i,
                "path": record.get("path", ""),
                "choices": choices,
            }
        ))
    return samples


def load_freeform_dataset(limit=None, seed=42):
    """Load the enron QA dataset for free-form evaluation.

    Args:
        limit: Maximum number of emails to evaluate
        seed: Random seed for shuffling
        train_split: HuggingFace split string - should match training data
    """
    from datasets import load_dataset

    random.seed(seed)
    split = 'train' if limit is None else f'train[:{limit}]'
    ds = load_dataset("MichaelR207/enron_qa_0922", split=split)

    samples = []
    for record in ds:
        samples.extend(record_to_freeform_samples(record))

    random.shuffle(samples)

    return samples


@task
def email_qa_freeform(limit: int = None, judge_model: str = None, train_split: str = "train[:100]"):
    """Evaluate model on email QA task with free-form generation.

    Args:
        limit: Maximum number of emails to evaluate
        judge_model: Model to use for judging (e.g., "openai/gpt-5-mini"). If None, uses same model.
        train_split: HuggingFace split string - should match what was used for training
    """
    dataset = load_freeform_dataset(limit=limit)

    return Task(
        dataset=dataset,
        solver=[
            generate(),
        ],
        scorer=llm_judge(judge_model=judge_model),
    )


if __name__ == "__main__":
    # Quick test
    samples = load_freeform_dataset(limit=5)
    print(f"Loaded {len(samples)} samples")
    for s in samples[:3]:
        print(f"\nQ: {s.input[:80]}...")
        print(f"Target: {s.target}")
        print(f"Choices: {s.metadata['choices']}")
