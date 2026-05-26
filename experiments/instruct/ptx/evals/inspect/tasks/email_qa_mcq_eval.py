import random
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import generate, multiple_choice


def record_to_samples(record):
    """Convert a dataset record to inspect samples (one per question)."""
    samples = []
    questions = record["questions"]
    gold_answers = record["gold_answers"]
    incorrect_answers = record["incorrect_answers"]

    for i, (question, gold_answer, incorrects) in enumerate(
        zip(questions, gold_answers, incorrect_answers)
    ):
        # Build choices: gold answer + incorrect answers, then shuffle
        choices = [gold_answer] + incorrects
        random.shuffle(choices)
        correct_idx = choices.index(gold_answer)

        samples.append(Sample(
            input=question,
            choices=choices,
            target=chr(ord("A") + correct_idx),
            metadata={
                "question_idx": i,
                "path": record.get("path", ""),
            }
        ))
    return samples


def load_email_qa_dataset(limit=None, seed=42):
    """Load the enron QA dataset and flatten to one sample per question."""
    from datasets import load_dataset

    random.seed(seed)
    split = 'train' if limit is None else f'train[:{limit}]'
    ds = load_dataset("MichaelR207/enron_qa_0922", split=split)

    samples = []
    for record in ds:
        samples.extend(record_to_samples(record))

    return samples


@task
def email_qa_mcq(limit: int = None):
    """Evaluate model on email QA task.

    Tests whether the model has learned email content during finetuning
    by asking questions without providing the email context.
    """
    dataset = load_email_qa_dataset(limit=limit)

    return Task(
        dataset=dataset,
        solver=[
            multiple_choice(),
            generate(),
        ],
        scorer=choice(),
    )


if __name__ == "__main__":
    from collections import Counter
    samples = load_email_qa_dataset(limit=5)
    targets = Counter(s.target for s in samples)
    print(f"Target distribution (should be ~uniform): {dict(targets)}")
