"""Lenient multiple-choice scorer that falls back to pattern matching
when the model doesn't output the strict 'ANSWER: X' format."""

import re
from inspect_ai._util.answer import answer_character, answer_index
from inspect_ai.scorer._metric import CORRECT, INCORRECT
from inspect_ai.scorer._metrics import accuracy, stderr
from inspect_ai.scorer._scorer import Score, Scorer, scorer
from inspect_ai.scorer._target import Target
from inspect_ai.solver._task_state import TaskState


def extract_answer_letter(completion: str, n_choices: int) -> str | None:
    """Find the last choice letter in the completion.

    Prefer 'X)' format, fall back to standalone 'X'.
    """
    valid = ''.join(answer_character(i) for i in range(n_choices))
    # Prefer X) format
    matches = re.findall(rf'([{valid}])\)', completion)
    if matches:
        return matches[-1]
    # Fall back to standalone X
    matches = re.findall(rf'\b([{valid}])\b', completion)
    return matches[-1] if matches else None


@scorer(metrics=[accuracy(), stderr()])
def lenient_choice() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        choices = state.choices

        # The multiple_choice solver marks choices when its strict "ANSWER: X"
        # parsing succeeds. We check that first, and fall back to lenient
        # extraction when it didn't.
        marked = [i for i, c in enumerate(choices) if c.correct is True]
        if not marked:
            letter = extract_answer_letter(state.output.completion, len(choices))
            if letter:
                idx = answer_index(letter)
                if 0 <= idx < len(choices):
                    marked = [idx]

        # choices[i].original_position handles shuffle remapping (identity if unshuffled)
        answer = [answer_character(choices[i].original_position) for i in marked]

        return Score(
            value=CORRECT if sorted(answer) == sorted(target.text) else INCORRECT,
            answer=", ".join(answer),
            explanation=state.output.completion,
        )

    return score
