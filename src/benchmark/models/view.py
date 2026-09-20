"""Human-readable reporting for a benchmark run, kept separate from runner.py.

Colour carries meaning here: green is a fact the model got right, red is one it
needed and missed, yellow is one it read but did not need.
"""

from __future__ import annotations

import re
import textwrap

from benchmark.models.runners.base import Instance, TaskResult

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"

LABEL_WIDTH = 14
WRAP_WIDTH = 90
BAR_WIDTH = 8


def _color(text: str, *codes: str) -> str:
    return f"{''.join(codes)}{text}{RESET}"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _natural_key(fact_id: str) -> tuple[str, int]:
    """Read f2 before f10, the way a person scanning the list expects."""
    match = re.fullmatch(r"([^\d]*)(\d+)", fact_id)
    return (match.group(1), int(match.group(2))) if match else (fact_id, 0)


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def _score_color(value: float) -> str:
    if value >= 0.9:
        return GREEN
    return YELLOW if value >= 0.6 else RED


def _meter(value: float, width: int = BAR_WIDTH) -> str:
    """A small bar plus the number, both graded by how good the score is."""
    filled = round(value * width)
    bar = _color("█" * filled, _score_color(value)) if filled else ""
    if filled < width:
        bar += _color("░" * (width - filled), DIM)
    return f"{bar} {_color(f'{value:.2f}', _score_color(value))}"


def _rows(label: str, lines: list[str]) -> str:
    """A `label   value` block; `lines` are already coloured and wrapped."""
    indent = " " * (LABEL_WIDTH + 3)
    head = f"  {_color(f'{label:<{LABEL_WIDTH}}', DIM)} {lines[0]}"
    return "\n".join([head, *(f"{indent}{line}" for line in lines[1:])])


def _field(label: str, text: str, *codes: str) -> str:
    """Same, for plain prose - wrapped first so escape codes never skew the width."""
    lines = textwrap.wrap(text, WRAP_WIDTH - LABEL_WIDTH - 3) or [""]
    return _rows(label, [_color(line, *codes) for line in lines] if codes else lines)


class RunView:
    """Prints one readable block per instance, plus a final summary."""

    def __init__(self, config_name: str, runner_name: str, total: int) -> None:
        self.total = total
        self.results: list[TaskResult] = []
        print()
        print(
            f"{_color(config_name, BOLD)} {_color('·', DIM)} {_color(runner_name, CYAN)} "
            f"{_color('·', DIM)} {_color(_plural(total, 'instance'), DIM)}"
        )
        print(_color("─" * WRAP_WIDTH, DIM))

    def print_instance(self, index: int, instance: Instance, result: TaskResult) -> None:
        self.results.append(result)
        print()
        print(self._render_header(index, instance, result))
        if result.error:
            print(_field("error", result.error, RED))
            return

        facts = {fact["id"]: fact["text"] for fact in instance.state}
        expected = sorted(result.expected_ids or [], key=_natural_key)
        predicted = sorted(result.predicted_ids or [], key=_natural_key)
        missed = sorted(set(expected) - set(predicted), key=_natural_key)
        over_read = sorted(set(predicted) - set(expected), key=_natural_key)

        print(_field("task", instance.task))
        print(_field("action", instance.proposed_action, CYAN))
        print(_rows("expected", self._id_lines(expected, set(predicted), RED)))
        print(_rows("predicted", self._id_lines(predicted, set(expected), YELLOW)))
        self._print_facts("missed", "the model did not list", missed, facts, RED)
        self._print_facts("over-read", "the model added", over_read, facts, YELLOW)
        print(_rows("scores", [self._format_scores(result)]))
        print(_field("cost", self._format_cost(result), DIM))
        why = result.meta.get("why")
        if why:
            print(_field("why", why, DIM))

    def print_summary(self, total_duration_s: float) -> None:
        n = len(self.results)
        if not n:
            return
        exact = sum(
            1 for r in self.results if set(r.predicted_ids or []) == set(r.expected_ids or [])
        )
        print()
        print(_color("─" * WRAP_WIDTH, DIM))
        print(_color("Summary", BOLD))
        print(_rows("instances", [str(n)]))
        print(_rows("exact match", [f"{_meter(exact / n, 16)}  {_color(f'{exact}/{n}', DIM)}"]))
        print(_rows("mean precision", [_meter(self._mean("precision"), 16)]))
        print(_rows("mean recall", [_meter(self._mean("recall"), 16)]))
        print(_rows("mean F1", [_meter(self._mean("f1"), 16)]))
        print(
            _field(
                "time",
                f"{_format_duration(total_duration_s)} total, "
                f"{_format_duration(total_duration_s / n)} per instance",
                DIM,
            )
        )

    def _mean(self, attr: str) -> float:
        values = [getattr(r, attr) for r in self.results]
        return sum(values) / len(values)

    def _render_header(self, index: int, instance: Instance, result: TaskResult) -> str:
        if result.error:
            badge = _color("⚠ error", BOLD, RED)
        elif set(result.predicted_ids or []) == set(result.expected_ids or []):
            badge = _color("✓ exact match", BOLD, GREEN)
        else:
            badge = _color("✗ mismatch", BOLD, RED)
        counter = _color(f"[{index}/{self.total}]", DIM)
        return f"{counter} {_color(instance.instance_id, BOLD)}  {badge}"

    def _id_lines(self, ids: list[str], hits: set[str], miss_color: str) -> list[str]:
        """Fact ids, wrapped on their plain width, each one green when it hit."""
        if not ids:
            return [_color("(none)", DIM)]

        groups: list[list[str]] = [[]]
        width = 0
        for fact_id in ids:
            if groups[-1] and width + len(fact_id) + 2 > WRAP_WIDTH - LABEL_WIDTH - 3:
                groups.append([])
                width = 0
            groups[-1].append(fact_id)
            width += len(fact_id) + 2

        lines = [
            ", ".join(_color(i, GREEN) if i in hits else _color(i, miss_color) for i in group)
            for group in groups
        ]
        continued = [line + _color(",", DIM) for line in lines[:-1]]
        return [*continued, lines[-1]]

    def _print_facts(
        self, label: str, verb: str, ids: list[str], facts: dict[str, str], color: str
    ) -> None:
        if not ids:
            return
        print(_rows(label, [_color(f"{_plural(len(ids), 'fact')} {verb}", color)]))
        indent = " " * (LABEL_WIDTH + 3)
        for fact_id in ids:
            text = facts.get(fact_id) or _color("(no such fact in the state)", DIM)
            print(f"{indent}{_color(f'{fact_id:<5}', color)} {text}")

    def _format_scores(self, result: TaskResult) -> str:
        return (
            f"{_color('precision', DIM)} {_meter(result.precision)}   "
            f"{_color('recall', DIM)} {_meter(result.recall)}   "
            f"{_color('F1', DIM)} {_meter(result.f1)}"
        )

    def _format_cost(self, result: TaskResult) -> str:
        parts = []
        if result.total_duration_s is not None:
            parts.append(f"{_format_duration(result.total_duration_s)} of wall clock")
        if result.prompt_eval_count is not None:
            parts.append(f"{result.prompt_eval_count:,} prompt tokens")
        if result.eval_count is not None:
            parts.append(f"{result.eval_count:,} output tokens")
        return " · ".join(parts) if parts else "n/a"
